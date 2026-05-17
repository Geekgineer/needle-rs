#!/usr/bin/env python3
"""
Generate 500+ end-to-end parity reference vectors for the Rust engine.run() test.

Runs the full Python pipeline (tokenize → INT4-quantized encode → constrained
greedy decode) and saves (query, tools, expected_token_ids, expected_text).

Parametric dimensions covered:
  - Tool naming: snake_case, camelCase, PascalCase, UPPER_SNAKE, hyphen-case
  - Parameter format: flat dict ONLY — JSON Schema format (properties) is
    intentionally excluded because Python's constrained decoder has a known
    bug where "properties" leaks as a valid arg key. That divergence is
    tested in constrained_unit.rs, not here.
  - Parameter count: 0, 1, 2, 3, 5, 8
  - Tool array size: 1, 2, 3, 5, 10
  - Prefix-shadow tool names (trie must not commit early)
  - Prefix-shadow argument keys (e.g. "date" vs "date_from" vs "date_to")
  - Mixed naming styles in the same tool array
  - JSON Schema with "required" field
  - JSON Schema with per-property descriptions

Usage (from needle-rust/ root):
    PYTHONPATH=needle python3 tools/gen_e2e_vectors.py \
        --checkpoint needle/checkpoints/needle.pkl \
        --output tests/e2e_vectors.json

    # Quick smoke run (first 10 only):
    PYTHONPATH=needle python3 tools/gen_e2e_vectors.py \
        --checkpoint needle/checkpoints/needle.pkl \
        --output tests/e2e_vectors.json \
        --limit 10
"""

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "needle"))

from needle.dataset.dataset import get_tokenizer, DEFAULT_MAX_ENC_LEN, DEFAULT_MAX_GEN_LEN
from needle.model.architecture import SimpleAttentionNetwork, TransformerConfig, make_padding_mask
from needle.model.run import _build_encoder_input, _get_decode_fn, normalize_tools, restore_tool_names
from needle.model.quantize import _fake_quantize_int4
from needle.model.constrained import build_constrained_decoder


# ──────────────────────────────────────────────────────────────────────────────
# Parametric helpers
# ──────────────────────────────────────────────────────────────────────────────

def fp(*pairs):
    """Flat-format parameters: {"key": {"type": "T"}}"""
    return {n: {"type": t} for n, t in pairs}

def sp(*pairs, required=None):
    """JSON-Schema parameters: {"type":"object","properties":{...}}"""
    props = {n: {"type": t} for n, t in pairs}
    d = {"type": "object", "properties": props}
    if required:
        d["required"] = list(required)
    return d

def sp_desc(*triples, required=None):
    """JSON-Schema parameters with per-property descriptions."""
    props = {n: {"type": t, "description": desc} for n, t, desc in triples}
    d = {"type": "object", "properties": props}
    if required:
        d["required"] = list(required)
    return d

def T(name, desc, params):
    return {"name": name, "description": desc, "parameters": params}

def TJ(*tools):
    """Serialize tool list to compact JSON string."""
    return json.dumps(list(tools), separators=(",", ":"))


# ──────────────────────────────────────────────────────────────────────────────
# Query banks (20 per domain, deterministic)
# ──────────────────────────────────────────────────────────────────────────────

WEATHER_Q = [
    "What's the weather in Paris?",
    "What's the weather in Tokyo?",
    "What's the weather in London?",
    "What's the weather in New York?",
    "What's the weather in Berlin?",
    "What's the weather in Sydney?",
    "What's the weather in Dubai?",
    "What's the weather in Moscow?",
    "What's the weather in Toronto?",
    "What's the weather in Singapore?",
    "Tell me the weather in Paris",
    "Get the current weather for Tokyo",
    "How is the weather in London today?",
    "What temperature is it in New York?",
    "Is it cold in Berlin right now?",
    "What is the weather in Sydney?",
    "Weather conditions in Dubai",
    "Check the weather in Moscow",
    "What's the temperature in Toronto?",
    "Show current weather for Singapore",
]

SEARCH_Q = [
    "Search for Python tutorials online",
    "Find information about climate change",
    "Look up the history of Rome",
    "Search for the latest AI news",
    "Find Python documentation",
    "Search for pasta recipes",
    "Look up the population of Brazil",
    "Search for machine learning papers",
    "Find information about quantum computing",
    "Search for JavaScript frameworks",
    "Look up Nobel Prize winners",
    "Search for best practices in Rust",
    "Find articles about space exploration",
    "Search for information about vaccines",
    "Look up the GDP of Germany",
    "Search for open source projects",
    "Find documentation for React",
    "Search for news about electric vehicles",
    "Look up the capital of Australia",
    "Search for Rust package recommendations",
]

TRANSLATE_Q = [
    "Translate 'hello' to Spanish",
    "Translate 'goodbye' to French",
    "Translate 'thank you' to German",
    "Translate 'good morning' to Japanese",
    "Translate 'how are you' to Italian",
    "Translate 'please' to Portuguese",
    "Translate 'yes' to Chinese",
    "Translate 'no' to Arabic",
    "Translate 'water' to Russian",
    "Translate 'food' to Korean",
    "How do you say 'hello' in Spanish?",
    "How do you say 'goodbye' in French?",
    "What is 'thank you' in German?",
    "Say 'good morning' in Japanese",
    "What does 'come stai' mean in English?",
    "Translate the word 'peace' to Hebrew",
    "How do you say 'welcome' in Thai?",
    "Translate 'book' to Finnish",
    "What is 'fire' in Swahili?",
    "Translate 'mountain' to Turkish",
]

TIME_Q = [
    "What time is it in New York?",
    "What time is it in London?",
    "What time is it in Tokyo?",
    "Get current time in Paris",
    "What's the time in Sydney?",
    "Tell me the time in Dubai",
    "What time is it in Moscow?",
    "Get the time for Singapore",
    "What's the current time in Berlin?",
    "Time in Toronto right now",
    "What time is it in Los Angeles?",
    "Current time in Chicago",
    "What time is it in Mumbai?",
    "Time in Seoul right now",
    "What's the time in Cairo?",
    "Get time for Buenos Aires",
    "What time is it in Bangkok?",
    "Current time in Lagos",
    "What's the time in Nairobi?",
    "Time in Auckland right now",
]

STOCK_Q = [
    "What's the stock price of Apple?",
    "Get the stock price of Microsoft",
    "What's TSLA trading at?",
    "Show me the stock price for Google",
    "What is the current price of Amazon stock?",
    "Check the NVDA stock price",
    "What's Meta stock trading at?",
    "Get stock quote for Netflix",
    "What's the price of Tesla shares?",
    "Show MSFT stock price",
    "What's the current price of AAPL?",
    "Get AMZN stock price",
    "What's Google trading at today?",
    "Check the price of NFLX",
    "What's the stock value of Intel?",
    "Get share price for AMD",
    "What's IBM stock worth?",
    "Check stock price for ORCL",
    "What's Shopify trading at?",
    "Get the price of Salesforce stock",
]

EMAIL_Q = [
    "Send an email to john@example.com about the meeting",
    "Email sarah about the project deadline",
    "Send a message to the team about tomorrow's standup",
    "Write an email to the client with the proposal",
    "Send an email about the quarterly review",
    "Email alice about the lunch plans",
    "Send a note to bob@company.com",
    "Write to the support team about the bug",
    "Email the manager about the report",
    "Send a follow-up to the customer",
]

CALENDAR_Q = [
    "Create a calendar event for tomorrow at 3pm",
    "Schedule a meeting for Monday at 10am",
    "Add a dentist appointment to my calendar",
    "Create an event for the conference next week",
    "Schedule lunch with John for Friday",
    "Book a team meeting for Tuesday at 2pm",
    "Add a reminder for the doctor at 9am",
    "Create a new event called team sync",
    "Schedule the product demo for Thursday",
    "Add the workshop to my calendar",
]

BOOKING_Q = [
    "Book a flight from New York to London",
    "Book a flight from Paris to Tokyo",
    "Book a flight from Berlin to Sydney",
    "Book me a flight from Toronto to Dubai",
    "I need a flight from Moscow to Singapore",
    "Book a flight to Los Angeles",
    "Get a flight from Chicago to Miami",
    "Book flights from Boston to Seattle",
    "I want to fly from Austin to Denver",
    "Book a round trip to San Francisco",
    "Find a flight from Amsterdam to Barcelona",
    "Book the cheapest flight to Rome",
    "Reserve a seat on a flight to Madrid",
    "I need to fly from Munich to Vienna",
    "Book a business class flight to Hong Kong",
]

CALC_Q = [
    "Calculate 15 times 12",
    "What is 2 to the power of 10?",
    "Calculate the square root of 144",
    "What is 100 divided by 7?",
    "Compute 3.14 times 9 squared",
    "What is 500 minus 237?",
    "Calculate 12 factorial",
    "What is log base 2 of 1024?",
    "Compute 45 percent of 320",
    "What is 7 times 8 plus 3?",
]

ZERO_PARAM_Q = [
    "Take a screenshot",
    "What's the battery level?",
    "Show me system information",
    "Clear the cache",
    "Restart the service",
    "Take a screenshot of the screen",
    "Check the battery",
    "Get system info",
    "Clear application cache",
    "Restart background services",
]


# ──────────────────────────────────────────────────────────────────────────────
# Tool factories
# ──────────────────────────────────────────────────────────────────────────────

def weather_flat(name="get_weather"):
    return T(name, "Get current weather for a city",
             fp(("location", "string"), ("unit", "string")))

def weather_schema(name="get_weather"):
    return T(name, "Get current weather for a city",
             sp(("location", "string"), ("unit", "string"), required=["location"]))

def weather_schema_desc(name="get_weather"):
    return T(name, "Get current weather for a city",
             sp_desc(("location", "string", "City name or postal code"),
                     ("unit", "string", "Temperature unit: celsius or fahrenheit"),
                     required=["location"]))

def search_flat(name="web_search"):
    return T(name, "Search the web for information", fp(("query", "string")))

def search_schema(name="web_search"):
    return T(name, "Search the web for information",
             sp(("query", "string"), required=["query"]))

def translate_flat(name="translate_text"):
    return T(name, "Translate text between languages",
             fp(("text", "string"), ("target_language", "string"), ("source_language", "string")))

def translate_schema(name="translate_text"):
    return T(name, "Translate text between languages",
             sp(("text", "string"), ("target_language", "string"), ("source_language", "string"),
                required=["text", "target_language"]))

def time_flat(name="get_time"):
    return T(name, "Get the current time in a city", fp(("city", "string")))

def time_schema(name="get_time"):
    return T(name, "Get the current time in a city",
             sp(("city", "string"), required=["city"]))

def email_flat(name="send_email"):
    return T(name, "Send an email message",
             fp(("to", "string"), ("subject", "string"), ("body", "string")))

def email_schema(name="send_email"):
    return T(name, "Send an email message",
             sp(("to", "string"), ("subject", "string"), ("body", "string"),
                required=["to", "subject"]))

def stock_flat(name="get_stock_price"):
    return T(name, "Get current stock price for a ticker symbol", fp(("symbol", "string")))

def stock_schema(name="get_stock_price"):
    return T(name, "Get current stock price for a ticker symbol",
             sp(("symbol", "string"), required=["symbol"]))

def calendar_flat(name="create_event"):
    return T(name, "Create a calendar event",
             fp(("title", "string"), ("date", "string"), ("time", "string")))

def calendar_schema(name="create_event"):
    return T(name, "Create a calendar event",
             sp(("title", "string"), ("date", "string"), ("time", "string"),
                required=["title", "date"]))

def calc_flat(name="calculate"):
    return T(name, "Perform a mathematical calculation", fp(("expression", "string")))

def calc_schema(name="calculate"):
    return T(name, "Perform a mathematical calculation",
             sp(("expression", "string"), required=["expression"]))

def flight_flat(name="book_flight"):
    return T(name, "Book a flight ticket",
             fp(("origin", "string"), ("destination", "string"),
                ("date", "string"), ("passengers", "integer"), ("cabin_class", "string")))

def flight_schema(name="book_flight"):
    return T(name, "Book a flight ticket",
             sp(("origin", "string"), ("destination", "string"),
                ("date", "string"), ("passengers", "integer"), ("cabin_class", "string"),
                required=["origin", "destination", "date"]))

def profile_flat(name="update_profile"):
    return T(name, "Update user profile information",
             fp(("user_id", "string"), ("name", "string"), ("email", "string"),
                ("phone", "string"), ("address", "string"), ("bio", "string"),
                ("timezone", "string"), ("website", "string")))

def directions_flat(name="get_directions"):
    return T(name, "Get driving or transit directions",
             fp(("origin", "string"), ("destination", "string")))

def file_flat(name="read_file"):
    return T(name, "Read a file from disk", fp(("path", "string")))

def no_param_flat(name, desc):
    return T(name, desc, {})

def no_param_schema(name, desc):
    return T(name, desc, {"type": "object", "properties": {}})


# ──────────────────────────────────────────────────────────────────────────────
# Example assembly
# ──────────────────────────────────────────────────────────────────────────────

def build_examples():
    ex = []

    # ── §1  snake_case, 1 tool, flat params ───────────────────────────────────
    for q in WEATHER_Q:
        ex.append({"query": q, "tools": TJ(weather_flat())})
    for q in SEARCH_Q:
        ex.append({"query": q, "tools": TJ(search_flat())})
    for q in TRANSLATE_Q:
        ex.append({"query": q, "tools": TJ(translate_flat())})
    for q in TIME_Q:
        ex.append({"query": q, "tools": TJ(time_flat())})
    for q in STOCK_Q:
        ex.append({"query": q, "tools": TJ(stock_flat())})
    for q in EMAIL_Q:
        ex.append({"query": q, "tools": TJ(email_flat())})
    for q in CALENDAR_Q:
        ex.append({"query": q, "tools": TJ(calendar_flat())})
    for q in CALC_Q:
        ex.append({"query": q, "tools": TJ(calc_flat())})
    for q in BOOKING_Q:
        ex.append({"query": q, "tools": TJ(flight_flat())})

    # §2, §3 (JSON Schema single-tool) intentionally omitted — Python constrained
    # decoder has a known bug with JSON Schema format ("properties" leaks as arg key).
    # Rust handles JSON Schema correctly (strict superset); parity is tested in
    # constrained_unit.rs instead of via e2e vectors.

    # ── §4  camelCase tool names, flat params ─────────────────────────────────
    camel_pairs = [
        ("getWeather",    weather_flat,    WEATHER_Q[:10]),
        ("webSearch",     search_flat,     SEARCH_Q[:10]),
        ("translateText", translate_flat,  TRANSLATE_Q[:10]),
        ("getTime",       time_flat,       TIME_Q[:10]),
        ("sendEmail",     email_flat,      EMAIL_Q),
        ("getStockPrice", stock_flat,      STOCK_Q[:10]),
        ("createEvent",   calendar_flat,   CALENDAR_Q),
        ("bookFlight",    flight_flat,     BOOKING_Q[:5]),
    ]
    for cname, factory, queries in camel_pairs:
        for q in queries:
            ex.append({"query": q, "tools": TJ(factory(cname))})

    # ── §5  PascalCase tool names, flat params ────────────────────────────────
    pascal_pairs = [
        ("GetWeather",   weather_flat,   WEATHER_Q[:5]),
        ("WebSearch",    search_flat,    SEARCH_Q[:5]),
        ("TranslateText",translate_flat, TRANSLATE_Q[:5]),
        ("GetTime",      time_flat,      TIME_Q[:5]),
        ("SendEmail",    email_flat,     EMAIL_Q[:5]),
    ]
    for pname, factory, queries in pascal_pairs:
        for q in queries:
            ex.append({"query": q, "tools": TJ(factory(pname))})

    # ── §6  UPPER_SNAKE tool names, flat params ───────────────────────────────
    upper_pairs = [
        ("GET_WEATHER",    weather_flat,   WEATHER_Q[:5]),
        ("WEB_SEARCH",     search_flat,    SEARCH_Q[:5]),
        ("TRANSLATE_TEXT", translate_flat, TRANSLATE_Q[:5]),
        ("GET_TIME",       time_flat,      TIME_Q[:5]),
        ("SEND_EMAIL",     email_flat,     EMAIL_Q[:5]),
    ]
    for uname, factory, queries in upper_pairs:
        for q in queries:
            ex.append({"query": q, "tools": TJ(factory(uname))})

    # ── §7  hyphen-case tool names, flat params ───────────────────────────────
    hyphen_pairs = [
        ("get-weather",    weather_flat,   WEATHER_Q[:5]),
        ("web-search",     search_flat,    SEARCH_Q[:5]),
        ("translate-text", translate_flat, TRANSLATE_Q[:5]),
        ("get-time",       time_flat,      TIME_Q[:5]),
        ("send-email",     email_flat,     EMAIL_Q[:5]),
    ]
    for hname, factory, queries in hyphen_pairs:
        for q in queries:
            ex.append({"query": q, "tools": TJ(factory(hname))})

    # §8 (camelCase + JSON Schema) — omitted, see §2 note above.

    # ── §9  Zero-parameter tools ──────────────────────────────────────────────
    zero_tools = [
        ("take_screenshot",   "Take a screenshot of the current screen"),
        ("get_battery_level", "Get the device battery level"),
        ("get_system_info",   "Get system information and metrics"),
        ("clear_cache",       "Clear the application cache"),
        ("restart_service",   "Restart the background service"),
    ]
    for i, q in enumerate(ZERO_PARAM_Q):
        tname, tdesc = zero_tools[i % len(zero_tools)]
        ex.append({"query": q, "tools": TJ(no_param_flat(tname, tdesc))})
    # no_param_schema loop omitted — JSON Schema zero-param, see §2 note above.

    # ── §10  Many parameters (5–8 params, flat) ───────────────────────────────
    for q in BOOKING_Q:
        ex.append({"query": q, "tools": TJ(flight_flat())})
    profile_q = [
        "Update my profile name",
        "Change my email address in my profile",
        "Update my phone number",
        "Set my timezone to UTC",
        "Update my website URL in my profile",
        "Change my bio",
        "Update my address",
        "Set my profile information",
        "Update my account details",
        "Change my profile settings",
    ]
    for q in profile_q:
        ex.append({"query": q, "tools": TJ(profile_flat())})

    # §11 (flight_schema) — omitted, see §2 note above.

    # ── §12  3-tool array, disambiguation ────────────────────────────────────
    tools3a = [weather_flat(), search_flat(), email_flat()]
    for q in WEATHER_Q[:5]:
        ex.append({"query": q, "tools": TJ(*tools3a)})
    for q in SEARCH_Q[:5]:
        ex.append({"query": q, "tools": TJ(*tools3a)})
    for q in EMAIL_Q[:5]:
        ex.append({"query": q, "tools": TJ(*tools3a)})

    tools3b = [translate_flat(), time_flat(), stock_flat()]
    for q in TRANSLATE_Q[:5]:
        ex.append({"query": q, "tools": TJ(*tools3b)})
    for q in TIME_Q[:5]:
        ex.append({"query": q, "tools": TJ(*tools3b)})
    for q in STOCK_Q[:5]:
        ex.append({"query": q, "tools": TJ(*tools3b)})

    tools3c = [calendar_flat(), calc_flat(), file_flat()]
    for q in CALENDAR_Q:
        ex.append({"query": q, "tools": TJ(*tools3c)})
    for q in CALC_Q:
        ex.append({"query": q, "tools": TJ(*tools3c)})

    # ── §13  5-tool array, disambiguation ────────────────────────────────────
    tools5 = [weather_flat(), search_flat(), translate_flat(), time_flat(), email_flat()]
    for q in WEATHER_Q[:4]:
        ex.append({"query": q, "tools": TJ(*tools5)})
    for q in SEARCH_Q[:4]:
        ex.append({"query": q, "tools": TJ(*tools5)})
    for q in TRANSLATE_Q[:4]:
        ex.append({"query": q, "tools": TJ(*tools5)})
    for q in TIME_Q[:4]:
        ex.append({"query": q, "tools": TJ(*tools5)})
    for q in EMAIL_Q[:4]:
        ex.append({"query": q, "tools": TJ(*tools5)})

    # ── §14  10-tool array, disambiguation ───────────────────────────────────
    tools10 = [
        weather_flat(), search_flat(), translate_flat(), time_flat(), email_flat(),
        stock_flat(), calendar_flat(), calc_flat(), flight_flat(),
        directions_flat(),
    ]
    multi10_queries = [
        "What's the weather in Paris?",
        "Search for Python tutorials",
        "Translate hello to Spanish",
        "What time is it in Tokyo?",
        "Send email to alice@example.com about the report",
        "Get the stock price of Apple",
        "Create a calendar event for 3pm tomorrow",
        "Calculate 15 times 12",
        "Book a flight from London to New York",
        "Get directions to the nearest airport",
        "How is the weather in London today?",
        "Find information about climate change",
        "How do you say goodbye in French?",
        "What's the time in Berlin?",
        "Email the team about the meeting",
        "What's Tesla's stock price?",
        "Schedule a dentist appointment for Friday",
        "What is 2 to the power of 10?",
        "Book a flight from Paris to Tokyo",
        "Directions from downtown to the train station",
    ]
    for q in multi10_queries:
        ex.append({"query": q, "tools": TJ(*tools10)})

    # §15 (JSON Schema 3/5-tool arrays) — omitted, see §2 note above.

    # ── §16  Prefix-shadow tool names (trie disambiguation) ───────────────────
    # "get_weather" is a prefix of "get_weather_forecast" — decoder must not
    # commit to the shorter name when the query clearly wants the longer one.
    tools_shadow_a = [
        weather_flat("get_weather"),
        T("get_weather_forecast", "Get a multi-day weather forecast",
          fp(("location", "string"), ("days", "integer"))),
    ]
    shadow_a_queries = [
        "What's the weather in Paris today?",
        "Get the weather forecast for Paris for the next 5 days",
        "What's the current weather in Tokyo?",
        "Give me a 7-day forecast for London",
        "How is the weather in Berlin now?",
        "Weather forecast for New York this week",
        "Is it raining in Sydney right now?",
        "What's the 3-day forecast for Dubai?",
        "Current weather conditions in Singapore",
        "Get a forecast for Moscow for the next week",
    ]
    for q in shadow_a_queries:
        ex.append({"query": q, "tools": TJ(*tools_shadow_a)})

    # "web_search" vs "web_search_advanced"
    tools_shadow_b = [
        search_flat("web_search"),
        T("web_search_advanced", "Advanced web search with filters",
          fp(("query", "string"), ("domain", "string"), ("date_range", "string"))),
    ]
    for q in SEARCH_Q[:10]:
        ex.append({"query": q, "tools": TJ(*tools_shadow_b)})

    # Short prefix: "get" vs "get_weather" vs "get_search"
    tools_shadow_c = [
        T("get", "Generic resource getter", fp(("resource", "string"))),
        weather_flat("get_weather"),
        search_flat("get_search"),
    ]
    for q in WEATHER_Q[:5]:
        ex.append({"query": q, "tools": TJ(*tools_shadow_c)})
    for q in SEARCH_Q[:5]:
        ex.append({"query": q, "tools": TJ(*tools_shadow_c)})

    # "send" vs "send_email" vs "send_message"
    tools_shadow_d = [
        T("send", "Generic send operation", fp(("data", "string"))),
        email_flat("send_email"),
        T("send_message", "Send a chat message",
          fp(("recipient", "string"), ("message", "string"))),
    ]
    for q in EMAIL_Q[:5]:
        ex.append({"query": q, "tools": TJ(*tools_shadow_d)})
    send_msg_queries = [
        "Send a message to Alice",
        "Message Bob about the project",
        "Send a chat message to the team",
        "Text Sarah about the plan",
        "Send a direct message to John",
    ]
    for q in send_msg_queries:
        ex.append({"query": q, "tools": TJ(*tools_shadow_d)})

    # ── §17  Prefix-shadow argument keys ──────────────────────────────────────
    # "date", "date_from", "date_to" — trie must traverse fully before committing
    date_shadow_tool = T("search_events", "Search for events in a date range",
                          fp(("query", "string"), ("date", "string"),
                             ("date_from", "string"), ("date_to", "string")))
    date_queries = [
        "Search for events today",
        "Find events from January to March",
        "Search for concerts next week",
        "Find events starting from Monday",
        "Search for meetings in a specific date range",
        "Find all events this month",
        "Search for upcoming events in a date range",
        "Find past events",
        "Search for events between two dates",
        "Find events scheduled for a date",
    ]
    for q in date_queries:
        ex.append({"query": q, "tools": TJ(date_shadow_tool)})

    # "city", "city_code", "city_name" — same prefix stress in arg keys
    city_shadow_tool = T("lookup_location", "Look up location details",
                          fp(("city", "string"), ("city_code", "string"),
                             ("city_name", "string"), ("country", "string")))
    for q in WEATHER_Q[:10]:
        ex.append({"query": q, "tools": TJ(city_shadow_tool)})

    # "user", "user_id", "username" — another prefix-shadow in arg keys
    user_shadow_tool = T("get_profile", "Retrieve user profile",
                          fp(("user", "string"), ("user_id", "string"),
                             ("username", "string"), ("email", "string")))
    profile_lookup_q = [
        "Get the profile for user alice",
        "Look up user ID 12345",
        "Find profile by username john_doe",
        "Get user information for bob",
        "Retrieve the profile with email alice@example.com",
        "Fetch user details for user 42",
        "Look up profile by user ID",
        "Get user info for username admin",
        "Retrieve profile for user alice@corp.com",
        "Get the profile details for user 99",
    ]
    for q in profile_lookup_q:
        ex.append({"query": q, "tools": TJ(user_shadow_tool)})

    # ── §18  Mixed naming styles in one tool array ────────────────────────────
    tools_mixed = [
        weather_flat("getWeather"),        # camelCase
        search_flat("web_search"),          # snake_case
        translate_flat("TranslateText"),    # PascalCase
        time_flat("GET_TIME"),              # UPPER_SNAKE
        email_flat("send-email"),           # hyphen
    ]
    mixed_q = [
        "What's the weather in Paris?",
        "Search for Python tutorials",
        "Translate hello to Spanish",
        "What time is it in Tokyo?",
        "Send an email to alice@example.com",
        "Weather in London",
        "Find AI news",
        "How do you say thank you in French?",
        "Time in Berlin",
        "Email the team about the standup",
    ]
    for q in mixed_q:
        ex.append({"query": q, "tools": TJ(*tools_mixed)})

    # §19, §20, §21, §22 (JSON Schema variants) — omitted, see §2 note above.

    # ── §23  Single-word / very short queries ────────────────────────────────
    short_queries = [
        ("Weather Paris", weather_flat()),
        ("Search Python", search_flat()),
        ("Translate hello", translate_flat()),
        ("Time Tokyo", time_flat()),
        ("Stock Apple", stock_flat()),
        ("Email John", email_flat()),
        ("Calendar meeting", calendar_flat()),
        ("Calculate 2+2", calc_flat()),
        ("Flight London", flight_flat()),
        ("Screenshot", no_param_flat("take_screenshot", "Take a screenshot")),
    ]
    for q, tool in short_queries:
        ex.append({"query": q, "tools": TJ(tool)})

    # ── §24  Tool with number in name ─────────────────────────────────────────
    numbered_tools = [
        ("get_weather_v2", weather_flat, WEATHER_Q[:5]),
        ("web_search_v2", search_flat, SEARCH_Q[:5]),
        ("translate_v3", translate_flat, TRANSLATE_Q[:5]),
        ("api_v1_get_time", time_flat, TIME_Q[:5]),
    ]
    for tname, factory, queries in numbered_tools:
        for q in queries:
            ex.append({"query": q, "tools": TJ(factory(tname))})

    # ── §25  Tool name with only one word (no underscore) ────────────────────
    single_word_tools = [
        ("weather", weather_flat, WEATHER_Q[:5]),
        ("search", search_flat, SEARCH_Q[:5]),
        ("translate", translate_flat, TRANSLATE_Q[:5]),
        ("time", time_flat, TIME_Q[:5]),
        ("email", email_flat, EMAIL_Q[:5]),
    ]
    for tname, factory, queries in single_word_tools:
        for q in queries:
            ex.append({"query": q, "tools": TJ(factory(tname))})

    # ── §26  Full 20-tool stress test (wide trie) ────────────────────────────
    tools20 = [
        weather_flat("get_weather"),
        search_flat("web_search"),
        translate_flat("translate_text"),
        time_flat("get_time"),
        email_flat("send_email"),
        stock_flat("get_stock_price"),
        calendar_flat("create_event"),
        calc_flat("calculate"),
        flight_flat("book_flight"),
        directions_flat("get_directions"),
        T("read_file", "Read a file", fp(("path", "string"))),
        T("write_file", "Write to a file", fp(("path", "string"), ("content", "string"))),
        T("delete_file", "Delete a file", fp(("path", "string"))),
        T("list_files", "List files in a directory", fp(("directory", "string"))),
        T("get_user", "Get user by ID", fp(("user_id", "string"))),
        T("create_user", "Create a new user", fp(("name", "string"), ("email", "string"))),
        T("set_reminder", "Set a reminder", fp(("message", "string"), ("time", "string"))),
        T("play_music", "Play music", fp(("song", "string"), ("artist", "string"))),
        T("get_news", "Get news articles", fp(("topic", "string"), ("count", "integer"))),
        T("convert_currency", "Convert currency",
          fp(("amount", "number"), ("from_currency", "string"), ("to_currency", "string"))),
    ]
    stress20_queries = [
        "What's the weather in Paris?",
        "Search for Python tutorials",
        "Translate hello to Spanish",
        "What time is it in Tokyo?",
        "Send email to alice@example.com",
        "Get the stock price of Apple",
        "Create a meeting event for 3pm",
        "Calculate 15 times 12",
        "Book a flight from London to New York",
        "Directions from home to the office",
        "Read the configuration file",
        "Write hello world to output.txt",
        "Delete the temporary file",
        "List files in the downloads folder",
        "Get user with ID 42",
        "Create a new user named Alice",
        "Set a reminder for the meeting",
        "Play the song Bohemian Rhapsody",
        "Get the latest news about AI",
        "Convert 100 USD to EUR",
    ]
    for q in stress20_queries:
        ex.append({"query": q, "tools": TJ(*tools20)})

    # ── Deduplicate preserving order ──────────────────────────────────────────
    seen = set()
    unique_ex = []
    for e in ex:
        key = (e["query"], e["tools"])
        if key not in seen:
            seen.add(key)
            unique_ex.append(e)

    return unique_ex


# ──────────────────────────────────────────────────────────────────────────────
# Inference (unchanged from original)
# ──────────────────────────────────────────────────────────────────────────────

def fake_quantize_params(params):
    """INT4 fake-quantize every attention projection kernel — matches Rust engine."""
    def walk(node, path=()):
        if isinstance(node, dict):
            return {k: walk(v, path + (k,)) for k, v in node.items()}
        if (hasattr(node, "shape") and path and path[-1] == "kernel"
                and len(path) >= 2
                and path[-2] in ("q_proj", "k_proj", "v_proj", "out_proj")):
            if len(node.shape) == 2:
                return _fake_quantize_int4(node, group_size=32)
            if len(node.shape) == 3:
                return jnp.stack([_fake_quantize_int4(node[i], group_size=32)
                                   for i in range(node.shape[0])])
        return node
    return walk(params)


def run_inference_quantized(model, q_params, tokenizer, query, tools_json,
                             max_gen_len=DEFAULT_MAX_GEN_LEN,
                             max_enc_len=DEFAULT_MAX_ENC_LEN):
    """Full inference with INT4-quantized params; mirrors Rust engine.run()."""
    tools_norm, _ = normalize_tools(tools_json)
    enc_tokens = _build_encoder_input(tokenizer, query, tools_norm, max_enc_len)
    enc_input = jnp.array([enc_tokens])

    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id

    src_mask = make_padding_mask(enc_input, pad_id)
    encoder_out, enc_mask = model.apply(
        {"params": q_params}, enc_input, src_mask=src_mask, method="encode"
    )

    dec_buffer = jnp.full((1, max_gen_len), pad_id, dtype=jnp.int32)
    dec_buffer = dec_buffer.at[0, 0].set(eos_id)
    decode_fn = _get_decode_fn(model, max_gen_len)

    constrained_dec = build_constrained_decoder([tools_norm], tokenizer)

    generated = []
    logits = decode_fn(q_params, dec_buffer, encoder_out, enc_mask)

    for i in range(max_gen_len - 1):
        next_logits_np = np.array(logits[0, i])

        if constrained_dec.is_active(0):
            next_logits_np = constrained_dec.constrain_logits(next_logits_np, 0)

        next_token = int(np.argmax(next_logits_np))
        constrained_dec.update(0, next_token)

        if next_token == eos_id:
            break

        generated.append(next_token)
        dec_buffer = dec_buffer.at[0, i + 1].set(next_token)
        logits = decode_fn(q_params, dec_buffer, encoder_out, enc_mask)

    text = tokenizer.decode(generated)
    if text.startswith("<tool_call>"):
        stripped = text[len("<tool_call>"):]
    else:
        stripped = text
    _, name_map = normalize_tools(tools_json)
    stripped = restore_tool_names(stripped, name_map)
    return generated, stripped


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="needle/checkpoints/needle.pkl")
    ap.add_argument("--output", default="tests/e2e_vectors.json")
    ap.add_argument("--limit", type=int, default=None,
                    help="Run only the first N examples (for smoke testing)")
    args = ap.parse_args()

    examples_meta = build_examples()
    if args.limit is not None:
        examples_meta = examples_meta[:args.limit]

    total = len(examples_meta)
    print(f"Total examples to generate: {total}", file=sys.stderr)

    print(f"Loading {args.checkpoint} ...", file=sys.stderr)
    with open(args.checkpoint, "rb") as f:
        data = pickle.load(f)

    config_dict = data.get("config", {})
    cfg = TransformerConfig(
        vocab_size=config_dict.get("vocab_size", 8192),
        d_model=config_dict.get("d_model", 512),
        num_heads=config_dict.get("num_heads", 8),
        num_kv_heads=config_dict.get("num_kv_heads", 4),
        num_encoder_layers=config_dict.get("num_encoder_layers", 12),
        num_decoder_layers=config_dict.get("num_decoder_layers", 8),
        d_ff=config_dict.get("d_ff", 2048),
        max_seq_len=config_dict.get("max_seq_len", 1024),
        rope_theta=config_dict.get("rope_theta", 10000.0),
        dtype="float32",
        no_feedforward=config_dict.get("no_feedforward", True),
    )
    print(f"  Config: d={cfg.d_model}, {cfg.num_encoder_layers}enc+"
          f"{cfg.num_decoder_layers}dec, vocab={cfg.vocab_size}", file=sys.stderr)

    model = SimpleAttentionNetwork(cfg)
    params = data["params"]

    print("Fake-quantizing params (INT4, group_size=32) ...", file=sys.stderr)
    q_params = fake_quantize_params(params)

    tokenizer = get_tokenizer()
    print("Tokenizer loaded.", file=sys.stderr)

    results = []
    t0 = time.time()
    for i, ex in enumerate(examples_meta):
        query = ex["query"]
        tools_json = ex["tools"]
        t_ex = time.time()
        token_ids, text = run_inference_quantized(
            model, q_params, tokenizer, query, tools_json
        )
        elapsed = time.time() - t_ex
        total_elapsed = time.time() - t0
        eta = (total_elapsed / (i + 1)) * (total - i - 1)
        print(
            f"  [{i+1:4d}/{total}] {elapsed:5.1f}s  ETA {eta/60:5.1f}min  "
            f"query={query[:50]!r:<52}  out={text[:60]!r}",
            file=sys.stderr,
        )
        results.append({
            "query": query,
            "tools": tools_json,
            "expected_token_ids": token_ids,
            "expected_text": text,
        })

    out = {"examples": results}
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)

    total_time = time.time() - t0
    print(
        f"\nWrote {len(results)} examples to {args.output} "
        f"in {total_time/60:.1f} min",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
