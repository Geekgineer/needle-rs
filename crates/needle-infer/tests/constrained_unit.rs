//! Focused unit tests for the constrained decoder.
//!
//! These tests exercise specific code paths that cross-language e2e parity
//! cannot reliably catch:
//!   - Tail buffer rollover and pattern detection across boundaries
//!   - `in_string` flag blocking re-triggers inside JSON value strings
//!   - Multiple argument keys in one call (state cycling)
//!   - Prefix-shadow tool names and argument keys (trie disambiguation)
//!   - Tool name normalisation (camelCase → snake_case, etc.)
//!   - JSON Schema `properties` path vs flat format
//!   - Empty-trie fallback (unconstrained recovery)
//!   - Escaped quote handling inside value strings
//!   - Nested object values (nesting depth tracking)
//!
//! Run: cargo test -p needle-infer --test constrained_unit -- --nocapture

use needle_infer::constrained::{ConstrainedDecoder, JsonState, JsonStateMachine, ToolDef};

// ──────────────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────────────

fn tok(pairs: &[(u32, &[u8])]) -> Vec<(u32, Vec<u8>)> {
    pairs.iter().map(|&(id, b)| (id, b.to_vec())).collect()
}

fn sm(s: &[u8]) -> JsonStateMachine {
    let mut m = JsonStateMachine::new();
    m.feed(s);
    m
}

fn make_decoder(tools_json: &str, vocab: &[(u32, &[u8])]) -> ConstrainedDecoder {
    let tools = ToolDef::from_json(tools_json);
    ConstrainedDecoder::new(&tools, tok(vocab))
}

fn drive(dec: &mut ConstrainedDecoder, bytes: &[u8]) {
    dec.feed_bytes(bytes);
}

// ──────────────────────────────────────────────────────────────────────────────
// JsonStateMachine — state transitions
// These use JsonStateMachine directly because its fields (state,
// current_function, constrained_buf) are pub.
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn sm_initial_state_is_free() {
    let m = JsonStateMachine::new();
    assert_eq!(m.state, JsonState::Free);
    assert!(m.current_function.is_empty());
    assert!(m.constrained_buf.is_empty());
}

#[test]
fn sm_enters_in_name_on_trigger() {
    let m = sm(b"[{\"name\":\"");
    assert_eq!(m.state, JsonState::InName);
    assert!(m.constrained_buf.is_empty());
}

#[test]
fn sm_captures_tool_name_and_returns_to_free() {
    let m = sm(br#"[{"name":"get_weather""#);
    assert_eq!(m.state, JsonState::Free);
    assert_eq!(m.current_function, "get_weather");
}

#[test]
fn sm_enters_in_arg_key_after_arguments_open() {
    // `{"` after `"arguments":{` should trigger InArgKey
    // One byte 'l' starts the arg key
    let m = sm(br#"[{"name":"get_weather","arguments":{"l"#);
    assert_eq!(m.state, JsonState::InArgKey);
    assert_eq!(m.constrained_buf, b"l");
}

#[test]
fn sm_arg_key_closed_on_quote() {
    let mut m = sm(br#"[{"name":"get_weather","arguments":{"location"#);
    assert_eq!(m.state, JsonState::InArgKey);
    m.feed_byte(b'"');
    assert_eq!(m.state, JsonState::Free);
}

#[test]
fn sm_multiple_arg_keys_cycle_correctly() {
    // First key: `location`, second key: `unit` — must cycle Free→InArgKey→Free→InArgKey
    let m = sm(br#"[{"name":"t","arguments":{"location":"Paris","u"#);
    // After `,"u` we should be in InArgKey with constrained_buf = "u"
    assert_eq!(m.state, JsonState::InArgKey);
    assert_eq!(m.constrained_buf, b"u");
}

#[test]
fn sm_in_string_blocks_name_trigger_inside_value() {
    // A value containing the literal `"name":"` must not re-trigger InName
    let output = br#"[{"name":"tool_a","arguments":{"q":"the \"name\":\"x\" text"}}]"#;
    let m = sm(output);
    assert_eq!(m.current_function, "tool_a");
    assert_eq!(m.state, JsonState::Free);
}

#[test]
fn sm_in_string_blocks_args_trigger_inside_value() {
    // A value that literally contains `"arguments":{` must not re-trigger
    let output = br#"[{"name":"tool_a","arguments":{"q":"some \"arguments\":{ data"}}]"#;
    let m = sm(output);
    assert_eq!(m.current_function, "tool_a");
    assert_eq!(m.state, JsonState::Free);
}

#[test]
fn sm_escaped_quote_does_not_close_value_string() {
    // `\"` inside a value should NOT close the string and must not trigger InArgKey
    let output = br#"[{"name":"tool","arguments":{"key":"val\"still_in_string"}}]"#;
    let m = sm(output);
    assert_eq!(m.state, JsonState::Free);
    assert_eq!(m.current_function, "tool");
}

#[test]
fn sm_nested_object_value_tracks_depth() {
    // Arg value is itself an object: nesting_depth must increase/decrease correctly
    // so in_arguments exits only at the right brace depth.
    let output = br#"[{"name":"tool","arguments":{"arg":{"nested":"v"}}}]"#;
    let m = sm(output);
    assert_eq!(m.current_function, "tool");
    assert_eq!(m.state, JsonState::Free);
}

#[test]
fn sm_tail_rollover_before_name_trigger() {
    // Pad with 10 bytes before the NAME_TRIGGER so the trigger arrives in
    // a window that crosses the TAIL_LEN=13 boundary.
    let mut input = b"[{\"x\":\"y\",".to_vec(); // 10 bytes preamble
    input.extend_from_slice(b"\"name\":\"");    // trigger starts at byte 10
    let m = sm(&input);
    assert_eq!(m.state, JsonState::InName, "InName must be entered after preamble");
}

#[test]
fn sm_args_trigger_exactly_fills_tail() {
    // ARGS_TRIGGER = `"arguments":{` = 13 bytes = TAIL_LEN.
    // When the trigger exactly fills the tail buffer, ends_with must still match.
    let m = sm(br#"[{"name":"t","arguments":{"#);
    // The `{"` at the end means we are right at the arg key start.
    // Feed one more byte to confirm InArgKey fires.
    let m2 = sm(br#"[{"name":"t","arguments":{"k"#);
    assert_eq!(m2.state, JsonState::InArgKey);
    let _ = m; // just check no panic
}

#[test]
fn sm_name_trigger_not_fired_inside_arguments_block() {
    // `"name":"` appearing as a value inside `arguments` must not change current_function
    let output = br#"[{"name":"real_tool","arguments":{"key":"\"name\":\"fake\""}}]"#;
    let m = sm(output);
    assert_eq!(m.current_function, "real_tool");
}

#[test]
fn sm_complete_empty_arguments_object() {
    let m = sm(br#"[{"name":"screenshotter","arguments":{}}]"#);
    assert_eq!(m.current_function, "screenshotter");
    assert_eq!(m.state, JsonState::Free);
}

#[test]
fn sm_tool_name_with_numbers() {
    let m = sm(br#"[{"name":"get_weather_v2","arguments":{}}]"#);
    assert_eq!(m.current_function, "get_weather_v2");
}

#[test]
fn sm_tool_name_single_word() {
    let m = sm(br#"[{"name":"search","arguments":{}}]"#);
    assert_eq!(m.current_function, "search");
}

// ──────────────────────────────────────────────────────────────────────────────
// ToolDef::from_json — parsing flat and JSON Schema formats
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn parse_flat_single_tool() {
    let tools = ToolDef::from_json(
        r#"[{"name":"get_weather","description":"Get weather","parameters":{"location":{"type":"string"},"unit":{"type":"string"}}}]"#,
    );
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].snake_name, "get_weather");
    let keys = &tools[0].param_keys;
    assert!(keys.contains(&"location".to_string()));
    assert!(keys.contains(&"unit".to_string()));
    assert!(!keys.contains(&"type".to_string()));
    assert!(!keys.contains(&"description".to_string()));
}

#[test]
fn parse_json_schema_single_tool() {
    let tools = ToolDef::from_json(
        r#"[{"name":"get_weather","description":"Get weather","parameters":{"type":"object","properties":{"location":{"type":"string"},"unit":{"type":"string"}},"required":["location"]}}]"#,
    );
    assert_eq!(tools.len(), 1);
    let keys = &tools[0].param_keys;
    assert!(keys.contains(&"location".to_string()));
    assert!(keys.contains(&"unit".to_string()));
    // Python bug fix: these structural keys must NOT appear as param keys
    assert!(!keys.contains(&"properties".to_string()), "\"properties\" must not be a param key");
    assert!(!keys.contains(&"type".to_string()));
    assert!(!keys.contains(&"required".to_string()));
}

#[test]
fn parse_json_schema_no_required_field() {
    let tools = ToolDef::from_json(
        r#"[{"name":"search","parameters":{"type":"object","properties":{"query":{"type":"string"}}}}]"#,
    );
    assert_eq!(tools[0].param_keys, vec!["query"]);
}

#[test]
fn parse_flat_empty_params() {
    let tools = ToolDef::from_json(
        r#"[{"name":"screenshot","description":"Take screenshot","parameters":{}}]"#,
    );
    assert_eq!(tools.len(), 1);
    assert!(tools[0].param_keys.is_empty());
}

#[test]
fn parse_json_schema_empty_properties() {
    let tools = ToolDef::from_json(
        r#"[{"name":"screenshot","parameters":{"type":"object","properties":{}}}]"#,
    );
    assert_eq!(tools.len(), 1);
    assert!(tools[0].param_keys.is_empty());
}

#[test]
fn parse_multiple_tools_mixed_format() {
    let tools = ToolDef::from_json(
        r#"[
          {"name":"get_weather","parameters":{"location":{"type":"string"}}},
          {"name":"web_search","parameters":{"type":"object","properties":{"query":{"type":"string"}}}},
          {"name":"screenshot","parameters":{}}
        ]"#,
    );
    assert_eq!(tools.len(), 3);
    assert!(tools[0].param_keys.contains(&"location".to_string()));
    assert!(tools[1].param_keys.contains(&"query".to_string()));
    assert!(tools[2].param_keys.is_empty());
}

#[test]
fn parse_camel_case_name_normalised() {
    let tools = ToolDef::from_json(
        r#"[{"name":"getWeather","parameters":{"location":{"type":"string"}}}]"#,
    );
    assert_eq!(tools[0].name, "getWeather");
    assert_eq!(tools[0].snake_name, "get_weather");
}

#[test]
fn parse_pascal_case_name_normalised() {
    let tools = ToolDef::from_json(
        r#"[{"name":"GetWeather","parameters":{"location":{"type":"string"}}}]"#,
    );
    assert_eq!(tools[0].snake_name, "get_weather");
}

#[test]
fn parse_upper_snake_name_normalised() {
    let tools = ToolDef::from_json(
        r#"[{"name":"GET_WEATHER","parameters":{"location":{"type":"string"}}}]"#,
    );
    assert_eq!(tools[0].snake_name, "get_weather");
}

#[test]
fn parse_hyphen_name_normalised() {
    let tools = ToolDef::from_json(
        r#"[{"name":"get-weather","parameters":{"location":{"type":"string"}}}]"#,
    );
    assert_eq!(tools[0].snake_name, "get_weather");
}

#[test]
fn parse_translate_text_camel_normalised() {
    let tools = ToolDef::from_json(
        r#"[{"name":"translateText","parameters":{"text":{"type":"string"},"target_language":{"type":"string"}}}]"#,
    );
    assert_eq!(tools[0].snake_name, "translate_text");
}

#[test]
fn parse_preserves_original_name() {
    let tools = ToolDef::from_json(
        r#"[{"name":"bookFlight","parameters":{"origin":{"type":"string"}}}]"#,
    );
    assert_eq!(tools[0].name, "bookFlight");
    assert_eq!(tools[0].snake_name, "book_flight");
}

#[test]
fn parse_json_schema_descriptions_in_properties_ignored() {
    let tools = ToolDef::from_json(
        r#"[{"name":"get_weather","parameters":{"type":"object","properties":{"location":{"type":"string","description":"City name"},"unit":{"type":"string","description":"celsius or fahrenheit"}}}}]"#,
    );
    let keys = &tools[0].param_keys;
    assert!(keys.contains(&"location".to_string()));
    assert!(keys.contains(&"unit".to_string()));
    assert!(!keys.contains(&"description".to_string()));
}

#[test]
fn parse_many_params_flat() {
    let tools = ToolDef::from_json(
        r#"[{"name":"book_flight","parameters":{"origin":{"type":"string"},"destination":{"type":"string"},"date":{"type":"string"},"passengers":{"type":"integer"},"cabin_class":{"type":"string"}}}]"#,
    );
    let keys = &tools[0].param_keys;
    assert_eq!(keys.len(), 5);
    for k in &["origin", "destination", "date", "passengers", "cabin_class"] {
        assert!(keys.contains(&k.to_string()), "missing key: {k}");
    }
}

#[test]
fn parse_many_params_schema() {
    let tools = ToolDef::from_json(
        r#"[{"name":"book_flight","parameters":{"type":"object","properties":{"origin":{"type":"string"},"destination":{"type":"string"},"date":{"type":"string"},"passengers":{"type":"integer"},"cabin_class":{"type":"string"}},"required":["origin","destination","date"]}}]"#,
    );
    let keys = &tools[0].param_keys;
    assert_eq!(keys.len(), 5);
    assert!(!keys.contains(&"required".to_string()));
    assert!(!keys.contains(&"properties".to_string()));
}

#[test]
fn parse_prefix_shadow_param_keys_all_present() {
    let tools = ToolDef::from_json(
        r#"[{"name":"search_events","parameters":{"query":{"type":"string"},"date":{"type":"string"},"date_from":{"type":"string"},"date_to":{"type":"string"}}}]"#,
    );
    let keys = &tools[0].param_keys;
    for k in &["query", "date", "date_from", "date_to"] {
        assert!(keys.contains(&k.to_string()), "missing: {k}");
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// ConstrainedDecoder — masking and prefix-shadow disambiguation
// Uses feed_bytes (public) + logit_mask (public); does not access private .sm.
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn prefix_shadow_short_name_allowed_at_terminal() {
    // Trie: "get_weather" and "get_weather_forecast".
    // After `[{"name":"`, token `get_weather"` is valid (terminal) AND so is
    // `get_weather_forecast"` (also terminal). Prefix tokens are also valid.
    let tools_json = r#"[
        {"name":"get_weather","parameters":{"location":{"type":"string"}}},
        {"name":"get_weather_forecast","parameters":{"location":{"type":"string"},"days":{"type":"integer"}}}
    ]"#;
    let vocab: &[(u32, &[u8])] = &[
        (10, b"get_weather\""),          // terminal: valid
        (11, b"get_weather_forecast\""), // terminal: valid
        (12, b"get_weath"),              // prefix: valid
        (13, b"get_weather_f"),          // prefix to _forecast: valid
        (14, b"get_weather_x\""),        // not in trie: invalid
        (15, b"set_weather\""),          // 's' not in trie: invalid
    ];
    let mut dec = make_decoder(tools_json, vocab);
    drive(&mut dec, b"[{\"name\":\"");

    let mask = dec.logit_mask(16);
    assert_eq!(mask[10], 0.0, "get_weather\" must be allowed (terminal)");
    assert_eq!(mask[11], 0.0, "get_weather_forecast\" must be allowed (terminal)");
    assert_eq!(mask[12], 0.0, "get_weath prefix must be allowed");
    assert_eq!(mask[13], 0.0, "get_weather_f prefix must be allowed");
    assert!(mask[14] < 0.0, "get_weather_x\" not in trie: must be blocked");
    assert!(mask[15] < 0.0, "set_weather\" not in trie: must be blocked");
}

#[test]
fn prefix_shadow_arg_keys() {
    // Arg key trie: "date", "date_from", "date_to", "query"
    let tools_json = r#"[{"name":"search_events","parameters":{"query":{"type":"string"},"date":{"type":"string"},"date_from":{"type":"string"},"date_to":{"type":"string"}}}]"#;
    let vocab: &[(u32, &[u8])] = &[
        (0, b"date\""),      // terminal "date"
        (1, b"date_from\""), // terminal "date_from"
        (2, b"date_to\""),   // terminal "date_to"
        (3, b"date_"),       // prefix: valid for both _from and _to
        (4, b"date_x\""),    // not in trie: invalid
        (5, b"query\""),     // terminal "query"
        (6, b"other\""),     // not in trie: invalid
    ];
    let mut dec = make_decoder(tools_json, vocab);
    drive(&mut dec, b"[{\"name\":\"search_events\",\"arguments\":{\"");

    let mask = dec.logit_mask(7);
    assert_eq!(mask[0], 0.0, "date\"");
    assert_eq!(mask[1], 0.0, "date_from\"");
    assert_eq!(mask[2], 0.0, "date_to\"");
    assert_eq!(mask[3], 0.0, "date_ prefix");
    assert!(mask[4] < 0.0, "date_x\" blocked");
    assert_eq!(mask[5], 0.0, "query\"");
    assert!(mask[6] < 0.0, "other\" blocked");
}

#[test]
fn arg_key_trie_switches_per_tool() {
    let tools_json = r#"[
        {"name":"get_weather","parameters":{"location":{"type":"string"},"unit":{"type":"string"}}},
        {"name":"web_search","parameters":{"query":{"type":"string"}}}
    ]"#;
    let vocab: &[(u32, &[u8])] = &[
        (0, b"location\""),
        (1, b"unit\""),
        (2, b"query\""),
        (3, b"other\""),
    ];

    let mut dec = make_decoder(tools_json, vocab);
    drive(&mut dec, b"[{\"name\":\"get_weather\",\"arguments\":{\"");
    let mask = dec.logit_mask(4);
    assert_eq!(mask[0], 0.0, "location valid for get_weather");
    assert_eq!(mask[1], 0.0, "unit valid for get_weather");
    assert!(mask[2] < 0.0, "query invalid for get_weather");
    assert!(mask[3] < 0.0, "other invalid");

    let mut dec2 = make_decoder(tools_json, vocab);
    drive(&mut dec2, b"[{\"name\":\"web_search\",\"arguments\":{\"");
    let mask2 = dec2.logit_mask(4);
    assert_eq!(mask2[2], 0.0, "query valid for web_search");
    assert!(mask2[0] < 0.0, "location invalid for web_search");
}

#[test]
fn empty_param_trie_falls_back_to_unconstrained() {
    let tools_json = r#"[{"name":"screenshot","parameters":{}}]"#;
    let vocab: &[(u32, &[u8])] = &[(0, b"any_token\""), (1, b"other\""), (2, b"xyz\"")];
    let mut dec = make_decoder(tools_json, vocab);
    drive(&mut dec, b"[{\"name\":\"screenshot\",\"arguments\":{\"");
    // Empty trie → build_mask_from_trie no_any_allowed path → all 0.0
    let mask = dec.logit_mask(3);
    assert!(
        mask.iter().all(|&v| v == 0.0),
        "empty trie must produce all-zero mask (unconstrained fallback)"
    );
}

#[test]
fn free_state_produces_all_zero_mask() {
    let tools_json = r#"[{"name":"get_weather","parameters":{"location":{"type":"string"}}}]"#;
    let vocab: &[(u32, &[u8])] = &[(0, b"hello"), (1, b"world"), (2, b"get_weather\"")];
    let dec = make_decoder(tools_json, vocab);
    // No bytes fed → Free state
    assert!(dec.logit_mask(3).iter().all(|&v| v == 0.0));
}

#[test]
fn name_mask_blocks_invalid_prefix() {
    let tools_json = r#"[{"name":"get_weather","parameters":{}}]"#;
    let vocab: &[(u32, &[u8])] = &[
        (0, b"get_weather\""), // valid terminal
        (1, b"set_weather\""), // 's' not in trie
        (2, b"get_"),          // valid prefix
        (3, b"get_x"),         // invalid — 'x' not in trie after "get_"
    ];
    let mut dec = make_decoder(tools_json, vocab);
    drive(&mut dec, b"[{\"name\":\"");
    let mask = dec.logit_mask(4);
    assert_eq!(mask[0], 0.0);
    assert!(mask[1] < 0.0);
    assert_eq!(mask[2], 0.0);
    assert!(mask[3] < 0.0);
}

#[test]
fn json_schema_tool_uses_correct_arg_keys() {
    // Ensure JSON Schema format produces same arg key trie as flat format
    let flat_tools = r#"[{"name":"get_weather","parameters":{"location":{"type":"string"},"unit":{"type":"string"}}}]"#;
    let schema_tools = r#"[{"name":"get_weather","parameters":{"type":"object","properties":{"location":{"type":"string"},"unit":{"type":"string"}}}}]"#;
    let vocab: &[(u32, &[u8])] = &[
        (0, b"location\""),
        (1, b"unit\""),
        (2, b"other\""),
        (3, b"properties\""), // must be blocked in both formats
    ];
    let trigger: &[u8] = b"[{\"name\":\"get_weather\",\"arguments\":{\"";


    let mut dec_flat = make_decoder(flat_tools, vocab);
    drive(&mut dec_flat, trigger);
    let mask_flat = dec_flat.logit_mask(4);

    let mut dec_schema = make_decoder(schema_tools, vocab);
    drive(&mut dec_schema, trigger);
    let mask_schema = dec_schema.logit_mask(4);

    assert_eq!(mask_flat[0], 0.0, "flat: location valid");
    assert_eq!(mask_flat[1], 0.0, "flat: unit valid");
    assert!(mask_flat[2] < 0.0, "flat: other blocked");
    assert!(mask_flat[3] < 0.0, "flat: properties blocked");

    assert_eq!(mask_schema[0], 0.0, "schema: location valid");
    assert_eq!(mask_schema[1], 0.0, "schema: unit valid");
    assert!(mask_schema[2] < 0.0, "schema: other blocked");
    assert!(mask_schema[3] < 0.0, "schema: properties blocked — Python bug fix");
}

// ──────────────────────────────────────────────────────────────────────────────
// to_snake_case — tested via ToolDef::from_json name normalisation
// ──────────────────────────────────────────────────────────────────────────────

fn snake(name: &str) -> String {
    ToolDef::from_json(&format!(r#"[{{"name":"{}","parameters":{{}}}}]"#, name))[0]
        .snake_name
        .clone()
}

#[test]
fn snake_passthrough() { assert_eq!(snake("get_weather"), "get_weather"); }

#[test]
fn snake_camel_two_words() { assert_eq!(snake("getWeather"), "get_weather"); }

#[test]
fn snake_camel_three_words() { assert_eq!(snake("bookFlightTicket"), "book_flight_ticket"); }

#[test]
fn snake_pascal_two_words() { assert_eq!(snake("GetWeather"), "get_weather"); }

#[test]
fn snake_pascal_three_words() { assert_eq!(snake("BookFlightTicket"), "book_flight_ticket"); }

#[test]
fn snake_upper_two_words() { assert_eq!(snake("GET_WEATHER"), "get_weather"); }

#[test]
fn snake_upper_three_words() { assert_eq!(snake("WEB_SEARCH_API"), "web_search_api"); }

#[test]
fn snake_hyphen_two_words() { assert_eq!(snake("get-weather"), "get_weather"); }

#[test]
fn snake_hyphen_three_words() { assert_eq!(snake("web-search-api"), "web_search_api"); }

#[test]
fn snake_single_word() { assert_eq!(snake("weather"), "weather"); }

#[test]
fn snake_with_number_suffix() { assert_eq!(snake("get_weather_v2"), "get_weather_v2"); }

#[test]
fn snake_camel_translate_text() { assert_eq!(snake("translateText"), "translate_text"); }

#[test]
fn snake_camel_stock_price() { assert_eq!(snake("getStockPrice"), "get_stock_price"); }

#[test]
fn snake_all_lowercase_unchanged() { assert_eq!(snake("search"), "search"); }

#[test]
fn snake_result_is_all_lowercase() {
    for name in &["GetWeather", "WEB_SEARCH", "bookFlight", "get-time", "translateText"] {
        let result = snake(name);
        assert!(
            result.chars().all(|c| c.is_lowercase() || c == '_'),
            "snake({name:?}) = {result:?} contains uppercase"
        );
    }
}

#[test]
fn sm_in_arg_key_state_after_opening_quote() {
    // The opening `"` of the first arg key (after `"arguments":{`) must put the SM into InArgKey.
    // Uses br##"..."## so `"#` inside is NOT treated as the raw-string closer.
    let m = sm(b"[{\"name\":\"get_weather\",\"arguments\":{\"");
    assert_eq!(m.state, JsonState::InArgKey);
    assert_eq!(m.current_function, "get_weather");
    assert!(m.constrained_buf.is_empty());
}
