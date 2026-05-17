#!/usr/bin/env python3
"""
Run Needle inference from Python using the needle-rs C ABI (ctypes).

No Python ML dependencies required — only the standard library + ctypes.
Calls libneedle_c.so (Linux) / libneedle_c.dylib (macOS) / needle_c.dll (Windows).

Usage:
    python infer.py \
        --lib ../../target/release/libneedle_c.so \
        --weights ../../weights/needle.safetensors \
        --vocab ../../weights/vocab.txt \
        --query "What is the weather in Paris?" \
        --tools '[{"name":"get_weather","description":"Get weather","parameters":{"type":"object","properties":{"location":{"type":"string"}}}}]'
"""

import argparse
import ctypes
import os
import platform
import sys

def load_library(path: str) -> ctypes.CDLL:
    lib = ctypes.CDLL(path)

    # needle_load(weights_path, vocab_path) -> *NeedleHandle
    lib.needle_load.restype = ctypes.c_void_p
    lib.needle_load.argtypes = [ctypes.c_char_p, ctypes.c_char_p]

    # needle_run(handle, query, tools_json) -> *char
    lib.needle_run.restype = ctypes.c_char_p
    lib.needle_run.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]

    # needle_run_stream(handle, query, tools_json, callback, user_data) -> *char
    CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_uint32, ctypes.c_char_p, ctypes.c_void_p)
    lib.needle_run_stream.restype = ctypes.c_char_p
    lib.needle_run_stream.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, CALLBACK, ctypes.c_void_p
    ]
    lib._stream_callback_type = CALLBACK  # keep reference alive

    # needle_free_str(s)
    lib.needle_free_str.restype = None
    lib.needle_free_str.argtypes = [ctypes.c_char_p]

    # needle_free(handle)
    lib.needle_free.restype = None
    lib.needle_free.argtypes = [ctypes.c_void_p]

    # needle_last_error() -> *const char
    lib.needle_last_error.restype = ctypes.c_char_p
    lib.needle_last_error.argtypes = []

    return lib


def default_lib_path() -> str:
    system = platform.system()
    repo_root = os.path.join(os.path.dirname(__file__), "..", "..")
    release_dir = os.path.join(repo_root, "target", "release")
    if system == "Linux":
        return os.path.join(release_dir, "libneedle_c.so")
    elif system == "Darwin":
        return os.path.join(release_dir, "libneedle_c.dylib")
    else:
        return os.path.join(release_dir, "needle_c.dll")


def main():
    parser = argparse.ArgumentParser(description="Needle inference via C ABI")
    parser.add_argument("--lib", default=default_lib_path())
    parser.add_argument("--weights", default="../../weights/needle.safetensors")
    parser.add_argument("--vocab", default="../../weights/vocab.txt")
    parser.add_argument("--query", default="What is the weather in Paris?")
    parser.add_argument("--tools", default=(
        '[{"name":"get_weather","description":"Get weather for a city",'
        '"parameters":{"type":"object","properties":{'
        '"location":{"type":"string","description":"City name"}}}}]'
    ))
    parser.add_argument("--stream", action="store_true", help="Print tokens as generated")
    args = parser.parse_args()

    if not os.path.exists(args.lib):
        print(f"Library not found: {args.lib}", file=sys.stderr)
        print("Build it first: cargo build --release -p needle-c", file=sys.stderr)
        sys.exit(1)

    lib = load_library(args.lib)

    handle = lib.needle_load(args.weights.encode(), args.vocab.encode())
    if not handle:
        err = lib.needle_last_error()
        print(f"Failed to load model: {err.decode() if err else '(unknown error)'}", file=sys.stderr)
        sys.exit(1)

    query = args.query.encode()
    tools = args.tools.encode()

    try:
        if args.stream:
            def on_token(token_id, piece, _user_data):
                print(piece.decode("utf-8", errors="replace"), end="", flush=True)

            cb = lib._stream_callback_type(on_token)
            result = lib.needle_run_stream(handle, query, tools, cb, None)
            print()  # newline after stream
        else:
            result = lib.needle_run(handle, query, tools)

        if result is None:
            err = lib.needle_last_error()
            print(f"Inference failed: {err.decode() if err else '(unknown error)'}", file=sys.stderr)
            sys.exit(1)

        print(result.decode())
    finally:
        lib.needle_free(handle)


if __name__ == "__main__":
    main()
