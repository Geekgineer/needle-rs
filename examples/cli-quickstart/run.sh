#!/usr/bin/env bash
# Run the needle-rs CLI against a single tool definition.
# Usage: ./run.sh [query]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CLI="$REPO_ROOT/target/release/needle-rs"
WEIGHTS="$REPO_ROOT/weights/needle.safetensors"
VOCAB="$REPO_ROOT/weights/vocab.txt"

if [ ! -f "$CLI" ]; then
  echo "Building CLI..."
  cargo build --release -p needle-cli --manifest-path "$REPO_ROOT/Cargo.toml"
fi

if [ ! -f "$WEIGHTS" ]; then
  echo "Weights not found at $WEIGHTS"
  echo "Export them first: PYTHONPATH=needle python tools/export.py --checkpoint needle/checkpoints/needle.pkl"
  exit 1
fi

QUERY="${1:-What is the weather in Paris?}"

TOOLS='[{
  "name": "get_weather",
  "description": "Get current weather for a city",
  "parameters": {
    "type": "object",
    "properties": {
      "location": {"type": "string", "description": "City name"},
      "unit": {"type": "string", "description": "celsius or fahrenheit"}
    }
  }
}]'

echo "Query: $QUERY"
echo "---"
"$CLI" "$WEIGHTS" "$VOCAB" "$QUERY" "$TOOLS"
