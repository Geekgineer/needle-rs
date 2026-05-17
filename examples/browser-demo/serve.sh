#!/usr/bin/env bash
# Assembles and serves the browser demo locally — mirrors the CI deployment to _site/.
# The demo imports ./pkg/needle_wasm.js; this script puts pkg/ adjacent to index.html.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

if [ ! -f "$REPO_ROOT/pkg/needle_wasm.js" ]; then
  echo "error: pkg/needle_wasm.js not found — run wasm-pack build first:" >&2
  echo "  wasm-pack build crates/needle-wasm --target web --release --out-dir ../../pkg/" >&2
  exit 1
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

cp -r "$REPO_ROOT/examples/browser-demo/." "$TMP/"
cp -r "$REPO_ROOT/pkg/" "$TMP/pkg/"

echo "Serving at http://localhost:8080  (Ctrl-C to stop)"
cd "$TMP" && python3 -m http.server 8080
