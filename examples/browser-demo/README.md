# Browser demo

A self-contained HTML page that runs needle-rs inference entirely in the browser.
No server. No Python. The WASM module is 260 KB; weights (22 MB) load from HuggingFace on demand.

This is also the source for the deployed Cloudflare Pages demo at **needle-rs.pages.dev**.

## What it shows

- WASM module initialization (`needle_wasm.js` + `needle_wasm_bg.wasm`)
- Weight loading from HuggingFace with a progress indicator
- Streaming inference: tokens appear as they're generated
- Tool-call JSON output in the browser

## Running locally

Build the WASM package first:

```bash
wasm-pack build crates/needle-wasm --target web --release --out-dir ../../pkg/
```

Then serve the demo (any static server works):

```bash
cd examples/browser-demo
python3 -m http.server 8080
# open http://localhost:8080
```

The page loads weights directly from HuggingFace — no local weights needed for the browser demo.

## Deployment

The `wasm-demo.yml` workflow builds and deploys this to Cloudflare Pages on every push to `main`.
See `.github/workflows/wasm-demo.yml` for setup instructions (Cloudflare API token required).
