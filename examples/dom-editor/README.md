# DOM editor (browser)

A real-life use case for needle-rs: a 22 MB tool-calling LLM that rewrites a
live web page from plain-English commands, entirely in the browser. No server,
no API key, no data leaving the tab.

Type one command (`"make the title red"`, `"set the lede background to teal"`).
The harness walks the current DOM, generates one tool per (element × action),
narrows the list with the model's contrastive head, then asks the model to
pick exactly one tool with its arguments filled in. The selected tool runs
against the DOM. Type the next command. The page evolves one step at a time.

## Why this is interesting

Most tool-calling demos give the model a hand-crafted, static list of tools.
This one **generates the tools fresh every turn from the current page state**,
which turns the targeting problem into the thing Needle is best at:
"given English + a list of clearly-named tools, pick one."

The model never sees a CSS selector or an abstract schema slot like
`property`. Each tool's *name* encodes which element it targets
(`set_title_color`, `set_lede_background`); the element itself lives in the
tool's closure on the JS side. Adding or removing DOM nodes in one command
automatically reshapes the tool list for the next.

## Pipeline

```
command → walk the DOM → generate ~N candidate tools
                          (one per interesting element × action)
        → engine.retrieve_tools()  ← stage 1: contrastive narrowing to top-K
        → engine.run()             ← stage 2: route to one tool, fill its args
        → execute against the DOM
```

Both stages run in a Web Worker (`src/worker.ts`) so the main thread stays
interactive while the WASM engine does its multi-second `run()`.

Regions of the page marked `data-no-edit` (the prompt input, the activity
log, the load panel, the limitations notice) are filtered out of the DOM
walk — the model can't wipe what someone is typing or scramble its own log.

## Quickstart

```bash
cd examples/dom-editor
npm install
npm run dev
```

Open the printed URL, click **Load model** (22 MB, cached by the browser after
the first run), then chain commands. The right-hand panel shows every tool
generated for the current command, with the top-K candidates highlighted and
the picked one marked in green.

The model and runtime are pulled from the published `needle-rs` npm package,
so this example builds standalone without a local Rust toolchain.

## Layout

```
index.html        — UI (prompt, log, tools panel, limitations notice)
src/
  main.ts         — UI wiring and rendering
  harness.ts      — two-stage dispatch (retrieve → route → execute)
  tools.ts        — DOM walker + per-element tool generator
  model.ts        — postMessage client for the engine worker
  worker.ts       — owns the NeedleWasm engine; streams progress + results
  style.css
```

## Design notes worth reading the source for

- `tools.ts` — how DOM elements are turned into model-friendly labels
  (`#app-title` → `title`, not `app_title`), why every tool name carries its
  target, and why descriptions deliberately *omit* the element's text preview
  (it dominated everything and the router couldn't discriminate between
  color/background/text).
- `harness.ts` — why the top-K is small (4) and what happens when the
  contrastive head isn't available (a JS term-overlap fallback ranker, so we
  don't degenerate to DOM order).
- `tools.ts` color/text parameter shapes — anchoring schemas with realistic
  examples (`"e.g. 'Welcome' or 'Hello world'"`) is the difference between
  the model emitting a valid call and returning `[]`.

## Caveats

This is a 26M-parameter router. It is good at single, clear commands and bad
at compound or ambiguous ones. The on-screen "Limitations" panel explains
what the model can and can't do — the same caveats apply to any production
use of needle-rs.
