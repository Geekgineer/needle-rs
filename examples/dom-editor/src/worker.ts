// Web Worker that owns the NeedleWasm engine.
//
// The engine's `run()` is a synchronous, multi-second wasm call. Running it
// on the main thread freezes the UI (no animation, no input, no scroll).
// Moving it here means the main thread is only blocked when serializing
// arguments and deserializing the result — milliseconds, not seconds.
//
// The DOM walker, tool execution, and UI all stay on the main thread; this
// worker only knows about the model.

import init, { NeedleWasm } from "needle-rs";

const HF_BASE = "https://huggingface.co/Abdalrahman/needle-rs-safetensors/resolve/main";
const WEIGHTS_URL = `${HF_BASE}/needle.safetensors`;
const VOCAB_URL = `${HF_BASE}/vocab.txt`;

type LoadStage = "init" | "weights" | "vocab" | "engine" | "ready";

type InMsg =
  | { id: number; type: "load" }
  | { id: number; type: "infer"; query: string; toolsJson: string }
  | { id: number; type: "retrieve"; query: string; descriptions: string[]; topK: number }
  | { id: number; type: "hasContrastive" };

type OutMsg =
  | { id: number; type: "progress"; stage: LoadStage; loadedBytes?: number; totalBytes?: number }
  | { id: number; type: "result"; data: unknown }
  | { id: number; type: "error"; message: string };

let engine: NeedleWasm | null = null;

function post(msg: OutMsg): void {
  (self as unknown as Worker).postMessage(msg);
}

async function fetchWithProgress(
  url: string,
  hintBytes: number,
  id: number,
  stage: LoadStage,
): Promise<Uint8Array> {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`fetch ${url} → ${resp.status}`);
  const total = parseInt(resp.headers.get("content-length") || "0") || hintBytes;
  const reader = resp.body!.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    post({ id, type: "progress", stage, loadedBytes: loaded, totalBytes: total });
  }
  const buf = new Uint8Array(loaded);
  let off = 0;
  for (const c of chunks) {
    buf.set(c, off);
    off += c.length;
  }
  return buf;
}

async function handleLoad(id: number): Promise<void> {
  post({ id, type: "progress", stage: "init" });
  await init();

  const weights = await fetchWithProgress(WEIGHTS_URL, 22 * 1024 * 1024, id, "weights");

  const vocabBytes = await fetchWithProgress(VOCAB_URL, 120 * 1024, id, "vocab");
  const vocab = new TextDecoder().decode(vocabBytes);

  post({ id, type: "progress", stage: "engine" });
  const e = NeedleWasm.load(weights, vocab);
  if (!e) throw new Error("NeedleWasm.load returned undefined");
  engine = e;

  post({ id, type: "progress", stage: "ready" });
  post({ id, type: "result", data: { contrastiveDim: e.contrastive_dim() } });
}

self.addEventListener("message", async (ev: MessageEvent<InMsg>) => {
  const msg = ev.data;
  try {
    switch (msg.type) {
      case "load":
        await handleLoad(msg.id);
        return;
      case "infer": {
        if (!engine) throw new Error("model not loaded");
        const result = engine.run(msg.query, msg.toolsJson);
        post({ id: msg.id, type: "result", data: result });
        return;
      }
      case "retrieve": {
        if (!engine) throw new Error("model not loaded");
        const raw = engine.retrieve_tools(msg.query, JSON.stringify(msg.descriptions), msg.topK);
        let parsed: Array<{ index: number; score: number }> = [];
        try {
          parsed = JSON.parse(raw);
        } catch {
          /* leave empty */
        }
        post({ id: msg.id, type: "result", data: parsed });
        return;
      }
      case "hasContrastive": {
        if (!engine) throw new Error("model not loaded");
        post({ id: msg.id, type: "result", data: engine.contrastive_dim() > 0 });
        return;
      }
    }
  } catch (e) {
    post({ id: msg.id, type: "error", message: (e as Error).message });
  }
});
