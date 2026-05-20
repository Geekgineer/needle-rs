import "./style.css";
import { NeedleModel, type LoadProgress } from "./model";
import { dispatch, type Step, type ToolSummary } from "./harness";

const $ = <T extends HTMLElement = HTMLElement>(id: string) =>
  document.getElementById(id) as T;

// Atomic commands — Needle is a single-turn router, so each command should be
// expressable as one tool call.
const EXAMPLES = [
  "Make the title red",
  "Change the title to Hello",
  "Set the lede background to teal",
  "Make the footer text yellow",
  "Add a paragraph saying hello world",
];

const statusPill = $("status-pill");
const loadBtn = $<HTMLButtonElement>("load-btn");
const progressEl = $("progress");
const progressLabel = $("progress-label");
const progressBytes = $("progress-bytes");
const progressFill = $("progress-fill");
const playPanel = $("play-panel");
const goalInput = $<HTMLInputElement>("goal-input");
const runBtn = $<HTMLButtonElement>("run-btn");
// The model edits the whole page. UI regions that should be off-limits (the
// command input, the activity log) carry `data-no-edit` in index.html and
// are filtered out by the DOM walker in tools.ts.
const editableRoot = document.body;
const log = $<HTMLOListElement>("log");
const examplesEl = $("examples");
const toolsList = $<HTMLUListElement>("tools-list");
const toolsEmpty = $("tools-empty");
const toolsCount = $("tools-count");

let model: NeedleModel | null = null;

function setStatus(state: "idle" | "loading" | "ready" | "running" | "error", text: string) {
  statusPill.className = `pill ${state === "idle" ? "" : state}`;
  statusPill.textContent = text;
}

function renderProgress(p: LoadProgress) {
  progressEl.classList.remove("hidden");
  const labels: Record<LoadProgress["stage"], string> = {
    init: "Initializing WASM…",
    weights: "Downloading weights…",
    vocab: "Downloading vocab…",
    engine: "Initializing engine…",
    ready: "Ready",
  };
  progressLabel.textContent = labels[p.stage];
  if (p.loadedBytes !== undefined && p.totalBytes) {
    const mb = (n: number) => (n / 1024 / 1024).toFixed(1);
    progressBytes.textContent = `${mb(p.loadedBytes)} / ${mb(p.totalBytes)} MB`;
    progressFill.style.width = `${(p.loadedBytes / p.totalBytes) * 100}%`;
  } else {
    progressBytes.textContent = "";
  }
}

function appendStep(command: string, step: Step) {
  const li = document.createElement("li");

  const cmdLine = document.createElement("div");
  cmdLine.className = "meta";
  cmdLine.textContent = `> ${command}`;

  const callLine = document.createElement("div");
  callLine.className = "call";
  const toolName = document.createElement("span");
  toolName.className = "tool-name";
  toolName.textContent = step.call.name;
  callLine.appendChild(toolName);
  callLine.appendChild(document.createTextNode(`(${JSON.stringify(step.call.arguments)})`));

  const resultLine = document.createElement("div");
  resultLine.className = `result ${step.result.ok ? "ok" : "err"}`;
  resultLine.textContent = step.result.ok ? `→ ${step.result.summary}` : `× ${step.result.error}`;

  const timing = document.createElement("div");
  timing.className = "meta";
  timing.textContent =
    `retrieve ${step.retrieveMs.toFixed(0)} ms · infer ${step.inferMs.toFixed(0)} ms · ` +
    `${step.selectedCount}/${step.candidateCount} tools`;

  li.appendChild(cmdLine);
  li.appendChild(callLine);
  li.appendChild(resultLine);
  li.appendChild(timing);
  log.appendChild(li);
  log.scrollTop = log.scrollHeight;
}

function appendError(command: string, message: string) {
  const li = document.createElement("li");
  const cmdLine = document.createElement("div");
  cmdLine.className = "meta";
  cmdLine.textContent = `> ${command}`;
  const err = document.createElement("div");
  err.className = "result err";
  err.textContent = `× ${message}`;
  li.appendChild(cmdLine);
  li.appendChild(err);
  log.appendChild(li);
  log.scrollTop = log.scrollHeight;
}

function renderTools(generated: ToolSummary[], selectedNames: string[], pickedName: string | null) {
  while (toolsList.firstChild) toolsList.removeChild(toolsList.firstChild);
  if (generated.length === 0) {
    toolsList.classList.add("hidden");
    toolsEmpty.classList.remove("hidden");
    toolsCount.textContent = "";
    return;
  }
  toolsEmpty.classList.add("hidden");
  toolsList.classList.remove("hidden");
  toolsCount.textContent = `${generated.length} generated · ${selectedNames.length} narrowed${pickedName ? " · 1 picked" : ""}`;
  const selected = new Set(selectedNames);
  for (const t of generated) {
    const li = document.createElement("li");
    const picked = t.name === pickedName;
    const isSelected = selected.has(t.name);
    li.className = "tool-row " + (picked ? "picked" : isSelected ? "selected" : "other");
    const name = document.createElement("span");
    name.className = "tool-name mono";
    name.textContent = t.name;
    const desc = document.createElement("span");
    desc.className = "tool-desc muted";
    desc.textContent = t.description;
    li.appendChild(name);
    li.appendChild(desc);
    toolsList.appendChild(li);
  }
}

function renderExamples() {
  for (const ex of EXAMPLES) {
    const btn = document.createElement("button");
    btn.className = "chip";
    btn.textContent = ex;
    btn.addEventListener("click", () => {
      goalInput.value = ex;
      goalInput.focus();
    });
    examplesEl.appendChild(btn);
  }
}

loadBtn.addEventListener("click", async () => {
  loadBtn.disabled = true;
  loadBtn.textContent = "Loading…";
  setStatus("loading", "loading model");
  try {
    model = await NeedleModel.load((p) => {
      renderProgress(p);
      if (p.stage === "engine") setStatus("loading", "initializing engine");
    });
    setStatus("ready", "model ready");
    progressEl.classList.add("hidden");
    loadBtn.classList.add("hidden");
    playPanel.classList.remove("hidden");
    goalInput.focus();
  } catch (e) {
    console.error(e);
    setStatus("error", "load failed");
    progressLabel.textContent = `Error: ${(e as Error).message}`;
    loadBtn.disabled = false;
    loadBtn.textContent = "Retry";
  }
});

async function run() {
  if (!model) return;
  const command = goalInput.value.trim();
  if (!command) return;

  runBtn.disabled = true;
  runBtn.textContent = "Running…";
  setStatus("running", "thinking");

  // Inference runs in a Web Worker (see src/worker.ts) so the main thread
  // stays interactive — no setTimeout-yield hack needed.
  const result = await dispatch(model, editableRoot, command);
  if (result.type === "step") {
    appendStep(command, result.step);
    renderTools(result.step.generated, result.step.selectedNames, result.step.call.name);
  } else {
    appendError(command, result.message);
    if (result.rawModelOutput) console.warn("raw model output:", result.rawModelOutput);
    if (result.generated) renderTools(result.generated, result.selectedNames ?? [], null);
  }
  goalInput.value = "";
  runBtn.disabled = false;
  runBtn.textContent = "Run →";
  setStatus("ready", "model ready");
  goalInput.focus();
}

runBtn.addEventListener("click", run);
goalInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !runBtn.disabled) run();
});

renderExamples();
setStatus("idle", "model not loaded");
