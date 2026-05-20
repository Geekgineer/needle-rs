// Two-stage dispatch pipeline.
//
//   command → generate tools from current DOM
//           → contrastively narrow to top-K (model call 1)
//           → route to one tool (model call 2)
//           → execute
//
// Each user command kicks off the whole pipeline fresh, so the tool universe
// reflects the *current* DOM. Elements added or removed by previous commands
// reshape the next command's tool list automatically.

import type { NeedleModel } from "./model";
import { generateTools, toolsSchemaJson, type GeneratedTool, type ToolCall, type ToolResult } from "./tools";

export type ToolSummary = { name: string; description: string };

export type Step = {
  call: ToolCall;
  result: ToolResult;
  rawModelOutput: string;
  // Pipeline observability — useful to surface in the UI / log.
  candidateCount: number;
  selectedCount: number;
  retrieveMs: number;
  inferMs: number;
  generated: ToolSummary[];
  selectedNames: string[];
};

export type DispatchResult =
  | { type: "step"; step: Step }
  | { type: "error"; message: string; rawModelOutput?: string; generated?: ToolSummary[]; selectedNames?: string[] };

// Keep this small. The Needle model overflows its context when given too many
// tools with verbose schemas — when that happens it returns `[]` instead of
// picking. Empirically, 4 works.
const TOP_K = 4;

// Stopwords stripped from queries/descriptions before ranking — they're noise
// for term overlap and would otherwise let any tool with "the" or "a" score.
const STOPWORDS = new Set([
  "a", "an", "the", "to", "of", "in", "on", "at", "and", "or", "for",
  "is", "are", "was", "were", "be", "make", "set", "change", "color",
  "text", "with", "this", "that", "it", "do", "please",
]);

function tokens(s: string): string[] {
  return s.toLowerCase().split(/[^a-z0-9]+/).filter((t) => t && !STOPWORDS.has(t));
}

// JS-side fallback ranker. Scores each tool by counting query tokens that
// appear in either the tool's *name* (split on underscores) or its
// description. Names matter because the element label ("title", "footer",
// "lede") lives there — descriptions often only carry the role word and
// the element's literal text. Without name-scoring, "change the title to
// Hello" finds no match and ties degenerate to DOM order.
function rankByTermOverlap(
  query: string,
  tools: Array<{ name: string; description: string }>,
  topK: number,
): number[] {
  const q = new Set(tokens(query));
  const scored = tools.map((t, i) => {
    const nameToks = tokens(t.name.replace(/_/g, " "));
    const descToks = tokens(t.description);
    let score = 0;
    for (const tok of nameToks) if (q.has(tok)) score++;
    for (const tok of descToks) if (q.has(tok)) score++;
    return { i, score };
  });
  scored.sort((a, b) => b.score - a.score || a.i - b.i);
  return scored.slice(0, topK).map((s) => s.i);
}

function parseToolCall(raw: string): ToolCall | null {
  try {
    const parsed = JSON.parse(raw);
    const obj = Array.isArray(parsed) ? parsed[0] : parsed;
    if (!obj || typeof obj.name !== "string") return null;
    const args = obj.arguments && typeof obj.arguments === "object" ? obj.arguments : {};
    return { name: obj.name, arguments: args as Record<string, unknown> };
  } catch {
    return null;
  }
}

export async function dispatch(
  model: NeedleModel,
  root: HTMLElement,
  command: string,
): Promise<DispatchResult> {
  const { tools } = generateTools(root);
  const generated: ToolSummary[] = tools.map((t) => ({ name: t.name, description: t.description }));
  if (tools.length === 0) {
    return { type: "error", message: "no editable elements found on the page", generated };
  }

  // Stage 1: narrow the candidate set to top-K. Prefer the model's contrastive
  // head if present; if the loaded weights don't include one, fall back to a
  // JS term-overlap ranker so we don't end up just picking the first K tools
  // in DOM order.
  const t0 = performance.now();
  let selected: GeneratedTool[];
  if (tools.length <= TOP_K) {
    selected = tools;
  } else {
    const descriptions = tools.map((t) => t.description);
    let indices: number[] = [];
    if (model.hasContrastiveHead()) {
      const ranked = await model.retrieve(command, descriptions, TOP_K);
      indices = ranked.map((r) => r.index);
    }
    if (indices.length === 0) {
      indices = rankByTermOverlap(command, tools, TOP_K);
    }
    selected = indices.map((i) => tools[i]).filter((t): t is GeneratedTool => Boolean(t));
  }
  const retrieveMs = performance.now() - t0;

  // Stage 2: routing. Hand the narrowed list to run() and let the model pick
  // exactly one with its arguments filled in.
  const t1 = performance.now();
  const selectedNames = selected.map((t) => t.name);
  let raw: string;
  try {
    raw = await model.infer(command, toolsSchemaJson(selected));
  } catch (e) {
    return { type: "error", message: `inference failed: ${(e as Error).message}`, generated, selectedNames };
  }
  const inferMs = performance.now() - t1;

  const call = parseToolCall(raw);
  if (!call) {
    // Surface enough context to debug from a console paste. The model
    // returning "[]" is the most common variant — usually a context overflow
    // or a query the ranker couldn't match to anything sensible.
    console.warn("[needle-playground] dispatch failure", {
      command,
      rawModelOutput: raw,
      narrowedTools: selected.map((t) => ({ name: t.name, description: t.description })),
    });
    return {
      type: "error",
      message: raw.trim() === "[]"
        ? "model returned no tool call (empty array) — see console for the 4 candidates it had to choose from"
        : "could not parse model output — see console",
      rawModelOutput: raw,
      generated,
      selectedNames,
    };
  }

  const tool = selected.find((t) => t.name === call.name);
  let result: ToolResult;
  if (!tool) {
    result = { ok: false, error: `unknown tool: ${call.name}` };
  } else {
    try {
      result = tool.execute(call.arguments);
    } catch (e) {
      result = { ok: false, error: `exec failed: ${(e as Error).message}` };
    }
  }

  return {
    type: "step",
    step: {
      call,
      result,
      rawModelOutput: raw,
      candidateCount: tools.length,
      selectedCount: selected.length,
      retrieveMs,
      inferMs,
      generated,
      selectedNames,
    },
  };
}
