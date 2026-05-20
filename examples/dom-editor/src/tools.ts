// Dynamic per-element tool generation.
//
// On every command we walk the editable root (the whole page, minus anything
// marked data-no-edit), find "interesting" elements, and emit one tool per
// (element × action). The tool *name* encodes which element it targets, so
// the model never has to think about selectors — it just picks a name from
// the list. The element itself lives in the tool's closure.
//
// Why this shape: the Needle model is a small single-turn router. It is good
// at "given English + a list of clearly-named tools, pick one." It is bad at
// CSS selectors, abstract schema slots, and history-stuffed prompts. Dynamic
// tool generation turns the targeting problem into the kind of choice the
// model is best at.

export type ToolCall = {
  name: string;
  arguments: Record<string, unknown>;
};

export type ToolResult =
  | { ok: true; summary: string }
  | { ok: false; error: string };

export type GeneratedTool = {
  name: string;
  description: string;
  parameters: {
    type: "object";
    properties: Record<string, { type: string; description?: string }>;
    required?: string[];
  };
  execute: (args: Record<string, unknown>) => ToolResult;
};

const BANNED_CSS = /url\s*\(|expression\s*\(|javascript:|@import/i;

function str(args: Record<string, unknown>, key: string): string | null {
  const v = args[key];
  return typeof v === "string" ? v : null;
}

// Keep the color example neutral. Earlier we listed "red, blue, teal, #ff0080"
// and the model would hallucinate "blue" any time the user didn't name a
// color — it was reaching for the first example in the schema.
const COLOR_PARAM = {
  type: "object" as const,
  properties: {
    color: { type: "string", description: "A CSS color name or hex code" },
  },
  required: ["color"],
};

// Anchor the text param with an explicit example. Needle's training favors
// typed entities (cities, colors, times); a bare "the new text" gives it
// nothing to grip on, which is why "Change the title to Hello" returned [].
const TEXT_PARAM = {
  type: "object" as const,
  properties: {
    text: {
      type: "string",
      description: "The new text content, e.g. 'Welcome' or 'Hello world'",
    },
  },
  required: ["text"],
};

// Choose a stable, model-friendly label for an element.
//   id > class > tag+index
// Lowercased, snake_cased, deduplicated. We strip generic chrome prefixes
// like `app_`, `page_`, `stage_` so `#app-title` becomes `title`, not
// `app_title` — otherwise the user's word "title" doesn't match the label
// the way the contrastive head needs it to.
const GENERIC_PREFIXES = ["app_", "page_", "stage_"];

function stripGenericPrefix(s: string): string {
  for (const p of GENERIC_PREFIXES) {
    if (s.startsWith(p) && s.length > p.length) return s.slice(p.length);
  }
  return s;
}

function labelFor(el: Element, fallbackIndex: number): string {
  const slug = (s: string) =>
    s.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "");
  if (el.id) return stripGenericPrefix(slug(el.id));
  if (el.className && typeof el.className === "string") {
    const first = el.className.split(/\s+/).find(Boolean);
    if (first) return `${stripGenericPrefix(slug(first))}_${fallbackIndex}`;
  }
  return `${el.tagName.toLowerCase()}_${fallbackIndex}`;
}

// Pick a human role word for descriptions. This word becomes the primary
// hook the model uses to match the user's natural-language target — so we
// pick the word a user is most likely to say. An h1 is "the title", not
// "the heading"; an h3 inside a card is a subheading; structural tags get
// real role words instead of "element".
function roleWord(el: Element): string {
  const tag = el.tagName.toLowerCase();
  if (tag === "h1") return "title";
  if (tag === "h2") return "heading";
  if (/^h[3-6]$/.test(tag)) return "subheading";
  if (tag === "button") return "button";
  if (tag === "p") return "paragraph";
  if (tag === "a") return "link";
  if (tag === "ul" || tag === "ol") return "list";
  if (tag === "li") return "list item";
  if (tag === "img") return "image";
  if (tag === "footer") return "footer";
  if (tag === "header") return "header";
  if (tag === "nav") return "navigation bar";
  if (tag === "section") return "section";
  if (tag === "article") return "article";
  if (tag === "aside") return "sidebar";
  return "element";
}

// Truncate text for a description so the schema doesn't blow up. Critically,
// we skip data-no-edit subtrees — otherwise an editable parent that *contains*
// non-editable UI (chips, status pills, run buttons) gets a textContent
// polluted by all that chrome and the ranker latches onto it.
function visibleText(el: Element): string {
  let acc = "";
  for (const child of Array.from(el.childNodes)) {
    if (child.nodeType === Node.TEXT_NODE) {
      acc += child.textContent || "";
    } else if (child.nodeType === Node.ELEMENT_NODE) {
      const childEl = child as Element;
      if (childEl.hasAttribute && childEl.hasAttribute("data-no-edit")) continue;
      acc += visibleText(childEl);
    }
  }
  return acc;
}

function textPreview(el: Element, max = 40): string {
  const t = visibleText(el).trim().replace(/\s+/g, " ");
  if (!t) return "";
  return t.length > max ? `"${t.slice(0, max)}…"` : `"${t}"`;
}

// Opt-out: any element marked data-no-edit (or living inside one) is hidden
// from the model. Used for the prompt input and the step log — we don't want
// the model wiping the user's command or scrambling its own activity feed.
function isOptedOut(el: Element): boolean {
  return el.closest("[data-no-edit]") !== null;
}

// Tags that are page-root containers, never discrete targets. Exposing them
// gives the model a tool that matches *any* query (its text preview contains
// the whole page) and drowns out the specific element tools.
const ROOT_CONTAINER_TAGS = new Set(["main", "body", "html"]);

// Decide whether an element is "interesting" enough to expose. We skip
// structural wrappers without identity to keep the tool list small, and
// skip user-input fields so the model can't wipe what someone's typing.
function isInteresting(el: Element): boolean {
  const tag = el.tagName.toLowerCase();
  if (ROOT_CONTAINER_TAGS.has(tag)) return false;
  if (["input", "textarea", "select", "script", "style", "link", "meta"].includes(tag)) return false;
  if (el.id) return true;
  if (el.className && typeof el.className === "string" && el.className.trim()) return true;
  if (/^h[1-6]$/.test(tag)) return true;
  if (["button", "a", "li", "img"].includes(tag)) return true;
  return false;
}

export type GenerateResult = {
  tools: GeneratedTool[];
  // For UI display: a flat list of (label, role, preview) for each surfaced element.
  inventory: Array<{ label: string; role: string; preview: string }>;
};

const MAX_ELEMENTS = 20;

export function generateTools(root: HTMLElement): GenerateResult {
  const elements: Array<{ el: HTMLElement; label: string; role: string; preview: string }> = [];
  const usedLabels = new Set<string>();
  let i = 0;

  for (const el of Array.from(root.querySelectorAll<HTMLElement>("*"))) {
    if (isOptedOut(el)) continue;
    if (!isInteresting(el)) continue;
    let label = labelFor(el, i);
    let n = 1;
    let unique = label;
    while (usedLabels.has(unique)) {
      n++;
      unique = `${label}_${n}`;
    }
    usedLabels.add(unique);
    elements.push({ el, label: unique, role: roleWord(el), preview: textPreview(el) });
    i++;
    if (elements.length >= MAX_ELEMENTS) break;
  }

  const tools: GeneratedTool[] = [];

  for (const { el, label, role } of elements) {
    // Tight, conventional descriptions. Needle was trained on routine routing
    // ("get weather", "book flight"); unusual verbs like "rewrite" cause it
    // to return [] instead of guessing. Stick to "Set the X of the Y" — the
    // model parses that shape reliably.
    //
    // We deliberately do NOT include the element's text preview here. With
    // 3 tools per element having near-identical descriptions, the preview
    // dominated everything and the model couldn't discriminate between
    // color/background/text. The action word ("color", "background", "text")
    // is the discriminator and needs to stand alone.
    tools.push({
      name: `set_${label}_color`,
      description: `Set the color of the ${role}.`,
      parameters: COLOR_PARAM,
      execute: (args) => {
        const color = str(args, "color");
        if (!color) return { ok: false, error: "color required" };
        if (BANNED_CSS.test(color)) return { ok: false, error: "color rejected" };
        el.style.setProperty("color", color);
        return { ok: true, summary: `${label}.color = ${color}` };
      },
    });

    tools.push({
      name: `set_${label}_background`,
      description: `Set the background of the ${role}.`,
      parameters: COLOR_PARAM,
      execute: (args) => {
        const color = str(args, "color");
        if (!color) return { ok: false, error: "color required" };
        if (BANNED_CSS.test(color)) return { ok: false, error: "color rejected" };
        el.style.setProperty("background-color", color);
        return { ok: true, summary: `${label}.background = ${color}` };
      },
    });

    tools.push({
      name: `set_${label}_text`,
      description: `Set, change, or replace the text of the ${role}.`,
      parameters: TEXT_PARAM,
      execute: (args) => {
        const text = str(args, "text");
        if (text === null) return { ok: false, error: "text required" };
        el.textContent = text;
        return { ok: true, summary: `${label}.text = "${text}"` };
      },
    });
  }

  // Stage-level tools that don't target a specific element.
  tools.push({
    name: "add_paragraph",
    description: "Append a new paragraph of text to the page.",
    parameters: TEXT_PARAM,
    execute: (args) => {
      const text = str(args, "text");
      if (text === null) return { ok: false, error: "text required" };
      const p = document.createElement("p");
      p.textContent = text;
      root.appendChild(p);
      return { ok: true, summary: `+ <p>"${text}"</p>` };
    },
  });

  tools.push({
    name: "add_button",
    description: "Append a new button to the page.",
    parameters: TEXT_PARAM,
    execute: (args) => {
      const text = str(args, "text");
      if (text === null) return { ok: false, error: "text required" };
      const b = document.createElement("button");
      b.textContent = text;
      root.appendChild(b);
      return { ok: true, summary: `+ <button>"${text}"</button>` };
    },
  });

  tools.push({
    name: "add_box",
    description: "Append a new box (a styled container) to the page.",
    parameters: {
      type: "object",
      properties: {
        text: { type: "string", description: "Optional text content for the box, e.g. 'A new box'" },
      },
    },
    execute: (args) => {
      const text = str(args, "text") || "";
      const div = document.createElement("div");
      div.className = "box";
      if (text) div.textContent = text;
      root.appendChild(div);
      return { ok: true, summary: text ? `+ <div.box>"${text}"</div>` : "+ <div.box>" };
    },
  });

  tools.push({
    name: "add_heading",
    description: "Append a new heading to the page.",
    parameters: TEXT_PARAM,
    execute: (args) => {
      const text = str(args, "text");
      if (text === null) return { ok: false, error: "text required" };
      const h = document.createElement("h2");
      h.textContent = text;
      root.appendChild(h);
      return { ok: true, summary: `+ <h2>"${text}"</h2>` };
    },
  });

  return {
    tools,
    inventory: elements.map(({ label, role, preview }) => ({ label, role, preview })),
  };
}

// Build the OpenAI-style schema string for a given subset of tools.
export function toolsSchemaJson(tools: GeneratedTool[]): string {
  return JSON.stringify(
    tools.map(({ name, description, parameters }) => ({ name, description, parameters })),
  );
}
