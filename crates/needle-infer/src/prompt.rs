//! The chat prompt, shared by Needle v2 and v3.
//!
//! Upstream assembles it identically in both generations
//! (`needle/model/finetune.py::render_example`), so it lives here rather than
//! in either engine.

/// Chat-template markers, as `needle/model/finetune.py` assembles them.
pub const IM_START: &str = "<|im_start|>";
pub const IM_END: &str = "<|im_end|>";
pub const TOOLS_START: &str = "<tools>";
pub const TOOLS_END: &str = "</tools>";
pub const TOOL_CALL_START: &str = "<tool_call>";
pub const TOOL_CALL_END: &str = "</tool_call>";
pub const THINK_START: &str = "<think>";
pub const THINK_END: &str = "</think>";

/// Assemble the chat prompt.
///
/// ```text
/// [<|im_start|>system\n{system}<|im_end|>\n]
/// <|im_start|>user\n<tools>{tools}</tools>\n{query}<|im_end|>\n<|im_start|>assistant\n
/// ```
///
/// `tools_json` is compacted before embedding. This matters: the model was
/// trained on compact schemas, and the indentation a caller naturally gets from
/// `JSON.stringify(x, null, 2)` or `json.dumps(x, indent=2)` is enough to change
/// the decision. On the shipped checkpoints the same query and schema yields a
/// real call compact and `[]` pretty-printed. Rather than make that a documented
/// gotcha, the whitespace is removed here.
pub fn build_prompt(query: &str, tools_json: &str, system: Option<&str>) -> String {
    let tools_json = &compact_json(tools_json);
    let mut p = String::new();
    if let Some(s) = system {
        p.push_str(IM_START);
        p.push_str("system\n");
        p.push_str(s);
        p.push_str(IM_END);
        p.push('\n');
    }
    p.push_str(IM_START);
    p.push_str("user\n");
    p.push_str(TOOLS_START);
    p.push_str(tools_json);
    p.push_str(TOOLS_END);
    p.push('\n');
    p.push_str(query);
    p.push_str(IM_END);
    p.push('\n');
    p.push_str(IM_START);
    p.push_str("assistant\n");
    p
}

/// Strip insignificant whitespace from JSON, leaving string literals untouched.
///
/// Deliberately not a parser: it does not validate, and anything it cannot
/// interpret it passes through byte for byte, so a malformed schema reaches the
/// model exactly as the caller wrote it rather than being silently mangled.
/// Already-compact input is returned unchanged, which is why this cannot move the
/// end-to-end parity fixtures.
pub fn compact_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut in_string = false;
    let mut escaped = false;
    for c in s.chars() {
        if in_string {
            out.push(c);
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_string = false;
            }
            continue;
        }
        match c {
            '"' => {
                in_string = true;
                out.push(c);
            }
            ' ' | '\t' | '\n' | '\r' => {}
            _ => out.push(c),
        }
    }
    out
}
