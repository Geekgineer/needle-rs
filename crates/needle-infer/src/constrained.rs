//! Grammar-constrained decoding via JSON state machine + character-level trie.
//!
//! Mirrors Python `needle/model/constrained.py`.
//!
//! Needle output format (compact JSON, no spaces):
//!   [{"name":"tool_name","arguments":{"key1":value1,...}}]
//!
//! Constrained regions:
//!   - After `"name":"` (not inside arguments) → InName
//!   - After `{"` or `,"` at arguments_depth → InArgKey
//!   - Closing `"` in either constrained state → Free

use std::collections::HashMap;

// ─── Trie ─────────────────────────────────────────────────────────────────────

#[derive(Default)]
struct TrieNode {
    children: HashMap<u8, usize>,
    is_terminal: bool,
}

pub struct Trie {
    nodes: Vec<TrieNode>,
}

impl Default for Trie {
    fn default() -> Self {
        Self::new()
    }
}

impl Trie {
    pub fn new() -> Self {
        Self {
            nodes: vec![TrieNode::default()],
        }
    }

    pub fn insert(&mut self, s: &[u8]) {
        let mut cur = 0;
        for &b in s {
            let next = self.nodes[cur].children.get(&b).copied();
            cur = match next {
                Some(n) => n,
                None => {
                    let n = self.nodes.len();
                    self.nodes.push(TrieNode::default());
                    self.nodes[cur].children.insert(b, n);
                    n
                }
            };
        }
        self.nodes[cur].is_terminal = true;
    }

    pub fn advance(&self, node: usize, bytes: &[u8]) -> Option<usize> {
        let mut cur = node;
        for &b in bytes {
            cur = *self.nodes[cur].children.get(&b)?;
        }
        Some(cur)
    }

    fn is_terminal(&self, node: usize) -> bool {
        self.nodes.get(node).is_some_and(|n| n.is_terminal)
    }

    fn child(&self, node: usize, b: u8) -> Option<usize> {
        self.nodes.get(node)?.children.get(&b).copied()
    }
}

/// Mirror Python `_check_token_valid`.
///
/// Walk `bytes` char-by-char through the trie from `start_node`.
/// `"` signals end-of-span — only valid if current node is terminal.
/// A byte absent from children → invalid.  Consuming all bytes without `"` → valid prefix.
fn check_token_valid(bytes: &[u8], trie: &Trie, start_node: usize) -> bool {
    let mut cur = start_node;
    for &b in bytes {
        if b == b'"' {
            return trie.is_terminal(cur);
        }
        match trie.child(cur, b) {
            Some(next) => cur = next,
            None => return false,
        }
    }
    true
}

fn build_mask_from_trie(
    trie: &Trie,
    node: usize,
    token_texts: &[Vec<u8>],
    vocab_size: usize,
) -> Vec<f32> {
    let mut mask = vec![-1e9f32; vocab_size];
    let mut any_allowed = false;
    for (id, text) in token_texts.iter().enumerate() {
        if id >= vocab_size {
            break;
        }
        if check_token_valid(text, trie, node) {
            mask[id] = 0.0;
            any_allowed = true;
        }
    }
    if !any_allowed {
        // Off-trie or empty trie — fall back to unconstrained (mirrors Python warning + return logits)
        mask.fill(0.0);
    }
    mask
}

// ─── JSON State Machine ───────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum JsonState {
    Free,
    InName,
    InArgKey,
}

// Pattern lengths
const NAME_TRIGGER: &[u8] = b"\"name\":\""; // 8 bytes
const ARGS_TRIGGER: &[u8] = b"\"arguments\":{"; // 13 bytes
const TAIL_LEN: usize = 13;

/// Tracks position in Needle's compact JSON output to constrain decoding.
///
/// Mirrors Python `JsonStateMachine._feed_char`.  Uses a fixed-size rolling
/// tail buffer (last 13 bytes) instead of a full output string, since only
/// recent bytes matter for pattern detection.
///
/// Key invariant: `in_string` tracks VALUE strings only (opened by `:"`) —
/// JSON key strings (`"name"`, `"arguments"`, …) are NOT tracked.  This is
/// identical to the Python implementation.
pub struct JsonStateMachine {
    pub state: JsonState,
    /// Tool name captured when leaving InName.
    pub current_function: String,
    /// Bytes accumulated inside the current constrained region.
    pub constrained_buf: Vec<u8>,
    in_arguments: bool,
    arguments_depth: usize,
    nesting_depth: usize,
    /// True while inside a JSON value string (opened by `:"`, closed by `"`).
    in_string: bool,
    prev_char_escape: bool,
    /// Rolling window of the last TAIL_LEN raw output bytes for pattern matching.
    tail: Vec<u8>,
}

impl Default for JsonStateMachine {
    fn default() -> Self {
        Self::new()
    }
}

impl JsonStateMachine {
    pub fn new() -> Self {
        Self {
            state: JsonState::Free,
            current_function: String::new(),
            constrained_buf: Vec::new(),
            in_arguments: false,
            arguments_depth: 0,
            nesting_depth: 0,
            in_string: false,
            prev_char_escape: false,
            tail: Vec::with_capacity(TAIL_LEN + 1),
        }
    }

    /// Feed one decoded byte (▁ already replaced with ' ').
    pub fn feed_byte(&mut self, b: u8) {
        // Constrained states: accumulate until closing `"`
        match self.state {
            JsonState::InName => {
                if b == b'"' {
                    self.current_function =
                        String::from_utf8_lossy(&self.constrained_buf).into_owned();
                    self.constrained_buf.clear();
                    self.state = JsonState::Free;
                } else {
                    self.constrained_buf.push(b);
                }
                return;
            }
            JsonState::InArgKey => {
                if b == b'"' {
                    self.constrained_buf.clear();
                    self.state = JsonState::Free;
                } else {
                    self.constrained_buf.push(b);
                }
                return;
            }
            JsonState::Free => {}
        }

        // Free state — append to rolling tail first
        self.push_tail(b);

        // Handle ongoing value string
        if self.in_string {
            if self.prev_char_escape {
                self.prev_char_escape = false;
                return;
            }
            if b == b'\\' {
                self.prev_char_escape = true;
                return;
            }
            if b == b'"' {
                self.in_string = false;
            }
            return;
        }

        // Not in string: track brace/bracket depth
        if b == b'{' || b == b'[' {
            self.nesting_depth += 1;
        } else if b == b'}' || b == b']' {
            self.nesting_depth = self.nesting_depth.saturating_sub(1);
            if b == b'}' && self.in_arguments && self.nesting_depth < self.arguments_depth {
                self.in_arguments = false;
            }
            return; // mirrors Python's early return for }]
        }

        // Trigger: "name":"  — enter InName (only outside arguments block)
        if !self.in_arguments && self.tail.ends_with(NAME_TRIGGER) {
            self.state = JsonState::InName;
            self.constrained_buf.clear();
            return;
        }

        // Trigger: "arguments":{  — begin tracking argument keys
        if self.tail.ends_with(ARGS_TRIGGER) {
            self.in_arguments = true;
            self.arguments_depth = self.nesting_depth;
            return;
        }

        // Trigger: {"  or `,"` at arguments_depth → arg key opening
        if self.in_arguments
            && self.nesting_depth == self.arguments_depth
            && self.at_arg_key_start()
        {
            self.state = JsonState::InArgKey;
            self.constrained_buf.clear();
            return;
        }

        // Value string: `:"` opens a string value — set in_string to skip its content
        if b == b'"' && self.is_value_quote() {
            self.in_string = true;
        }
    }

    pub fn feed(&mut self, text: &[u8]) {
        for &b in text {
            self.feed_byte(b);
        }
    }

    fn push_tail(&mut self, b: u8) {
        self.tail.push(b);
        if self.tail.len() > TAIL_LEN {
            self.tail.drain(0..self.tail.len() - TAIL_LEN);
        }
    }

    /// Mirrors Python `_at_arg_key_start`: last 2 bytes are `{"` or `,"`.
    fn at_arg_key_start(&self) -> bool {
        let n = self.tail.len();
        if n < 2 {
            return false;
        }
        (self.tail[n - 2] == b'{' || self.tail[n - 2] == b',') && self.tail[n - 1] == b'"'
    }

    /// Mirrors Python `_is_value_quote`: preceding non-whitespace byte is `:`.
    /// Compact JSON has no spaces, so tail[n-2] is always the directly preceding byte.
    fn is_value_quote(&self) -> bool {
        let n = self.tail.len();
        n >= 2 && self.tail[n - 2] == b':'
    }
}

// ─── Tool Definitions ─────────────────────────────────────────────────────────

pub struct ToolDef {
    pub name: String,
    pub snake_name: String,
    pub param_keys: Vec<String>,
}

impl ToolDef {
    pub fn from_json(json: &str) -> Vec<Self> {
        parse_tools_json(json)
    }
}

// ─── Constrained Decoder ──────────────────────────────────────────────────────

pub struct ConstrainedDecoder {
    name_trie: Trie,
    param_tries: HashMap<String, Trie>,
    sm: JsonStateMachine,
    /// Decoded bytes for each vocab token (▁ → ' '), indexed by token ID.
    token_texts: Vec<Vec<u8>>,
}

impl ConstrainedDecoder {
    pub fn new(tool_defs: &[ToolDef], token_bytes: Vec<(u32, Vec<u8>)>) -> Self {
        let mut name_trie = Trie::new();
        let mut param_tries = HashMap::new();

        for tool in tool_defs {
            name_trie.insert(tool.snake_name.as_bytes());
            let mut key_trie = Trie::new();
            for key in &tool.param_keys {
                key_trie.insert(key.as_bytes());
            }
            param_tries.insert(tool.snake_name.clone(), key_trie);
        }

        let max_id = token_bytes
            .iter()
            .map(|(id, _)| *id as usize)
            .max()
            .unwrap_or(0);
        let mut token_texts = vec![Vec::new(); max_id + 1];
        for (id, bytes) in token_bytes {
            if (id as usize) <= max_id {
                token_texts[id as usize] = bytes;
            }
        }

        Self {
            name_trie,
            param_tries,
            sm: JsonStateMachine::new(),
            token_texts,
        }
    }

    /// Advance the state machine after emitting `token_id`.
    pub fn update(&mut self, token_id: u32) {
        if let Some(text) = self.token_texts.get(token_id as usize) {
            let text = text.clone();
            self.sm.feed(&text);
        }
    }

    /// Feed raw output bytes directly into the state machine.
    /// Useful for driving the decoder to a specific state in tests or streaming
    /// contexts where byte sequences arrive outside the normal token-update path.
    pub fn feed_bytes(&mut self, bytes: &[u8]) {
        self.sm.feed(bytes);
    }

    /// Build additive logit bias mask for the current state.
    /// Free → all 0.0.  Constrained → 0.0 for valid tokens, -1e9 for invalid.
    pub fn logit_mask(&self, vocab_size: usize) -> Vec<f32> {
        let texts = &self.token_texts;
        match &self.sm.state {
            JsonState::Free => vec![0.0f32; vocab_size],
            JsonState::InName => match self.name_trie.advance(0, &self.sm.constrained_buf) {
                Some(node) => build_mask_from_trie(&self.name_trie, node, texts, vocab_size),
                None => vec![0.0f32; vocab_size],
            },
            JsonState::InArgKey => match self.param_tries.get(&self.sm.current_function) {
                Some(trie) => match trie.advance(0, &self.sm.constrained_buf) {
                    Some(node) => build_mask_from_trie(trie, node, texts, vocab_size),
                    None => vec![0.0f32; vocab_size],
                },
                None => vec![0.0f32; vocab_size],
            },
        }
    }
}

// ─── Tool JSON Parser ─────────────────────────────────────────────────────────

fn parse_tools_json(json: &str) -> Vec<ToolDef> {
    let bytes = json.as_bytes();
    let mut tools = Vec::new();
    let mut i = 0;

    // Skip to opening '[' or '{'
    while i < bytes.len() && bytes[i] != b'[' && bytes[i] != b'{' {
        i += 1;
    }
    if i >= bytes.len() {
        return tools;
    }
    if bytes[i] == b'[' {
        i += 1;
    }

    while i < bytes.len() {
        while i < bytes.len() && (bytes[i] == b',' || bytes[i].is_ascii_whitespace()) {
            i += 1;
        }
        if i >= bytes.len() || bytes[i] == b']' {
            break;
        }
        if bytes[i] != b'{' {
            i += 1;
            continue;
        }

        let obj_end = json_find_matching(bytes, i, b'{', b'}');
        let obj_str = &json[i..obj_end + 1];

        if let Some(tool) = parse_single_tool(obj_str) {
            tools.push(tool);
        }

        i = obj_end + 1;
    }

    tools
}

fn parse_single_tool(obj: &str) -> Option<ToolDef> {
    let name = json_extract_string(obj, "name")?;
    let snake_name = crate::tokenizer::to_snake_case(&name);
    let param_keys = extract_param_keys(obj);
    Some(ToolDef {
        name,
        snake_name,
        param_keys,
    })
}

/// Extract param keys from a tool object.
///
/// Mirrors Python `ToolConstraints.__init__`:
///   `for key, val in params.items(): if isinstance(val, dict): trie.insert(key)`
///
/// Also handles JSON Schema format: if `parameters` has a `"properties"` key,
/// use that object's top-level keys instead.
fn extract_param_keys(tool_obj: &str) -> Vec<String> {
    let params = match json_extract_object(tool_obj, "parameters") {
        Some(p) => p,
        None => return Vec::new(),
    };

    // JSON Schema: "parameters": {"type":"object","properties":{...},...}
    if let Some(props) = json_extract_object(&params, "properties") {
        return json_top_level_keys(&props);
    }

    // Flat format: "parameters": {"key1":{...},"key2":{...}}
    json_keys_with_object_values(&params)
}

/// Find the closing delimiter matching the opening delimiter at `start`.
/// Correctly skips over quoted strings.
fn json_find_matching(bytes: &[u8], start: usize, open: u8, close: u8) -> usize {
    let mut depth = 0usize;
    let mut i = start;
    let mut in_str = false;
    let mut escape = false;
    while i < bytes.len() {
        let b = bytes[i];
        if in_str {
            if escape {
                escape = false;
            } else if b == b'\\' {
                escape = true;
            } else if b == b'"' {
                in_str = false;
            }
        } else if b == b'"' {
            in_str = true;
        } else if b == open {
            depth += 1;
        } else if b == close {
            depth = depth.saturating_sub(1);
            if depth == 0 {
                return i;
            }
        }
        i += 1;
    }
    bytes.len().saturating_sub(1)
}

/// Extract the value of a string field: `"field":"value"` → `"value"`.
fn json_extract_string(obj: &str, field: &str) -> Option<String> {
    let needle = format!("\"{}\":\"", field);
    let pos = obj.find(&needle)?;
    let after = &obj[pos + needle.len()..];
    let bytes = after.as_bytes();
    let mut i = 0;
    let mut escape = false;
    while i < bytes.len() {
        if escape {
            escape = false;
        } else if bytes[i] == b'\\' {
            escape = true;
        } else if bytes[i] == b'"' {
            return Some(after[..i].to_string());
        }
        i += 1;
    }
    None
}

/// Extract the value of an object field: `"field":{...}` → `"{...}"`.
fn json_extract_object(obj: &str, field: &str) -> Option<String> {
    let needle = format!("\"{}\":", field);
    let pos = obj.find(&needle)?;
    let after = obj[pos + needle.len()..].trim_start();
    let bytes = after.as_bytes();
    if bytes.is_empty() || bytes[0] != b'{' {
        return None;
    }
    let end = json_find_matching(bytes, 0, b'{', b'}');
    Some(after[..end + 1].to_string())
}

/// Return all top-level string keys of a JSON object `{...}`.
fn json_top_level_keys(obj: &str) -> Vec<String> {
    let bytes = obj.as_bytes();
    let mut keys = Vec::new();
    let mut i = 0;

    if i < bytes.len() && bytes[i] == b'{' {
        i += 1;
    }

    while i < bytes.len() {
        while i < bytes.len() && (bytes[i] == b',' || bytes[i].is_ascii_whitespace()) {
            i += 1;
        }
        if i >= bytes.len() || bytes[i] == b'}' {
            break;
        }
        if bytes[i] != b'"' {
            i += 1;
            continue;
        }

        // Read key
        i += 1;
        let key_start = i;
        let mut escape = false;
        while i < bytes.len() {
            if escape {
                escape = false;
            } else if bytes[i] == b'\\' {
                escape = true;
            } else if bytes[i] == b'"' {
                break;
            }
            i += 1;
        }
        let key = obj[key_start..i].to_string();
        i += 1; // closing "

        // Skip ':'
        while i < bytes.len() && bytes[i] != b':' {
            i += 1;
        }
        i += 1;
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }

        if !key.is_empty() {
            keys.push(key);
        }

        // Skip value
        i = json_skip_value(bytes, i);
    }

    keys
}

/// Return top-level keys of a JSON object whose values are objects (dicts).
/// Mirrors Python: `for key, val in params.items(): if isinstance(val, dict): insert(key)`.
fn json_keys_with_object_values(obj: &str) -> Vec<String> {
    let bytes = obj.as_bytes();
    let mut keys = Vec::new();
    let mut i = 0;

    if i < bytes.len() && bytes[i] == b'{' {
        i += 1;
    }

    while i < bytes.len() {
        while i < bytes.len() && (bytes[i] == b',' || bytes[i].is_ascii_whitespace()) {
            i += 1;
        }
        if i >= bytes.len() || bytes[i] == b'}' {
            break;
        }
        if bytes[i] != b'"' {
            i += 1;
            continue;
        }

        // Read key
        i += 1;
        let key_start = i;
        let mut escape = false;
        while i < bytes.len() {
            if escape {
                escape = false;
            } else if bytes[i] == b'\\' {
                escape = true;
            } else if bytes[i] == b'"' {
                break;
            }
            i += 1;
        }
        let key = obj[key_start..i].to_string();
        i += 1; // closing "

        // Skip ':'
        while i < bytes.len() && bytes[i] != b':' {
            i += 1;
        }
        i += 1;
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }

        if i < bytes.len() && bytes[i] == b'{' {
            if !key.is_empty() {
                keys.push(key);
            }
            let end = json_find_matching(bytes, i, b'{', b'}');
            i = end + 1;
        } else {
            i = json_skip_value(bytes, i);
        }
    }

    keys
}

/// Skip over a JSON value starting at `i`, return position after it.
fn json_skip_value(bytes: &[u8], i: usize) -> usize {
    let mut i = i;
    if i >= bytes.len() {
        return i;
    }
    match bytes[i] {
        b'{' => json_find_matching(bytes, i, b'{', b'}') + 1,
        b'[' => json_find_matching(bytes, i, b'[', b']') + 1,
        b'"' => {
            i += 1;
            let mut escape = false;
            while i < bytes.len() {
                if escape {
                    escape = false;
                } else if bytes[i] == b'\\' {
                    escape = true;
                } else if bytes[i] == b'"' {
                    i += 1;
                    return i;
                }
                i += 1;
            }
            i
        }
        _ => {
            while i < bytes.len() && bytes[i] != b',' && bytes[i] != b'}' && bytes[i] != b']' {
                i += 1;
            }
            i
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trie_basic() {
        let mut t = Trie::new();
        t.insert(b"get_weather");
        t.insert(b"get_time");

        assert!(check_token_valid(b"get", &t, 0));
        assert!(check_token_valid(b"get_w", &t, 0));
        assert!(check_token_valid(b"get_weather\"", &t, 0));
        // Chars after `"` are structural JSON handled by the state machine — ignored here
        assert!(check_token_valid(b"get_weather\",", &t, 0));
        assert!(!check_token_valid(b"set_", &t, 0));
        assert!(!check_token_valid(b"get_x", &t, 0));
    }

    #[test]
    fn test_state_machine_tool_name() {
        let mut sm = JsonStateMachine::new();
        sm.feed(b"[{\"name\":\"get_weather\",\"arguments\":{}}]");
        assert_eq!(sm.current_function, "get_weather");
    }

    #[test]
    fn test_state_machine_arg_key() {
        let mut sm = JsonStateMachine::new();
        // Feed up to (but not including) the arg key
        sm.feed(b"[{\"name\":\"get_weather\",\"arguments\":{\"");
        assert_eq!(sm.state, JsonState::InArgKey);
        assert!(sm.constrained_buf.is_empty());
    }

    #[test]
    fn test_state_machine_full_sequence() {
        let mut sm = JsonStateMachine::new();
        // Simulate token-by-token feeding
        let output =
            br#"[{"name":"get_weather","arguments":{"location":"Paris","unit":"celsius"}}]"#;
        for &b in output.iter() {
            sm.feed_byte(b);
        }
        assert_eq!(sm.state, JsonState::Free);
        assert_eq!(sm.current_function, "get_weather");
    }

    #[test]
    fn test_parse_tools_flat() {
        let json = r#"[{"name":"get_weather","description":"Get weather","parameters":{"location":{"type":"string"},"unit":{"type":"string"}}}]"#;
        let tools = parse_tools_json(json);
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].snake_name, "get_weather");
        assert!(tools[0].param_keys.contains(&"location".to_string()));
        assert!(tools[0].param_keys.contains(&"unit".to_string()));
    }

    #[test]
    fn test_parse_tools_json_schema() {
        let json = r#"[{"name":"get_weather","parameters":{"type":"object","properties":{"location":{"type":"string"},"unit":{"type":"string"}},"required":["location"]}}]"#;
        let tools = parse_tools_json(json);
        assert_eq!(tools.len(), 1);
        assert!(tools[0].param_keys.contains(&"location".to_string()));
        assert!(tools[0].param_keys.contains(&"unit".to_string()));
        // "type", "required" must NOT be in param_keys
        assert!(!tools[0].param_keys.contains(&"type".to_string()));
        assert!(!tools[0].param_keys.contains(&"required".to_string()));
    }

    #[test]
    fn test_constrained_decoder_name_mask() {
        let tools = vec![ToolDef {
            name: "get_weather".to_string(),
            snake_name: "get_weather".to_string(),
            param_keys: vec!["location".to_string(), "unit".to_string()],
        }];
        // token 0 = "get", token 1 = " get", token 2 = "set"
        let token_bytes: Vec<(u32, Vec<u8>)> = vec![
            (0, b"get".to_vec()),
            (1, b" get".to_vec()), // space prefix — invalid (space not in trie)
            (2, b"set".to_vec()),
        ];
        let mut dec = ConstrainedDecoder::new(&tools, token_bytes);

        // Drive state machine into InName by feeding `[{"name":"`
        dec.sm.feed(b"[{\"name\":\"");
        assert_eq!(dec.sm.state, JsonState::InName);

        let mask = dec.logit_mask(3);
        assert_eq!(mask[0], 0.0, "\"get\" should be allowed");
        assert!(mask[1] < 0.0, "\" get\" (space-prefixed) should be blocked");
        assert!(mask[2] < 0.0, "\"set\" should be blocked (not in trie)");
    }
}
