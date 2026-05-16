//! Grammar-constrained decoding via character-level trie.
//!
//! Mirrors Python `needle/model/constrained.py`.
//! Constrains the tool name slot and argument key slots to only valid tokens
//! from the set of provided tool definitions.
//!
//! Output format: [{"name":"<tool_name>","arguments":{"<key>":value}}]
//!
//! The decoder starts at `ConstraintState::ToolName` after generating `<tool_call>`.

use std::collections::HashMap;

/// Node in the character-level trie.
#[derive(Default)]
struct TrieNode {
    children: HashMap<u8, usize>, // byte → child node index
    is_terminal: bool,
}

pub struct Trie {
    nodes: Vec<TrieNode>,
}

impl Trie {
    pub fn new() -> Self {
        Self { nodes: vec![TrieNode::default()] }
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

    /// Return all terminal nodes reachable from `node` by consuming `bytes`.
    /// Returns None if the prefix is not in the trie.
    pub fn advance(&self, node: usize, bytes: &[u8]) -> Option<usize> {
        let mut cur = node;
        for &b in bytes {
            cur = *self.nodes[cur].children.get(&b)?;
        }
        Some(cur)
    }

    pub fn is_terminal(&self, node: usize) -> bool {
        self.nodes[node].is_terminal
    }

    /// Collect all token IDs whose string representation is a valid prefix
    /// from `node`. Used to build the allowed-token mask.
    pub fn valid_next_tokens<'a>(
        &self,
        node: usize,
        vocab: &'a [(u32, Vec<u8>)], // (token_id, bytes)
    ) -> Vec<u32> {
        vocab.iter()
            .filter_map(|(id, bytes)| {
                if self.advance(node, bytes).is_some() {
                    Some(*id)
                } else {
                    None
                }
            })
            .collect()
    }
}

/// Constraint state machine — tracks where in the output format we are.
#[derive(Debug, Clone, PartialEq)]
pub enum ConstraintState {
    /// Free generation (no constraint)
    Free,
    /// Generating the tool name value after `"name":"`
    ToolName { trie_node: usize },
    /// Generating an argument key after `{"` or `,"` in the arguments object
    ArgKey { trie_node: usize },
}

pub struct ConstrainedDecoder {
    tool_name_trie: Trie,
    arg_key_tries: HashMap<String, Trie>, // per-tool argument key trie
    /// Byte representation of each vocabulary token (for trie lookup)
    token_bytes: Vec<(u32, Vec<u8>)>,
}

impl ConstrainedDecoder {
    pub fn new(
        tool_defs: &[ToolDef],
        token_bytes: Vec<(u32, Vec<u8>)>,
    ) -> Self {
        let mut tool_name_trie = Trie::new();
        let mut arg_key_tries = HashMap::new();

        for tool in tool_defs {
            tool_name_trie.insert(tool.snake_name.as_bytes());

            let mut key_trie = Trie::new();
            for key in &tool.param_keys {
                key_trie.insert(key.as_bytes());
            }
            arg_key_tries.insert(tool.snake_name.clone(), key_trie);
        }

        Self { tool_name_trie, arg_key_tries, token_bytes }
    }

    /// Given the current constraint state, compute additive logit bias mask.
    /// Returns a sparse list of (token_id, bias) — bias is 0.0 for allowed, -1e9 for blocked.
    pub fn logit_mask(&self, state: &ConstraintState, vocab_size: usize) -> Vec<f32> {
        match state {
            ConstraintState::Free => vec![0.0f32; vocab_size],
            ConstraintState::ToolName { trie_node } => {
                self.build_mask(*trie_node, &self.tool_name_trie, vocab_size)
            }
            ConstraintState::ArgKey { trie_node } => {
                // We don't know which tool yet — allow any key from any tool.
                // TODO: track current tool name and use per-tool trie.
                self.build_mask(*trie_node, &self.tool_name_trie, vocab_size)
            }
        }
    }

    fn build_mask(&self, node: usize, trie: &Trie, vocab_size: usize) -> Vec<f32> {
        let mut mask = vec![-1e9f32; vocab_size];
        let allowed = trie.valid_next_tokens(node, &self.token_bytes);
        for id in allowed {
            if (id as usize) < vocab_size {
                mask[id as usize] = 0.0;
            }
        }
        mask
    }
}

/// Parsed tool definition (from JSON string).
pub struct ToolDef {
    pub name: String,
    pub snake_name: String,
    pub param_keys: Vec<String>,
}

impl ToolDef {
    pub fn from_json(json: &str) -> Vec<Self> {
        // Minimal parser for tool JSON array.
        // Full JSON parsing belongs in the CLI; this just extracts names and parameter keys.
        parse_tools_json(json)
    }
}

/// Parse tool definitions from JSON string (minimal, no external deps).
fn parse_tools_json(json: &str) -> Vec<ToolDef> {
    let mut tools = Vec::new();
    // Find each {"name":"..."} object
    let mut rest = json;
    while let Some(name_pos) = rest.find("\"name\":\"") {
        let after = &rest[name_pos + 8..];
        let end = after.find('"').unwrap_or(after.len());
        let name = &after[..end];
        let snake = crate::tokenizer::to_snake_case(name);

        // Find parameter keys: look for "parameters":{"properties":{...}}
        let param_keys = extract_param_keys(rest);

        tools.push(ToolDef {
            name: name.to_string(),
            snake_name: snake,
            param_keys,
        });

        // Advance past this tool
        rest = &rest[name_pos + 8 + end..];
    }
    tools
}

fn extract_param_keys(json: &str) -> Vec<String> {
    let mut keys = Vec::new();
    if let Some(props_pos) = json.find("\"properties\":{") {
        let after = &json[props_pos + 14..];
        // Extract keys from {"key1":{...}, "key2":{...}}
        let mut rest = after;
        while rest.starts_with('"') || rest.trim_start().starts_with('"') {
            let rest2 = rest.trim_start();
            if !rest2.starts_with('"') {
                break;
            }
            let inner = &rest2[1..];
            let end = inner.find('"').unwrap_or(inner.len());
            let key = &inner[..end];
            if key == "}" || key.is_empty() {
                break;
            }
            keys.push(key.to_string());
            rest = &inner[end + 1..];
            // Skip past value
            if let Some(next_key) = rest.find("\",\"") {
                rest = &rest[next_key + 2..];
            } else {
                break;
            }
        }
    }
    keys
}
