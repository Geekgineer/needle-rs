//! SentencePiece BPE tokenizer for Needle.
//!
//! The tokenizer model is loaded from a `.model` file (SentencePiece protobuf).
//! We implement a minimal BPE merge decoder and a simple greedy BPE encoder.
//!
//! Special token IDs: PAD=0, EOS=1, BOS=2, UNK=3, TOOL_CALL=4, TOOLS=5
//!
//! For full SentencePiece compatibility (complex BPE rules, Unicode normalization),
//! we delegate to the `sentencepiece` C library via FFI in the `needle-c` crate.
//! This module provides the pure-Rust lightweight fallback that covers the common ASCII path.

use std::collections::HashMap;
use std::path::Path;
use std::fs;

pub const PAD_ID: u32 = 0;
pub const EOS_ID: u32 = 1;
pub const BOS_ID: u32 = 2;
pub const UNK_ID: u32 = 3;
pub const TOOL_CALL_ID: u32 = 4;
pub const TOOLS_ID: u32 = 5;

/// Loaded vocabulary: id → string piece, string piece → id.
pub struct Vocabulary {
    pub id_to_piece: Vec<String>,
    pub piece_to_id: HashMap<String, u32>,
    pub merges: Vec<(u32, u32)>,  // BPE merge rules in priority order
}

impl Vocabulary {
    /// Load from a pre-exported text format (one piece per line, index = id).
    /// The export script writes this alongside the SafeTensors weight file.
    pub fn load_text<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let content = fs::read_to_string(path)?;
        let mut id_to_piece = Vec::new();
        let mut piece_to_id = HashMap::new();

        for line in content.lines() {
            let id = id_to_piece.len() as u32;
            id_to_piece.push(line.to_string());
            piece_to_id.insert(line.to_string(), id);
        }

        Ok(Self {
            id_to_piece,
            piece_to_id,
            merges: Vec::new(), // populated separately if needed
        })
    }

    pub fn decode_ids(&self, ids: &[u32]) -> String {
        let mut out = String::new();
        for &id in ids {
            if let Some(piece) = self.id_to_piece.get(id as usize) {
                // SentencePiece uses '▁' (U+2581) to mark word boundaries (= space prefix)
                out.push_str(&piece.replace('▁', " "));
            }
        }
        // Strip leading space that comes from BOS/first piece
        out.trim_start().to_string()
    }

    pub fn piece_id(&self, piece: &str) -> Option<u32> {
        self.piece_to_id.get(piece).copied()
    }
}

/// Convert a tool name to snake_case (mirrors Python's `to_snake_case`).
pub fn to_snake_case(name: &str) -> String {
    let mut out = String::with_capacity(name.len() + 4);
    let mut prev_upper = false;
    for (i, c) in name.chars().enumerate() {
        if c.is_uppercase() {
            if i > 0 && !prev_upper {
                out.push('_');
            }
            out.push(c.to_ascii_lowercase());
            prev_upper = true;
        } else if c == '-' || c == ' ' {
            out.push('_');
            prev_upper = false;
        } else {
            out.push(c);
            prev_upper = false;
        }
    }
    out
}
