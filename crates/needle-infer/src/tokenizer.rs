//! SentencePiece BPE tokenizer for Needle.
//!
//! Implements greedy longest-match tokenization (matching SentencePiece BPE output)
//! using only the exported vocabulary text file — no .model protobuf required.
//!
//! Normalization (matches SentencePiece identity + byte_fallback=true):
//!   1. Trim leading/trailing whitespace.
//!   2. Prepend ▁ (U+2581) to the whole string.
//!   3. Replace each run of internal whitespace with a single ▁.
//!   4. Greedy longest-match over the piece vocabulary.
//!   5. Byte fallback: any byte not covered by a piece emits <0xXX>.
//!
//! Special token IDs: PAD=0, EOS=1, BOS=2, UNK=3, TOOL_CALL=4, TOOLS=5

use std::collections::HashMap;
use std::path::Path;
use std::fs;

pub const PAD_ID: u32 = 0;
pub const EOS_ID: u32 = 1;
pub const BOS_ID: u32 = 2;
pub const UNK_ID: u32 = 3;
pub const TOOL_CALL_ID: u32 = 4;
pub const TOOLS_ID: u32 = 5;

/// ▁ (U+2581 LOWER ONE EIGHTH BLOCK) used by SentencePiece for word boundaries.
const SP_SPACE: char = '\u{2581}';

/// Loaded vocabulary: id → piece string, piece string → id.
pub struct Vocabulary {
    pub id_to_piece: Vec<String>,
    pub piece_to_id: HashMap<String, u32>,
    /// BPE merge rules in priority order (optional; empty if not exported).
    pub merges: Vec<(u32, u32)>,
    /// Maximum piece length in bytes (for greedy search window).
    max_piece_bytes: usize,
}

impl Vocabulary {
    /// Load from a pre-exported text format (one piece per line, index = id).
    pub fn load_text<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let content = fs::read_to_string(path)?;
        let mut id_to_piece = Vec::new();
        let mut piece_to_id = HashMap::new();
        let mut max_piece_bytes = 0usize;

        for line in content.lines() {
            let id = id_to_piece.len() as u32;
            let piece = line.to_string();
            max_piece_bytes = max_piece_bytes.max(piece.len());
            id_to_piece.push(piece.clone());
            piece_to_id.insert(piece, id);
        }

        Ok(Self { id_to_piece, piece_to_id, merges: Vec::new(), max_piece_bytes })
    }

    /// Tokenize raw text to token IDs using greedy longest-match BPE.
    ///
    /// Matches SentencePiece BPE with `normalization_rule_name="identity"` and
    /// `byte_fallback=true`. Produces the same token IDs as `sp.Encode(text)` for
    /// the Needle tokenizer on common ASCII + JSON inputs.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let text = text.trim();
        if text.is_empty() {
            return Vec::new();
        }

        // Normalize: prepend ▁, replace whitespace runs with ▁
        let normalized = normalize_sp(text);
        let bytes = normalized.as_bytes();
        let n = bytes.len();

        let mut ids = Vec::with_capacity(n / 2 + 4);
        let mut pos = 0;

        while pos < n {
            // Try lengths from max_piece_bytes down to 1, at valid UTF-8 boundaries.
            let window_end = (pos + self.max_piece_bytes).min(n);

            // Snap window_end to a valid UTF-8 char start.
            // `end == n` is always valid (exclusive bound); only check bytes[end] when end < n.
            let mut end = window_end;
            while end > pos && end < n && is_utf8_continuation(bytes[end]) {
                end -= 1;
            }

            let mut found = false;
            while end > pos {
                // Safety: we only slice at valid UTF-8 char boundaries
                if let Ok(piece) = std::str::from_utf8(&bytes[pos..end]) {
                    if let Some(&id) = self.piece_to_id.get(piece) {
                        ids.push(id);
                        pos += end - pos;
                        found = true;
                        break;
                    }
                }
                // Step back by one char
                end -= 1;
                while end > pos && is_utf8_continuation(bytes[end]) {
                    end -= 1;
                }
            }

            if !found {
                // Byte fallback: emit <0xXX> piece for the raw byte value
                let byte = bytes[pos];
                let fallback = byte_fallback_piece(byte);
                ids.push(self.piece_to_id.get(&fallback).copied().unwrap_or(UNK_ID));
                pos += 1;
            }
        }

        ids
    }

    /// Decode token IDs back to text (for output from the decoder).
    pub fn decode_ids(&self, ids: &[u32]) -> String {
        let mut out = String::new();
        for &id in ids {
            if let Some(piece) = self.id_to_piece.get(id as usize) {
                out.push_str(&piece.replace(SP_SPACE, " "));
            }
        }
        out.trim_start().to_string()
    }

    pub fn piece_id(&self, piece: &str) -> Option<u32> {
        self.piece_to_id.get(piece).copied()
    }
}

/// SentencePiece identity normalization:
/// prepend ▁, collapse whitespace runs to single ▁.
fn normalize_sp(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + SP_SPACE.len_utf8());
    out.push(SP_SPACE);
    let mut prev_space = false;
    for c in text.chars() {
        if c.is_whitespace() {
            if !prev_space {
                out.push(SP_SPACE);
                prev_space = true;
            }
        } else {
            out.push(c);
            prev_space = false;
        }
    }
    out
}

#[inline(always)]
fn is_utf8_continuation(byte: u8) -> bool {
    (byte & 0xC0) == 0x80
}

fn byte_fallback_piece(byte: u8) -> String {
    // SentencePiece format: <0xXX> with exactly 2 uppercase hex digits
    format!("<0x{byte:02X}>")
}

/// Convert a tool name to snake_case (mirrors Python's `to_snake_case`).
/// Handles camelCase, PascalCase, dot.notation, and hyphen-case.
pub fn to_snake_case(name: &str) -> String {
    // Replace non-alphanumeric/underscore runs with _
    let mut out = String::with_capacity(name.len() + 4);
    let mut prev_underscore = false;
    let mut prev_upper = false;
    let chars: Vec<char> = name.chars().collect();
    let n = chars.len();
    for i in 0..n {
        let c = chars[i];
        if c.is_alphanumeric() || c == '_' {
            if c == '_' {
                if !prev_underscore && !out.is_empty() {
                    out.push('_');
                    prev_underscore = true;
                }
                prev_upper = false;
            } else if c.is_uppercase() {
                // Insert _ before uppercase if following lowercase/digit
                let prev_lower = i > 0 && (chars[i-1].is_lowercase() || chars[i-1].is_ascii_digit());
                // Insert _ between run of uppers and next upper+lower (e.g. HTMLParser → HTML_Parser)
                let next_lower = i + 1 < n && chars[i+1].is_lowercase();
                if i > 0 && !prev_underscore && (prev_lower || (prev_upper && next_lower)) {
                    out.push('_');
                }
                out.push(c.to_ascii_lowercase());
                prev_underscore = false;
                prev_upper = true;
            } else {
                out.push(c);
                prev_underscore = false;
                prev_upper = false;
            }
        } else {
            // Non-alphanumeric: replace with _
            if !prev_underscore && !out.is_empty() {
                out.push('_');
                prev_underscore = true;
            }
            prev_upper = false;
        }
    }
    // Strip trailing underscores
    while out.ends_with('_') {
        out.pop();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const VOCAB_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../weights/vocab.txt");

    fn load_vocab() -> Option<Vocabulary> {
        if !std::path::Path::new(VOCAB_PATH).exists() {
            return None;
        }
        Vocabulary::load_text(VOCAB_PATH).ok()
    }

    /// Verified against Python: sp.Encode(text, out_type=int) with needle.model
    #[test]
    fn test_encode_weather_query() {
        let Some(vocab) = load_vocab() else {
            eprintln!("SKIP: vocab.txt not found");
            return;
        };
        // Python: "What's the weather?" → [4279, 8066, 8046, 302, 1149, 8105]
        // pieces: ['▁What', "'", 's', '▁the', '▁weather', '?']
        let got = vocab.encode("What's the weather?");
        assert_eq!(got, vec![4279, 8066, 8046, 302, 1149, 8105],
            "pieces: {:?}", got.iter().map(|&i| &vocab.id_to_piece[i as usize]).collect::<Vec<_>>());
    }

    #[test]
    fn test_encode_single_piece() {
        let Some(vocab) = load_vocab() else { return };
        // Python: "description" → [1483]   pieces: ['▁description']
        let got = vocab.encode("description");
        assert_eq!(got, vec![1483]);
    }

    #[test]
    fn test_encode_snake_case_name() {
        let Some(vocab) = load_vocab() else { return };
        // Python: "get_weather" → [1734, 8062, 1331]  pieces: ['▁get', '_', 'weather']
        let got = vocab.encode("get_weather");
        assert_eq!(got, vec![1734, 8062, 1331],
            "pieces: {:?}", got.iter().map(|&i| &vocab.id_to_piece[i as usize]).collect::<Vec<_>>());
    }

    #[test]
    fn test_encode_json_fragment() {
        let Some(vocab) = load_vocab() else { return };
        // Python: '{"name":"get_weather"}' → [857, 294, 264, 358, 8062, 1331, 8039, 8059]
        // pieces: ['▁{"', 'name', '":"', 'get', '_', 'weather', '"', '}']
        let got = vocab.encode(r#"{"name":"get_weather"}"#);
        assert_eq!(got, vec![857, 294, 264, 358, 8062, 1331, 8039, 8059],
            "pieces: {:?}", got.iter().map(|&i| &vocab.id_to_piece[i as usize]).collect::<Vec<_>>());
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let Some(vocab) = load_vocab() else { return };
        let text = "get weather information";
        let ids = vocab.encode(text);
        let decoded = vocab.decode_ids(&ids);
        assert_eq!(decoded.trim(), text,
            "roundtrip failed: encoded={ids:?} decoded={decoded:?}");
    }

    #[test]
    fn test_to_snake_case() {
        assert_eq!(to_snake_case("getWeather"), "get_weather");
        assert_eq!(to_snake_case("GetWeather"), "get_weather");
        assert_eq!(to_snake_case("get-weather"), "get_weather");
        assert_eq!(to_snake_case("get.weather"), "get_weather");
        assert_eq!(to_snake_case("HTMLParser"), "html_parser");
        assert_eq!(to_snake_case("already_snake"), "already_snake");
    }
}
