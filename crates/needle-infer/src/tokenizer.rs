//! SentencePiece BPE tokenizer for Needle.
//!
//! Implements the correct BPE merge algorithm, matching SentencePiece exactly:
//!   1. Trim + normalize: prepend ▁, collapse whitespace runs to single ▁.
//!   2. Split into initial units: for chars in vocab → single-char piece;
//!      for chars not in vocab → individual UTF-8 bytes as <0xXX> fallback pieces.
//!   3. Iteratively merge the adjacent pair with the highest piece score
//!      (score = -(merge order), so 0.0 = earliest/highest priority).
//!   4. Stop when no scoreable adjacent pair remains.
//!
//! Vocab file format (tab-separated, one piece per line):
//!   <piece>\t<score>
//! Legacy format (no tab) is still accepted; falls back to greedy longest-match.
//!
//! Special token IDs: PAD=0, EOS=1, BOS=2, UNK=3, TOOL_CALL=4, TOOLS=5

use std::collections::HashMap;
use std::fs;
use std::path::Path;

pub const PAD_ID: u32 = 0;
pub const EOS_ID: u32 = 1;
pub const BOS_ID: u32 = 2;
pub const UNK_ID: u32 = 3;
pub const TOOL_CALL_ID: u32 = 4;
pub const TOOLS_ID: u32 = 5;

/// ▁ (U+2581 LOWER ONE EIGHTH BLOCK) used by SentencePiece for word boundaries.
const SP_SPACE: char = '\u{2581}';

/// Loaded vocabulary: id → piece string, piece string → id, and BPE scores.
pub struct Vocabulary {
    pub id_to_piece: Vec<String>,
    pub piece_to_id: HashMap<String, u32>,
    /// BPE merge scores, indexed by piece id. Empty if vocab file has no scores (legacy).
    scores: Vec<f32>,
    /// Maximum piece length in bytes (for greedy fallback).
    max_piece_bytes: usize,
}

impl Vocabulary {
    /// Load from the exported text format (one `piece\tscore` per line, index = id).
    pub fn load_text<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let content = fs::read_to_string(path)?;
        Ok(Self::parse(&content))
    }

    /// Parse from an already-loaded vocabulary string (WASM / in-memory use).
    pub fn parse(content: &str) -> Self {
        let mut id_to_piece = Vec::new();
        let mut piece_to_id = HashMap::new();
        let mut scores: Vec<f32> = Vec::new();
        let mut max_piece_bytes = 0usize;
        let mut has_scores = false;

        for line in content.lines() {
            let id = id_to_piece.len() as u32;
            let (piece, score) = if let Some(tab) = line.find('\t') {
                has_scores = true;
                let p = &line[..tab];
                let s: f32 = line[tab + 1..].trim().parse().unwrap_or(0.0);
                (p.to_string(), s)
            } else {
                (line.to_string(), 0.0f32)
            };
            max_piece_bytes = max_piece_bytes.max(piece.len());
            piece_to_id.insert(piece.clone(), id);
            id_to_piece.push(piece);
            scores.push(score);
        }

        if !has_scores {
            scores.clear(); // signal: use greedy
        }

        Self {
            id_to_piece,
            piece_to_id,
            scores,
            max_piece_bytes,
        }
    }

    /// Tokenize raw text to token IDs.
    ///
    /// Uses BPE merge algorithm when scores are available (new vocab format),
    /// otherwise falls back to greedy longest-match (legacy vocab format).
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let text = text.trim();
        if text.is_empty() {
            return Vec::new();
        }
        let normalized = normalize_sp(text);
        if self.scores.is_empty() {
            self.encode_greedy(normalized.as_bytes())
        } else {
            self.encode_bpe(&normalized)
        }
    }

    /// BPE merge algorithm matching SentencePiece exactly.
    fn encode_bpe(&self, normalized: &str) -> Vec<u32> {
        // Step 1: build initial sequence — one entry per character, with byte-fallback
        // for chars not directly in the vocabulary.
        let mut pieces: Vec<u32> = Vec::with_capacity(normalized.len());
        let mut piece_strings: Vec<String> = Vec::with_capacity(normalized.len());

        for c in normalized.chars() {
            let s = c.to_string();
            if let Some(&id) = self.piece_to_id.get(&s) {
                piece_strings.push(s);
                pieces.push(id);
            } else {
                // Byte fallback: emit one <0xXX> token per byte of the UTF-8 encoding.
                for &b in s.as_bytes() {
                    let fb = byte_fallback_piece(b);
                    let id = self.piece_to_id.get(&fb).copied().unwrap_or(UNK_ID);
                    piece_strings.push(fb);
                    pieces.push(id);
                }
            }
        }

        // Step 2: iteratively merge the adjacent pair with the highest BPE score.
        // Complexity: O(n²) over piece count — fine for sequences < 1024 chars.
        loop {
            let mut best_score = f32::NEG_INFINITY;
            let mut best_i = usize::MAX;
            let mut best_id = 0u32;

            for i in 0..piece_strings.len().saturating_sub(1) {
                // Concatenate the two adjacent pieces
                let mut merged = piece_strings[i].clone();
                merged.push_str(&piece_strings[i + 1]);
                if let Some(&id) = self.piece_to_id.get(&merged) {
                    let score = self.scores[id as usize];
                    if score > best_score {
                        best_score = score;
                        best_i = i;
                        best_id = id;
                    }
                }
            }

            if best_i == usize::MAX {
                break; // no more merges applicable
            }

            // Merge: replace entries at best_i and best_i+1 with merged piece.
            let merged_str = {
                let mut s = piece_strings[best_i].clone();
                s.push_str(&piece_strings[best_i + 1]);
                s
            };
            piece_strings[best_i] = merged_str;
            piece_strings.remove(best_i + 1);
            pieces[best_i] = best_id;
            pieces.remove(best_i + 1);
        }

        pieces
    }

    /// Greedy longest-match (legacy fallback for vocab files without scores).
    fn encode_greedy(&self, bytes: &[u8]) -> Vec<u32> {
        let n = bytes.len();
        let mut ids = Vec::with_capacity(n / 2 + 4);
        let mut pos = 0;

        while pos < n {
            let window_end = (pos + self.max_piece_bytes).min(n);
            let mut end = window_end;
            while end > pos && end < n && is_utf8_continuation(bytes[end]) {
                end -= 1;
            }

            let mut found = false;
            while end > pos {
                if let Ok(piece) = std::str::from_utf8(&bytes[pos..end]) {
                    if let Some(&id) = self.piece_to_id.get(piece) {
                        ids.push(id);
                        pos = end;
                        found = true;
                        break;
                    }
                }
                end -= 1;
                while end > pos && is_utf8_continuation(bytes[end]) {
                    end -= 1;
                }
            }

            if !found {
                let fallback = byte_fallback_piece(bytes[pos]);
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
    format!("<0x{byte:02X}>")
}

/// Convert a tool name to snake_case (mirrors Python's `to_snake_case`).
/// Handles camelCase, PascalCase, dot.notation, and hyphen-case.
pub fn to_snake_case(name: &str) -> String {
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
                let prev_lower =
                    i > 0 && (chars[i - 1].is_lowercase() || chars[i - 1].is_ascii_digit());
                let next_lower = i + 1 < n && chars[i + 1].is_lowercase();
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
            if !prev_underscore && !out.is_empty() {
                out.push('_');
                prev_underscore = true;
            }
            prev_upper = false;
        }
    }
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
        let got = vocab.encode("What's the weather?");
        assert_eq!(
            got,
            vec![4279, 8066, 8046, 302, 1149, 8105],
            "pieces: {:?}",
            got.iter()
                .map(|&i| &vocab.id_to_piece[i as usize])
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_encode_single_piece() {
        let Some(vocab) = load_vocab() else { return };
        let got = vocab.encode("description");
        assert_eq!(got, vec![1483]);
    }

    #[test]
    fn test_encode_snake_case_name() {
        let Some(vocab) = load_vocab() else { return };
        let got = vocab.encode("get_weather");
        assert_eq!(
            got,
            vec![1734, 8062, 1331],
            "pieces: {:?}",
            got.iter()
                .map(|&i| &vocab.id_to_piece[i as usize])
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_encode_json_fragment() {
        let Some(vocab) = load_vocab() else { return };
        let got = vocab.encode(r#"{"name":"get_weather"}"#);
        assert_eq!(
            got,
            vec![857, 294, 264, 358, 8062, 1331, 8039, 8059],
            "pieces: {:?}",
            got.iter()
                .map(|&i| &vocab.id_to_piece[i as usize])
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let Some(vocab) = load_vocab() else { return };
        let text = "get weather information";
        let ids = vocab.encode(text);
        let decoded = vocab.decode_ids(&ids);
        assert_eq!(
            decoded.trim(),
            text,
            "roundtrip failed: encoded={ids:?} decoded={decoded:?}"
        );
    }

    /// BPE-sensitive: greedy picks "ran"+"ce", BPE correctly picks "r"+"ance".
    #[test]
    fn test_encode_bpe_france() {
        let Some(vocab) = load_vocab() else { return };
        // Python: sp.Encode("What is the capital of France?")
        // = ['▁What', '▁is', '▁the', '▁cap', 'ital', '▁of', '▁F', 'r', 'ance', '?']
        let got = vocab.encode("What is the capital of France?");
        assert_eq!(
            got,
            vec![4279, 743, 302, 4682, 1720, 326, 1295, 8042, 454, 8105],
            "pieces: {:?}",
            got.iter()
                .map(|&i| &vocab.id_to_piece[i as usize])
                .collect::<Vec<_>>()
        );
    }

    /// BPE-sensitive: "Python" → ['▁P', 'y', 'th', 'on'], not greedy ['▁P', 'yt', 'ho', 'n'].
    #[test]
    fn test_encode_bpe_python() {
        let Some(vocab) = load_vocab() else { return };
        // Python: sp.Encode("Search for Python tutorials online")
        // = ['▁Search', '▁for', '▁P', 'y', 'th', 'on', '▁tu', 'to', 'rial', 's', '▁online']
        let got = vocab.encode("Search for Python tutorials online");
        assert_eq!(
            got,
            vec![5252, 345, 953, 8061, 548, 268, 3843, 369, 2149, 8046, 2856],
            "pieces: {:?}",
            got.iter()
                .map(|&i| &vocab.id_to_piece[i as usize])
                .collect::<Vec<_>>()
        );
    }

    /// BPE-sensitive: "New York" → ['▁New', '▁', 'York'], not greedy ['▁New', '▁Y', 'ork'].
    #[test]
    fn test_encode_bpe_new_york() {
        let Some(vocab) = load_vocab() else { return };
        // Python: sp.Encode("Please book a flight from New York")
        // = ['▁Please', '▁book', '▁a', '▁flight', '▁from', '▁New', '▁', 'York']
        let got = vocab.encode("Please book a flight from New York");
        assert_eq!(
            got,
            vec![2975, 5091, 289, 1364, 564, 6753, 8041, 3210],
            "pieces: {:?}",
            got.iter()
                .map(|&i| &vocab.id_to_piece[i as usize])
                .collect::<Vec<_>>()
        );
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
