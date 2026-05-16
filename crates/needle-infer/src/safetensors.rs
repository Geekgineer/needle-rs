//! Minimal SafeTensors reader — zero external dependencies.
//!
//! Format: 8-byte LE u64 header length, then JSON header, then tensor data.
//! Header JSON: { "tensor_name": { "dtype": "F32"|"BF16"|"F16"|"I8", "shape": [...], "data_offsets": [start, end] }, "__metadata__": {...} }
//!
//! We support: F32, BF16, F16, I8, I4 (our custom packed format).

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct TensorMeta {
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub data_start: usize,
    pub data_end: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DType {
    F32,
    BF16,
    F16,
    I8,
    I4,     // custom: packed nibbles
}

pub struct SafeTensors {
    data: Vec<u8>,
    pub tensors: HashMap<String, TensorMeta>,
}

impl SafeTensors {
    pub fn load<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let mut f = File::open(path)?;

        // Read 8-byte header length
        let mut len_buf = [0u8; 8];
        f.read_exact(&mut len_buf)?;
        let header_len = u64::from_le_bytes(len_buf) as usize;

        // Read JSON header
        let mut header_bytes = vec![0u8; header_len];
        f.read_exact(&mut header_bytes)?;
        let header_str = std::str::from_utf8(&header_bytes)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "invalid UTF-8 header"))?;

        let tensors = parse_header(header_str)?;

        // Read remainder (tensor data)
        let mut data = Vec::new();
        f.read_to_end(&mut data)?;

        Ok(Self { data, tensors })
    }

    /// Return tensor data as f32 slice, converting from stored dtype.
    pub fn get_f32(&self, name: &str) -> Option<Vec<f32>> {
        let meta = self.tensors.get(name)?;
        let raw = &self.data[meta.data_start..meta.data_end];
        let out = match meta.dtype {
            DType::F32 => {
                let mut v = vec![0.0f32; raw.len() / 4];
                for (i, chunk) in raw.chunks_exact(4).enumerate() {
                    v[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                }
                v
            }
            DType::BF16 => {
                let mut v = vec![0.0f32; raw.len() / 2];
                for (i, chunk) in raw.chunks_exact(2).enumerate() {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]) as u32;
                    v[i] = f32::from_bits(bits << 16);
                }
                v
            }
            DType::F16 => {
                let mut v = vec![0.0f32; raw.len() / 2];
                for (i, chunk) in raw.chunks_exact(2).enumerate() {
                    v[i] = f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]));
                }
                v
            }
            DType::I8 => {
                raw.iter().map(|&b| b as i8 as f32).collect()
            }
            DType::I4 => {
                // Packed nibbles: 2 values per byte
                let mut v = Vec::with_capacity(raw.len() * 2);
                for &byte in raw {
                    let lo = (byte & 0x0F) as u8;
                    let hi = (byte >> 4) as u8;
                    v.push(sign_extend4(lo) as f32);
                    v.push(sign_extend4(hi) as f32);
                }
                v
            }
        };
        Some(out)
    }

    /// Return raw bytes for a tensor (for quantized weights with separate scale).
    pub fn get_raw(&self, name: &str) -> Option<&[u8]> {
        let meta = self.tensors.get(name)?;
        Some(&self.data[meta.data_start..meta.data_end])
    }

    pub fn meta(&self, name: &str) -> Option<&TensorMeta> {
        self.tensors.get(name)
    }
}

fn parse_header(json: &str) -> io::Result<HashMap<String, TensorMeta>> {
    let mut tensors = HashMap::new();

    // Minimal JSON parser — SafeTensors header is a flat object with known structure.
    // We don't want a full JSON library, so parse manually.
    let json = json.trim();
    if !json.starts_with('{') || !json.ends_with('}') {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "expected JSON object"));
    }

    // Split on top-level '"key":' patterns
    let inner = &json[1..json.len() - 1];
    let entries = split_top_level_entries(inner);

    for (key, val) in entries {
        if key == "__metadata__" {
            continue;
        }
        // val looks like: { "dtype": "BF16", "shape": [512, 512], "data_offsets": [0, 524288] }
        let dtype = extract_str_field(val, "dtype").ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, format!("missing dtype for {key}"))
        })?;
        let shape = extract_usize_array(val, "shape").ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, format!("missing shape for {key}"))
        })?;
        let offsets = extract_usize_array(val, "data_offsets").ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, format!("missing data_offsets for {key}"))
        })?;
        if offsets.len() != 2 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "data_offsets must have 2 elements"));
        }

        let dtype = match dtype {
            "F32" => DType::F32,
            "BF16" => DType::BF16,
            "F16" => DType::F16,
            "I8" => DType::I8,
            "I4" => DType::I4,
            other => return Err(io::Error::new(io::ErrorKind::InvalidData, format!("unknown dtype {other}"))),
        };

        tensors.insert(key.to_string(), TensorMeta {
            dtype,
            shape,
            data_start: offsets[0],
            data_end: offsets[1],
        });
    }

    Ok(tensors)
}

/// Split a JSON object body into (key, value_str) pairs at the top level.
fn split_top_level_entries(s: &str) -> Vec<(&str, &str)> {
    let mut result = Vec::new();
    let bytes = s.as_bytes();
    let mut i = 0;

    while i < bytes.len() {
        // Skip whitespace and commas
        while i < bytes.len() && (bytes[i] == b',' || bytes[i].is_ascii_whitespace()) {
            i += 1;
        }
        if i >= bytes.len() {
            break;
        }

        // Expect opening quote for key
        if bytes[i] != b'"' {
            break;
        }
        i += 1;
        let key_start = i;
        while i < bytes.len() && bytes[i] != b'"' {
            i += 1;
        }
        let key = &s[key_start..i];
        i += 1; // skip closing quote

        // Skip ':'
        while i < bytes.len() && bytes[i] != b':' {
            i += 1;
        }
        i += 1;

        // Skip whitespace
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }

        // Read value (object or string)
        let val_start = i;
        let val = if bytes[i] == b'{' {
            let end = find_matching_brace(s, i);
            i = end + 1;
            &s[val_start..end + 1]
        } else if bytes[i] == b'"' {
            i += 1;
            while i < bytes.len() && bytes[i] != b'"' {
                i += 1;
            }
            i += 1;
            &s[val_start..i]
        } else {
            while i < bytes.len() && bytes[i] != b',' && bytes[i] != b'}' {
                i += 1;
            }
            &s[val_start..i]
        };

        result.push((key, val));
    }

    result
}

fn find_matching_brace(s: &str, start: usize) -> usize {
    let bytes = s.as_bytes();
    let mut depth = 0;
    let mut i = start;
    while i < bytes.len() {
        match bytes[i] {
            b'{' | b'[' => depth += 1,
            b'}' | b']' => {
                depth -= 1;
                if depth == 0 {
                    return i;
                }
            }
            _ => {}
        }
        i += 1;
    }
    s.len() - 1
}

fn extract_str_field<'a>(obj: &'a str, field: &str) -> Option<&'a str> {
    let needle = format!("\"{field}\"");
    let pos = obj.find(&needle)?;
    let after = &obj[pos + needle.len()..];
    let colon = after.find(':')? + 1;
    let after = after[colon..].trim_start();
    if after.starts_with('"') {
        let inner = &after[1..];
        let end = inner.find('"')?;
        Some(&inner[..end])
    } else {
        None
    }
}

fn extract_usize_array(obj: &str, field: &str) -> Option<Vec<usize>> {
    let needle = format!("\"{field}\"");
    let pos = obj.find(&needle)?;
    let after = &obj[pos + needle.len()..];
    let bracket = after.find('[')? + 1;
    let after = &after[bracket..];
    let end = after.find(']')?;
    let inner = &after[..end];

    let values: Option<Vec<usize>> = inner.split(',')
        .map(|s| s.trim().parse::<usize>().ok())
        .collect();
    values
}

#[inline]
fn sign_extend4(nibble: u8) -> i8 {
    if nibble & 0x8 != 0 {
        (nibble | 0xF0) as i8
    } else {
        nibble as i8
    }
}

/// Convert IEEE 754 half-precision f16 to f32.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) as u32) << 31;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;

    if exp == 0 {
        // Subnormal
        let val = mant as f32 / (1 << 24) as f32;
        let out = if sign != 0 { -val } else { val };
        return out;
    }
    if exp == 31 {
        // Inf or NaN
        return f32::from_bits(sign | 0x7F80_0000 | (mant << 13));
    }

    let exp32 = (exp + 127 - 15) << 23;
    f32::from_bits(sign | exp32 | (mant << 13))
}
