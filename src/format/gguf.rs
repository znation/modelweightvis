//! GGUF parser, layered on top of candle-core's `quantized::gguf_file::Content`.
//!
//! GGUF lays out a file as four sequential regions:
//!   - magic + version + counts (24 bytes)
//!   - metadata KV table (variable size; ends where the tensor info table starts)
//!   - tensor info table (variable size; ends at `tensor_data_offset`)
//!   - tensor data (rest of file)
//!
//! Unlike safetensors there is no length prefix telling us where the metadata
//! ends — the only way to find `tensor_data_offset` is to parse through the
//! KV table and the tensor info table. Callers therefore feed enough of the
//! file prefix to `parse_header` to cover both tables; on overshoot we report
//! a clear error so the data layer can retry with a bigger fetch.

use std::collections::HashMap;
use std::io::Cursor;

use candle_core::quantized::gguf_file::{Content, Value};
use image::Rgb;

use super::dtype::{gguf_tensor_byte_range, Dtype};
use super::types::TensorMeta;

/// Parsed GGUF header — everything except tensor data.
///
/// Wraps candle-core's `Content` to keep the rest of the codebase
/// independent of candle's types. `metadata` is the raw KV table; the
/// architectural layout's `ModelConfig::from_gguf_metadata` reads from it.
pub struct GgufHeader {
    pub tensors: Vec<TensorMeta>,
    /// Absolute byte offset where tensor data starts (after KV + info tables
    /// + any alignment padding).
    pub tensor_data_offset: u64,
    pub metadata: HashMap<String, Value>,
}

/// Canonical name for a raw GGML tensor dtype code that candle-core's
/// `GgmlDType::from_u32` rejects.
///
/// candle implements only F32/F16/BF16 and the Q-series / K-series quants
/// (even on `main`); every other code — the whole IQ ("imatrix") family,
/// the integer types, F64, ternary, MXFP4 — bails out of `Content::read`.
/// We map the rejected codes to names purely so the *warning* can say what
/// the file actually uses; we still can't decode them. Codes candle *does*
/// support are intentionally absent (we only ever look up rejected ones).
/// Values per ggml's `enum ggml_type`.
fn unsupported_ggml_dtype_name(code: u32) -> Option<&'static str> {
    Some(match code {
        16 => "IQ2_XXS",
        17 => "IQ2_XS",
        18 => "IQ3_XXS",
        19 => "IQ1_S",
        20 => "IQ4_NL",
        21 => "IQ3_S",
        22 => "IQ2_S",
        23 => "IQ4_XS",
        24 => "I8",
        25 => "I16",
        26 => "I32",
        27 => "I64",
        28 => "F64",
        29 => "IQ1_M",
        34 => "TQ1_0",
        35 => "TQ2_0",
        39 => "MXFP4",
        _ => return None,
    })
}

/// If `candle_msg` is candle's "unknown dtype for tensor {code}" error,
/// produce a clearer replacement message.
///
/// candle's wording is misleading twice over: the trailing number is the raw
/// GGML *dtype code*, not a tensor index, and "unknown" reads like a corrupt
/// file when it really means "candle doesn't implement this quant". candle
/// also appends a backtrace to the Display string, so we scan for the marker
/// and take only the leading digits after it. Returns `None` for any other
/// error (e.g. the EOF "failed to fill whole buffer") so the caller keeps the
/// original.
fn rewrite_unknown_dtype_message(candle_msg: &str) -> Option<String> {
    const MARK: &str = "unknown dtype for tensor ";
    let pos = candle_msg.find(MARK)?;
    let code: u32 = candle_msg[pos + MARK.len()..]
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect::<String>()
        .parse()
        .ok()?;
    // No "falls back to plain binary" clause here: arbvis's generic
    // plugin-failure handler already appends "— treating as plain binary", so
    // we only state the cause and let that suffix complete the sentence.
    Some(match unsupported_ggml_dtype_name(code) {
        Some(name) => {
            format!("gguf: tensor uses GGML dtype {name} (code {code}), which candle cannot decode")
        }
        None => format!(
            "gguf: tensor uses unsupported GGML dtype code {code}, which candle cannot decode"
        ),
    })
}

/// Translate candle-core's GGUF parse error into something actionable, then
/// wrap it as an `anyhow::Error`. The IQ-family case (see
/// [`rewrite_unknown_dtype_message`]) gets a precise message; everything else
/// keeps candle's original text.
fn explain_gguf_error(e: candle_core::Error) -> anyhow::Error {
    match rewrite_unknown_dtype_message(&e.to_string()) {
        Some(m) => anyhow::anyhow!(m),
        None => anyhow::anyhow!("gguf: header parse failed: {}", e),
    }
}

/// Parse a GGUF file header from raw bytes. The bytes must cover the magic,
/// the KV table, and the tensor info table — at minimum
/// `tensor_data_offset` bytes. `Content::read` errors if the prefix is too
/// short; callers should catch that and retry with a larger fetch.
pub fn parse_header(data: &[u8]) -> anyhow::Result<GgufHeader> {
    let mut reader = Cursor::new(data);
    let content = Content::read(&mut reader).map_err(explain_gguf_error)?;

    let mut tensors = Vec::with_capacity(content.tensor_infos.len());
    for (name, info) in &content.tensor_infos {
        let dtype = Dtype::from_ggml(info.ggml_dtype);
        let (file_start, file_end) = gguf_tensor_byte_range(info, content.tensor_data_offset);
        let shape: Vec<u64> = info.shape.dims().iter().map(|&d| d as u64).collect();
        tensors.push(TensorMeta {
            name: name.clone(),
            dtype,
            shape,
            file_start,
            file_end,
            packed_sidecars: None,
        });
    }
    tensors.sort_by_key(|t| t.file_start);

    Ok(GgufHeader {
        tensors,
        tensor_data_offset: content.tensor_data_offset,
        metadata: content.metadata,
    })
}

/// Build display color ranges for a GGUF file. Three header sub-regions are
/// shown in slightly different greys so the KV table, the info table, and
/// any alignment padding read as distinct bands in the legacy Hilbert dtype
/// view. Tensor data uses the per-tensor dtype color; gaps between tensors
/// are black.
pub fn build_color_ranges(
    tensors: &[TensorMeta],
    tensor_data_offset: u64,
    file_size: u64,
) -> Vec<(u64, u64, Rgb<u8>)> {
    let mut ranges: Vec<(u64, u64, Rgb<u8>)> = Vec::with_capacity(tensors.len() + 2);
    // Combined header region (magic + KV + info table + alignment padding).
    // We don't bother sub-dividing — without re-running the parser we can't
    // tell where the KV table ends and the info table begins, and the
    // visual payoff is small relative to the renderer plumbing.
    if tensor_data_offset > 0 {
        ranges.push((0, tensor_data_offset, Rgb([100, 100, 100])));
    }
    let mut pos = tensor_data_offset;
    for t in tensors {
        if t.file_start > pos {
            ranges.push((pos, t.file_start, Rgb([0, 0, 0])));
        }
        if t.file_end > t.file_start {
            ranges.push((t.file_start, t.file_end, t.dtype.to_color()));
        }
        pos = t.file_end;
    }
    if pos < file_size {
        ranges.push((pos, file_size, Rgb([0, 0, 0])));
    }
    ranges
}

/// Look up a metadata KV value as a u64. Handles the various integer types
/// GGUF KV pairs may use. Returns `None` if the key is missing or has a
/// non-integer type.
pub fn metadata_u64(metadata: &HashMap<String, Value>, key: &str) -> Option<u64> {
    match metadata.get(key)? {
        Value::U8(v) => Some(*v as u64),
        Value::U16(v) => Some(*v as u64),
        Value::U32(v) => Some(*v as u64),
        Value::U64(v) => Some(*v),
        Value::I8(v) if *v >= 0 => Some(*v as u64),
        Value::I16(v) if *v >= 0 => Some(*v as u64),
        Value::I32(v) if *v >= 0 => Some(*v as u64),
        Value::I64(v) if *v >= 0 => Some(*v as u64),
        _ => None,
    }
}

/// Look up a metadata KV value as a string.
pub fn metadata_string<'a>(metadata: &'a HashMap<String, Value>, key: &str) -> Option<&'a str> {
    match metadata.get(key)? {
        Value::String(s) => Some(s.as_str()),
        _ => None,
    }
}

/// Look up the length of an array-valued KV. Useful for e.g.
/// `tokenizer.ggml.tokens` (vocab size = array length).
pub fn metadata_array_len(metadata: &HashMap<String, Value>, key: &str) -> Option<usize> {
    match metadata.get(key)? {
        Value::Array(v) => Some(v.len()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal valid GGUF v2 byte stream with a single string KV pair
    /// (`general.architecture` → `"llama"`) and zero tensors. Exercises the
    /// magic/version/counts parser plus one KV decode path.
    fn synthetic_no_tensor_gguf() -> Vec<u8> {
        let mut bytes = Vec::new();
        // magic = "GGUF" (LE u32)
        bytes.extend_from_slice(&0x46554747u32.to_le_bytes());
        // version 2
        bytes.extend_from_slice(&2u32.to_le_bytes());
        // tensor_count = 0
        bytes.extend_from_slice(&0u64.to_le_bytes());
        // metadata_kv_count = 1
        bytes.extend_from_slice(&1u64.to_le_bytes());
        // KV: key string, type, value
        let key = b"general.architecture";
        bytes.extend_from_slice(&(key.len() as u64).to_le_bytes());
        bytes.extend_from_slice(key);
        // value_type = 8 (String)
        bytes.extend_from_slice(&8u32.to_le_bytes());
        let val = b"llama";
        bytes.extend_from_slice(&(val.len() as u64).to_le_bytes());
        bytes.extend_from_slice(val);
        // Tensor info table is empty.
        // Pad to 32-byte alignment so tensor_data_offset is reachable.
        while bytes.len() % 32 != 0 {
            bytes.push(0);
        }
        bytes
    }

    #[test]
    fn parse_header_minimal_no_tensors() {
        let bytes = synthetic_no_tensor_gguf();
        let header = parse_header(&bytes).expect("parses");
        assert!(header.tensors.is_empty());
        assert_eq!(
            metadata_string(&header.metadata, "general.architecture"),
            Some("llama")
        );
        // `tensor_data_offset` lands at the end of the header (which is also
        // where we padded to; both will be at the same 32-byte boundary).
        assert!(header.tensor_data_offset > 0);
        assert!(header.tensor_data_offset.is_multiple_of(32));
    }

    #[test]
    fn rewrites_iq_dtype_error_with_quant_name() {
        // candle's Display prints the bail! message then a backtrace; we pull
        // the dtype code out of the leading line and name the quant. Code 16
        // is IQ2_XXS — exactly the Unsloth UD-IQ2_XXS case.
        let msg = "unknown dtype for tensor 16\n   1: candle_core::error::Error::bt";
        let out = rewrite_unknown_dtype_message(msg).expect("rewrites");
        assert!(out.contains("IQ2_XXS"), "got {out}");
        assert!(out.contains("code 16"), "got {out}");
        assert!(out.contains("cannot decode"), "got {out}");
    }

    #[test]
    fn rewrites_unmapped_dtype_code_without_a_name() {
        let out = rewrite_unknown_dtype_message("unknown dtype for tensor 250").expect("rewrites");
        assert!(out.contains("250"), "got {out}");
        assert!(!out.contains("IQ"), "got {out}");
    }

    #[test]
    fn leaves_non_dtype_errors_untouched() {
        // The EOF case (prefix too short) must NOT be rewritten — the data
        // layer relies on it to trigger a larger fetch.
        assert!(rewrite_unknown_dtype_message("failed to fill whole buffer").is_none());
    }

    #[test]
    fn build_color_ranges_no_tensors_yields_header_only() {
        let r = build_color_ranges(&[], 64, 128);
        // One header band [0..64) and a trailing-gap band [64..128).
        assert_eq!(r.len(), 2);
        assert_eq!(r[0].0, 0);
        assert_eq!(r[0].1, 64);
        assert_eq!(r[1].0, 64);
        assert_eq!(r[1].1, 128);
    }
}
