//! Format-agnostic dtype: plain (safetensors-style, fixed bytes/element) +
//! block-quantized (GGUF-style, fixed bytes/block, fixed elements/block).
//!
//! `TensorElementReader` is the single per-element decode path the renderer
//! and diff math both use. It hides whether an element comes from a plain
//! little-endian read or from a block dequant: callers ask for "element K"
//! and get an `f32`.

use std::borrow::Cow;

use candle_core::quantized::gguf_file::TensorInfo;
use candle_core::quantized::GgmlDType;
use candle_core::CpuStorage;
use image::Rgb;

use super::types::{ABS_LOG_MAX, ABS_LOG_MIN, K_RMS_SAT, RMS_FLOOR};
use super::DiffMetric;

/// How an element is addressed within a contiguous tensor byte buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElementStride {
    /// Fixed bytes per element (every safetensors dtype and the unquantised
    /// GGUF dtypes F32/F16/BF16). Element `k` lives at byte offset
    /// `k * bytes_per_element`.
    Fixed(usize),
    /// Block-quantized: each block of `block_bytes` covers `block_elements`
    /// logical elements. Element `k` lives in block `k / block_elements` at
    /// in-block position `k % block_elements`; the block bytes start at
    /// `(k / block_elements) * block_bytes`.
    Block {
        block_bytes: usize,
        block_elements: usize,
    },
    /// AWQ/GPTQ-style bit-packed integers with out-of-band scales/zeros.
    ///
    /// `bits` low-bits-per-element are packed into a `pack_dtype_bytes`-wide
    /// integer (typically int32, so `pack_dtype_bytes = 4`). Every
    /// `group_size` consecutive elements share one scale (and, for GPTQ,
    /// one zero-point) drawn from sidecar tensors. The dequant formula is
    /// `(q - zero) * scale` for GPTQ, `q * scale` for AWQ.
    ///
    /// Unlike `Block`, the scale/zero data is NOT inline with the quants —
    /// it lives in separately-named tensors. The
    /// [`TensorElementReader::with_sidecars`] builder threads those buffers
    /// through. Element `k` resolves to packed slot
    /// `k / (pack_dtype_bytes * 8 / bits)` with intra-slot position
    /// `k % (pack_dtype_bytes * 8 / bits)`.
    Packed {
        bits: u8,
        pack_dtype_bytes: u8,
        group_size: u32,
    },
}

impl ElementStride {
    /// Bytes spanned by `n` consecutive elements starting from a block
    /// boundary. For fixed: `n * bytes_per_element`. For block: rounded up
    /// to the nearest whole block. For packed: rounded up to the nearest
    /// whole pack-slot (e.g. 8 elements for int4-in-int32).
    /// `#[allow(dead_code)]` is deliberate: no current caller walks byte
    /// ranges in ElementStride terms, but the helper is the intended unit
    /// for the sidecar byte-range math.
    #[allow(dead_code)]
    pub fn bytes_for_elements(self, n: usize) -> usize {
        match self {
            ElementStride::Fixed(b) => n * b,
            ElementStride::Block {
                block_bytes,
                block_elements,
            } => {
                if block_elements == 0 {
                    0
                } else {
                    n.div_ceil(block_elements) * block_bytes
                }
            }
            ElementStride::Packed {
                bits,
                pack_dtype_bytes,
                ..
            } => {
                if bits == 0 {
                    0
                } else {
                    let elems_per_slot = (pack_dtype_bytes as usize * 8) / bits as usize;
                    if elems_per_slot == 0 {
                        0
                    } else {
                        n.div_ceil(elems_per_slot) * pack_dtype_bytes as usize
                    }
                }
            }
        }
    }

    /// Bytes spanned by one full row of `cols` elements in a row-major tensor.
    /// For fixed: `cols * bytes_per_element`. For block/packed: the whole-row
    /// stride, assuming `cols` is a multiple of the block/slot element count
    /// (true for canonical GGUF and AWQ/GPTQ layouts). Used to address the
    /// start of an arbitrary row without re-deriving the packing math.
    pub fn bytes_per_row(self, cols: u64) -> u64 {
        match self {
            ElementStride::Fixed(b) => cols * b as u64,
            ElementStride::Block {
                block_bytes,
                block_elements,
            } => {
                if block_elements == 0 {
                    0
                } else {
                    (cols / block_elements as u64) * block_bytes as u64
                }
            }
            ElementStride::Packed {
                bits,
                pack_dtype_bytes,
                ..
            } => {
                if bits == 0 {
                    return 0;
                }
                let elems_per_slot = (pack_dtype_bytes as u64 * 8) / bits as u64;
                cols.checked_div(elems_per_slot)
                    .map(|n| n * pack_dtype_bytes as u64)
                    .unwrap_or(0)
            }
        }
    }
}

/// Tensor element data type.
///
/// Covers safetensors plain dtypes (fixed bytes/element) plus the GGUF
/// block-quantized variants. `element_size()` returns 1 for quantized types
/// — callers that need real stride information must use `stride()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    // Plain types (safetensors + GGUF F32/F16/BF16).
    F64,
    F32,
    F16,
    BF16,
    F8E4M3,
    F8E5M2,
    I64,
    I32,
    I16,
    I8,
    U64,
    U32,
    U16,
    U8,
    Bool,
    Unknown,
    // GGUF block-quantized.
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    // AWQ/GPTQ/EXL2-style bit-packed quants. The element bytes live
    // inline; the per-group `scales` and `zeros` tensors live as
    // sidecar tensors named alongside the quant tensor (see
    // [`crate::format::safetensors::fuse_packed_quant_triples`]).
    /// 4 bits per element, packed 8-per-int32 with f16/f32 scales (+ zeros).
    Int4Packed,
    /// 3 bits per element, packed ~10-per-int32 with f16 scales (+ zeros).
    /// Used by EXL2 mixed-precision layouts. The 4-bit case dominates the
    /// HF corpus today; 3-bit is held for future EXL2 / GPTQ detection.
    #[allow(dead_code)]
    Int3Packed,
    /// 8 bits per element, packed 4-per-int32 with f16 scales (+ zeros).
    /// Rarer on the Hub but reserved for future format detection.
    #[allow(dead_code)]
    Int8Packed,
}

impl Dtype {
    /// Parse a safetensors header dtype string (`"F32"`, `"BF16"`, …).
    /// Returns `Dtype::Unknown` for anything unrecognised — safetensors
    /// itself can ship custom dtype strings, so we don't bail on unknowns.
    pub fn from_safetensors_str(s: &str) -> Self {
        match s {
            "F64" => Dtype::F64,
            "F32" => Dtype::F32,
            "F16" => Dtype::F16,
            "BF16" => Dtype::BF16,
            "F8_E4M3" | "F8_E4M3FN" | "F8_E4M3FNUZ" => Dtype::F8E4M3,
            "F8_E5M2" | "F8_E5M2FNUZ" => Dtype::F8E5M2,
            "I64" => Dtype::I64,
            "I32" => Dtype::I32,
            "I16" => Dtype::I16,
            "I8" => Dtype::I8,
            "U64" => Dtype::U64,
            "U32" => Dtype::U32,
            "U16" => Dtype::U16,
            "U8" => Dtype::U8,
            "BOOL" => Dtype::Bool,
            _ => Dtype::Unknown,
        }
    }

    /// Translate a candle-core `GgmlDType` (from a parsed GGUF tensor info)
    /// into the arbvis `Dtype`. Non-quantized GGUF dtypes (`F32`/`F16`/`BF16`)
    /// map onto the plain safetensors variants so the renderer treats them
    /// identically.
    pub fn from_ggml(ty: GgmlDType) -> Self {
        match ty {
            GgmlDType::F32 => Dtype::F32,
            GgmlDType::F16 => Dtype::F16,
            GgmlDType::BF16 => Dtype::BF16,
            GgmlDType::Q4_0 => Dtype::Q4_0,
            GgmlDType::Q4_1 => Dtype::Q4_1,
            GgmlDType::Q5_0 => Dtype::Q5_0,
            GgmlDType::Q5_1 => Dtype::Q5_1,
            GgmlDType::Q8_0 => Dtype::Q8_0,
            GgmlDType::Q8_1 => Dtype::Q8_1,
            GgmlDType::Q2K => Dtype::Q2K,
            GgmlDType::Q3K => Dtype::Q3K,
            GgmlDType::Q4K => Dtype::Q4K,
            GgmlDType::Q5K => Dtype::Q5K,
            GgmlDType::Q6K => Dtype::Q6K,
            GgmlDType::Q8K => Dtype::Q8K,
        }
    }

    /// Inverse of [`Dtype::from_ggml`] for types that have a candle-side
    /// representation. Returns `None` for safetensors-only dtypes like F8/I*.
    pub fn to_ggml(self) -> Option<GgmlDType> {
        Some(match self {
            Dtype::F32 => GgmlDType::F32,
            Dtype::F16 => GgmlDType::F16,
            Dtype::BF16 => GgmlDType::BF16,
            Dtype::Q4_0 => GgmlDType::Q4_0,
            Dtype::Q4_1 => GgmlDType::Q4_1,
            Dtype::Q5_0 => GgmlDType::Q5_0,
            Dtype::Q5_1 => GgmlDType::Q5_1,
            Dtype::Q8_0 => GgmlDType::Q8_0,
            Dtype::Q8_1 => GgmlDType::Q8_1,
            Dtype::Q2K => GgmlDType::Q2K,
            Dtype::Q3K => GgmlDType::Q3K,
            Dtype::Q4K => GgmlDType::Q4K,
            Dtype::Q5K => GgmlDType::Q5K,
            Dtype::Q6K => GgmlDType::Q6K,
            Dtype::Q8K => GgmlDType::Q8K,
            _ => return None,
        })
    }

    /// Bytes per logical element, or `1` for block-quantized dtypes.
    ///
    /// Kept for legacy call sites that compute byte ranges as
    /// `count * dtype.element_size()` and have no way to address blocks.
    /// New code should use [`Dtype::stride`].
    pub fn element_size(self) -> usize {
        match self {
            Dtype::F64 | Dtype::I64 | Dtype::U64 => 8,
            Dtype::F32 | Dtype::I32 | Dtype::U32 => 4,
            Dtype::F16 | Dtype::BF16 | Dtype::I16 | Dtype::U16 => 2,
            Dtype::F8E4M3
            | Dtype::F8E5M2
            | Dtype::I8
            | Dtype::U8
            | Dtype::Bool
            | Dtype::Unknown => 1,
            // Block-quantised: every quantised dtype is sub-byte-per-element
            // (Q4_0 = 0.5625 bytes/elem). Return 1 so legacy callers that do
            // `n * element_size()` don't divide by zero; the real stride lives
            // in `Dtype::stride()`.
            Dtype::Q4_0
            | Dtype::Q4_1
            | Dtype::Q5_0
            | Dtype::Q5_1
            | Dtype::Q8_0
            | Dtype::Q8_1
            | Dtype::Q2K
            | Dtype::Q3K
            | Dtype::Q4K
            | Dtype::Q5K
            | Dtype::Q6K
            | Dtype::Q8K
            // Same rationale for packed-int dtypes — the real stride lives
            // in `Dtype::stride()`. (Int4Packed is 0.5 bytes/elem, etc.)
            | Dtype::Int4Packed
            | Dtype::Int3Packed
            | Dtype::Int8Packed => 1,
        }
    }

    /// Real stride information — `Fixed(n)` for plain dtypes, `Block { … }`
    /// for the GGUF quantised types, `Packed { … }` for AWQ/GPTQ-style
    /// bit-packed ints. AWQ/GPTQ canonical layout is int4 packed into int32
    /// with 128-element groups; sub-byte int3/int8 use the same packing
    /// dtype but different bit widths.
    pub fn stride(self) -> ElementStride {
        match self {
            Dtype::Int4Packed => ElementStride::Packed {
                bits: 4,
                pack_dtype_bytes: 4,
                group_size: 128,
            },
            Dtype::Int3Packed => ElementStride::Packed {
                bits: 3,
                pack_dtype_bytes: 4,
                group_size: 128,
            },
            Dtype::Int8Packed => ElementStride::Packed {
                bits: 8,
                pack_dtype_bytes: 4,
                group_size: 128,
            },
            _ => {
                if let Some(g) = self.to_ggml() {
                    // For F32/F16/BF16, type_size == 4/2/2 and block_size == 1,
                    // so this collapses to `Fixed(type_size)`.
                    let bytes = g.type_size();
                    let elems = g.block_size();
                    if elems <= 1 {
                        ElementStride::Fixed(bytes)
                    } else {
                        ElementStride::Block {
                            block_bytes: bytes,
                            block_elements: elems,
                        }
                    }
                } else {
                    ElementStride::Fixed(self.element_size())
                }
            }
        }
    }

    /// True when the dtype is a GGUF quant or an AWQ/GPTQ-style packed
    /// integer, i.e. raw bytes cannot be read as plain f32/f16/bf16
    /// elements.
    /// `#[allow(dead_code)]` is deliberate: kept as the canonical dtype
    /// classifier for the AWQ/GPTQ sidecar wire-up tracked in BUGS.md.
    #[allow(dead_code)]
    pub fn is_quantized(self) -> bool {
        matches!(
            self,
            Dtype::Q4_0
                | Dtype::Q4_1
                | Dtype::Q5_0
                | Dtype::Q5_1
                | Dtype::Q8_0
                | Dtype::Q8_1
                | Dtype::Q2K
                | Dtype::Q3K
                | Dtype::Q4K
                | Dtype::Q5K
                | Dtype::Q6K
                | Dtype::Q8K
                | Dtype::Int4Packed
                | Dtype::Int3Packed
                | Dtype::Int8Packed
        )
    }

    /// True for AWQ/GPTQ-style packed integers whose dequant requires
    /// out-of-band scales (and, for GPTQ, zero-points) — see
    /// [`ElementStride::Packed`].
    /// `#[allow(dead_code)]` is deliberate: the AWQ/GPTQ diff/xet/voxel
    /// paths (BUGS.md Open) are the intended consumers.
    #[allow(dead_code)]
    pub fn is_packed(self) -> bool {
        matches!(
            self,
            Dtype::Int4Packed | Dtype::Int3Packed | Dtype::Int8Packed
        )
    }

    /// Display color for `build_color_ranges` and the architectural arch
    /// legend. Plain dtypes keep their original safetensors palette; each
    /// quantised type gets a distinct cool hue so a tile of mixed-precision
    /// GGUF tensors reads as a striated rainbow.
    pub fn to_color(self) -> Rgb<u8> {
        match self {
            Dtype::F32 => Rgb([255, 120, 50]),
            Dtype::F16 => Rgb([255, 210, 60]),
            Dtype::BF16 => Rgb([180, 255, 60]),
            Dtype::F64 => Rgb([255, 50, 50]),
            Dtype::F8E4M3 => Rgb([50, 220, 255]),
            Dtype::F8E5M2 => Rgb([50, 255, 200]),
            Dtype::I8 | Dtype::U8 => Rgb([60, 60, 255]),
            Dtype::I16 | Dtype::U16 => Rgb([130, 60, 255]),
            Dtype::I32 | Dtype::U32 | Dtype::I64 | Dtype::U64 | Dtype::Bool => Rgb([220, 60, 255]),
            Dtype::Unknown => Rgb([0, 0, 0]),
            // GGUF quantised palette. Stepped through teal→indigo so the
            // visual band scales with precision (Q2 darkest, Q8 brightest).
            Dtype::Q2K => Rgb([0, 80, 110]),
            Dtype::Q3K => Rgb([0, 110, 140]),
            Dtype::Q4_0 => Rgb([0, 140, 170]),
            Dtype::Q4_1 => Rgb([0, 160, 190]),
            Dtype::Q4K => Rgb([0, 180, 200]),
            Dtype::Q5_0 => Rgb([60, 170, 220]),
            Dtype::Q5_1 => Rgb([80, 180, 230]),
            Dtype::Q5K => Rgb([100, 190, 240]),
            Dtype::Q6K => Rgb([130, 180, 240]),
            Dtype::Q8_0 => Rgb([160, 170, 240]),
            Dtype::Q8_1 => Rgb([180, 160, 240]),
            Dtype::Q8K => Rgb([200, 150, 240]),
            // AWQ/GPTQ packed-int palette. Magenta family — visually distinct
            // from the cool GGUF range so a mixed AWQ + base diff reads
            // unambiguously. Stepped by bit width: int3 darkest, int8 brightest.
            Dtype::Int3Packed => Rgb([160, 40, 180]),
            Dtype::Int4Packed => Rgb([200, 60, 200]),
            Dtype::Int8Packed => Rgb([230, 100, 210]),
        }
    }

    /// Short string used in tensor labels (legend, tooltips). Matches the
    /// safetensors header convention for plain dtypes; for quantised dtypes
    /// uses the canonical GGML name.
    pub fn label(self) -> &'static str {
        match self {
            Dtype::F32 => "F32",
            Dtype::F16 => "F16",
            Dtype::BF16 => "BF16",
            Dtype::F64 => "F64",
            Dtype::F8E4M3 => "F8E4M3",
            Dtype::F8E5M2 => "F8E5M2",
            Dtype::I8 => "I8",
            Dtype::U8 => "U8",
            Dtype::I16 => "I16",
            Dtype::U16 => "U16",
            Dtype::I32 => "I32",
            Dtype::U32 => "U32",
            Dtype::I64 => "I64",
            Dtype::U64 => "U64",
            Dtype::Bool => "BOOL",
            Dtype::Unknown => "?",
            Dtype::Q4_0 => "Q4_0",
            Dtype::Q4_1 => "Q4_1",
            Dtype::Q5_0 => "Q5_0",
            Dtype::Q5_1 => "Q5_1",
            Dtype::Q8_0 => "Q8_0",
            Dtype::Q8_1 => "Q8_1",
            Dtype::Q2K => "Q2K",
            Dtype::Q3K => "Q3K",
            Dtype::Q4K => "Q4K",
            Dtype::Q5K => "Q5K",
            Dtype::Q6K => "Q6K",
            Dtype::Q8K => "Q8K",
            Dtype::Int3Packed => "I3_pack",
            Dtype::Int4Packed => "I4_pack",
            Dtype::Int8Packed => "I8_pack",
        }
    }

    /// Compute the signed diff between matched elements, returning one u8
    /// per element pair.
    ///
    /// `self` is the dtype of the `orig` buffer; `mod_dtype` is the dtype of
    /// the `mod_` buffer (they may differ — e.g. a cross-format diff between
    /// a safetensors F16 base and a GGUF Q4_K finetune). `scale_orig` is the
    /// per-tensor scale (RMS of `orig`) used by `DiffMetric::Rms`.
    ///
    /// `orig_start_elem`/`mod_start_elem` let the caller supply a buffer
    /// that begins on a block boundary (for quantised tensors) but skip
    /// `orig_start_elem` elements into it before pairing. For safetensors
    /// fixed-stride buffers pass `0`.
    ///
    /// `orig_sc`/`mod_sc` carry optional AWQ/GPTQ sidecar context (borrowed
    /// whole-tensor scales/qzeros buffers, as fetched by the caller); they
    /// are only consulted when that side's dtype is packed, letting packed
    /// sides dequantise instead of decoding to NaN sentinels.
    /// `orig_anchor`/`mod_anchor` are the absolute `(row, col)` of each
    /// buffer's first element within its tensor — for the row-aligned
    /// fetches the diff source issues, `col` is 0 and `row` is the fetched
    /// buffer's first row. Non-packed sides ignore them; pass `None` and
    /// `(0, 0)` for plain buffers.
    ///
    /// Encoding: 127 = no change, 128–254 = increased, 0–126 = decreased,
    /// 255 = non-finite.
    #[allow(clippy::too_many_arguments)]
    pub fn diff_to_u8_sidecars(
        self,
        orig: &[u8],
        orig_start_elem: usize,
        mod_dtype: Dtype,
        mod_: &[u8],
        mod_start_elem: usize,
        metric: DiffMetric,
        scale_orig: f32,
        elem_count: usize,
        orig_sc: Option<PackedSidecarRefs<'_>>,
        orig_anchor: (usize, usize),
        mod_sc: Option<PackedSidecarRefs<'_>>,
        mod_anchor: (usize, usize),
    ) -> Vec<u8> {
        let (rms_denom, log_min, log_max) = diff_metric_norms(scale_orig);
        // Fixed-stride fast path: chunked little-endian reads instead of the
        // per-element reader dispatch — the stride match and bounds check are
        // hoisted out of the loop. Only taken when both buffers fully cover
        // the requested element range, so out-of-range pixels keep flowing
        // through the reader's NaN sentinel below.
        if let (ElementStride::Fixed(bo), ElementStride::Fixed(bm)) =
            (self.stride(), mod_dtype.stride())
        {
            let need_o = orig_start_elem
                .saturating_mul(bo)
                .saturating_add(elem_count.saturating_mul(bo));
            let need_m = mod_start_elem
                .saturating_mul(bm)
                .saturating_add(elem_count.saturating_mul(bm));
            if need_o <= orig.len() && need_m <= mod_.len() {
                let mut out = Vec::with_capacity(elem_count);
                for k in 0..elem_count {
                    let o = decode_element(
                        self,
                        &orig[orig_start_elem * bo + k * bo..orig_start_elem * bo + (k + 1) * bo],
                    );
                    let m = decode_element(
                        mod_dtype,
                        &mod_[mod_start_elem * bm + k * bm..mod_start_elem * bm + (k + 1) * bm],
                    );
                    if !o.is_finite() || !m.is_finite() {
                        out.push(255u8);
                        continue;
                    }
                    push_diff_pixel(&mut out, o, m, metric, rms_denom, log_min, log_max);
                }
                return out;
            }
        }
        let mut o_reader = TensorElementReader::new(self, orig);
        if let Some(sc) = orig_sc {
            o_reader = o_reader.with_sidecars(sc).with_anchor(orig_anchor.0, orig_anchor.1);
        }
        let mut m_reader = TensorElementReader::new(mod_dtype, mod_);
        if let Some(sc) = mod_sc {
            m_reader = m_reader.with_sidecars(sc).with_anchor(mod_anchor.0, mod_anchor.1);
        }
        let mut out = Vec::with_capacity(elem_count);
        for k in 0..elem_count {
            let o = o_reader.element(orig_start_elem + k);
            let m = m_reader.element(mod_start_elem + k);
            if !o.is_finite() || !m.is_finite() {
                out.push(255u8);
                continue;
            }
            push_diff_pixel(&mut out, o, m, metric, rms_denom, log_min, log_max);
        }
        out
    }
}

/// Per-tensor constants the diff metrics derive from `scale_orig` (the RMS
/// of the `orig` tensor): `(rms_denom, log_min, log_max)` for
/// [`DiffMetric::Rms`] / [`DiffMetric::AbsLog`]. Shared by `diff_to_u8` and
/// the element-wise LUT encoder in `layout::render` so the two stay
/// byte-identical.
pub fn diff_metric_norms(scale_orig: f32) -> (f32, f32, f32) {
    (
        (K_RMS_SAT * scale_orig.max(RMS_FLOOR)).max(f32::MIN_POSITIVE),
        ABS_LOG_MIN.log10(),
        ABS_LOG_MAX.log10(),
    )
}

/// Normalize one delta into a signed value in `[-1, 1]` under `metric`,
/// given the per-tensor norms from [`diff_metric_norms`]. Shared by both
/// diff encoders (byte encoding in [`Dtype::diff_to_u8`] and the LUT colour
/// in `layout::render`) so the metric arithmetic lives in one place.
pub fn diff_signed(
    delta: f32,
    metric: DiffMetric,
    rms_denom: f32,
    log_min: f32,
    log_max: f32,
) -> f32 {
    match metric {
        DiffMetric::Rms => (delta / rms_denom).clamp(-1.0, 1.0),
        DiffMetric::AbsLog => {
            let abs_d = delta.abs();
            if abs_d <= ABS_LOG_MIN {
                0.0
            } else {
                let norm = ((abs_d.log10() - log_min) / (log_max - log_min)).clamp(0.0, 1.0);
                if delta >= 0.0 {
                    norm
                } else {
                    -norm
                }
            }
        }
        DiffMetric::Exact => {
            if delta == 0.0 {
                0.0
            } else if delta > 0.0 {
                1.0
            } else {
                -1.0
            }
        }
    }
}

/// Signed-delta metric + sign-magnitude byte encode for one diff pixel.
/// Shared by the fixed-stride fast path and the reader fallback in
/// [`Dtype::diff_to_u8`] so the two loops stay byte-identical.
fn push_diff_pixel(
    out: &mut Vec<u8>,
    o: f32,
    m: f32,
    metric: DiffMetric,
    rms_denom: f32,
    log_min: f32,
    log_max: f32,
) {
    let delta = m - o;
    let signed = diff_signed(delta, metric, rms_denom, log_min, log_max);
    let brightness = (signed.abs() * 127.0).round() as u8;
    let byte = if signed >= 0.0 {
        127u8.saturating_add(brightness)
    } else {
        127u8.saturating_sub(brightness)
    };
    out.push(byte);
}

/// Decode one plain (non-quantized) element from a little-endian byte slice.
/// Panics for quantized dtypes — callers must use [`TensorElementReader`].
pub fn decode_element(dtype: Dtype, bytes: &[u8]) -> f32 {
    match dtype {
        Dtype::F32 => f32::from_le_bytes(bytes.try_into().unwrap()),
        Dtype::F16 => half::f16::from_le_bytes(bytes.try_into().unwrap()).to_f32(),
        Dtype::BF16 => half::bf16::from_le_bytes(bytes.try_into().unwrap()).to_f32(),
        Dtype::F64 => f64::from_le_bytes(bytes.try_into().unwrap()) as f32,
        Dtype::I8 => (bytes[0] as i8) as f32,
        Dtype::U8 | Dtype::Bool => bytes[0] as f32,
        Dtype::I16 => i16::from_le_bytes(bytes.try_into().unwrap()) as f32,
        Dtype::U16 => u16::from_le_bytes(bytes.try_into().unwrap()) as f32,
        Dtype::I32 => i32::from_le_bytes(bytes.try_into().unwrap()) as f32,
        Dtype::U32 => u32::from_le_bytes(bytes.try_into().unwrap()) as f32,
        Dtype::I64 => i64::from_le_bytes(bytes.try_into().unwrap()) as f32,
        Dtype::U64 => u64::from_le_bytes(bytes.try_into().unwrap()) as f32,
        Dtype::F8E4M3 | Dtype::F8E5M2 | Dtype::Unknown => bytes[0] as f32,
        // Quantized dtypes need block context — see TensorElementReader.
        Dtype::Q4_0
        | Dtype::Q4_1
        | Dtype::Q5_0
        | Dtype::Q5_1
        | Dtype::Q8_0
        | Dtype::Q8_1
        | Dtype::Q2K
        | Dtype::Q3K
        | Dtype::Q4K
        | Dtype::Q5K
        | Dtype::Q6K
        | Dtype::Q8K
        // Packed dtypes need sidecar context — see TensorElementReader::with_sidecars.
        | Dtype::Int4Packed
        | Dtype::Int3Packed
        | Dtype::Int8Packed => {
            panic!(
                "decode_element: quantized dtype {:?} needs TensorElementReader",
                dtype
            )
        }
    }
}

/// Decode one element from an AWQ/GPTQ-style packed-int tensor.
///
/// `qweight` is the row-major byte buffer of packed ints (typically int32);
/// `sc` carries the matching scales / qzeros byte buffers. Element `k` is
/// the logical (unpacked) element index — row-major across the unpacked
/// `[rows, cols]` shape, where `cols = sc.cols`.
///
/// Layout assumed (GPTQ / AWQ canonical):
///   - quant slots are arranged row-major across columns, so the slot
///     containing logical column `c` of row `r` lives at
///     `(r * (cols / elems_per_slot) + c / elems_per_slot)` packed ints
///   - within a slot, the `bits`-wide field for in-slot index `i` is at
///     bit positions `[i*bits, i*bits + bits)`
///   - scales are `[rows / group_size, cols]` (one scale per (group, col))
///   - qzeros mirror scales but bit-packed the same way as qweight
///
/// Returns NaN on any out-of-range access (the renderer treats that as a
/// "padding / partial fetch" pixel).
fn packed_element(
    qweight: &[u8],
    sc: PackedSidecarRefs<'_>,
    k: usize,
    bits: u8,
    pack_dtype_bytes: u8,
    group_size: usize,
    anchor: (usize, usize),
) -> f32 {
    if bits == 0 || pack_dtype_bytes == 0 || sc.cols == 0 {
        return f32::NAN;
    }
    let bits = bits as usize;
    let slot_bytes = pack_dtype_bytes as usize;
    let elems_per_slot = (slot_bytes * 8) / bits;
    if elems_per_slot == 0 {
        return f32::NAN;
    }
    let cols = sc.cols as usize;
    if cols == 0 || !cols.is_multiple_of(elems_per_slot) {
        return f32::NAN;
    }
    // Buffer-relative row / in-row position: the qweight slot read below is
    // purely relative to the fetched buffer. `anchor` lifts both to the
    // element's absolute tensor position for the sidecar lookups.
    let row = k / cols;
    let col = k % cols;
    let abs_row = anchor.0.saturating_add(row);
    let abs_col = anchor.1.saturating_add(col);
    let slots_per_row = cols / elems_per_slot;
    let slot_idx = row * slots_per_row + col / elems_per_slot;
    let in_slot = col % elems_per_slot;

    // Read packed int (little-endian) from qweight.
    let off = slot_idx * slot_bytes;
    if off + slot_bytes > qweight.len() {
        return f32::NAN;
    }
    let packed = read_u32_le(&qweight[off..off + slot_bytes]);
    let mask: u32 = if bits >= 32 {
        u32::MAX
    } else {
        (1u32 << bits) - 1
    };
    let q = ((packed >> (in_slot * bits)) & mask) as i32;

    // Group index and scale lookup — absolute position, since the scales
    // tensor covers the whole unpacked tensor.
    let group_idx = abs_row.checked_div(group_size).unwrap_or(0);
    let scale_elem_idx = group_idx.saturating_mul(cols).saturating_add(abs_col);
    let scale = read_scalar(sc.scales, sc.scales_dtype, scale_elem_idx);
    if !scale.is_finite() {
        return f32::NAN;
    }

    // Zero lookup. For AWQ asymmetric: zeros buffer is packed the same way
    // as qweight; for symmetric variants (no zeros) treat zero as 0.
    let zero = if let Some(zb) = sc.zeros {
        if sc.zeros_dtype == Dtype::F16 || sc.zeros_dtype == Dtype::F32 {
            // EXL2 stores zeros as floats; same indexing as scales.
            let z = read_scalar(zb, sc.zeros_dtype, scale_elem_idx);
            if !z.is_finite() {
                return f32::NAN;
            }
            z
        } else {
            // GPTQ/AWQ store zeros bit-packed identically to qweight; the
            // zeros buffer is whole-tensor, so index it by absolute column
            // slot.
            let zslot = group_idx * slots_per_row + abs_col / elems_per_slot;
            let zoff = zslot * slot_bytes;
            if zoff + slot_bytes > zb.len() {
                return f32::NAN;
            }
            let zp = read_u32_le(&zb[zoff..zoff + slot_bytes]);
            let z = ((zp >> (in_slot * bits)) & mask) as i32;
            z as f32
        }
    } else {
        0.0
    };

    ((q as f32) - zero) * scale
}

/// Decode the first `n` elements of `bytes` to f32 and fold them left with
/// `f`.
///
/// Fixed-stride dtypes take a chunked fast path: the per-element stride
/// match and bounds check are hoisted out of the loop (the little-endian
/// reads then vectorize), which matters for the multi-million-element full
/// buffer scans behind the summary stats and the CKA panel decode.
/// Block-quantized and packed dtypes go through [`TensorElementReader`],
/// keeping the block dequant cache. The fallback also covers a fixed-stride
/// buffer that doesn't cover `n` elements — the reader then paints the
/// out-of-range tail as NaN, same as before.
pub(crate) fn fold_prefix_f32<T>(
    dtype: Dtype,
    bytes: &[u8],
    n: usize,
    init: T,
    mut f: impl FnMut(T, f32) -> T,
) -> T {
    if let ElementStride::Fixed(bpe) = dtype.stride() {
        if n.saturating_mul(bpe) <= bytes.len() {
            return bytes[..n * bpe]
                .chunks_exact(bpe)
                .fold(init, |acc, chunk| f(acc, decode_element(dtype, chunk)));
        }
    }
    let mut reader = TensorElementReader::new(dtype, bytes);
    (0..n).fold(init, |acc, k| f(acc, reader.element(k)))
}

/// Decode the first `n` elements of `bytes` into a fresh f32 vec.
pub fn decode_prefix_f32(dtype: Dtype, bytes: &[u8], n: usize) -> Vec<f32> {
    fold_prefix_f32(
        dtype,
        bytes,
        n,
        Vec::with_capacity(n),
        |mut v, x| {
            v.push(x);
            v
        },
    )
}

/// Decode the first `n` elements of `bytes` into a fresh f32 vec, with
/// optional AWQ/GPTQ sidecar context for `ElementStride::Packed` dtypes.
/// With `Some(refs)` and a packed dtype, elements dequantise through the
/// scales/qzeros buffers (the buffer covers the whole tensor, so the
/// reader's default anchor `(0, 0)` is correct). With `None` — or a
/// non-packed dtype — this is exactly [`decode_prefix_f32`], and packed
/// dtypes decode to NaN as before.
pub fn decode_prefix_f32_sidecars(
    dtype: Dtype,
    bytes: &[u8],
    n: usize,
    refs: Option<PackedSidecarRefs<'_>>,
) -> Vec<f32> {
    if let (ElementStride::Packed { .. }, Some(refs)) = (dtype.stride(), refs) {
        let mut reader = TensorElementReader::new(dtype, bytes).with_sidecars(refs);
        return (0..n).map(|k| reader.element(k)).collect();
    }
    decode_prefix_f32(dtype, bytes, n)
}


fn read_u32_le(b: &[u8]) -> u32 {
    let mut v = 0u32;
    for (i, &x) in b.iter().take(4).enumerate() {
        v |= (x as u32) << (i * 8);
    }
    v
}

fn read_scalar(bytes: &[u8], dtype: Dtype, elem_idx: usize) -> f32 {
    let bpe = dtype.element_size();
    let off = elem_idx * bpe;
    if off + bpe > bytes.len() {
        return f32::NAN;
    }
    decode_element(dtype, &bytes[off..off + bpe])
}

/// Per-element f32 decode with a single-block dequant cache.
///
/// The renderer and diff math both iterate elements 0..N within a byte buffer
/// that starts at element 0 of a tensor. `element(k)` returns the f32 value
/// of element K. For plain dtypes this is a fixed-stride little-endian read.
/// For quantized dtypes the reader keeps the most recently dequantized block
/// in `cache`; sequential `element(k)` calls in the hot per-tile loop hit
/// cache on every call after the first within each block.
///
/// For [`ElementStride::Packed`] dtypes (AWQ / GPTQ) the reader additionally
/// needs the matching `scales` and `qzeros` byte buffers. Attach them via
/// [`TensorElementReader::with_sidecars`]; without sidecars, packed dtypes
/// decode to NaN (so the renderer paints them as a sentinel rather than
/// returning a wrong f32).
pub struct TensorElementReader<'a> {
    dtype: Dtype,
    bytes: &'a [u8],
    /// Cached `(block_index, dequantized_block_floats)`. Only populated for
    /// quantized dtypes.
    cache: Option<(usize, Vec<f32>)>,
    /// AWQ/GPTQ sidecars. `None` for non-packed dtypes; required (else
    /// `element` returns NaN) for `ElementStride::Packed`.
    sidecars: Option<PackedSidecarRefs<'a>>,
    /// Absolute `(row, col)` of the byte buffer's first element within its
    /// tensor. Only read by the packed-int path: the qweight slot lookup
    /// stays buffer-relative, but the scales/qzeros sidecar lookups need
    /// the element's absolute position. `(0, 0)` for whole-tensor buffers.
    anchor: (usize, usize),
}

/// Borrowed sidecar buffers for one packed-int tensor. Attached to a
/// [`TensorElementReader`] via [`TensorElementReader::with_sidecars`].
///
/// Lifetime is tied to the reader so the caller can hold the fetched byte
/// buffers on the stack alongside the reader without lifetime gymnastics.
#[derive(Clone, Copy)]
pub struct PackedSidecarRefs<'a> {
    pub scales: &'a [u8],
    pub scales_dtype: Dtype,
    /// `None` for symmetric quants (e.g. AWQ without zero-points).
    pub zeros: Option<&'a [u8]>,
    pub zeros_dtype: Dtype,
    /// Number of output columns in the unpacked tensor — needed to map a
    /// logical element index to its (group_index, in_group_pos) pair when
    /// `group_size` doesn't evenly divide the row stride.
    pub cols: u32,
}

impl<'a> TensorElementReader<'a> {
    /// Reader over one tensor's raw bytes; dequantises on demand via [`decode_element`].
    pub fn new(dtype: Dtype, bytes: &'a [u8]) -> Self {
        Self {
            dtype,
            bytes,
            cache: None,
            sidecars: None,
            anchor: (0, 0),
        }
    }

    /// Builder: attach AWQ/GPTQ sidecar tensors. For plain / Block dtypes
    /// this is a no-op (the sidecars are simply unused).
    ///
    /// Production wire-up: the architectural tile loader
    /// (`crate::tiled::leaf_arch::load_arch_tile_regions`) reads
    /// `TensorMeta::packed_sidecars` and threads the fetched scales/qzeros
    /// buffers here.
    pub fn with_sidecars(mut self, refs: PackedSidecarRefs<'a>) -> Self {
        self.sidecars = Some(refs);
        self
    }

    /// Builder: declare the absolute `(row, col)` of the byte buffer's first
    /// element within its tensor. Used by the tile render path, whose fetched
    /// buffer starts at a region's `(row_first, col_first_aligned)` rather
    /// than element (0, 0): the packed-int decoder reads qweight slots
    /// buffer-relatively but indexes the scales/qzeros sidecars at the
    /// element's absolute position. Defaults to `(0, 0)` (whole-tensor
    /// buffers), which keeps every existing call site's behaviour.
    pub fn with_anchor(mut self, row: usize, col: usize) -> Self {
        self.anchor = (row, col);
        self
    }

    /// f32 value of element `k`. Returns NaN if `k` is out of range so the
    /// caller's `is_finite()` check paints the pixel as a NaN sentinel
    /// instead of panicking on a partial fetch.
    pub fn element(&mut self, k: usize) -> f32 {
        match self.dtype.stride() {
            ElementStride::Fixed(bpe) => {
                let off = k * bpe;
                if off + bpe > self.bytes.len() {
                    return f32::NAN;
                }
                decode_element(self.dtype, &self.bytes[off..off + bpe])
            }
            ElementStride::Block {
                block_bytes,
                block_elements,
            } => {
                if block_elements == 0 {
                    return f32::NAN;
                }
                let blk_idx = k / block_elements;
                let in_blk = k % block_elements;
                if !matches!(self.cache, Some((idx, _)) if idx == blk_idx) {
                    let start = blk_idx * block_bytes;
                    if start + block_bytes > self.bytes.len() {
                        return f32::NAN;
                    }
                    let g = self.dtype.to_ggml().expect("quantized dtype has GgmlDType");
                    // `from_data` panics on misalignment. Copy the block bytes
                    // into a fresh Vec<u8> — heap-aligned to 8 bytes, which
                    // covers every block struct's alignment (≤ 2 in practice).
                    let raw = self.bytes[start..start + block_bytes].to_vec();
                    let qt = g.from_data(Cow::Owned(raw));
                    match qt.dequantize(block_elements) {
                        Ok(CpuStorage::F32(v)) => {
                            self.cache = Some((blk_idx, v));
                        }
                        _ => return f32::NAN,
                    }
                }
                self.cache
                    .as_ref()
                    .map(|(_, v)| v[in_blk])
                    .unwrap_or(f32::NAN)
            }
            ElementStride::Packed {
                bits,
                pack_dtype_bytes,
                group_size,
            } => {
                let Some(sc) = self.sidecars else {
                    // Without sidecars, packed dtypes can't be dequantized;
                    // return NaN so the renderer paints a sentinel.
                    return f32::NAN;
                };
                packed_element(
                    self.bytes,
                    sc,
                    k,
                    bits,
                    pack_dtype_bytes,
                    group_size as usize,
                    self.anchor,
                )
            }
        }
    }

    /// Estimate `rms = sqrt(mean(x²))` over the first `sample_elements`
    /// elements (or the whole buffer if shorter). Skips non-finite values.
    pub fn rms_estimate(&mut self, sample_elements: usize) -> f32 {
        // Same stride math as [`element_count_for_buf`], clamped to the
        // requested sample budget.
        let n = element_count_for_buf(self.dtype, self.bytes.len()).min(sample_elements);
        if n == 0 {
            return 0.0;
        }
        let (sum_sq, count) = fold_prefix_f32(
            self.dtype,
            self.bytes,
            n,
            (0.0f64, 0u64),
            |(sum_sq, count), v| {
                if v.is_finite() {
                    (sum_sq + (v as f64) * (v as f64), count + 1)
                } else {
                    (sum_sq, count)
                }
            },
        );
        if count == 0 {
            0.0
        } else {
            (sum_sq / count as f64).sqrt() as f32
        }
    }
}

/// Logical element count addressable from `bytes` at this dtype's stride.
/// Plain dtypes: `bytes.len() / element_size`. Quantized block dtypes:
/// `(bytes.len() / block_bytes) * block_elements`. Packed (e.g. Q4_K)
/// dtypes: slots * (pack_bits / bits).
fn element_count_for_buf(dtype: Dtype, bytes_len: usize) -> usize {
    match dtype.stride() {
        ElementStride::Fixed(bpe) => bytes_len.checked_div(bpe).unwrap_or(0),
        ElementStride::Block {
            block_bytes,
            block_elements,
        } => bytes_len
            .checked_div(block_bytes)
            .map(|nb| nb * block_elements)
            .unwrap_or(0),
        ElementStride::Packed {
            bits,
            pack_dtype_bytes,
            ..
        } => {
            if bits == 0 {
                0
            } else {
                let elems_per_slot = (pack_dtype_bytes as usize * 8) / bits as usize;
                bytes_len
                    .checked_div(pack_dtype_bytes as usize)
                    .map(|slots| slots * elems_per_slot)
                    .unwrap_or(0)
            }
        }
    }
}

/// Estimate RMS over a contiguous tensor byte slice. Convenience wrapper
/// around [`TensorElementReader::rms_estimate`] for call sites that don't
/// already hold a reader.
///
/// `sample_elements` is the *logical* element count, not bytes — for plain
/// dtypes that's `bytes.len() / element_size`; for quantized it's
/// `(bytes.len() / block_bytes) * block_elements`.
pub fn rms_from_buf(dtype: Dtype, bytes: &[u8]) -> f32 {
    let n = element_count_for_buf(dtype, bytes.len());
    let mut reader = TensorElementReader::new(dtype, bytes);
    reader.rms_estimate(n)
}

/// Frobenius norm: `sqrt(sum(x²))` over every element in the buffer.
/// Single pass, skips non-finite values. Honest about total magnitude
/// (varies with tensor size, unlike [`rms_from_buf`]). Returns `0.0` for
/// an empty buffer or one with no finite elements.
pub fn frobenius_from_buf(dtype: Dtype, bytes: &[u8]) -> f32 {
    let n = element_count_for_buf(dtype, bytes.len());
    if n == 0 {
        return 0.0;
    }
    let sum_sq = fold_prefix_f32(dtype, bytes, n, 0.0f64, |s, v| {
        if v.is_finite() {
            s + (v as f64) * (v as f64)
        } else {
            s
        }
    });
    (sum_sq.sqrt()) as f32
}

/// Mean absolute value: `mean(|x|)` over every element in the buffer.
/// Single pass, skips non-finite values. Stable across tensor sizes;
/// dominated by typical-magnitude entries rather than extreme outliers
/// the way [`frobenius_from_buf`] is. Returns `0.0` for an empty buffer
/// or one with no finite elements.
pub fn mean_abs_from_buf(dtype: Dtype, bytes: &[u8]) -> f32 {
    let n = element_count_for_buf(dtype, bytes.len());
    if n == 0 {
        return 0.0;
    }
    let (sum_abs, count) = fold_prefix_f32(dtype, bytes, n, (0.0f64, 0u64), |(s, c), v| {
        if v.is_finite() {
            (s + (v as f64).abs(), c + 1)
        } else {
            (s, c)
        }
    });
    if count == 0 {
        0.0
    } else {
        (sum_abs / count as f64) as f32
    }
}

/// Fraction of entries with `|x| < eps`. Single pass; non-finite values
/// count as non-sparse (they're not "near zero"). Returns `0.0` for an
/// empty buffer. Useful for spotting dead / near-dead experts.
pub fn sparsity_from_buf(dtype: Dtype, bytes: &[u8], eps: f32) -> f32 {
    let n = element_count_for_buf(dtype, bytes.len());
    if n == 0 {
        return 0.0;
    }
    let near_zero = fold_prefix_f32(dtype, bytes, n, 0u64, |nz, v| {
        if v.is_finite() && v.abs() < eps {
            nz + 1
        } else {
            nz
        }
    });
    (near_zero as f64 / n as f64) as f32
}

/// Convenience: given a GGUF `TensorInfo`, the in-file byte range
/// `[file_start, file_end)`. `file_start` is `tensor_data_offset +
/// info.offset`; `file_end` is `file_start + byte_size_for(info)`. Used by
/// the GGUF parser when translating into `TensorMeta`.
pub fn gguf_tensor_byte_range(info: &TensorInfo, tensor_data_offset: u64) -> (u64, u64) {
    let elem_count: usize = info.shape.elem_count();
    let g = info.ggml_dtype;
    let bytes = elem_count / g.block_size() * g.type_size();
    let start = tensor_data_offset + info.offset;
    (start, start + bytes as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::DiffMetric;

    #[test]
    fn dtype_from_str_roundtrip() {
        assert!(matches!(Dtype::from_safetensors_str("BF16"), Dtype::BF16));
        assert!(matches!(Dtype::from_safetensors_str("F16"), Dtype::F16));
        assert!(matches!(
            Dtype::from_safetensors_str("F8_E4M3"),
            Dtype::F8E4M3
        ));
    }

    fn f32_bytes(vals: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(vals.len() * 4);
        for &v in vals {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    #[test]
    fn reader_plain_f32() {
        let bytes = f32_bytes(&[1.0, 2.0, -3.5]);
        let mut r = TensorElementReader::new(Dtype::F32, &bytes);
        assert_eq!(r.element(0), 1.0);
        assert_eq!(r.element(1), 2.0);
        assert_eq!(r.element(2), -3.5);
        assert!(r.element(3).is_nan());
    }

    #[test]
    fn reader_quantized_q8_0_does_not_crash_on_padded_block() {
        // Smoke-test the quantised dequant path: a syntactically valid Q8_0
        // block (1 f16 scale + 32 i8 quants = 34 bytes) goes through the
        // candle dequant kernel without panicking, returns finite values,
        // and out-of-range reads return NaN as the renderer expects.
        //
        // We don't assert specific dequant values because candle's
        // `GgmlDType::from_data` reads block scales/quants via a slice
        // reinterpretation whose results are correct in production (via
        // `BlockQ8_0::from_float`) but harder to construct from raw bytes
        // here without going through private struct fields. End-to-end
        // correctness is verified by the GGUF render path in the
        // verification plan.
        //
        // The block scale must be a nonzero f16: candle's dequant kernel
        // computes quant * scale, and a zero scale can yield NaN in its
        // SIMD paths, making the `is_finite` assertion depend on candle's
        // zero-scale handling (the flake behind this test's original
        // all-zero-bytes version). Scale 1.0 with zero quants dequantises
        // to exactly 0.0, finite everywhere.
        let mut bytes = vec![0u8; 34];
        bytes[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
        let mut r = TensorElementReader::new(Dtype::Q8_0, &bytes);
        for k in 0..32 {
            let v = r.element(k);
            assert!(v.is_finite(), "elem {k}: got {v}, expected finite");
            assert_eq!(v, 0.0, "elem {k}: zero quants with scale 1.0");
        }
        // Out-of-block read returns NaN sentinel.
        assert!(r.element(32).is_nan());
    }

    #[test]
    fn diff_rms_zero_delta_paints_black() {
        let o = f32_bytes(&[0.1, -0.2, 0.3]);
        let m = o.clone();
        let out = Dtype::F32.diff_to_u8_sidecars(&o, 0, Dtype::F32, &m, 0, DiffMetric::Rms, 0.1, 3, None, (0, 0), None, (0, 0));
        assert_eq!(out, vec![127, 127, 127]);
    }

    #[test]
    fn diff_rms_half_stddev_saturates() {
        let rms: f32 = 0.04;
        let o = f32_bytes(&[0.1, 0.1]);
        let m = f32_bytes(&[0.1 + 0.5 * rms, 0.1 - 0.5 * rms]);
        let out = Dtype::F32.diff_to_u8_sidecars(&o, 0, Dtype::F32, &m, 0, DiffMetric::Rms, rms, 2, None, (0, 0), None, (0, 0));
        assert_eq!(out, vec![254, 0]);
    }

    #[test]
    fn diff_non_finite_paints_white() {
        let o = f32_bytes(&[0.1, f32::NAN, 0.1]);
        let m = f32_bytes(&[0.1, 0.1, f32::INFINITY]);
        for metric in [DiffMetric::Rms, DiffMetric::AbsLog, DiffMetric::Exact] {
            let out = Dtype::F32.diff_to_u8_sidecars(&o, 0, Dtype::F32, &m, 0, metric, 0.1, 3, None, (0, 0), None, (0, 0));
            assert_eq!(out[0], 127, "{metric:?} same value");
            assert_eq!(out[1], 255, "{metric:?} NaN in orig");
            assert_eq!(out[2], 255, "{metric:?} Inf in mod");
        }
    }

    #[test]
    fn rms_from_buf_basic() {
        let b = f32_bytes(&[1.0, -1.0, 1.0, -1.0]);
        let r = rms_from_buf(Dtype::F32, &b);
        assert!((r - 1.0).abs() < 1e-6, "got {r}");
    }

    #[test]
    fn rms_from_buf_ignores_non_finite() {
        let b = f32_bytes(&[2.0, f32::NAN, -2.0, f32::INFINITY, 2.0]);
        let r = rms_from_buf(Dtype::F32, &b);
        assert!((r - 2.0).abs() < 1e-6, "got {r}");
    }

    #[test]
    fn rms_from_buf_empty_is_zero() {
        let r = rms_from_buf(Dtype::F32, &[]);
        assert_eq!(r, 0.0);
    }

    /// The fixed-stride chunked fast path must agree element-for-element with
    /// the `TensorElementReader` fallback, including the NaN-painted tail of a
    /// buffer that doesn't cover all requested elements.
    #[test]
    fn decode_prefix_f32_fast_path_matches_reader() {
        for (dtype, bpe) in [
            (Dtype::F32, 4usize),
            (Dtype::F16, 2),
            (Dtype::BF16, 2),
            (Dtype::I32, 4),
            (Dtype::U8, 1),
        ] {
            let raw: Vec<u8> = (0..9usize)
                .flat_map(|i| (0..bpe).map(|j| ((i * 31 + j * 7) % 251) as u8).collect::<Vec<_>>())
                .collect();
            let mut reader = TensorElementReader::new(dtype, &raw);
            for n in [0usize, 1, 4, 9, 12] {
                let expect: Vec<f32> = (0..n).map(|k| reader.element(k)).collect();
                let got = decode_prefix_f32(dtype, &raw, n);
                assert_eq!(got.len(), n, "{dtype:?} n={n}");
                for (a, b) in got.iter().zip(expect.iter()) {
                    assert_eq!(
                        a.is_nan(), b.is_nan(),
                        "{dtype:?} n={n}: NaN agreement"
                    );
                    if !a.is_nan() {
                        assert_eq!(a, b, "{dtype:?} n={n}");
                    }
                }
            }
        }
    }

    /// `diff_to_u8`'s fixed-stride fast path must produce the same bytes as
    /// the reader fallback, both for fully covered ranges and for a range
    /// that overruns either buffer (NaN sentinel tail → 255).
    #[test]
    fn diff_to_u8_fixed_fast_path_matches_reader_fallback() {
        let orig = f32_bytes(&[1.0, 2.0, 4.0, 8.0]);
        let modd = f32_bytes(&[1.5, 2.0, 5.0]);
        for metric in [
            DiffMetric::Rms,
            DiffMetric::AbsLog,
            DiffMetric::Exact,
        ] {
            for elem_count in 3usize..6 {
                // Force the reader fallback by requesting more elements than
                // the modified buffer covers; the fast path must decline and
                // both must agree.
                let got = Dtype::F32.diff_to_u8_sidecars(
                    &orig, 0, Dtype::F32, &modd, 0, metric, 1.0, elem_count,
                    None, (0, 0), None, (0, 0),
                );
                assert_eq!(got.len(), elem_count, "len n={elem_count}");
                for (k, &byte) in got.iter().enumerate() {
                    let tail = k >= 3;
                    if tail {
                        assert_eq!(byte, 255, "NaN sentinel at k={k}");
                    }
                }
            }
        }
        // Fully covered range: exact metric on equal pixels → 127 (mid-grey).
        let a = f32_bytes(&[1.0, 2.0]);
        let got = Dtype::F32.diff_to_u8_sidecars(&a, 0, Dtype::F32, &a, 0, DiffMetric::Exact, 1.0, 2, None, (0, 0), None, (0, 0));
        assert_eq!(got, vec![127u8, 127]);
    }

    #[test]
    fn diff_to_u8_sidecars_matches_dequantized_ground_truth() {
        // 4-row × 8-col packed int4 tensor (one int32 slot per row),
        // symmetric (no zeros), scale 1.0 → element (r, c) dequantises to
        // its int value. orig holds c+1, mod holds c+4, so every element's
        // delta is +3.
        let row_word = |add: u32| {
            (0..8u32)
                .map(|c| (c + 1 + add) << (4 * c))
                .fold(0u32, |a, v| a | v)
        };
        let qweight: Vec<u8> = (0..4u32)
            .flat_map(|_| row_word(0).to_le_bytes())
            .collect();
        let mod_qweight: Vec<u8> = (0..4u32)
            .flat_map(|_| row_word(3).to_le_bytes())
            .collect();
        let scales: Vec<u8> = std::iter::repeat(1.0f32.to_le_bytes())
            .take(8)
            .flatten()
            .collect();
        let refs = PackedSidecarRefs {
            scales: &scales,
            scales_dtype: Dtype::F32,
            zeros: None,
            zeros_dtype: Dtype::Unknown,
            cols: 8,
        };

        // Ground truth: dequantise both sides, diff as f32 buffers.
        let o_f32 = decode_prefix_f32_sidecars(Dtype::Int4Packed, &qweight, 32, Some(refs));
        let m_f32 = decode_prefix_f32_sidecars(Dtype::Int4Packed, &mod_qweight, 32, Some(refs));
        let expected = Dtype::F32.diff_to_u8_sidecars(
            &o_f32.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>(),
            0,
            Dtype::F32,
            &m_f32.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>(),
            0,
            DiffMetric::Exact,
            1.0,
            32,
            None, (0, 0), None, (0, 0),
        );
        let got = Dtype::Int4Packed.diff_to_u8_sidecars(
            &qweight,
            0,
            Dtype::Int4Packed,
            &mod_qweight,
            0,
            DiffMetric::Exact,
            1.0,
            32,
            Some(refs),
            (0, 0),
            Some(refs),
            (0, 0),
        );
        assert_eq!(got, expected);
        // All deltas are +3 → every byte is the same non-sentinel value.
        assert!(!got.iter().any(|&b| b == 255));
        assert!(got.windows(2).all(|w| w[0] == w[1]));

        // Without sidecars the packed sides decode to NaN sentinels.
        let got = Dtype::Int4Packed.diff_to_u8_sidecars(
            &qweight, 0, Dtype::Int4Packed, &mod_qweight, 0, DiffMetric::Exact, 1.0, 32,
            None, (0, 0), None, (0, 0),
        );
        assert_eq!(got, vec![255u8; 32]);
    }

    #[test]
    fn diff_to_u8_sidecars_row_aligned_fetch_matches_full_buffer() {
        // Same fixture as above; simulate the diff source's row-aligned
        // partial fetch: rows 1..4 of the qweight, starting mid-row at
        // absolute element 9, anchored at (row 1, col 0). Must byte-match
        // the same absolute range decoded from the whole-tensor buffers.
        let qweight: Vec<u8> = (0..4u32)
            .flat_map(|_| {
                (0..8u32)
                    .map(|c| (c + 1) << (4 * c))
                    .fold(0u32, |a, v| a | v)
                    .to_le_bytes()
            })
            .collect();
        let mod_qweight: Vec<u8> = (0..4u32)
            .flat_map(|_| {
                (0..8u32)
                    .map(|c| (c + 4) << (4 * c))
                    .fold(0u32, |a, v| a | v)
                    .to_le_bytes()
            })
            .collect();
        let scales: Vec<u8> = std::iter::repeat(1.0f32.to_le_bytes())
            .take(8)
            .flatten()
            .collect();
        let refs = PackedSidecarRefs {
            scales: &scales,
            scales_dtype: Dtype::F32,
            zeros: None,
            zeros_dtype: Dtype::Unknown,
            cols: 8,
        };
        let expected = Dtype::Int4Packed.diff_to_u8_sidecars(
            &qweight, 9, Dtype::Int4Packed, &mod_qweight, 9, DiffMetric::AbsLog, 1.0, 6,
            Some(refs), (0, 0), Some(refs), (0, 0),
        );
        // Row-aligned fetch: buffer starts at byte 4 (row 1), covering
        // rows 1..4; requested start element 9 is elem_off 1 into it,
        // anchor (1, 0).
        let got = Dtype::Int4Packed.diff_to_u8_sidecars(
            &qweight[4..],
            1,
            Dtype::Int4Packed,
            &mod_qweight[4..],
            1,
            DiffMetric::AbsLog,
            1.0,
            6,
            Some(refs),
            (1, 0),
            Some(refs),
            (1, 0),
        );
        assert_eq!(got, expected);
    }

    #[test]
    fn frobenius_from_buf_matches_naive() {
        let b = f32_bytes(&[3.0, 4.0]);
        let r = frobenius_from_buf(Dtype::F32, &b);
        assert!((r - 5.0).abs() < 1e-6, "got {r}");
    }

    #[test]
    fn frobenius_from_buf_skips_non_finite() {
        let b = f32_bytes(&[3.0, f32::NAN, 4.0, f32::INFINITY]);
        let r = frobenius_from_buf(Dtype::F32, &b);
        assert!((r - 5.0).abs() < 1e-6, "got {r}");
    }

    #[test]
    fn frobenius_from_buf_empty_is_zero() {
        assert_eq!(frobenius_from_buf(Dtype::F32, &[]), 0.0);
    }

    #[test]
    fn mean_abs_from_buf_basic() {
        let b = f32_bytes(&[1.0, -3.0, 2.0, -2.0]);
        let r = mean_abs_from_buf(Dtype::F32, &b);
        // mean(|1|, |-3|, |2|, |-2|) = 8/4 = 2.0
        assert!((r - 2.0).abs() < 1e-6, "got {r}");
    }

    #[test]
    fn mean_abs_from_buf_skips_non_finite() {
        // Non-finite values are excluded from both numerator and count, so
        // the mean is over the remaining 3 finite entries.
        let b = f32_bytes(&[1.0, f32::NAN, -3.0, 2.0, f32::INFINITY]);
        let r = mean_abs_from_buf(Dtype::F32, &b);
        assert!((r - 2.0).abs() < 1e-6, "got {r}");
    }

    #[test]
    fn mean_abs_from_buf_empty_is_zero() {
        assert_eq!(mean_abs_from_buf(Dtype::F32, &[]), 0.0);
    }

    #[test]
    fn sparsity_from_buf_counts_near_zero() {
        // 3 of 5 entries have |x| < 1e-6.
        let b = f32_bytes(&[0.0, 1.0, 1e-9, -1e-12, 2.0]);
        let r = sparsity_from_buf(Dtype::F32, &b, 1e-6);
        assert!((r - 0.6).abs() < 1e-6, "got {r}");
    }

    #[test]
    fn sparsity_from_buf_non_finite_not_sparse() {
        // NaN/Inf are not "near zero" — they don't contribute to the
        // sparsity count. Denominator stays the full element count
        // (sparsity is a fraction of all entries, not just finite ones —
        // otherwise a tensor of all-NaN would report 100% sparse, which is
        // misleading).
        let b = f32_bytes(&[0.0, f32::NAN, f32::INFINITY, 1.0]);
        let r = sparsity_from_buf(Dtype::F32, &b, 1e-6);
        // 1 near-zero out of 4 entries.
        assert!((r - 0.25).abs() < 1e-6, "got {r}");
    }

    #[test]
    fn sparsity_from_buf_empty_is_zero() {
        assert_eq!(sparsity_from_buf(Dtype::F32, &[], 1e-6), 0.0);
    }

    fn f16_bytes(vs: &[f32]) -> Vec<u8> {
        use half::f16;
        vs.iter()
            .flat_map(|&v| f16::from_f32(v).to_le_bytes())
            .collect()
    }

    fn bf16_bytes(vs: &[f32]) -> Vec<u8> {
        use half::bf16;
        vs.iter()
            .flat_map(|&v| bf16::from_f32(v).to_le_bytes())
            .collect()
    }

    #[test]
    fn scalar_helpers_handle_f16_bf16() {
        // Same expected values as the F32 cases — half-precision rounds but
        // the assertion thresholds are loose enough to absorb it.
        for (name, bytes_fn) in [
            ("f16", f16_bytes as fn(&[f32]) -> Vec<u8>),
            ("bf16", bf16_bytes as fn(&[f32]) -> Vec<u8>),
        ] {
            let dt = if name == "f16" {
                Dtype::F16
            } else {
                Dtype::BF16
            };
            let r_rms = rms_from_buf(dt, &bytes_fn(&[1.0, -1.0, 1.0, -1.0]));
            assert!((r_rms - 1.0).abs() < 1e-2, "{name} rms: {r_rms}");
            let r_fro = frobenius_from_buf(dt, &bytes_fn(&[3.0, 4.0]));
            assert!((r_fro - 5.0).abs() < 1e-2, "{name} frobenius: {r_fro}");
            let r_ma = mean_abs_from_buf(dt, &bytes_fn(&[1.0, -3.0, 2.0, -2.0]));
            assert!((r_ma - 2.0).abs() < 1e-2, "{name} mean_abs: {r_ma}");
        }
    }

    #[test]
    fn stride_fixed_for_plain() {
        assert_eq!(Dtype::F32.stride(), ElementStride::Fixed(4));
        assert_eq!(Dtype::F16.stride(), ElementStride::Fixed(2));
        assert_eq!(Dtype::I8.stride(), ElementStride::Fixed(1));
    }

    #[test]
    fn stride_packed_for_awq_dtypes() {
        match Dtype::Int4Packed.stride() {
            ElementStride::Packed {
                bits,
                pack_dtype_bytes,
                group_size,
            } => {
                assert_eq!(bits, 4);
                assert_eq!(pack_dtype_bytes, 4);
                assert_eq!(group_size, 128);
            }
            other => panic!("expected Packed for Int4Packed, got {other:?}"),
        }
    }

    #[test]
    fn packed_int4_dequant_known_values() {
        // Build a 2-row × 8-col tensor of int4 quants:
        //   row 0: [1, 2, 3, 4, 5, 6, 7, 8]   (one int32 slot = 0x87654321)
        //   row 1: [0, 1, 0, 2, 0, 3, 0, 4]   (one int32 slot = 0x40302010)
        // Two group_size=2 groups per row (well, group_size really applies on
        // rows for canonical layouts — for this micro test we set group_size = 2
        // so we exercise the group lookup). Scale = 1.0, zero = 0 ⇒ dequant
        // returns the int4 value as-is.
        //
        // We construct sidecars manually:
        //   scales: 2 groups × 8 cols, all = 1.0 (f32)
        //   zeros:  None (symmetric)
        let qweight: Vec<u8> = {
            let row0: u32 = 0x8765_4321;
            let row1: u32 = 0x4030_2010;
            let mut v = Vec::new();
            v.extend_from_slice(&row0.to_le_bytes());
            v.extend_from_slice(&row1.to_le_bytes());
            v
        };
        let scales: Vec<u8> = {
            let mut v = Vec::new();
            // 2 groups × 8 cols, but for this single-row-per-group layout
            // we just need group 0 (covers row 0) and group 1 (covers row 1).
            // Each group has 8 cols. So 16 scales total.
            for _ in 0..16 {
                v.extend_from_slice(&1.0f32.to_le_bytes());
            }
            v
        };
        let sc = PackedSidecarRefs {
            scales: &scales,
            scales_dtype: Dtype::F32,
            zeros: None,
            zeros_dtype: Dtype::Unknown,
            cols: 8,
        };

        let mut r = TensorElementReader::new(Dtype::Int4Packed, &qweight).with_sidecars(sc);
        // Override the stride's group_size via construction is not exposed,
        // but the stride() method returns group_size=128 for Int4Packed by
        // default. For our 2-row test we set the scales table to a full
        // 16-element layout (groups 0/1 × cols 0..8) so the group lookup at
        // row 0 picks scale group 0 = 1.0, row 1 picks group 0 again (since
        // row 1 / 128 = 0). Either way, scale=1.0.
        // Row 0 values:
        assert_eq!(r.element(0), 1.0);
        assert_eq!(r.element(1), 2.0);
        assert_eq!(r.element(7), 8.0);
        // Row 1 values:
        assert_eq!(r.element(8), 0.0);
        assert_eq!(r.element(9), 1.0);
        assert_eq!(r.element(15), 4.0);
    }

    #[test]
    fn decode_prefix_f32_sidecars_matches_reader() {
        // Packed int4 tensor + sidecars → the helper dequantises through the
        // sidecars (same values as TensorElementReader with them attached).
        let qweight: Vec<u8> = {
            let mut v = Vec::new();
            v.extend_from_slice(&0x8765_4321u32.to_le_bytes());
            v
        };
        let scales: Vec<u8> = std::iter::repeat_n(1.0f32.to_le_bytes(), 8)
            .flatten()
            .collect();
        let refs = PackedSidecarRefs {
            scales: &scales,
            scales_dtype: Dtype::F32,
            zeros: None,
            zeros_dtype: Dtype::Unknown,
            cols: 8,
        };
        let v = decode_prefix_f32_sidecars(Dtype::Int4Packed, &qweight, 8, Some(refs));
        assert_eq!(v.len(), 8);
        for (i, expect) in [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0].iter().enumerate() {
            assert_eq!(v[i], *expect, "element {i}");
        }

        // Without sidecars the packed dtype decodes to NaN (unchanged
        // behaviour), and a non-packed dtype ignores the refs entirely.
        let v = decode_prefix_f32_sidecars(Dtype::Int4Packed, &qweight, 8, None);
        assert!(v.iter().all(|x| x.is_nan()));
        let plain = [7.0f32.to_le_bytes().as_slice()].concat();
        let v = decode_prefix_f32_sidecars(Dtype::F32, &plain, 1, Some(refs));
        assert_eq!(v, vec![7.0]);
    }

    #[test]
    fn packed_without_sidecars_returns_nan() {
        // Same packed buffer, no sidecars attached → NaN per element.
        let qweight: Vec<u8> = vec![0xFFu8; 4];
        let mut r = TensorElementReader::new(Dtype::Int4Packed, &qweight);
        for k in 0..8 {
            assert!(
                r.element(k).is_nan(),
                "k={k}: expected NaN without sidecars"
            );
        }
    }

    #[test]
    fn stride_block_for_quantized() {
        match Dtype::Q4_0.stride() {
            ElementStride::Block {
                block_bytes,
                block_elements,
            } => {
                assert_eq!(block_elements, 32);
                assert_eq!(block_bytes, 18);
            }
            other => panic!("expected Block, got {other:?}"),
        }
        match Dtype::Q4K.stride() {
            ElementStride::Block {
                block_bytes,
                block_elements,
            } => {
                assert_eq!(block_elements, 256);
                // Q4K is 144 bytes per 256-element block.
                assert_eq!(block_bytes, 144);
            }
            other => panic!("expected Block, got {other:?}"),
        }
    }
    #[test]
    fn packed_sidecar_anchor_row_region() {
        // 4 rows x 8 cols int4 packed (one int32 slot per row). Slot value
        // for row r is 0xRRRRRRRR so every nibble is r. Scales are f32,
        // scale(col) = col + 1. Dequantised value at (r, c) = r * (c + 1).
        let qweight: Vec<u8> = [0u32, 1, 2, 3]
            .iter()
            .map(|r| r * 0x1111_1111) // every nibble of slot r is r
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let scales: Vec<u8> = (0..8)
            .flat_map(|c| ((c as f32) + 1.0).to_le_bytes())
            .collect();
        let sc = PackedSidecarRefs {
            scales: &scales,
            scales_dtype: Dtype::F32,
            zeros: None,
            zeros_dtype: Dtype::Unknown,
            cols: 8,
        };
        // Whole-tensor buffer, anchor (0, 0): baseline behaviour.
        let mut whole = TensorElementReader::new(Dtype::Int4Packed, &qweight)
            .with_anchor(0, 0)
            .with_sidecars(sc);
        for (k, expect) in (0..32).map(|k| {
            let (r, c) = (k / 8, k % 8);
            (k, r as f32 * (c as f32 + 1.0))
        }) {
            assert_eq!(whole.element(k), expect, "whole-tensor k={k}");
        }

        // Tile-region slice: only row 2's slot was fetched (bytes 8..12).
        // Without the anchor the decoder would read row 0's scales and
        // return 0.0 * (c + 1) = 0 for every element; with it the sidecar
        // lookups lift to the absolute row.
        let region = &qweight[8..12];
        let mut r2 = TensorElementReader::new(Dtype::Int4Packed, region)
            .with_anchor(2, 0)
            .with_sidecars(sc);
        for (k, expect) in (0..8).map(|k| (k, 2.0f32 * (k as f32 + 1.0))) {
            assert_eq!(r2.element(k), expect, "row-anchored region k={k}");
        }
    }

    #[test]
    fn packed_sidecar_anchor_col_region() {
        // 1 row x 16 cols (two int32 slots per row), slot 0 packed with
        // nibble value 7, slot 1 with 5. A region covering only slot 1
        // (bytes 4..8) with anchor col 8 must read slot 1's nibbles and
        // slot 1's scales (cols 8..16).
        let slot = |v: u32| v.to_le_bytes();
        let qweight: Vec<u8> = [slot(0x7777_7777), slot(0x5555_5555)].concat();
        let scales: Vec<u8> = (0..16)
            .flat_map(|c| ((c as f32) + 1.0).to_le_bytes())
            .collect();
        let sc = PackedSidecarRefs {
            scales: &scales,
            scales_dtype: Dtype::F32,
            zeros: None,
            zeros_dtype: Dtype::Unknown,
            cols: 16,
        };
        let region = &qweight[4..8];
        let mut r = TensorElementReader::new(Dtype::Int4Packed, region)
            .with_anchor(0, 8)
            .with_sidecars(sc);
        for (k, expect) in (0..8).map(|k| (k, 5.0f32 * (k as f32 + 9.0))) {
            assert_eq!(r.element(k), expect, "col-anchored region k={k}");
        }
    }
}

