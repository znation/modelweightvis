//! Format-agnostic tensor / model types.
//!
//! `TensorMeta`, `ModelInfo`, `DiffFill`, and `DiffMetric` are shared across
//! every model format (safetensors, GGUF, future). They carry no format-
//! specific state — `file_start`/`file_end` are absolute byte ranges in the
//! underlying file so the renderer can treat both formats identically.

use image::Rgb;

use super::dtype::Dtype;
use super::SourceFormat;

// DiffFill and DiffMetric moved to arbvis (byte-foundation). The
// `format::DiffMetric` / `format::DiffFill` names that the per-format
// parsers use are now re-exported from `format/mod.rs`.

/// Saturation threshold for `DiffMetric::Rms`: an element whose delta equals
/// `K_RMS_SAT * rms(orig)` paints at full brightness. 0.5 means "half a
/// tensor-stddev is fully saturated"; a typical LoRA-merge moves median
/// elements by ~0.005 stddevs (subtle), an aggressive full-finetune by ~0.05
/// stddevs (clearly visible), an uncorrelated init by ~1 stddev (saturated).
pub const K_RMS_SAT: f32 = 0.5;

/// Floor for `rms(orig)` in `DiffMetric::Rms`, used to avoid divide-by-zero
/// on all-zero tensors and to cap sensitivity on near-zero tensors.
pub const RMS_FLOOR: f32 = 1e-6;

/// Log-brightness range endpoints for `DiffMetric::AbsLog`. Deltas with
/// `|delta| < ABS_LOG_MIN` paint black; `|delta| >= ABS_LOG_MAX` saturate.
/// The span covers the typical range of useful bf16 finetune deltas.
pub const ABS_LOG_MIN: f32 = 1e-6;
pub const ABS_LOG_MAX: f32 = 1e-1;

/// Per-tensor metadata, format-agnostic.
///
/// For safetensors: built from the JSON header at file open. For GGUF: built
/// from the tensor info table. `file_start`/`file_end` are absolute byte
/// offsets into the underlying file; the renderer does not need to know which
/// format produced them.
#[derive(Debug, Clone)]
pub struct TensorMeta {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<u64>,
    /// Absolute byte positions in the file [start, end)
    pub file_start: u64,
    pub file_end: u64,
    /// AWQ / GPTQ packed-int sidecar references. `Some(...)` for tensors
    /// whose [`Dtype`] is one of `IntNPacked`; `None` for plain / Block
    /// dtypes. The qweight byte range lives in `file_start`/`file_end`; the
    /// sidecar struct carries the parallel byte ranges and dtypes for the
    /// `scales` and `qzeros` tensors needed to dequantise.
    pub packed_sidecars: Option<PackedSidecars>,
}

/// Validate that every tensor's absolute byte range fits within the file.
///
/// A truncated file (interrupted download or copy) can carry a fully valid
/// header while its tensor data is cut off; without this check the header
/// parse succeeds and later per-tensor reads run past the end of the file
/// (faulting on an mmap, or erroring per tile). `file_size` is the real
/// length of the data the header was parsed from. Format-agnostic: applies
/// to any source whose [`TensorMeta`] carries absolute byte ranges.
pub fn validate_tensor_offsets(tensors: &[TensorMeta], file_size: u64) -> anyhow::Result<()> {
    for t in tensors {
        if t.file_end > file_size {
            anyhow::bail!(
                "file is truncated — tensor '{}' needs bytes {}..{} but the data is only \
                 {} bytes (interrupted download or copy?)",
                t.name,
                t.file_start,
                t.file_end,
                file_size
            );
        }
        // AWQ/GPTQ fused tensors carry their scales/qzeros byte ranges in
        // `packed_sidecars`. The fusion step removes the sidecar entries from
        // the tensor list before validation, so this is the only place their
        // ranges are checked against the real file size.
        if let Some(sc) = &t.packed_sidecars {
            if sc.scales_end > file_size {
                anyhow::bail!(
                    "file is truncated — tensor '{}' declares scales bytes {}..{} but \
                     the data is only {} bytes (interrupted download or copy?)",
                    t.name,
                    sc.scales_start,
                    sc.scales_end,
                    file_size
                );
            }
            if let Some(ze) = sc.zeros_end {
                if ze > file_size {
                    anyhow::bail!(
                        "file is truncated — tensor '{}' declares qzeros bytes ending at {} \
                         but the data is only {} bytes (interrupted download or copy?)",
                        t.name,
                        ze,
                        file_size
                    );
                }
            }
        }
    }
    Ok(())
}

/// Byte ranges and dtypes for the `scales` / `qzeros` sidecar tensors that
/// accompany an AWQ/GPTQ-style packed-int `qweight` tensor in the same file.
/// Populated by [`crate::format::safetensors::fuse_packed_quant_triples`].
#[derive(Debug, Clone)]
pub struct PackedSidecars {
    pub scales_start: u64,
    pub scales_end: u64,
    pub scales_dtype: Dtype,
    /// `None` for AWQ symmetric quants without zero-points. `Some` for
    /// GPTQ / EXL2 / AWQ-with-qzeros.
    pub zeros_start: Option<u64>,
    pub zeros_end: Option<u64>,
    pub zeros_dtype: Dtype,
    /// Number of output columns in the unpacked tensor — needed when
    /// indexing per-element across rows.
    pub cols: u32,
}

impl TensorMeta {
    /// 2D pixel-grid shape used by the architectural layout. Distinct from
    /// `shape`, which is the raw tensor shape: this collapses to exactly two
    /// dimensions so a tensor occupies a flat rectangle on the canvas.
    ///
    /// - 0-D (scalar) → `(1, 1)`
    /// - 1-D `(n)` → `(1, n)` (one-pixel-tall strip)
    /// - 2-D `(r, c)` → `(r, c)` (preserved)
    /// - ≥3-D `(a, b, c, …)` → `(a, b*c*…)` (last dims collapsed into the
    ///   column axis). The element index within the resulting rect uses
    ///   row-major order, which matches the byte order in the underlying
    ///   file: element `(row, col)` lives at the logical position
    ///   `row*cols + col`.
    pub fn element_shape(&self) -> (u64, u64) {
        match self.shape.len() {
            0 => (1, 1),
            1 => (1, self.shape[0]),
            2 => (self.shape[0], self.shape[1]),
            _ => {
                let rows = self.shape[0];
                let cols: u64 = self.shape[1..].iter().product();
                (rows, cols)
            }
        }
    }

    /// Human-readable one-line description: `name dtype [rows, cols]`.
    pub fn label(&self) -> String {
        let shape_str: Vec<String> = self.shape.iter().map(|d| d.to_string()).collect();
        format!(
            "{} [{}, {}]",
            self.name,
            self.dtype.label(),
            shape_str.join("×")
        )
    }
}

/// Format-aware metadata attached to a `Source` whose underlying file is a
/// recognised model format. The tensor list drives the architectural layout;
/// `color_ranges` drives the legacy Hilbert dtype-mode coloring.
///
/// `format` records which parser produced this — read by cross-format diff
/// matching to canonicalise tensor names before pairing.
#[cfg(test)]
mod tests {
    use super::*;

    fn meta(shape: &[u64]) -> TensorMeta {
        TensorMeta {
            name: "t".into(),
            dtype: Dtype::F32,
            shape: shape.to_vec(),
            file_start: 0,
            file_end: 0,
            packed_sidecars: None,
        }
    }

    #[test]
    fn element_shape_scalar_is_one_by_one() {
        assert_eq!(meta(&[]).element_shape(), (1, 1));
    }

    #[test]
    fn element_shape_1d_is_one_row_strip() {
        assert_eq!(meta(&[7]).element_shape(), (1, 7));
        assert_eq!(meta(&[0]).element_shape(), (1, 0));
    }

    #[test]
    fn element_shape_2d_preserved() {
        assert_eq!(meta(&[3, 5]).element_shape(), (3, 5));
    }

    #[test]
    fn element_shape_nd_collapses_last_dims() {
        assert_eq!(meta(&[2, 3, 4]).element_shape(), (2, 12));
        assert_eq!(meta(&[2, 3, 4, 5]).element_shape(), (2, 60));
        assert_eq!(meta(&[2, 0, 4]).element_shape(), (2, 0));
    }

    #[test]
    fn label_names_dtype_and_shape() {
        let mut m = meta(&[2, 3]);
        m.dtype = Dtype::BF16;
        m.name = "model.layers.1.q_proj.weight".into();
        assert_eq!(m.label(), "model.layers.1.q_proj.weight [BF16, 2×3]");
    }

    #[test]
    fn label_scalar_has_empty_axis_list() {
        assert_eq!(meta(&[]).label(), "t [F32, ]");
    }

    fn mk_t(name: &str, start: u64, end: u64) -> TensorMeta {
        TensorMeta {
            name: name.to_string(),
            dtype: Dtype::Int4Packed,
            shape: vec![4096, 4096],
            file_start: start,
            file_end: end,
            packed_sidecars: None,
        }
    }

    fn sidecars(scales_end: u64, zeros_end: Option<u64>) -> PackedSidecars {
        PackedSidecars {
            scales_start: 1_000_000,
            scales_end,
            scales_dtype: Dtype::F16,
            zeros_start: zeros_end.map(|_| 2_000_000),
            zeros_end,
            zeros_dtype: Dtype::I32,
            cols: 4096,
        }
    }

    #[test]
    fn validate_offsets_rejects_sidecar_past_eof() {
        // The fused qweight entry is in-bounds, but the scales range was
        // declared past EOF (hostile header or truncated download). The
        // sidecar entries were consumed at fusion time, so this validator is
        // the only boundary the scales range crosses before fetch_range.
        let mut t = mk_t("w", 0, 1_000_000);
        t.packed_sidecars = Some(sidecars(3_000_000, Some(2_500_000)));
        let err = validate_tensor_offsets(&[t], 2_000_000).unwrap_err();
        assert!(format!("{err:#}").contains("scales"));
    }

    #[test]
    fn validate_offsets_rejects_qzeros_past_eof() {
        let mut t = mk_t("w", 0, 1_000_000);
        t.packed_sidecars = Some(sidecars(1_500_000, Some(9_999_999)));
        let err = validate_tensor_offsets(&[t], 2_000_000).unwrap_err();
        assert!(format!("{err:#}").contains("qzeros"));
    }

    #[test]
    fn validate_offsets_accepts_in_bounds_sidecars() {
        let mut t = mk_t("w", 0, 1_000_000);
        t.packed_sidecars = Some(sidecars(1_500_000, Some(2_000_000)));
        assert!(validate_tensor_offsets(std::slice::from_ref(&t), 2_000_000).is_ok());
    }

    #[test]
    fn validate_ranges_strips_sidecar_past_eof_but_keeps_tensor() {
        let mut t = mk_t("w", 0, 1_000_000);
        t.packed_sidecars = Some(sidecars(3_000_000, Some(3_500_000)));
        let mut v = vec![t];
        let dropped = validate_tensor_ranges(&mut v, 2_000_000);
        assert_eq!(dropped, 0, "the qweight entry itself is in-bounds");
        assert!(v[0].packed_sidecars.is_none());
    }

    #[test]
    fn validate_ranges_keeps_in_bounds_sidecar() {
        let mut v = vec![mk_t("w", 0, 1_000_000)];
        v[0].packed_sidecars = Some(sidecars(1_500_000, None));
        validate_tensor_ranges(&mut v, 2_000_000);
        assert!(v[0].packed_sidecars.is_some());
    }
}

#[derive(Debug, Clone)]
pub struct ModelInfo {
    #[allow(dead_code)]
    pub format: SourceFormat,
    pub tensors: Vec<TensorMeta>,
    /// Per-file `(start, end, color)` LUT covering the metadata-header
    /// region and per-tensor byte ranges. Populated by the per-format
    /// `build_color_ranges` helpers and consumed by [`color_for_pos`]
    /// in this crate's tests. The split-out byte-Hilbert pipeline lives
    /// in arbvis (which doesn't carry [`ModelInfo`]), and the arch
    /// renderer keys off `TensorMeta` directly — so this field is
    /// currently only exercised by tests. Kept for a future arch-mode
    /// Hilbert overlay (one-line wire-up) and for read-back by curious
    /// downstream tooling.
    #[allow(dead_code)]
    pub color_ranges: Vec<(u64, u64, Rgb<u8>)>,
}

/// Validate parsed tensors' absolute byte ranges against the file's actual
/// size, dropping entries that fall outside it.
///
/// A malicious or truncated model file (safetensors and GGUF alike) can
/// declare tensor ranges beyond the end of the file (or `end < start`):
/// safetensors carries `data_offsets` straight from its JSON header, and
/// GGUF tensor infos carry raw u64 `offset` fields that candle's
/// `Content::read` never checks against the real file. Downstream readers
/// slice the backing bytes at those offsets, and an out-of-range slice
/// panics, so the header cannot be trusted on its own: this is the
/// boundary where the declared ranges meet the real file length. Returns
/// how many tensors were dropped and logs one warning per drop.
pub fn validate_tensor_ranges(tensors: &mut Vec<TensorMeta>, file_size: u64) -> usize {
    let before = tensors.len();
    tensors.retain(|t| {
        let in_bounds = t.file_end <= file_size && t.file_start < t.file_end;
        if !in_bounds {
            log::warn!(
                "dropping tensor '{}' with declared byte range [{}..{}) \
                 outside file size {}",
                t.name,
                t.file_start,
                t.file_end,
                file_size
            );
        }
        in_bounds
    });
    // Sidecar ranges cannot be range-checked at fusion time (the sidecar
    // entries were consumed there), and the fused qweight entry can be
    // in-bounds while its scales/qzeros are not — strip the sidecar
    // reference rather than the whole tensor; the renderer paints NaN
    // sentinels without sidecars, the same as an incomplete quant triple.
    for t in tensors.iter_mut() {
        if let Some(sc) = &t.packed_sidecars {
            let zeros_bad = sc.zeros_end.map_or(false, |ze| ze > file_size);
            if sc.scales_end > file_size || zeros_bad {
                log::warn!(
                    "dropping packed-int sidecar ranges for tensor '{}' — declared \
                     bytes fall outside file size {}",
                    t.name,
                    file_size
                );
                t.packed_sidecars = None;
            }
        }
    }
    before - tensors.len()
}
