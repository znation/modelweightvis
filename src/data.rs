//! Tensor-aware data helpers extracted from `arbvis::data`.
//!
//! These items use `format::*` (Dtype, TensorMeta, etc.) and the
//! safetensors / GGUF / pickle parsers — all of which live in
//! `modelweightvis::format`. arbvis itself stays format-agnostic; it
//! reaches these helpers via the `MoeScenesPrep`, `RepoDiffPrep`,
//! `DirectoryTensorDiffPrep`, and (per-Source) `FormatPlugin` hooks
//! registered by `modelweightvis::register_all`.

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Context;
use futures::stream::{self, StreamExt};
use indicatif::ProgressBar;
use memmap2::Mmap;

use arbvis::hf_url::{self, RemoteFileSpec, RemoteRepo};
use arbvis::xet::XetReader;
use arbvis::{CustomSource, Data, DiffFill, Extensions, Source, SourceKind};

use crate::format::{self, moe::ExpertWeight, ModelInfo, SourceFormat};
use crate::layout::model_config::{ModelConfig, SafetensorsIndex};

/// Bounded concurrency for setup-time HTTP loops. Mirrors arbvis's
/// `SETUP_FETCH_CONCURRENCY` constant; the global throttle is the real cap.
const SETUP_FETCH_CONCURRENCY: usize = 16;

/// Per-tensor RMS sampling concurrency. Smaller than SETUP because the
/// per-call descriptor download (~50–65 MB before dedup) is heavy.
const RMS_SAMPLE_FETCH_CONCURRENCY: usize = 4;

/// Initial header-fetch size for GGUF. Most files fit in 1 MiB; if
/// `Content::read` errors with "unexpected EOF" we retry with the larger
/// size below.
const GGUF_HEADER_FETCH_INITIAL: usize = 1024 * 1024;
const GGUF_HEADER_FETCH_LARGE: usize = 8 * 1024 * 1024;

/// Build a one-shot indicatif progress bar attached to a fresh
/// `MultiProgress` (modelweightvis-local — the moved-from helpers all
/// needed the arbvis-side one which isn't pub-exposed). The pb just
/// emits to stderr; behaviour matches arbvis byte-for-byte modulo the
/// bar group it lives in.
fn setup_progress(label: &str, total: u64) -> Option<ProgressBar> {
    let pb = ProgressBar::new(total);
    pb.set_message(label.to_string());
    pb.enable_steady_tick(std::time::Duration::from_millis(100));
    Some(pb)
}

/// Recursively collect all files under `root`, sorted.
fn collect_files_recursive(root: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    collect_recursive(root, &mut files);
    files.sort();
    files
}

fn collect_recursive(dir: &Path, files: &mut Vec<PathBuf>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) => {
            log::warn!("{}: {} — skipping", dir.display(), e);
            return;
        }
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_file() {
            files.push(path);
        } else if path.is_dir() {
            collect_recursive(&path, files);
        }
    }
}

/// Download every remote `Arc<Data>` in `arcs` (those for which `is_local()`
/// returns false) to a local mmap-backed file and swap them in place. Used
/// by the multi-shard tensor diff to amortise xet setup cost. Lifted from
/// arbvis::data unchanged; the xet/HTTP behaviour is identical.
async fn materialize_remote_arcs(
    arcs: &mut [Arc<Data>],
    specs: &[RemoteFileSpec],
    progress_label: &str,
) -> anyhow::Result<()> {
    assert_eq!(
        arcs.len(),
        specs.len(),
        "materialize_remote_arcs: arcs and specs must be the same length"
    );

    let to_download: Vec<(usize, RemoteFileSpec)> = arcs
        .iter()
        .zip(specs.iter())
        .enumerate()
        .filter_map(|(i, (arc, spec))| {
            if arc.is_local() {
                None
            } else {
                Some((i, spec.clone()))
            }
        })
        .collect();

    if to_download.is_empty() {
        return Ok(());
    }

    let indices: Vec<usize> = to_download.iter().map(|(i, _)| *i).collect();
    let specs_subset: Vec<RemoteFileSpec> = to_download.into_iter().map(|(_, s)| s).collect();
    let paths = arbvis::data::download_specs_to_paths(&specs_subset, progress_label).await?;

    for (i, path) in indices.into_iter().zip(paths) {
        let f = File::open(&path).with_context(|| format!("opening {}", path.display()))?;
        let mmap =
            unsafe { Mmap::map(&f) }.with_context(|| format!("mmap'ing {}", path.display()))?;
        arcs[i] = Arc::new(Data::Mapped(mmap));
    }
    Ok(())
}

// === extracted from arbvis::data (block_aligned_byte_range) ===
/// Block-aligned byte range for reading `len` consecutive elements starting
/// at element `start` from a tensor's data.
///
/// Returns `(byte_offset_from_tensor_start, byte_length, elem_offset_in_buffer)`.
/// For fixed-stride dtypes the element offset is always 0; for block-stride
/// it's the in-block position of `start`, since the buffer must begin on a
/// block boundary so [`crate::format::TensorElementReader`] can decode it.
fn block_aligned_byte_range(dtype: format::Dtype, start: u64, len: u64) -> (u64, usize, usize) {
    use format::ElementStride;
    match dtype.stride() {
        ElementStride::Fixed(bpe) => {
            let bpe = bpe as u64;
            (start * bpe, (len * bpe) as usize, 0)
        }
        ElementStride::Block {
            block_bytes,
            block_elements,
        } => {
            let be = block_elements as u64;
            let bb = block_bytes as u64;
            let first_block = start / be;
            let last_block_excl = (start + len).div_ceil(be);
            let byte_off = first_block * bb;
            let byte_len = (last_block_excl - first_block) * bb;
            let elem_off = (start - first_block * be) as usize;
            (byte_off, byte_len as usize, elem_off)
        }
        // Packed dtypes (AWQ / GPTQ): treat the packed-int slot as the
        // atomic unit. Snap down to the slot containing `start` and up to
        // the slot just past `start + len`; report the in-slot element
        // offset so the reader can skip over `elem_off` elements before
        // returning the requested range. The sidecar scales/zeros buffers
        // are fetched separately by the render path — this function only
        // sizes the qweight read.
        ElementStride::Packed {
            bits,
            pack_dtype_bytes,
            ..
        } => {
            if bits == 0 {
                return (0, 0, 0);
            }
            let elems_per_slot = ((pack_dtype_bytes as u64) * 8) / bits as u64;
            let Some(first_slot) = start.checked_div(elems_per_slot) else {
                return (0, 0, 0);
            };
            let slot_bytes = pack_dtype_bytes as u64;
            let last_slot_excl = (start + len).div_ceil(elems_per_slot);
            let byte_off = first_slot * slot_bytes;
            let byte_len = (last_slot_excl - first_slot) * slot_bytes;
            let elem_off = (start - first_slot * elems_per_slot) as usize;
            (byte_off, byte_len as usize, elem_off)
        }
    }
}

// `LazyFetcher` re-exported from arbvis::data (single source of truth).

// === extracted from arbvis::data (TensorDiffSource_struct,TensorDiffSource_impl) ===
/// Per-tensor diff buffer. The byte stream this exposes is computed lazily
/// from two whole-file `Data` sources whenever the render pipeline calls
/// `fetch_range` on the resulting `Data::LazyDiff`. `metric` selects how
/// per-element deltas are encoded; `scale_orig` carries any per-tensor
/// statistic the metric needs (RMS of `orig` for `DiffMetric::Rms`),
/// pre-computed at setup so the per-tile path stays pure-streaming.
pub struct TensorDiffSource {
    pub orig: Arc<Data>,
    pub mod_: Arc<Data>,
    pub orig_start: u64,
    pub mod_start: u64,
    pub orig_dtype: format::Dtype,
    pub mod_dtype: format::Dtype,
    pub metric: format::DiffMetric,
    pub scale_orig: f32,
    pub byte_size: u64,
}

impl CustomSource for TensorDiffSource {
    fn id(&self) -> &'static str {
        "tensor-diff"
    }

    fn byte_size(&self) -> u64 {
        self.byte_size
    }

    fn open(&self) -> anyhow::Result<Data> {
        let orig = Arc::clone(&self.orig);
        let mod_ = Arc::clone(&self.mod_);
        let orig_start = self.orig_start;
        let mod_start = self.mod_start;
        let orig_dtype = self.orig_dtype;
        let mod_dtype = self.mod_dtype;
        let metric = self.metric;
        let scale_orig = self.scale_orig;
        Ok(Data::LazyDiff(Arc::new(move |start: u64, len: usize| {
            let orig = Arc::clone(&orig);
            let mod_ = Arc::clone(&mod_);
            Box::pin(async move {
                // Block-aligned byte ranges: for fixed-stride dtypes
                // these are `start * elem`/`len * elem`; for block-
                // stride they snap down/up to block boundaries and the
                // returned `elem_off` is how many elements into the
                // fetched buffer the requested start element lives.
                let (o_byte_off, o_byte_len, o_elem_off) =
                    block_aligned_byte_range(orig_dtype, start, len as u64);
                let (m_byte_off, m_byte_len, m_elem_off) =
                    block_aligned_byte_range(mod_dtype, start, len as u64);
                let ob = orig
                    .fetch_range(orig_start + o_byte_off, o_byte_len)
                    .await?;
                let mb = mod_.fetch_range(mod_start + m_byte_off, m_byte_len).await?;
                Ok(orig_dtype.diff_to_u8(
                    &ob, o_elem_off, mod_dtype, &mb, m_elem_off, metric, scale_orig, len,
                ))
            })
        })))
    }
}

/// Per-panel tag attached to each `--moe-summary` source. One source per
/// `ExpertWeight` variant present in the model (gate/up/down/router); the
/// source's bytes are a `n_layers × n_experts` U8 heatmap, row-major
/// (rows = layers, cols = experts), normalized to `0..=255` per panel.
/// Presence of this extension on any source routes `select_layout` to the
/// MoE-summary layout plugin.
#[derive(Debug, Clone, Copy)]
pub struct MoeSummaryPanel {
    pub weight: ExpertWeight,
    pub n_layers: u32,
    pub n_experts: u32,
}

/// Per-panel tag attached to each `--moe-cka` source. One source per
/// `(layer, weight)` pair present in the model; the source's bytes
/// are an `n_experts × n_experts` U8 CKA-similarity heatmap, row-major,
/// values in `0..=255` (`255` = CKA 1.0). Presence of this extension
/// on any source routes `select_layout` to the MoE-CKA layout plugin.
#[derive(Debug, Clone, Copy)]
pub struct MoeCkaPanel {
    pub layer: u32,
    pub weight: ExpertWeight,
    pub n_experts: u32,
}

/// Which behavioral stat a probe panel carries. Drives the panel's
/// label and (later) colormap choices in the layout. `RoutingFreq` is
/// the `n_layers × n_experts` per-expert stat attached to `--moe-summary`;
/// `RoutingCoactivation` is the per-layer `n_experts × n_experts` co-routing
/// matrix attached to `--moe-cka`. The enum keeps the door open for
/// `RoutingWeightMean`, `ExpertOutputNorm`, etc. in follow-ups without
/// re-shuffling the extension types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeStat {
    /// Fraction of probe tokens routed to this expert by the router's
    /// top-k decision in this layer. Range: `[0, k / n_experts]` in
    /// expectation under uniform routing; real MoEs concentrate.
    RoutingFreq,
    /// Fraction of probe tokens whose top-k decision in this layer
    /// included both experts `i` and `j` (the diagonal is each expert's
    /// own routing frequency). Symmetric; range `[0, 1]` per cell.
    RoutingCoactivation,
}

impl ProbeStat {
    pub fn label(self) -> &'static str {
        match self {
            ProbeStat::RoutingFreq => "routing_freq",
            ProbeStat::RoutingCoactivation => "routing_coactivation",
        }
    }
}

/// Per-panel tag attached to each `--moe-summary --probe` behavioral
/// source. One source per `ProbeStat`; the source's bytes are an
/// `n_layers × n_experts` U8 heatmap of the stat (per-panel
/// normalised to `0..=255`). The layout plugin merges these panels
/// alongside the static `MoeSummaryPanel` columns.
#[derive(Debug, Clone, Copy)]
pub struct MoeProbePanel {
    pub stat: ProbeStat,
    pub n_layers: u32,
    pub n_experts: u32,
}

/// Per-layer tag attached to each `--moe-cka --probe` co-activation
/// source. One source per MoE layer; the source's bytes are an
/// `n_experts × n_experts` U8 co-routing heatmap (cell `(i,j)` = fraction
/// of probe tokens whose top-k included both expert `i` and `j`; the
/// diagonal `(i,i)` is expert `i`'s routing frequency), per-panel
/// normalised to `0..=255`. Same shape as [`MoeCkaPanel`], so the CKA
/// layout slots it in as an extra column per layer.
#[derive(Debug, Clone, Copy)]
pub struct MoeCkaProbePanel {
    pub layer: u32,
    pub n_experts: u32,
}

// === extracted from arbvis::data (load_model_info,fetch_model_header) ===
/// Read just the header of a recognised model file and return parsed
/// metadata. Dispatches on `format`.
pub fn load_model_info(
    path: &Path,
    file_size: u64,
    fmt: SourceFormat,
) -> anyhow::Result<ModelInfo> {
    match fmt {
        SourceFormat::Safetensors => {
            // Read the first 8 bytes to get header_size, then read header_size more bytes.
            let mut f = File::open(path)?;
            let mut size_buf = [0u8; 8];
            f.read_exact(&mut size_buf)?;
            let header_size = u64::from_le_bytes(size_buf);
            if header_size > 100 * 1024 * 1024 {
                anyhow::bail!("header_size={} exceeds 100 MB safety limit", header_size);
            }
            let total_header = 8 + header_size as usize;
            let mut header_buf = vec![0u8; total_header];
            header_buf[..8].copy_from_slice(&size_buf);
            f.read_exact(&mut header_buf[8..])?;
            let (tensors, header_end) = format::safetensors::parse_header(&header_buf)?;
            let color_ranges =
                format::safetensors::build_color_ranges(&tensors, header_end, file_size);
            Ok(ModelInfo {
                format: SourceFormat::Safetensors,
                tensors,
                color_ranges,
            })
        }
        SourceFormat::Gguf => {
            // For GGUF the header size isn't known up-front. Read a 1 MiB
            // prefix first; if parsing fails on EOF, retry with 8 MiB.
            let mut buf = vec![0u8; GGUF_HEADER_FETCH_INITIAL.min(file_size as usize)];
            let mut f = File::open(path)?;
            f.read_exact(&mut buf)?;
            let header = match format::gguf::parse_header(&buf) {
                Ok(h) => h,
                Err(_) if file_size as usize > buf.len() => {
                    let mut bigger = vec![0u8; GGUF_HEADER_FETCH_LARGE.min(file_size as usize)];
                    let mut f = File::open(path)?;
                    f.read_exact(&mut bigger)?;
                    format::gguf::parse_header(&bigger)?
                }
                Err(e) => return Err(e),
            };
            let color_ranges = format::gguf::build_color_ranges(
                &header.tensors,
                header.tensor_data_offset,
                file_size,
            );
            Ok(ModelInfo {
                format: SourceFormat::Gguf,
                tensors: header.tensors,
                color_ranges,
            })
        }
        SourceFormat::Pickle => {
            // Pickle's zip end-of-central-directory lives at the END of the
            // file, so unlike safetensors/GGUF we can't parse from a prefix.
            // candle's pickle reader opens the file by path; pass it through.
            let header = format::pickle::parse_header(path)?;
            let color_ranges = format::pickle::build_color_ranges(
                &header.tensors,
                header.tensor_data_offset,
                file_size,
            );
            Ok(ModelInfo {
                format: SourceFormat::Pickle,
                tensors: header.tensors,
                color_ranges,
            })
        }
    }
}

/// Async sibling of [`load_model_info`] for `Data` sources (Http/Xet/etc.).
/// Wraps [`fetch_model_header`] and builds the same [`ModelInfo`] shape.
///
/// Pickle files cannot be parsed from a remote prefix (the zip
/// end-of-central-directory lives at the tail), so this returns an error
/// for [`SourceFormat::Pickle`] — callers fall back to "treat as plain
/// bytes" the same way they would for an unrecognised file.
pub async fn load_model_info_async(
    data: &Data,
    byte_size: u64,
    fmt: SourceFormat,
) -> anyhow::Result<ModelInfo> {
    let (tensors, header_end) = fetch_model_header(data, fmt).await?;
    let color_ranges = match fmt {
        SourceFormat::Safetensors => {
            format::safetensors::build_color_ranges(&tensors, header_end, byte_size)
        }
        SourceFormat::Gguf => format::gguf::build_color_ranges(&tensors, header_end, byte_size),
        SourceFormat::Pickle => format::pickle::build_color_ranges(&tensors, header_end, byte_size),
    };
    Ok(ModelInfo {
        format: fmt,
        tensors,
        color_ranges,
    })
}

/// Fetch and parse the header from any `Data` source. For local sources
/// (Mapped/Owned) the fetches are zero-copy slices. For remote sources
/// (Http/Xet) this issues one or two range requests.
async fn fetch_model_header(
    data: &Data,
    fmt: SourceFormat,
) -> anyhow::Result<(Vec<format::TensorMeta>, u64)> {
    match fmt {
        SourceFormat::Safetensors => {
            let size_bytes = data.fetch_range(0, 8).await?;
            let header_size = u64::from_le_bytes(size_bytes[..8].try_into().unwrap());
            if header_size > 100 * 1024 * 1024 {
                anyhow::bail!(
                    "safetensors header_size={} exceeds 100 MB safety limit",
                    header_size
                );
            }
            let total_header = 8 + header_size as usize;
            let header_bytes = data.fetch_range(0, total_header).await?;
            format::safetensors::parse_header(&header_bytes)
        }
        SourceFormat::Gguf => {
            let buf = data.fetch_range(0, GGUF_HEADER_FETCH_INITIAL).await?;
            let header = match format::gguf::parse_header(&buf) {
                Ok(h) => h,
                Err(_) => {
                    let bigger = data.fetch_range(0, GGUF_HEADER_FETCH_LARGE).await?;
                    format::gguf::parse_header(&bigger)?
                }
            };
            Ok((header.tensors, header.tensor_data_offset))
        }
        SourceFormat::Pickle => {
            // The zip end-of-central-directory record lives at the END of a
            // pickle file (.bin / .pth / .pt), so we can't parse from a head
            // prefix the way we do for safetensors/GGUF. Supporting this
            // would need a tail-first range fetch plus an in-memory zip
            // parser; for v1 we surface a clear error and let the caller
            // fall back to "treat as plain bytes".
            anyhow::bail!("pickle: remote header fetch not yet supported — download the file first")
        }
    }
}

// === extracted from arbvis::data (prepare_diff_sources_from_http,strip_prefix_components,find_strip_depths,TensorMatch,match_under_strip_depths,find_matched_tensor_pairs,fetch_rms_estimates,build_multi_safetensors_diff_sources_inner,build_multi_safetensors_diff_sources,build_multi_safetensors_diff_sources_from_http,build_safetensors_diff_sources,SourceMeta,try_load_source_meta,load_meta_for_sources,fetch_hf_sidecar) ===
/// Build diff sources from two repos listed as HTTP specs (no download).
///
/// Safetensors files are diffed lazily via range requests — no model weights
/// are downloaded to disk or held in RAM. Small non-safetensors files (≤16 MB)
/// are downloaded eagerly and binary-diffed; larger ones are skipped with a warning.
pub async fn prepare_diff_sources_from_http(
    orig_specs: &[(String, RemoteFileSpec)],
    mod_specs: &[(String, RemoteFileSpec)],
    is_finetune: bool,
    metric: format::DiffMetric,
    stream: bool,
) -> anyhow::Result<(Vec<Source>, u64)> {
    let is_st = |name: &str| name.ends_with(".safetensors");

    let orig_st: Vec<&(String, RemoteFileSpec)> =
        orig_specs.iter().filter(|(n, _)| is_st(n)).collect();
    let mod_st: Vec<&(String, RemoteFileSpec)> =
        mod_specs.iter().filter(|(n, _)| is_st(n)).collect();

    let mut sources: Vec<Source> = Vec::new();
    let mut total = 0u64;

    // Safetensors diff — disk-backed by default; pass `stream=true` to keep
    // shards remote and diff each tile via HTTP range requests instead.
    if !orig_st.is_empty() || !mod_st.is_empty() {
        match build_multi_safetensors_diff_sources_from_http(
            &orig_st,
            &mod_st,
            is_finetune,
            metric,
            stream,
        )
        .await
        {
            Ok((mut tensor_sources, bytes)) => {
                let base_idx = sources.len();
                for s in &mut tensor_sources {
                    s.file_idx += base_idx;
                }
                sources.extend(tensor_sources);
                total += bytes;
            }
            Err(e) => log::warn!("safetensors diff failed: {e} — skipping"),
        }
    }

    // Non-safetensors files: match by filename. Same-size pairs become a byte
    // diff (downloaded if small); different-size or single-side files become
    // crosshatched unmatched regions so they remain visible. Large files are
    // sized but rendered as the orig-fill kind (we can't byte-diff something
    // we won't download).
    const MAX_EAGER_SIZE: u64 = 16 * 1024 * 1024;
    let orig_fill_kind = if is_finetune {
        DiffFill::Grey
    } else {
        DiffFill::Red
    };
    let orig_non: HashMap<&str, &RemoteFileSpec> = orig_specs
        .iter()
        .filter(|(n, _)| !is_st(n))
        .map(|(n, s)| (n.as_str(), s))
        .collect();
    let mod_non: HashMap<&str, &RemoteFileSpec> = mod_specs
        .iter()
        .filter(|(n, _)| !is_st(n))
        .map(|(n, s)| (n.as_str(), s))
        .collect();

    let mut mod_only_files: Vec<&str> = mod_non
        .keys()
        .copied()
        .filter(|k| !orig_non.contains_key(k))
        .collect();
    mod_only_files.sort();
    if is_finetune && !mod_only_files.is_empty() {
        log::warn!(
            "--diff --finetune: modified side has {} file(s) with no counterpart on the \
             original/base side — rendering as green crosshatch: {}",
            mod_only_files.len(),
            mod_only_files.join(", ")
        );
    }

    let mut sorted: Vec<&str> = orig_non.keys().copied().collect();
    sorted.sort();

    // First pass (sync): partition into byte-diff jobs vs unmatched-region
    // sources. Diff jobs are downloaded in parallel afterwards.
    let mut diff_jobs: Vec<(String, RemoteFileSpec, RemoteFileSpec)> = Vec::new();
    let mut unmatched_orig: Vec<(String, u64, DiffFill)> = Vec::new();
    let mut unmatched_mod: Vec<(String, u64, DiffFill)> = Vec::new();
    // Files present in both repos but too large to eagerly byte-diff. Rendered
    // as a single crosshatch source per file (not split across both sides).
    let mut unmatched_modified: Vec<(String, u64, DiffFill)> = Vec::new();
    for fname in sorted {
        let orig_spec = &orig_non[fname];
        let mod_spec = match mod_non.get(fname) {
            Some(s) => s,
            None => {
                if orig_spec.size > 0 {
                    unmatched_orig.push((fname.to_string(), orig_spec.size, orig_fill_kind));
                }
                continue;
            }
        };
        if orig_spec.size != mod_spec.size {
            if is_finetune {
                log::warn!(
                    "--diff --finetune: size mismatch for {fname} ({} vs {} bytes) — \
                     byte-diffing with zero-padding on the shorter side",
                    orig_spec.size,
                    mod_spec.size
                );
            } else {
                log::warn!(
                    "size mismatch for {fname} ({} vs {} bytes) — byte-diffing with zero-padding",
                    orig_spec.size,
                    mod_spec.size
                );
            }
        }
        let max_size = orig_spec.size.max(mod_spec.size);
        if max_size == 0 {
            continue;
        }
        if max_size > MAX_EAGER_SIZE {
            // Too large to byte-diff eagerly; surface as a single crosshatched
            // region sized to max(orig, mod) so the file appears once in the
            // canvas, regardless of whether sizes match.
            log::warn!(
                "{fname} exceeds {} MB — rendering as crosshatched region instead of byte diff",
                MAX_EAGER_SIZE / 1024 / 1024
            );
            unmatched_modified.push((fname.to_string(), max_size, orig_fill_kind));
            continue;
        }
        diff_jobs.push((fname.to_string(), (*orig_spec).clone(), (*mod_spec).clone()));
    }
    for fname in &mod_only_files {
        let spec = &mod_non[fname];
        if spec.size > 0 {
            unmatched_mod.push((fname.to_string(), spec.size, DiffFill::Green));
        }
    }

    let pb = setup_progress(
        "file pairs (non-safetensors diff downloads)",
        diff_jobs.len() as u64,
    );
    let pb_for_workers = pb.clone();
    let diffs: Vec<anyhow::Result<(String, Vec<u8>)>> = stream::iter(diff_jobs)
        .map(|(fname, orig_spec, mod_spec)| {
            let pb = pb_for_workers.clone();
            async move {
                let orig_data = Data::Http {
                    repo: orig_spec.repo.clone(),
                    filename: Arc::clone(&orig_spec.filename),
                    revision: Arc::clone(&orig_spec.revision),
                };
                let mod_data = Data::Http {
                    repo: mod_spec.repo.clone(),
                    filename: Arc::clone(&mod_spec.filename),
                    revision: Arc::clone(&mod_spec.revision),
                };
                let ob = orig_data.fetch_range(0, orig_spec.size as usize).await?;
                let mb = mod_data.fetch_range(0, mod_spec.size as usize).await?;
                // Pad the shorter side with zeros so size-mismatched but
                // same-named files share one diff source. The longer side's
                // tail diffs against zero, which renders as deltas indicating
                // bytes that exist on only one side.
                let len = ob.len().max(mb.len());
                let diff: Vec<u8> = (0..len)
                    .map(|i| {
                        let a = ob.get(i).copied().unwrap_or(0);
                        let b = mb.get(i).copied().unwrap_or(0);
                        let delta = b as i16 - a as i16;
                        let brightness =
                            (delta.unsigned_abs() as f32 / 255.0 * 127.0).round() as u8;
                        if delta >= 0 {
                            127u8 + brightness
                        } else {
                            127u8 - brightness
                        }
                    })
                    .collect();
                if let Some(pb) = pb.as_ref() {
                    pb.inc(1);
                }
                Ok((fname, diff))
            }
        })
        .buffer_unordered(SETUP_FETCH_CONCURRENCY)
        .collect()
        .await;
    if let Some(pb) = pb.as_ref() {
        pb.finish_and_clear();
    }
    // Re-sort by filename so the Source order is deterministic.
    let mut diffs: Vec<(String, Vec<u8>)> =
        diffs.into_iter().collect::<anyhow::Result<Vec<_>>>()?;
    diffs.sort_by(|a, b| a.0.cmp(&b.0));
    for (fname, diff) in diffs {
        let size = diff.len() as u64;
        sources.push(Source {
            file_idx: sources.len(),
            kind: SourceKind::Buffered(diff),
            byte_size: size,
            name_override: Some(fname),
            xet_terms: None,
            extensions: Extensions::default(),
        });
        total += size;
    }

    // Unmatched / oversize / size-mismatch files surface as crosshatched
    // regions so the user sees they exist even though no byte diff was
    // computed.
    unmatched_orig.sort();
    for (fname, size, fill) in unmatched_orig {
        sources.push(Source {
            file_idx: sources.len(),
            kind: SourceKind::UnmatchedRegion { fill },
            byte_size: size,
            name_override: Some(format!("[only in original] {fname}")),
            xet_terms: None,
            extensions: Extensions::default(),
        });
        total += size;
    }
    unmatched_mod.sort();
    for (fname, size, fill) in unmatched_mod {
        sources.push(Source {
            file_idx: sources.len(),
            kind: SourceKind::UnmatchedRegion { fill },
            byte_size: size,
            name_override: Some(format!("[only in modified] {fname}")),
            xet_terms: None,
            extensions: Extensions::default(),
        });
        total += size;
    }
    unmatched_modified.sort();
    for (fname, size, fill) in unmatched_modified {
        sources.push(Source {
            file_idx: sources.len(),
            kind: SourceKind::UnmatchedRegion { fill },
            byte_size: size,
            name_override: Some(fname),
            xet_terms: None,
            extensions: Extensions::default(),
        });
        total += size;
    }

    if sources.is_empty() {
        anyhow::bail!("--diff: no matching file pairs found between the two repos");
    }
    Ok((sources, total))
}

/// Strip the first `n` dot-delimited path components from `name`.
/// Returns the remainder after the n-th dot, or `None` if `name` has fewer than n+1 components.
fn strip_prefix_components(name: &str, n: usize) -> Option<&str> {
    let mut idx = 0;
    for _ in 0..n {
        idx += name[idx..].find('.')? + 1;
    }
    Some(&name[idx..])
}

/// Find (strip_orig, strip_mod) prefix depths that maximise unique 1-to-1 tensor name matches.
/// Returns (0, 0) if exact matching already produces matches.
fn find_strip_depths(orig_names: &[String], mod_names: &[String]) -> (usize, usize) {
    // Count unique-suffix occurrences for a set of names stripped by `n` components.
    let unique_suffixes = |names: &[String], n: usize| -> HashMap<String, usize> {
        let mut counts: HashMap<String, usize> = HashMap::new();
        for name in names {
            if let Some(s) = strip_prefix_components(name, n) {
                if !s.is_empty() {
                    *counts.entry(s.to_owned()).or_insert(0) += 1;
                }
            }
        }
        counts
    };

    let mut best = (0usize, 0usize, 0usize); // (strip_orig, strip_mod, match_count)
    for so in 0usize..=5 {
        let orig_counts = unique_suffixes(orig_names, so);
        for sm in 0usize..=5 {
            if so == 0 && sm == 0 {
                continue;
            }
            let mod_counts = unique_suffixes(mod_names, sm);
            let matches = orig_counts
                .iter()
                .filter(|(s, &oc)| oc == 1 && mod_counts.get(s.as_str()) == Some(&1))
                .count();
            if matches > best.2 {
                best = (so, sm, matches);
            }
        }
    }
    (best.0, best.1)
}

/// Result of tensor-name matching across the two sides of a diff.
pub struct TensorMatch {
    /// 1-to-1 matched pairs `(orig_full, mod_full)`, sorted by `orig_full`.
    pub pairs: Vec<(String, String)>,
    /// Tensor full names present only on the original side, sorted.
    pub orig_only: Vec<String>,
    /// Tensor full names present only on the modified side, sorted.
    pub mod_only: Vec<String>,
}

/// Match tensors under a fixed `(strip_o, strip_m)` strip pair: build the
/// stripped-suffix maps (with collisions blanked), then pair up unique 1-to-1
/// matches. Returns the matched `(orig_full, mod_full)` pairs only — caller
/// drives iteration and tracks the unmatched residual.
fn match_under_strip_depths(
    orig_names: &[String],
    mod_names: &[String],
    strip_o: usize,
    strip_m: usize,
) -> Vec<(String, String)> {
    let orig_by_stripped: HashMap<String, &str> = orig_names
        .iter()
        .filter_map(|n| {
            strip_prefix_components(n, strip_o)
                .filter(|s| !s.is_empty())
                .map(|s| (s.to_owned(), n.as_str()))
        })
        .fold(HashMap::new(), |mut acc, (stripped, full)| {
            acc.entry(stripped).and_modify(|v| *v = "").or_insert(full);
            acc
        });

    let mod_by_stripped: HashMap<String, &str> = mod_names
        .iter()
        .filter_map(|n| {
            strip_prefix_components(n, strip_m)
                .filter(|s| !s.is_empty())
                .map(|s| (s.to_owned(), n.as_str()))
        })
        .fold(HashMap::new(), |mut acc, (stripped, full)| {
            acc.entry(stripped).and_modify(|v| *v = "").or_insert(full);
            acc
        });

    let mut sorted_orig: Vec<&str> = orig_by_stripped
        .values()
        .copied()
        .filter(|s| !s.is_empty())
        .collect();
    sorted_orig.sort();

    let mut pairs = Vec::new();
    for orig_full in sorted_orig {
        let stripped = match strip_prefix_components(orig_full, strip_o) {
            Some(s) if !s.is_empty() => s,
            _ => continue,
        };
        if let Some(&mod_full) = mod_by_stripped.get(stripped) {
            if !mod_full.is_empty() {
                pairs.push((orig_full.to_owned(), mod_full.to_owned()));
            }
        }
    }
    pairs
}

/// Find matched + unmatched tensor name groupings between two name sets.
///
/// **Multi-pass strip heuristic.** Real-world model files frequently mix
/// multiple wrapper-induced prefix nestings — e.g. GRaPE-2-Nano's language
/// tensors live under `model.language_model.language_model.language_model.*`
/// (matching the base's `model.language_model.*` at strip depths `(1, 3)`)
/// while its vision tensors live under `model.language_model.visual.*`
/// (matching the base's `model.visual.*` at strip depths `(2, 3)`). A single
/// `(strip_o, strip_m)` pair can't capture both — so we iterate:
///
/// 1. Pull out all exact-name matches first.
/// 2. Greedily pick the best `(strip_o, strip_m)` over the remaining
///    unmatched tensors, apply those matches, repeat.
/// 3. Stop when no further pair yields any matches.
///
/// Unmatched tensors are returned so callers can surface them (e.g. as
/// crosshatch fill) rather than silently dropping them.
fn find_matched_tensor_pairs(orig_names: &[String], mod_names: &[String]) -> TensorMatch {
    use std::collections::HashSet;
    let mut remaining_orig: HashSet<String> = orig_names.iter().cloned().collect();
    let mut remaining_mod: HashSet<String> = mod_names.iter().cloned().collect();

    let mut pairs: Vec<(String, String)> = Vec::new();

    // Pass 0: exact-name overlap. Iterate the input order so the resulting
    // log line is deterministic on repeat runs.
    for name in orig_names {
        if remaining_orig.contains(name) && remaining_mod.contains(name) {
            remaining_orig.remove(name);
            remaining_mod.remove(name);
            pairs.push((name.clone(), name.clone()));
        }
    }
    if !pairs.is_empty() {
        log::debug!(
            "safetensors diff: pass 0 exact match — {} pairs",
            pairs.len()
        );
    }

    // Subsequent passes: greedy multi-pass strip search. Each pass picks the
    // (strip_o, strip_m) that yields the most matches over the *remaining*
    // unmatched tensors, applies those matches, and continues. Bounded by
    // the strip search range (find_strip_depths) and by termination once no
    // pair yields any matches.
    let mut pass = 0usize;
    loop {
        if remaining_orig.is_empty() || remaining_mod.is_empty() {
            break;
        }
        let orig_vec: Vec<String> = remaining_orig.iter().cloned().collect();
        let mod_vec: Vec<String> = remaining_mod.iter().cloned().collect();
        let (strip_o, strip_m) = find_strip_depths(&orig_vec, &mod_vec);
        if strip_o == 0 && strip_m == 0 {
            break;
        }
        let new_pairs = match_under_strip_depths(&orig_vec, &mod_vec, strip_o, strip_m);
        if new_pairs.is_empty() {
            break;
        }
        pass += 1;
        log::info!(
            "safetensors diff: strip-match pass {}: stripping {} component(s) from original and {} from modified — {} new pair(s)",
            pass, strip_o, strip_m, new_pairs.len()
        );
        for (o, m) in &new_pairs {
            remaining_orig.remove(o);
            remaining_mod.remove(m);
        }
        pairs.extend(new_pairs);
    }

    let mut orig_only: Vec<String> = remaining_orig.into_iter().collect();
    let mut mod_only: Vec<String> = remaining_mod.into_iter().collect();
    orig_only.sort();
    mod_only.sort();
    pairs.sort_by(|a, b| a.0.cmp(&b.0));

    TensorMatch {
        pairs,
        orig_only,
        mod_only,
    }
}

/// Sample each matched orig tensor's bytes and compute its RMS, used as the
/// per-tensor scale for `DiffMetric::Rms`. A 64 KB contiguous prefix is more
/// than enough for a stable estimate; for HTTP/Xet sources this is one extra
/// range fetch per matched tensor at setup, parallelised over the throttle.
async fn fetch_rms_estimates(
    paired_ok: &[(String, String)],
    orig_map: &HashMap<String, (usize, format::TensorMeta)>,
    orig_data: &[Arc<Data>],
) -> anyhow::Result<Vec<f32>> {
    const SCALE_SAMPLE_BYTES: u64 = 64 * 1024;
    let pb = setup_progress("orig tensor RMS samples", paired_ok.len() as u64);
    let inputs: Vec<(usize, usize, u64, u64, format::Dtype)> = paired_ok
        .iter()
        .enumerate()
        .map(|(idx, (orig_full, _))| {
            let (oi, orig_t) = &orig_map[orig_full];
            let elem = orig_t.dtype.element_size() as u64;
            let tensor_bytes = orig_t.file_end.saturating_sub(orig_t.file_start);
            let want = SCALE_SAMPLE_BYTES.min(tensor_bytes);
            // Align sample length down to a whole element so rms_from_buf
            // only sees complete values.
            let len = want.checked_div(elem).map(|q| q * elem).unwrap_or(0);
            (idx, *oi, orig_t.file_start, len, orig_t.dtype)
        })
        .collect();
    let mut out: Vec<(usize, f32)> = stream::iter(inputs)
        .map(|(idx, oi, start, len, dtype)| {
            let d = Arc::clone(&orig_data[oi]);
            let pb = pb.clone();
            async move {
                let scale = if len == 0 {
                    0.0
                } else {
                    match d.fetch_range(start, len as usize).await {
                        Ok(bytes) => format::rms_from_buf(dtype, &bytes),
                        Err(e) => {
                            log::warn!("safetensors diff: orig RMS sample failed ({e}); using 0.0");
                            0.0
                        }
                    }
                };
                if let Some(pb) = pb.as_ref() {
                    pb.inc(1);
                }
                (idx, scale)
            }
        })
        .buffer_unordered(RMS_SAMPLE_FETCH_CONCURRENCY)
        .collect()
        .await;
    if let Some(pb) = pb.as_ref() {
        pb.finish_and_clear();
    }
    out.sort_by_key(|(i, _)| *i);
    Ok(out.into_iter().map(|(_, s)| s).collect())
}

/// Core tensor diff builder: given two parallel lists of whole-file Data sources,
/// build per-tensor TensorDiff Source entries without reading any tensor bytes.
async fn build_multi_safetensors_diff_sources_inner(
    orig_data: &[Arc<Data>],
    orig_fmts: &[SourceFormat],
    mod_data: &[Arc<Data>],
    mod_fmts: &[SourceFormat],
    is_finetune: bool,
    metric: format::DiffMetric,
) -> anyhow::Result<(Vec<Source>, u64)> {
    // Two HTTP range requests per file (8-byte preamble + variable header
    // for safetensors; one or two prefix fetches for GGUF). For sharded
    // models this is dozens of files per side; serializing them wastes the
    // throttle.
    let total = (orig_data.len() + mod_data.len()) as u64;
    let pb = setup_progress("source files (model headers)", total);

    async fn fetch_all(
        data: &[Arc<Data>],
        fmts: &[SourceFormat],
        side: &'static str,
        pb: &Option<ProgressBar>,
    ) -> anyhow::Result<Vec<(usize, Vec<format::TensorMeta>)>> {
        let pb = pb.clone();
        // Collect owned `(Arc<Data>, SourceFormat)` pairs up front so the
        // per-item closure below doesn't borrow from `data`/`fmts` — that
        // borrow's lifetime is too narrow to satisfy the higher-ranked
        // bound required when this helper is reached from
        // `DiffSourceBuilder::try_build`'s async-trait future.
        let items: Vec<(Arc<Data>, SourceFormat)> =
            data.iter().cloned().zip(fmts.iter().copied()).collect();
        let mut out: Vec<(usize, anyhow::Result<Vec<format::TensorMeta>>)> =
            stream::iter(items.into_iter().enumerate())
                .map(|(i, (d, fmt))| {
                    let pb = pb.clone();
                    async move {
                        let r = fetch_model_header(&d, fmt)
                            .await
                            .map(|(t, _)| t)
                            .with_context(|| format!("reading {fmt:?} header for {side} file {i}"));
                        if let Some(pb) = pb.as_ref() {
                            pb.inc(1);
                        }
                        (i, r)
                    }
                })
                .buffer_unordered(SETUP_FETCH_CONCURRENCY)
                .collect()
                .await;
        out.sort_by_key(|(i, _)| *i);
        out.into_iter().map(|(i, r)| r.map(|t| (i, t))).collect()
    }

    let orig_headers = fetch_all(orig_data, orig_fmts, "orig", &pb).await?;
    let mod_headers = fetch_all(mod_data, mod_fmts, "mod", &pb).await?;
    if let Some(pb) = pb.as_ref() {
        pb.finish_and_clear();
    }

    // Use the first source's format on each side for name canonicalisation.
    // In practice all files on one side of a diff share a format; the
    // canonical name table is identity for safetensors and translates GGUF
    // tensor names to their HF equivalents.
    let orig_canon_fmt = orig_fmts
        .first()
        .copied()
        .unwrap_or(SourceFormat::Safetensors);
    let mod_canon_fmt = mod_fmts
        .first()
        .copied()
        .unwrap_or(SourceFormat::Safetensors);

    let mut orig_map: HashMap<String, (usize, format::TensorMeta)> = HashMap::new();
    for (i, tensors) in orig_headers {
        for t in tensors {
            orig_map.entry(t.name.clone()).or_insert((i, t));
        }
    }
    let mut mod_map: HashMap<String, (usize, format::TensorMeta)> = HashMap::new();
    for (i, tensors) in mod_headers {
        for t in tensors {
            mod_map.entry(t.name.clone()).or_insert((i, t));
        }
    }

    // Canonical name table: maps raw → canonical for each side.
    let orig_canon: HashMap<String, String> = orig_map
        .keys()
        .map(|n| (n.clone(), orig_canon_fmt.canonical_name(n)))
        .collect();
    let mod_canon: HashMap<String, String> = mod_map
        .keys()
        .map(|n| (n.clone(), mod_canon_fmt.canonical_name(n)))
        .collect();
    // Reverse maps: canonical → raw, so after matching on canonical names we
    // can look the original name back up.
    let orig_by_canon: HashMap<String, String> = orig_canon
        .iter()
        .map(|(raw, can)| (can.clone(), raw.clone()))
        .collect();
    let mod_by_canon: HashMap<String, String> = mod_canon
        .iter()
        .map(|(raw, can)| (can.clone(), raw.clone()))
        .collect();
    let orig_canon_names: Vec<String> = orig_by_canon.keys().cloned().collect();
    let mod_canon_names: Vec<String> = mod_by_canon.keys().cloned().collect();
    let TensorMatch {
        pairs: canon_pairs,
        orig_only: canon_orig_only,
        mod_only: canon_mod_only,
    } = find_matched_tensor_pairs(&orig_canon_names, &mod_canon_names);
    // Translate back to raw names for the rest of the pipeline.
    let pairs: Vec<(String, String)> = canon_pairs
        .into_iter()
        .filter_map(|(co, cm)| {
            let o = orig_by_canon.get(&co)?.clone();
            let m = mod_by_canon.get(&cm)?.clone();
            Some((o, m))
        })
        .collect();
    let orig_only: Vec<String> = canon_orig_only
        .into_iter()
        .filter_map(|c| orig_by_canon.get(&c).cloned())
        .collect();
    let mod_only: Vec<String> = canon_mod_only
        .into_iter()
        .filter_map(|c| mod_by_canon.get(&c).cloned())
        .collect();
    let _ = (&orig_canon, &mod_canon); // intentional: keep maps alive above

    // Tensors present in both, but with incompatible shapes, can't be diffed
    // element-wise. Treat each side independently: in non-finetune mode both
    // sides surface as unmatched (red on orig, green on mod). In finetune
    // mode the modified side is an error (see below), so we fail fast.
    let mut shape_mismatch: Vec<(String, String)> = Vec::new();
    let mut paired_ok: Vec<(String, String)> = Vec::with_capacity(pairs.len());
    for (orig_full, mod_full) in pairs {
        let orig_t = &orig_map[&orig_full].1;
        let mod_t = &mod_map[&mod_full].1;
        if orig_t.shape != mod_t.shape {
            shape_mismatch.push((orig_full, mod_full));
        } else {
            paired_ok.push((orig_full, mod_full));
        }
    }

    // Finetune contract: every tensor the finetune ships should exist (with
    // the same shape) on the base side. Real-world models sometimes break
    // this — e.g. wrapper-saved finetunes that nest a vision tower under a
    // language_model prefix the base doesn't share — so we warn rather than
    // bail, and surface the offending tensors as green crosshatch via the
    // normal only-in-modified rendering below.
    if is_finetune {
        let mut mod_extras: Vec<String> = mod_only.clone();
        for (_, mod_full) in &shape_mismatch {
            mod_extras.push(mod_full.clone());
        }
        if !mod_extras.is_empty() {
            mod_extras.sort();
            log::warn!(
                "safetensors diff --finetune: modified side has {} tensor(s) not present \
                 (or with mismatched shape) on the original/base side — rendering as green \
                 crosshatch: {}",
                mod_extras.len(),
                mod_extras.join(", ")
            );
        }
    }

    // For DiffMetric::Rms we need a per-tensor scale (RMS of orig). Sample
    // up to RMS_SAMPLE_ELEMS elements per tensor via a single range fetch;
    // for HTTP sources this is one extra request per tensor at setup time,
    // for local mmap it's free. AbsLog and Exact don't need a scale.
    let scales: Vec<f32> = if matches!(metric, format::DiffMetric::Rms) {
        fetch_rms_estimates(&paired_ok, &orig_map, orig_data).await?
    } else {
        vec![0.0; paired_ok.len()]
    };

    let mut sources: Vec<Source> = Vec::new();
    let mut total = 0u64;

    for ((orig_full, mod_full), scale_orig) in paired_ok.iter().zip(scales.iter()) {
        let (oi, orig_t) = &orig_map[orig_full];
        let (mi, mod_t) = &mod_map[mod_full];

        let nelem: u64 = orig_t.shape.iter().product();
        // Describe the diff buffer as a synthetic 1-byte-per-element tensor so
        // the architectural layout can place it at its natural 2D shape. The
        // dtype is U8 (the output of `Dtype::diff_to_u8`); the *element shape*
        // tracks the original tensor's so layer-N q_proj stacks pixel-aligned
        // with layer-N+1 q_proj.
        let diff_meta = format::TensorMeta {
            name: orig_t.name.clone(),
            dtype: format::Dtype::U8,
            shape: orig_t.shape.clone(),
            file_start: 0,
            file_end: nelem,
            packed_sidecars: None,
        };
        let mut extensions = Extensions::default();
        extensions.insert(ModelInfo {
            format: SourceFormat::Safetensors,
            tensors: vec![diff_meta],
            color_ranges: Vec::new(),
        });
        sources.push(Source {
            file_idx: sources.len(),
            kind: SourceKind::Custom(Box::new(TensorDiffSource {
                orig: Arc::clone(&orig_data[*oi]),
                mod_: Arc::clone(&mod_data[*mi]),
                orig_start: orig_t.file_start,
                mod_start: mod_t.file_start,
                orig_dtype: orig_t.dtype,
                mod_dtype: mod_t.dtype,
                metric,
                scale_orig: *scale_orig,
                byte_size: nelem,
            })),
            byte_size: nelem,
            name_override: Some(orig_t.label()),
            xet_terms: None,
            extensions,
        });
        total += nelem;
    }

    // Unmatched / shape-mismatched tensors become crosshatched canvas regions.
    // In finetune mode only orig-only entries can survive (mod-side errors
    // were already raised above), and they render as informational grey.
    let orig_fill = if is_finetune {
        DiffFill::Grey
    } else {
        DiffFill::Red
    };

    let mut orig_unmatched: Vec<&format::TensorMeta> = Vec::new();
    for name in &orig_only {
        orig_unmatched.push(&orig_map[name].1);
    }
    for (orig_full, _) in &shape_mismatch {
        orig_unmatched.push(&orig_map[orig_full].1);
    }
    for t in orig_unmatched {
        let nelem: u64 = t.shape.iter().product();
        if nelem == 0 {
            continue;
        }
        // Carry the original shape so the gate in `select_layout` treats this
        // as a safetensors-aware source. The arch layout currently *skips*
        // UnmatchedRegion entries (they're drawn via the crosshatch overlay in
        // the Hilbert path); attaching the synthetic meta keeps future arch
        // crosshatch wiring straightforward.
        let unmatched_meta = format::TensorMeta {
            name: t.name.clone(),
            dtype: format::Dtype::U8,
            shape: t.shape.clone(),
            file_start: 0,
            file_end: nelem,
            packed_sidecars: None,
        };
        let mut extensions = Extensions::default();
        extensions.insert(ModelInfo {
            format: SourceFormat::Safetensors,
            tensors: vec![unmatched_meta],
            color_ranges: Vec::new(),
        });
        sources.push(Source {
            file_idx: sources.len(),
            kind: SourceKind::UnmatchedRegion { fill: orig_fill },
            byte_size: nelem,
            name_override: Some(format!("[only in original] {}", t.label())),
            xet_terms: None,
            extensions,
        });
        total += nelem;
    }

    // mod-only tensors render as green crosshatch in both modes. In finetune
    // mode the warning above already flagged the contract violation; the
    // green crosshatch surfaces it visually too.
    for name in &mod_only {
        let t = &mod_map[name].1;
        let nelem: u64 = t.shape.iter().product();
        if nelem == 0 {
            continue;
        }
        let unmatched_meta = format::TensorMeta {
            name: t.name.clone(),
            dtype: format::Dtype::U8,
            shape: t.shape.clone(),
            file_start: 0,
            file_end: nelem,
            packed_sidecars: None,
        };
        let mut extensions = Extensions::default();
        extensions.insert(ModelInfo {
            format: SourceFormat::Safetensors,
            tensors: vec![unmatched_meta],
            color_ranges: Vec::new(),
        });
        sources.push(Source {
            file_idx: sources.len(),
            kind: SourceKind::UnmatchedRegion {
                fill: DiffFill::Green,
            },
            byte_size: nelem,
            name_override: Some(format!("[only in modified] {}", t.label())),
            xet_terms: None,
            extensions,
        });
        total += nelem;
    }

    if !orig_only.is_empty() || !mod_only.is_empty() || !shape_mismatch.is_empty() {
        log::info!(
            "safetensors diff: {} matched, {} only in original, {} only in modified, {} shape-mismatch",
            sources
                .iter()
                .filter(|s| matches!(&s.kind, SourceKind::Custom(c) if c.id() == "tensor-diff"))
                .count(),
            orig_only.len(),
            mod_only.len(),
            shape_mismatch.len()
        );
    }

    Ok((sources, total))
}

/// Build per-tensor diff Sources from multiple local model files on each
/// side. Each path's extension determines its format (`.safetensors` or
/// `.gguf`); mixed-format pairs are routed through the cross-format name
/// canonicaliser.
pub async fn build_multi_safetensors_diff_sources(
    orig_files: &[PathBuf],
    mod_files: &[PathBuf],
    is_finetune: bool,
    metric: format::DiffMetric,
) -> anyhow::Result<(Vec<Source>, u64)> {
    let open_arcs = |files: &[PathBuf]| -> anyhow::Result<(Vec<Arc<Data>>, Vec<SourceFormat>)> {
        let mut datas = Vec::with_capacity(files.len());
        let mut fmts = Vec::with_capacity(files.len());
        for p in files {
            let f = File::open(p).with_context(|| format!("opening {}", p.display()))?;
            datas.push(Arc::new(Data::Mapped(unsafe { Mmap::map(&f) }?)));
            fmts.push(SourceFormat::from_path(p).unwrap_or(SourceFormat::Safetensors));
        }
        Ok((datas, fmts))
    };
    let (orig_data, orig_fmts) = open_arcs(orig_files)?;
    let (mod_data, mod_fmts) = open_arcs(mod_files)?;
    build_multi_safetensors_diff_sources_inner(
        &orig_data,
        &orig_fmts,
        &mod_data,
        &mod_fmts,
        is_finetune,
        metric,
    )
    .await
}

/// Build per-tensor diff Sources from multiple remote .safetensors files on each side.
///
/// When `stream` is false (the default) every shard is downloaded to the
/// local HF cache (via the `hf` CLI) and mmapped before any source is constructed, so the
/// per-tile diff path is pure memcpy. When `stream` is true each xet-backed
/// file gets a `Data::Xet` reader (one V2 reconstruction fetch per file,
/// then direct-CAS range fetches afterward — see `XetReader`) and non-xet
/// remote files fall back to `Data::Http`; every per-tile diff then issues
/// an HTTP range request.
async fn build_multi_safetensors_diff_sources_from_http(
    orig_specs: &[&(String, RemoteFileSpec)],
    mod_specs: &[&(String, RemoteFileSpec)],
    is_finetune: bool,
    metric: format::DiffMetric,
    stream: bool,
) -> anyhow::Result<(Vec<Source>, u64)> {
    let orig_fmts: Vec<SourceFormat> = orig_specs
        .iter()
        .map(|(n, _)| SourceFormat::from_name(n).unwrap_or(SourceFormat::Safetensors))
        .collect();
    let mod_fmts: Vec<SourceFormat> = mod_specs
        .iter()
        .map(|(n, _)| SourceFormat::from_name(n).unwrap_or(SourceFormat::Safetensors))
        .collect();
    // Streaming arc construction: per-spec xet reconstruction + Http fallback.
    // Only used when `stream` is true; the disk-backed branch skips it and
    // goes straight through `materialize_remote_arcs`.
    async fn make_streaming_arcs(specs: Vec<RemoteFileSpec>) -> anyhow::Result<Vec<Arc<Data>>> {
        let total = specs.len() as u64;
        let pb = setup_progress("source files (xet reconstruction for diff)", total);
        let pb_for_workers = pb.clone();
        let mut out: Vec<(usize, anyhow::Result<Arc<Data>>)> =
            stream::iter(specs.into_iter().enumerate())
                .map(|(i, spec)| {
                    let pb = pb_for_workers.clone();
                    async move {
                        let r: anyhow::Result<Arc<Data>> = if spec.xet_hash.is_some() {
                            match XetReader::new(&spec).await {
                                Ok(reader) => Ok(Arc::new(Data::Xet(reader))),
                                Err(e) => {
                                    log::warn!(
                                    "{}: XetReader build failed ({e}); falling back to Data::Http",
                                    spec.filename,
                                );
                                    Ok(Arc::new(Data::Http {
                                        repo: spec.repo.clone(),
                                        filename: Arc::clone(&spec.filename),
                                        revision: Arc::clone(&spec.revision),
                                    }))
                                }
                            }
                        } else {
                            Ok(Arc::new(Data::Http {
                                repo: spec.repo.clone(),
                                filename: Arc::clone(&spec.filename),
                                revision: Arc::clone(&spec.revision),
                            }))
                        };
                        if let Some(pb) = pb.as_ref() {
                            pb.inc(1);
                        }
                        (i, r)
                    }
                })
                .buffer_unordered(SETUP_FETCH_CONCURRENCY)
                .collect()
                .await;
        if let Some(pb) = pb.as_ref() {
            pb.finish_and_clear();
        }
        out.sort_by_key(|(i, _)| *i);
        out.into_iter()
            .map(|(_, r)| r)
            .collect::<anyhow::Result<Vec<_>>>()
    }
    let orig_specs_owned: Vec<RemoteFileSpec> = orig_specs.iter().map(|(_, s)| s.clone()).collect();
    let mod_specs_owned: Vec<RemoteFileSpec> = mod_specs.iter().map(|(_, s)| s.clone()).collect();
    let (mut orig_data, mut mod_data) = if stream {
        let orig_data = make_streaming_arcs(orig_specs_owned.clone()).await?;
        let mod_data = make_streaming_arcs(mod_specs_owned.clone()).await?;
        (orig_data, mod_data)
    } else {
        // Placeholder arcs swapped in place by materialize_remote_arcs.
        let placeholder = |specs: &[RemoteFileSpec]| -> Vec<Arc<Data>> {
            specs
                .iter()
                .map(|spec| {
                    Arc::new(Data::Http {
                        repo: spec.repo.clone(),
                        filename: Arc::clone(&spec.filename),
                        revision: Arc::clone(&spec.revision),
                    })
                })
                .collect()
        };
        (
            placeholder(&orig_specs_owned),
            placeholder(&mod_specs_owned),
        )
    };
    if !stream {
        materialize_remote_arcs(
            &mut orig_data,
            &orig_specs_owned,
            "source files (downloading for diff: orig side)",
        )
        .await?;
        materialize_remote_arcs(
            &mut mod_data,
            &mod_specs_owned,
            "source files (downloading for diff: modified side)",
        )
        .await?;
    }
    build_multi_safetensors_diff_sources_inner(
        &orig_data,
        &orig_fmts,
        &mod_data,
        &mod_fmts,
        is_finetune,
        metric,
    )
    .await
}

/// `(layer, weight, expert)` key for one per-expert summary scalar.
type ScalarKey = (u32, crate::format::moe::ExpertWeight, u32);
/// A unit of scalar work: which bytes to read and how to interpret them —
/// `(key, shard_idx, file_start, byte_len, dtype)`. Produced both by the
/// per-expert tensor scan and by [`build_fused_expert_jobs`].
type ScalarJob = (ScalarKey, usize, u64, u64, format::Dtype);

/// Whether `dtype` stores elements in blocks / bit-packed groups rather than
/// a fixed number of contiguous bytes per element. Such tensors can't be
/// sliced on arbitrary element boundaries, so the fused per-expert slicer
/// skips them.
fn is_block_or_packed_dtype(dtype: format::Dtype) -> bool {
    use format::Dtype::*;
    matches!(
        dtype,
        Q4_0 | Q4_1
            | Q5_0
            | Q5_1
            | Q8_0
            | Q8_1
            | Q2K
            | Q3K
            | Q4K
            | Q5K
            | Q6K
            | Q8K
            | Int4Packed
            | Int3Packed
            | Int8Packed
    )
}

/// Slice batched fused-expert tensors into per-expert [`ScalarJob`]s.
///
/// `mlp.experts.gate_up_proj` is `[E, 2·inter, H]` (gate rows then up rows
/// along dim 1); `mlp.experts.down_proj` is `[E, H, inter]`. Each expert's
/// weight occupies a contiguous element-major byte sub-range of the batched
/// tensor, so we emit one job per `(layer, weight, expert)` pointing at that
/// range — the existing `scalar_from_buf` path then reads it unchanged. This
/// mirrors how [`crate::probe::mixtral`] slices the same tensors for the
/// forward pass.
///
/// Returns the jobs, the largest per-layer expert count seen (for canvas
/// sizing), and the set of layers that carried fused tensors. Block-quantized
/// / bit-packed dtypes can't be sub-sliced on element boundaries, so such
/// tensors are logged and skipped rather than mis-read.
fn build_fused_expert_jobs(
    fused: &std::collections::BTreeMap<
        (u32, crate::format::moe::FusedExpertTensor),
        (usize, format::TensorMeta),
    >,
) -> (Vec<ScalarJob>, u32, std::collections::BTreeSet<u32>) {
    use crate::format::moe::{ExpertWeight as EW, FusedExpertTensor};
    let mut jobs: Vec<ScalarJob> = Vec::new();
    let mut n_experts: u32 = 0;
    let mut layers: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();

    for (&(layer_idx, kind), (shard, meta)) in fused {
        if is_block_or_packed_dtype(meta.dtype) {
            log::warn!(
                "moe-summary: fused expert tensor at layer {} has block/packed dtype {:?}; \
                 cannot slice per-expert — skipping",
                layer_idx,
                meta.dtype,
            );
            continue;
        }
        if meta.shape.len() != 3 {
            log::warn!(
                "moe-summary: fused expert tensor at layer {} has rank {} (expected 3); skipping",
                layer_idx,
                meta.shape.len(),
            );
            continue;
        }
        let e = meta.shape[0];
        let elem = meta.dtype.element_size() as u64;
        // Per-expert element count + which (weight, sub-offset, sub-len) slots
        // live inside one expert's block, in *elements*.
        let (per_expert_elems, slots): (u64, Vec<(EW, u64, u64)>) = match kind {
            FusedExpertTensor::GateUp => {
                let two_inter = meta.shape[1];
                let h = meta.shape[2];
                if two_inter % 2 != 0 {
                    log::warn!(
                        "moe-summary: fused gate_up_proj at layer {} has odd dim1={} \
                         (expected 2*intermediate); skipping",
                        layer_idx,
                        two_inter,
                    );
                    continue;
                }
                let half = (two_inter / 2) * h; // elements in one of {gate, up}
                (
                    two_inter * h,
                    vec![(EW::GateProj, 0, half), (EW::UpProj, half, half)],
                )
            }
            FusedExpertTensor::Down => {
                let block = meta.shape[1] * meta.shape[2]; // H * inter
                (block, vec![(EW::DownProj, 0, block)])
            }
        };
        n_experts = n_experts.max(e as u32);
        layers.insert(layer_idx);
        let per_expert_bytes = per_expert_elems * elem;
        for expert_idx in 0..e {
            let expert_base = meta.file_start + expert_idx * per_expert_bytes;
            for &(weight, sub_off_elems, sub_len_elems) in &slots {
                let start = expert_base + sub_off_elems * elem;
                let len = sub_len_elems * elem;
                // Defensive: never read past the tensor's own byte range.
                if start + len > meta.file_end {
                    log::warn!(
                        "moe-summary: fused {} slice for layer {} expert {} out of range \
                         (start={} len={} end={}); skipping",
                        weight.label(),
                        layer_idx,
                        expert_idx,
                        start,
                        len,
                        meta.file_end,
                    );
                    continue;
                }
                jobs.push((
                    (layer_idx, weight, expert_idx as u32),
                    *shard,
                    start,
                    len,
                    meta.dtype,
                ));
            }
        }
    }
    (jobs, n_experts, layers)
}

/// Model data shared by both `--moe` scene builders: the opened/materialized
/// per-shard [`Data`] handles plus their parsed tensor headers. Built once by
/// [`open_moe_model_sources`] so a combined `--moe` render loads the model a
/// single time. `datas` is the heavy part (mmaps / downloaded shards) — both
/// builders `.clone()` the `Vec<Arc<Data>>`, which only bumps refcounts and
/// shares the underlying bytes; `headers` is light tensor metadata.
struct LoadedMoe {
    datas: Vec<Arc<Data>>,
    headers: Vec<(usize, Vec<format::TensorMeta>)>,
}

/// Open + (for remote, materialize) every `.safetensors` / `.gguf` shard of a
/// single MoE model and parse each shard's tensor header. Shared verbatim by
/// the summary and CKA scene builders — this is the formerly-duplicated
/// file-open + header-fetch preamble, factored out so `--moe` loads once.
async fn open_moe_model_sources(input: &str, stream: bool) -> anyhow::Result<LoadedMoe> {
    // === File opening ===================================================
    let (datas, fmts, file_names) = if hf_url::is_repo_level(input)? {
        let listed = hf_url::list_repo_as_http_specs(input)
            .await
            .with_context(|| format!("listing files in {input}"))?;
        let st_specs: Vec<&(String, hf_url::RemoteFileSpec)> = listed
            .iter()
            .filter(|(n, _)| SourceFormat::from_name(n).is_some())
            .collect();
        if st_specs.is_empty() {
            anyhow::bail!("--moe: no .safetensors / .gguf files in {input}");
        }
        let fmts: Vec<SourceFormat> = st_specs
            .iter()
            .map(|(n, _)| SourceFormat::from_name(n).unwrap_or(SourceFormat::Safetensors))
            .collect();
        let names: Vec<String> = st_specs.iter().map(|(n, _)| n.clone()).collect();
        let specs_owned: Vec<RemoteFileSpec> = st_specs.iter().map(|(_, s)| s.clone()).collect();
        let mut datas: Vec<Arc<Data>> = if stream {
            let pb = setup_progress(
                "source files (xet reconstruction for moe)",
                specs_owned.len() as u64,
            );
            let pb_for_workers = pb.clone();
            let mut out: Vec<(usize, anyhow::Result<Arc<Data>>)> =
                stream::iter(specs_owned.iter().cloned().enumerate())
                    .map(|(i, spec)| {
                        let pb = pb_for_workers.clone();
                        async move {
                            let r: anyhow::Result<Arc<Data>> = if spec.xet_hash.is_some() {
                                match XetReader::new(&spec).await {
                                    Ok(reader) => Ok(Arc::new(Data::Xet(reader))),
                                    Err(e) => {
                                        log::warn!(
                                    "{}: XetReader build failed ({e}); falling back to Data::Http",
                                    spec.filename,
                                );
                                        Ok(Arc::new(Data::Http {
                                            repo: spec.repo.clone(),
                                            filename: Arc::clone(&spec.filename),
                                            revision: Arc::clone(&spec.revision),
                                        }))
                                    }
                                }
                            } else {
                                Ok(Arc::new(Data::Http {
                                    repo: spec.repo.clone(),
                                    filename: Arc::clone(&spec.filename),
                                    revision: Arc::clone(&spec.revision),
                                }))
                            };
                            if let Some(pb) = pb.as_ref() {
                                pb.inc(1);
                            }
                            (i, r)
                        }
                    })
                    .buffer_unordered(SETUP_FETCH_CONCURRENCY)
                    .collect()
                    .await;
            if let Some(pb) = pb.as_ref() {
                pb.finish_and_clear();
            }
            out.sort_by_key(|(i, _)| *i);
            out.into_iter()
                .map(|(_, r)| r)
                .collect::<anyhow::Result<Vec<_>>>()?
        } else {
            specs_owned
                .iter()
                .map(|spec| {
                    Arc::new(Data::Http {
                        repo: spec.repo.clone(),
                        filename: Arc::clone(&spec.filename),
                        revision: Arc::clone(&spec.revision),
                    })
                })
                .collect()
        };
        if !stream {
            materialize_remote_arcs(
                &mut datas,
                &specs_owned,
                "source files (downloading for moe)",
            )
            .await?;
        }
        (datas, fmts, names)
    } else {
        let resolved = hf_url::resolve(Path::new(input))
            .await
            .with_context(|| format!("resolving {input}"))?;
        let files: Vec<PathBuf> = if resolved.is_dir() {
            collect_files_recursive(&resolved)
                .into_iter()
                .filter(|p| SourceFormat::from_path(p).is_some())
                .collect()
        } else {
            vec![resolved]
        };
        if files.is_empty() {
            anyhow::bail!("--moe: no recognised model files (.safetensors / .gguf) in {input}");
        }
        let mut datas = Vec::with_capacity(files.len());
        let mut fmts = Vec::with_capacity(files.len());
        let mut names = Vec::with_capacity(files.len());
        for p in &files {
            let f = File::open(p).with_context(|| format!("opening {}", p.display()))?;
            datas.push(Arc::new(Data::Mapped(unsafe { Mmap::map(&f) }?)));
            fmts.push(SourceFormat::from_path(p).unwrap_or(SourceFormat::Safetensors));
            names.push(
                p.file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default(),
            );
        }
        (datas, fmts, names)
    };

    // === Header fetch ===================================================
    let total_headers = datas.len() as u64;
    let pb = setup_progress("source files (model headers)", total_headers);
    let headers: Vec<(usize, Vec<format::TensorMeta>)> = {
        let pb = pb.clone();
        let items: Vec<(usize, Arc<Data>, SourceFormat)> = datas
            .iter()
            .zip(fmts.iter())
            .enumerate()
            .map(|(i, (d, fmt))| (i, Arc::clone(d), *fmt))
            .collect();
        let mut out: Vec<(usize, anyhow::Result<Vec<format::TensorMeta>>)> = stream::iter(items)
            .map(|(i, d, fmt)| {
                let pb = pb.clone();
                async move {
                    let r = fetch_model_header(&d, fmt)
                        .await
                        .map(|(t, _)| t)
                        .with_context(|| format!("reading {fmt:?} header for moe file {i}"));
                    if let Some(pb) = pb.as_ref() {
                        pb.inc(1);
                    }
                    (i, r)
                }
            })
            .buffer_unordered(SETUP_FETCH_CONCURRENCY)
            .collect()
            .await;
        out.sort_by_key(|(i, _)| *i);
        out.into_iter()
            .map(|(i, r)| r.map(|t| (i, t)))
            .collect::<anyhow::Result<Vec<_>>>()?
    };
    if let Some(pb) = pb.as_ref() {
        pb.finish_and_clear();
    }

    // GGUF fused-expert rejection — not yet supported by either scene.
    for (shard_idx, tensors) in &headers {
        if matches!(fmts[*shard_idx], SourceFormat::Gguf)
            && tensors
                .iter()
                .any(|t| crate::format::moe::is_fused_gguf_expert(&t.name))
        {
            anyhow::bail!(
                "--moe: GGUF fused expert tensors are not yet supported \
                 (found `ffn_*_exps.weight` in {}).",
                file_names
                    .get(*shard_idx)
                    .map(String::as_str)
                    .unwrap_or("<unknown>"),
            );
        }
    }

    Ok(LoadedMoe { datas, headers })
}

/// Build every `--moe` scene for `input` in one pass: open the model once,
/// then render the summary and CKA scenes from the shared [`LoadedMoe`],
/// tagging each scene's `Source`s with an [`arbvis::SceneTag`] so the tiler
/// splits them into independent, tab-switchable pyramids.
///
/// Scenes are independent: a failure in one (e.g. the CKA scene declines a
/// fused-layout checkpoint, which it doesn't yet support) is non-fatal — it's
/// logged and skipped so the other scene still renders. Only an
/// all-scenes-failed run is a hard error.
///
/// When `--probe` is enabled the routing-faithful forward runs exactly **once**
/// here — a single [`crate::probe::RoutingCapture`] carries both the
/// routing-frequency (summary) and co-activation (CKA) signals — and the
/// capture is shared with both scene builders.
pub async fn prepare_moe_scenes_sources(
    input: &str,
    stat: format::SummaryStat,
    sample: u32,
    stream: bool,
    probe: &arbvis::ProbeOpts,
) -> anyhow::Result<(Vec<Source>, u64)> {
    let loaded = open_moe_model_sources(input, stream).await?;

    // Run the probe forward at most once and share its capture across scenes.
    // Non-fatal: an unsupported architecture or a failed forward logs and drops
    // the behavioral panels; the static scenes still render.
    let capture = if probe.enabled {
        match run_probe_capture(input, probe).await {
            Ok(Some(c)) => Some(c),
            Ok(None) => {
                log::warn!(
                    "--moe: --probe requested but the architecture isn't supported; \
                     skipping behavioral panels"
                );
                None
            }
            Err(e) => {
                log::warn!("--moe: --probe forward failed ({e:#}); skipping behavioral panels");
                None
            }
        }
    } else {
        None
    };

    let mut sources: Vec<Source> = Vec::new();

    match build_moe_summary_sources(&loaded, stat, capture.as_ref(), input).await {
        Ok(mut summary) => {
            tag_scene(&mut summary, "summary", "Summary", 0);
            sources.append(&mut summary);
        }
        Err(e) => log::warn!("--moe: summary scene could not be built ({e:#}); skipping it"),
    }

    match build_moe_cka_sources(&loaded, sample, capture.as_ref(), input).await {
        Ok(mut cka) => {
            tag_scene(&mut cka, "cka", "CKA", 1);
            sources.append(&mut cka);
        }
        Err(e) => log::warn!("--moe: CKA scene could not be built ({e:#}); skipping it"),
    }

    if sources.is_empty() {
        anyhow::bail!(
            "--moe: neither the summary nor the CKA scene could be built from {input} \
             (no recognised per-expert / fused MoE tensors?)"
        );
    }

    // Per-scene byte totals are recomputed by the tiler's scene partition; this
    // top-level figure is just the sum the dispatch / progress layer expects.
    let total: u64 = sources.iter().map(|s| s.byte_size).sum();
    Ok((sources, total))
}

/// Stamp every source in `sources` with an [`arbvis::SceneTag`] so the tiler
/// renders them as their own tab-switchable pyramid under `tiles/<key>/`.
fn tag_scene(sources: &mut [Source], key: &str, label: &str, order: u32) {
    for s in sources {
        s.extensions.insert(arbvis::SceneTag {
            key: key.to_string(),
            label: label.to_string(),
            order,
        });
    }
}

/// Build the `--moe` **summary** scene from the shared [`LoadedMoe`]: per-expert
/// scalar heatmaps, one `Source` per per-weight panel (gate / up / down, plus
/// router when the checkpoint has `mlp.gate.weight` tensors) carrying a
/// `MoeSummaryPanel` tag and an `n_layers × n_experts` U8 heatmap that
/// [`crate::MoeSummaryLayoutPlugin`] lays out side-by-side. Returns the panels
/// untagged — the caller stamps the [`arbvis::SceneTag`].
async fn build_moe_summary_sources(
    loaded: &LoadedMoe,
    stat: format::SummaryStat,
    probe_capture: Option<&crate::probe::RoutingCapture>,
    input: &str,
) -> anyhow::Result<Vec<Source>> {
    // Cheap clones: `datas` only bumps Arc refcounts (shared shard bytes),
    // `headers` is light tensor metadata. Keeps the compute below identical
    // to the pre-refactor single-mode body.
    let datas = loaded.datas.clone();
    let headers = loaded.headers.clone();

    // === Group per-expert + router tensors ================================
    use crate::format::moe::{
        parse_hf_expert, parse_hf_fused_expert, parse_hf_router, ExpertWeight as EW,
        FusedExpertTensor,
    };
    use std::collections::BTreeMap;
    type LayerKey = (u32, EW);
    let mut expert_groups: BTreeMap<LayerKey, BTreeMap<u32, (usize, format::TensorMeta)>> =
        BTreeMap::new();
    let mut routers: BTreeMap<u32, (usize, format::TensorMeta)> = BTreeMap::new();
    // Batched fused-expert tensors (newer transformers export): one entry per
    // `(layer, GateUp|Down)`, sliced into per-expert byte ranges below.
    let mut fused: BTreeMap<(u32, FusedExpertTensor), (usize, format::TensorMeta)> =
        BTreeMap::new();
    for (shard_idx, tensors) in &headers {
        for t in tensors {
            if let Some(r) = parse_hf_expert(&t.name) {
                expert_groups
                    .entry((r.layer_idx, r.weight))
                    .or_default()
                    .insert(r.expert_idx, (*shard_idx, t.clone()));
            } else if let Some((layer_idx, kind)) = parse_hf_fused_expert(&t.name) {
                fused.insert((layer_idx, kind), (*shard_idx, t.clone()));
            } else if let Some(layer_idx) = parse_hf_router(&t.name) {
                routers.insert(layer_idx, (*shard_idx, t.clone()));
            }
        }
    }

    // Slice the batched fused tensors into per-expert scalar jobs. Each
    // expert's gate/up/down weights occupy a contiguous byte sub-range of
    // the batched tensor (row-major over `[E, …, …]`), so the existing
    // `scalar_from_buf` machinery applies unchanged — we just hand it the
    // right offset and length. Returns the jobs plus the per-layer expert
    // count discovered from the tensor shapes.
    let (fused_jobs, fused_n_experts, fused_layers) = build_fused_expert_jobs(&fused);

    if expert_groups.is_empty() && fused_jobs.is_empty() {
        anyhow::bail!(
            "--moe-summary: no per-expert tensors found in {input} \
             (expected `model.layers.{{L}}.mlp.experts.{{E}}.{{gate|up|down}}_proj.weight` \
             or batched `model.layers.{{L}}.mlp.experts.{{gate_up_proj|down_proj}}`)"
        );
    }

    // === Determine canvas dimensions ======================================
    let layer_ids: Vec<u32> = {
        let mut set: std::collections::BTreeSet<u32> =
            expert_groups.keys().map(|(l, _)| *l).collect();
        // Include router-only layers so the canvas covers them too — but in
        // practice every MoE layer with a router also has experts, so this
        // is belt-and-braces.
        for l in routers.keys() {
            set.insert(*l);
        }
        // Fused checkpoints carry no per-expert tensors, so their layers come
        // entirely from the batched-tensor scan.
        for l in &fused_layers {
            set.insert(*l);
        }
        set.into_iter().collect()
    };
    let n_experts: u32 = expert_groups
        .values()
        .flat_map(|m| m.keys().copied())
        .max()
        .map(|m| m + 1)
        .unwrap_or(0)
        .max(fused_n_experts);
    let n_layers = layer_ids.len() as u32;
    if n_layers == 0 || n_experts == 0 {
        anyhow::bail!("--moe-summary: derived n_layers=0 or n_experts=0 from {input}");
    }
    // layer_idx → row index in the output matrix (layer ids may be sparse).
    let layer_row: BTreeMap<u32, usize> =
        layer_ids.iter().enumerate().map(|(i, &l)| (l, i)).collect();

    // === Compute per-(layer,weight,expert) scalars ========================
    // For each panel, walk experts sequentially per layer and stream the
    // whole tensor through `scalar_from_buf` (single pass, dtype-aware).
    // Experts within one layer are independent → parallelised per panel
    // via `buffer_unordered`.
    let mut jobs: Vec<ScalarJob> = Vec::new();
    for ((layer_idx, weight), experts) in &expert_groups {
        for (&expert_idx, (shard, meta)) in experts {
            let bytes_len = meta.file_end.saturating_sub(meta.file_start);
            jobs.push((
                (*layer_idx, *weight, expert_idx),
                *shard,
                meta.file_start,
                bytes_len,
                meta.dtype,
            ));
        }
    }
    // Per-expert jobs sliced out of the batched fused tensors join the same
    // pool — downstream only cares about `(layer, weight, expert) → scalar`.
    jobs.extend(fused_jobs);
    let pb = setup_progress("moe-summary expert scalars", jobs.len() as u64);
    let datas_ref = &datas;
    let pb_for_workers = pb.clone();
    let expert_results: Vec<(ScalarKey, f32)> = stream::iter(jobs)
        .map(|(key, shard, start, len, dtype)| {
            let d = Arc::clone(&datas_ref[shard]);
            let pb = pb_for_workers.clone();
            async move {
                let v = if len == 0 {
                    0.0
                } else {
                    match d.fetch_range(start, len as usize).await {
                        Ok(bytes) => scalar_from_buf(stat, dtype, &bytes),
                        Err(e) => {
                            log::warn!(
                                "moe-summary: tensor fetch failed for layer {} {} expert {} ({e}); using 0.0",
                                key.0, key.1.label(), key.2,
                            );
                            0.0
                        }
                    }
                };
                if let Some(pb) = pb.as_ref() {
                    pb.inc(1);
                }
                (key, v)
            }
        })
        .buffer_unordered(SETUP_FETCH_CONCURRENCY)
        .collect()
        .await;
    if let Some(pb) = pb.as_ref() {
        pb.finish_and_clear();
    }

    // === Per-row router slicing ===========================================
    // For each layer's `mlp.gate.weight` (shape `[n_experts, hidden_dim]`),
    // fetch the whole tensor once then slice each row as one expert's gate
    // vector. Skip layers whose router shape doesn't match the inferred
    // n_experts (defensive: a mis-shaped router would mis-index the heatmap).
    let mut router_scalars: BTreeMap<(u32, u32), f32> = BTreeMap::new();
    if !routers.is_empty() {
        let pb = setup_progress("moe-summary router scalars", routers.len() as u64);
        let pb_for_workers = pb.clone();
        let router_jobs: Vec<(u32, usize, format::TensorMeta)> = routers
            .iter()
            .map(|(l, (shard, meta))| (*l, *shard, meta.clone()))
            .collect();
        let results: Vec<(u32, Vec<(u32, f32)>)> = stream::iter(router_jobs)
            .map(|(layer_idx, shard, meta)| {
                let d = Arc::clone(&datas_ref[shard]);
                let pb = pb_for_workers.clone();
                async move {
                    let mut out: Vec<(u32, f32)> = Vec::new();
                    if meta.shape.len() != 2 {
                        log::warn!(
                            "moe-summary: router at layer {} has rank {} (expected 2); skipping",
                            layer_idx,
                            meta.shape.len(),
                        );
                    } else if meta.shape[0] as u32 != n_experts {
                        log::warn!(
                            "moe-summary: router at layer {} has shape[0]={} but inferred \
                             n_experts={}; skipping (likely a router for a different \
                             expert count)",
                            layer_idx,
                            meta.shape[0],
                            n_experts,
                        );
                    } else {
                        let cols = meta.shape[1];
                        let elem = meta.dtype.element_size() as u64;
                        let row_bytes = cols.saturating_mul(elem);
                        let total = meta.file_end.saturating_sub(meta.file_start);
                        match d.fetch_range(meta.file_start, total as usize).await {
                            Ok(bytes) => {
                                for e in 0..n_experts as u64 {
                                    let off = (e * row_bytes) as usize;
                                    let end = off + row_bytes as usize;
                                    if end > bytes.len() {
                                        log::warn!(
                                            "moe-summary: router row {} of layer {} out of \
                                             bounds (off={} end={} len={}); padding with 0",
                                            e,
                                            layer_idx,
                                            off,
                                            end,
                                            bytes.len(),
                                        );
                                        out.push((e as u32, 0.0));
                                    } else {
                                        out.push((
                                            e as u32,
                                            scalar_from_buf(stat, meta.dtype, &bytes[off..end]),
                                        ));
                                    }
                                }
                            }
                            Err(e) => {
                                log::warn!(
                                    "moe-summary: router fetch failed for layer {} ({e}); \
                                     padding with 0",
                                    layer_idx,
                                );
                            }
                        }
                    }
                    if let Some(pb) = pb.as_ref() {
                        pb.inc(1);
                    }
                    (layer_idx, out)
                }
            })
            .buffer_unordered(SETUP_FETCH_CONCURRENCY)
            .collect()
            .await;
        if let Some(pb) = pb.as_ref() {
            pb.finish_and_clear();
        }
        for (layer, rows) in results {
            for (e, v) in rows {
                router_scalars.insert((layer, e), v);
            }
        }
    }

    // === Build per-panel U8 heatmaps + emit Sources =======================
    // One source per panel. Each panel's bytes are (n_layers × n_experts)
    // U8 row-major, normalised so the panel's max scalar maps to 255 and
    // 0 maps to 0. Per-panel scaling means a quiet panel doesn't get
    // crushed by a noisy one — each colormap gradient occupies its own
    // dynamic range.
    let panels: Vec<EW> = {
        let mut v = vec![EW::GateProj, EW::UpProj, EW::DownProj];
        if !routers.is_empty() {
            v.push(EW::Router);
        }
        v
    };
    let mut sources: Vec<Source> = Vec::new();
    let mut total_bytes: u64 = 0;
    for weight in panels {
        let panel_size = (n_layers as usize) * (n_experts as usize);
        let mut scalars: Vec<f32> = vec![0.0; panel_size];
        if matches!(weight, EW::Router) {
            for (&(layer, expert), &v) in &router_scalars {
                if let (Some(&row), col) = (layer_row.get(&layer), expert as usize) {
                    if col < n_experts as usize {
                        scalars[row * n_experts as usize + col] = v;
                    }
                }
            }
        } else {
            for &((layer, w, expert), v) in &expert_results {
                if w != weight {
                    continue;
                }
                if let (Some(&row), col) = (layer_row.get(&layer), expert as usize) {
                    if col < n_experts as usize {
                        scalars[row * n_experts as usize + col] = v;
                    }
                }
            }
        }
        let max = scalars.iter().copied().fold(0.0_f32, f32::max).max(1e-12);
        let bytes_u8: Vec<u8> = scalars
            .iter()
            .map(|&s| ((s / max).clamp(0.0, 1.0) * 255.0) as u8)
            .collect();
        let nbytes = bytes_u8.len() as u64;
        let label = match weight {
            EW::Router => format!(
                "MoE summary · router ({} layers × {} experts)",
                n_layers, n_experts
            ),
            _ => format!(
                "MoE summary · {} ({} layers × {} experts)",
                weight.label(),
                n_layers,
                n_experts,
            ),
        };
        let synthetic = format::TensorMeta {
            name: format!("moe-summary::{}", weight.label()),
            dtype: format::Dtype::U8,
            shape: vec![n_layers as u64, n_experts as u64],
            file_start: 0,
            file_end: nbytes,
            packed_sidecars: None,
        };
        let mut extensions = Extensions::default();
        extensions.insert(ModelInfo {
            format: SourceFormat::Safetensors,
            tensors: vec![synthetic],
            color_ranges: Vec::new(),
        });
        extensions.insert(MoeSummaryPanel {
            weight,
            n_layers,
            n_experts,
        });
        sources.push(Source {
            file_idx: sources.len(),
            kind: SourceKind::Buffered(bytes_u8),
            byte_size: nbytes,
            name_override: Some(label),
            xet_terms: None,
            extensions,
        });
        total_bytes += nbytes;
    }

    // === Optional probe panel ============================================
    // The probe forward is run once by `prepare_moe_scenes_sources`; if it
    // produced a capture, turn its routing-frequency signal into one extra
    // `MoeProbePanel` source. The layout merges it alongside the static panels.
    if let Some(capture) = probe_capture {
        match attach_probe_panel(capture, n_layers, n_experts) {
            Ok(source) => {
                total_bytes += source.byte_size;
                sources.push(source);
            }
            Err(e) => {
                // A dim mismatch is non-fatal: log and skip the panel so the
                // static summary still renders.
                log::warn!("moe-summary: probe panel skipped ({e:#})");
            }
        }
    }

    log::info!(
        "moe-summary: {} layer(s) × {} expert(s), {} panel(s) emitted, {} total byte(s); stat = {:?}",
        n_layers,
        n_experts,
        sources.len(),
        total_bytes,
        stat,
    );

    Ok(sources)
}

/// Resolve a local model directory, detect the architecture, and run the
/// routing-faithful probe forward. Shared by [`attach_probe_panel`]
/// (`--moe-summary`) and [`attach_cka_probe_panels`] (`--moe-cka`).
///
/// Returns `Ok(None)` when the architecture isn't supported by the probe
/// (the caller logs a warning and renders the static panels anyway).
/// Errors on any other failure (bad input, missing shards, forward panic).
async fn run_probe_capture(
    input: &str,
    probe: &arbvis::ProbeOpts,
) -> anyhow::Result<Option<crate::probe::RoutingCapture>> {
    // Currently only local-directory inputs are supported for the probe:
    // we need tokenizer.json + per-shard paths to hand to candle's
    // VarBuilder. hf:// inputs go via the HF cache, but threading those
    // cache paths through the prep function is left for a follow-up —
    // surface a clear error.
    if hf_url::is_repo_level(input)? {
        anyhow::bail!(
            "--probe: hf:// repo inputs not yet supported; resolve to a local dir first \
             (`hf download …`) and pass that path."
        );
    }
    let model_dir = hf_url::resolve(Path::new(input))
        .await
        .with_context(|| format!("--probe: resolving {input}"))?;
    if !model_dir.is_dir() {
        anyhow::bail!(
            "--probe: input {input} resolved to {} (expected a directory containing \
             tokenizer.json + safetensors shards)",
            model_dir.display(),
        );
    }

    // Architecture detection from config.json. Skip probe with a None
    // (caller logs) if the arch isn't supported.
    let config =
        crate::layout::model_config::ModelConfig::try_from_dir(&model_dir).ok_or_else(|| {
            anyhow::anyhow!(
                "--probe: missing or unparseable config.json in {}",
                model_dir.display()
            )
        })?;
    let Some(arch) = crate::probe::detect_arch(&config) else {
        log::warn!(
            "--probe: architecture {:?} not supported (need Qwen2MoeForCausalLM or MixtralForCausalLM)",
            config.architectures,
        );
        return Ok(None);
    };

    // List safetensors shards in `model_dir`. Sorted for deterministic
    // VarBuilder loading order.
    let mut weight_paths: Vec<PathBuf> = collect_files_recursive(&model_dir)
        .into_iter()
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("safetensors"))
        .collect();
    weight_paths.sort();
    if weight_paths.is_empty() {
        anyhow::bail!("--probe: no .safetensors shards in {}", model_dir.display(),);
    }

    // Resolve probe text.
    let probe_text = crate::probe::text::resolve(&probe.source).await?;
    if probe_text.trim().is_empty() {
        anyhow::bail!("--probe: resolved probe text is empty");
    }

    // Forward pass. May take a couple of minutes on a real-sized MoE.
    let capture = tokio::task::spawn_blocking({
        let model_dir = model_dir.clone();
        let weight_paths = weight_paths.clone();
        let config = config.clone();
        let probe_text = probe_text.clone();
        move || crate::probe::run(arch, &model_dir, &weight_paths, &config, &probe_text)
    })
    .await
    .with_context(|| "--probe: forward pass panicked")??;

    Ok(Some(capture))
}

/// Build the summary scene's behavioral panel — a per-`(layer, expert)`
/// routing-frequency heatmap — from an already-captured routing forward (run
/// once by [`prepare_moe_scenes_sources`] and shared across scenes). Tagged
/// with `MoeProbePanel`. Errors only on a capture/summary dimension mismatch.
fn attach_probe_panel(
    capture: &crate::probe::RoutingCapture,
    n_layers: u32,
    n_experts: u32,
) -> anyhow::Result<Source> {
    // Normalize freq to U8 per-panel (same convention as the static
    // summary panels): max → 255, 0 → 0.
    let max = capture
        .freq
        .iter()
        .copied()
        .fold(0.0_f32, f32::max)
        .max(1e-12);
    let bytes_u8: Vec<u8> = capture
        .freq
        .iter()
        .map(|&v| ((v / max).clamp(0.0, 1.0) * 255.0) as u8)
        .collect();
    let nbytes = bytes_u8.len() as u64;
    log::info!(
        "--probe: routing-frequency capture from {} tokens ({} layers × {} experts), \
         max_freq = {max:.3}",
        capture.n_tokens,
        capture.n_layers,
        capture.n_experts,
    );

    // Refuse to attach if dimensions don't match the static summary
    // panels — would mis-render the heatmap.
    if capture.n_layers != n_layers || capture.n_experts != n_experts {
        anyhow::bail!(
            "--probe: capture dims ({}×{}) differ from summary dims ({}×{}); refusing to attach",
            capture.n_layers,
            capture.n_experts,
            n_layers,
            n_experts,
        );
    }

    let synthetic = format::TensorMeta {
        name: format!("moe-summary::probe::{}", ProbeStat::RoutingFreq.label()),
        dtype: format::Dtype::U8,
        shape: vec![n_layers as u64, n_experts as u64],
        file_start: 0,
        file_end: nbytes,
        packed_sidecars: None,
    };
    let label = format!(
        "MoE probe · routing_freq ({} layers × {} experts, {} tokens)",
        n_layers, n_experts, capture.n_tokens,
    );
    let mut extensions = Extensions::default();
    extensions.insert(ModelInfo {
        format: SourceFormat::Safetensors,
        tensors: vec![synthetic],
        color_ranges: Vec::new(),
    });
    extensions.insert(MoeProbePanel {
        stat: ProbeStat::RoutingFreq,
        n_layers,
        n_experts,
    });
    Ok(Source {
        // file_idx is overwritten by the caller before push.
        file_idx: 0,
        kind: SourceKind::Buffered(bytes_u8),
        byte_size: nbytes,
        name_override: Some(label),
        xet_terms: None,
        extensions,
    })
}

/// Build the CKA scene's behavioral panels — one per-layer `n_experts ×
/// n_experts` routing co-activation matrix — from an already-captured routing
/// forward (run once by [`prepare_moe_scenes_sources`] and shared across
/// scenes). Each is tagged with `MoeCkaProbePanel`; the CKA layout slots them
/// in as an extra column. Errors only on a capture/CKA dimension mismatch.
fn attach_cka_probe_panels(
    capture: &crate::probe::RoutingCapture,
    n_layers: u32,
    n_experts: u32,
) -> anyhow::Result<Vec<Source>> {
    // Refuse to attach if dimensions don't match the static CKA panels —
    // would mis-place / mis-render the heatmaps in the grid.
    if capture.n_layers != n_layers || capture.n_experts != n_experts {
        anyhow::bail!(
            "--probe: capture dims ({}×{}) differ from moe-cka dims ({}×{}); refusing to attach",
            capture.n_layers,
            capture.n_experts,
            n_layers,
            n_experts,
        );
    }
    log::info!(
        "--probe: routing co-activation capture from {} tokens ({} layers × {} experts)",
        capture.n_tokens,
        capture.n_layers,
        capture.n_experts,
    );

    let e = n_experts as usize;
    let mut out: Vec<Source> = Vec::with_capacity(n_layers as usize);
    for layer in 0..n_layers as usize {
        let block = &capture.coact[layer * e * e..(layer + 1) * e * e];
        // Per-panel (per-layer) max → 255 normalisation, matching the
        // summary probe panel and `compute_cka_panel`'s 0..=255 scaling.
        // The brightest cell is the most-co-fired expert pair in the layer.
        let max = block.iter().copied().fold(0.0_f32, f32::max).max(1e-12);
        let bytes_u8: Vec<u8> = block
            .iter()
            .map(|&v| ((v / max).clamp(0.0, 1.0) * 255.0) as u8)
            .collect();
        let nbytes = bytes_u8.len() as u64;
        let synthetic = format::TensorMeta {
            name: format!(
                "moe-cka::L{layer}::{}",
                ProbeStat::RoutingCoactivation.label()
            ),
            dtype: format::Dtype::U8,
            shape: vec![n_experts as u64, n_experts as u64],
            file_start: 0,
            file_end: nbytes,
            packed_sidecars: None,
        };
        let label = format!(
            "MoE co-activation · layer {layer} ({n_experts}×{n_experts}, {} tokens)",
            capture.n_tokens,
        );
        let mut extensions = Extensions::default();
        extensions.insert(ModelInfo {
            format: SourceFormat::Safetensors,
            tensors: vec![synthetic],
            color_ranges: Vec::new(),
        });
        extensions.insert(MoeCkaProbePanel {
            layer: layer as u32,
            n_experts,
        });
        out.push(Source {
            // file_idx is overwritten by the caller before push.
            file_idx: 0,
            kind: SourceKind::Buffered(bytes_u8),
            byte_size: nbytes,
            name_override: Some(label),
            xet_terms: None,
            extensions,
        });
    }
    Ok(out)
}

/// Dispatch a [`format::SummaryStat`] over a contiguous tensor byte slice.
/// Single-pass, dtype-aware. Used by [`build_moe_summary_sources`].
fn scalar_from_buf(stat: format::SummaryStat, dtype: format::Dtype, bytes: &[u8]) -> f32 {
    match stat {
        format::SummaryStat::Rms => format::rms_from_buf(dtype, bytes),
        format::SummaryStat::Frobenius => format::frobenius_from_buf(dtype, bytes),
        format::SummaryStat::MeanAbs => format::mean_abs_from_buf(dtype, bytes),
        // 1e-6 is well above bf16 underflow but small enough to catch
        // "essentially zero" entries. Hardcoded for now — promote to a CLI
        // knob if a user reports it being wrong for their dtype mix.
        format::SummaryStat::Sparsity => format::sparsity_from_buf(dtype, bytes, 1e-6),
    }
}

/// Decode `bytes` (one expert's whole weight tensor, dtype `dtype`) to
/// a fresh `Vec<f32>` of length `n_elements`. Non-finite values are
/// preserved (caller's projection / inner-product loops happen to be
/// finite-safe — the random projection of a NaN row stays NaN, but
/// the only way to hit one is a malformed checkpoint, in which case
/// CKA polluted by NaN is still a useful diagnostic).
fn decode_tensor_to_f32(dtype: format::Dtype, bytes: &[u8], n_elements: usize) -> Vec<f32> {
    let mut reader = format::TensorElementReader::new(dtype, bytes);
    let mut out = vec![0.0f32; n_elements];
    for (k, slot) in out.iter_mut().enumerate() {
        *slot = reader.element(k);
    }
    out
}

/// Build the `--moe` **CKA** scene from the shared [`LoadedMoe`]: per-`(layer,
/// weight)` `n_experts × n_experts` linear-CKA similarity matrices. Emits one
/// `Source` per panel carrying a [`MoeCkaPanel`] tag and an in-memory U8 heatmap
/// of linear CKA between every expert pair; [`crate::MoeCkaLayoutPlugin`] lays
/// them out in an `n_layers × 3` grid (one row per layer; gate / up / down
/// columns). Returns the panels untagged — the caller stamps the
/// [`arbvis::SceneTag`].
///
/// Uses Gaussian random projection on the input dim (controlled by `sample`,
/// defaulting to 128 via the CLI) so the per-pair compute drops from
/// `O(d_in² · d_out)` to `O(k² · d_out)` — see [`crate::cka`] for the math and
/// accuracy notes.
///
/// Router weights are intentionally excluded: CKA needs a shared input space,
/// and the router maps tokens → expert logits (a different space from the FFN
/// weights). The router's per-row stat lives in the summary scene instead;
/// pairwise router-row CKA would be well-defined but doesn't fit the
/// per-`(layer, weight)` panel shape, so it's deferred.
async fn build_moe_cka_sources(
    loaded: &LoadedMoe,
    sample: u32,
    probe_capture: Option<&crate::probe::RoutingCapture>,
    input: &str,
) -> anyhow::Result<Vec<Source>> {
    // Cheap clones: see [`build_moe_summary_sources`].
    let datas = loaded.datas.clone();
    let headers = loaded.headers.clone();

    // === Expert grouping ==================================================
    use crate::format::moe::{parse_hf_expert, ExpertWeight as EW};
    use std::collections::BTreeMap;
    type LayerKey = (u32, EW);
    let mut groups: BTreeMap<LayerKey, BTreeMap<u32, (usize, format::TensorMeta)>> =
        BTreeMap::new();
    for (shard_idx, tensors) in &headers {
        for t in tensors {
            if let Some(r) = parse_hf_expert(&t.name) {
                groups
                    .entry((r.layer_idx, r.weight))
                    .or_default()
                    .insert(r.expert_idx, (*shard_idx, t.clone()));
            }
        }
    }
    if groups.is_empty() {
        anyhow::bail!(
            "--moe-cka: no per-expert tensors found in {input} \
             (expected `model.layers.{{L}}.mlp.experts.{{E}}.{{gate|up|down}}_proj.weight`)"
        );
    }

    let n_experts: u32 = groups
        .values()
        .flat_map(|m| m.keys().copied())
        .max()
        .map(|m| m + 1)
        .unwrap_or(0);
    if n_experts == 0 {
        anyhow::bail!("--moe-cka: derived n_experts=0 from {input}");
    }
    // Layer count for the optional probe co-activation panels (one per
    // layer). Derived from the static panel grid; the dims guard in
    // `attach_cka_probe_panels` rejects any mismatch with the probe's
    // config-derived layer count, so a partial-shard CKA run declines the
    // panels rather than mis-rendering.
    let n_layers: u32 = groups
        .keys()
        .map(|(layer, _)| *layer)
        .max()
        .map(|m| m + 1)
        .unwrap_or(0);

    // === Per-(layer, weight): project all experts, then pair CKA ==========
    // Process groups serially so peak memory stays bounded — each group
    // peaks at `n_experts × d_out × k × 4 bytes` of f32 projected matrices
    // (~43 MB for Qwen at d_out=1408, k=128, n_experts=60). Within a
    // group, expert projection is parallel (rayon over experts), and
    // pair-CKA is parallel (rayon over the upper-triangle index set).
    //
    // Random projection seed is fixed at module level (per-group seed
    // derives from layer + weight) so reruns produce identical heatmaps.
    let datas_ref = &datas;
    let pb = setup_progress("moe-cka panels", groups.len() as u64);
    let pb_for_groups = pb.clone();
    let panel_results: Vec<((u32, EW), anyhow::Result<Vec<u8>>)> =
        stream::iter(groups.iter().map(|(k, v)| (*k, v.clone())))
            .map(|((layer_idx, weight), experts)| {
                let pb = pb_for_groups.clone();
                async move {
                    let r = compute_cka_panel(
                        layer_idx,
                        weight,
                        &experts,
                        datas_ref,
                        n_experts,
                        sample as usize,
                    )
                    .await;
                    if let Some(pb) = pb.as_ref() {
                        pb.inc(1);
                    }
                    ((layer_idx, weight), r)
                }
            })
            // Serial — one group at a time. Within each group the heavy
            // lifting (projection + pair loop) is multi-threaded via rayon;
            // running multiple groups concurrently would oversubscribe the
            // CPU and blow peak memory.
            .buffered(1)
            .collect()
            .await;
    if let Some(pb) = pb.as_ref() {
        pb.finish_and_clear();
    }

    // Materialise sources in stable `(layer, weight)` order.
    let mut sources: Vec<Source> = Vec::new();
    let mut total_bytes: u64 = 0;
    for ((layer_idx, weight), result) in panel_results {
        let bytes_u8 = result.with_context(|| {
            format!(
                "computing CKA panel for layer {layer_idx} {}",
                weight.label()
            )
        })?;
        let nbytes = bytes_u8.len() as u64;
        let synthetic = format::TensorMeta {
            name: format!("moe-cka::L{layer_idx}::{}", weight.label()),
            dtype: format::Dtype::U8,
            shape: vec![n_experts as u64, n_experts as u64],
            file_start: 0,
            file_end: nbytes,
            packed_sidecars: None,
        };
        let label = format!(
            "MoE CKA · layer {layer_idx} · {} ({n_experts}×{n_experts})",
            weight.label(),
        );
        let mut extensions = Extensions::default();
        extensions.insert(ModelInfo {
            format: SourceFormat::Safetensors,
            tensors: vec![synthetic],
            color_ranges: Vec::new(),
        });
        extensions.insert(MoeCkaPanel {
            layer: layer_idx,
            weight,
            n_experts,
        });
        sources.push(Source {
            file_idx: sources.len(),
            kind: SourceKind::Buffered(bytes_u8),
            byte_size: nbytes,
            name_override: Some(label),
            xet_terms: None,
            extensions,
        });
        total_bytes += nbytes;
    }

    // === Optional probe forward pass =====================================
    // When `--probe` is set, run a routing-faithful forward on the resolved
    // probe text and emit one extra Source per layer (a routing co-activation
    // matrix) tagged with `MoeCkaProbePanel`. The CKA layout merges these in
    // as an extra column. Probe failures are non-fatal: log and skip so the
    // static CKA grid still renders.
    if let Some(capture) = probe_capture {
        match attach_cka_probe_panels(capture, n_layers, n_experts) {
            Ok(panels) => {
                for mut source in panels {
                    source.file_idx = sources.len();
                    total_bytes += source.byte_size;
                    sources.push(source);
                }
            }
            Err(e) => {
                log::warn!("moe-cka: co-activation panels skipped ({e:#})");
            }
        }
    }

    log::info!(
        "moe-cka: {} panel(s) emitted, {} total byte(s); sample = {}",
        sources.len(),
        total_bytes,
        sample,
    );

    Ok(sources)
}

/// One panel of CKA: project every expert in `experts` and compute the
/// pairwise similarity matrix. Returns the row-major `n_experts × n_experts`
/// U8 heatmap. CKA values in `[0, 1]` map to `0..=255` linearly.
async fn compute_cka_panel(
    layer_idx: u32,
    weight: crate::format::moe::ExpertWeight,
    experts: &std::collections::BTreeMap<u32, (usize, format::TensorMeta)>,
    datas: &[Arc<Data>],
    n_experts: u32,
    k: usize,
) -> anyhow::Result<Vec<u8>> {
    use rayon::prelude::*;

    // Pick d_out, d_in from the first expert's shape — all experts of one
    // (layer, weight) share shape by construction.
    let Some(first) = experts.values().next() else {
        return Ok(vec![0u8; (n_experts as usize) * (n_experts as usize)]);
    };
    let meta = &first.1;
    if meta.shape.len() != 2 {
        anyhow::bail!(
            "moe-cka: layer {} {} expects rank-2 weights, got shape {:?}",
            layer_idx,
            weight.label(),
            meta.shape,
        );
    }
    let d_out = meta.shape[0] as usize;
    let d_in = meta.shape[1] as usize;
    let dtype = meta.dtype;
    let n_elements = d_out * d_in;

    // Seed: layer × big-prime + weight discriminant. Stable across
    // reruns, distinct across panels.
    let weight_idx: u64 = match weight {
        crate::format::moe::ExpertWeight::GateProj => 0,
        crate::format::moe::ExpertWeight::UpProj => 1,
        crate::format::moe::ExpertWeight::DownProj => 2,
        crate::format::moe::ExpertWeight::Router => 3,
    };
    let seed = (layer_idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ weight_idx;
    let r = crate::cka::gaussian_projection(k, d_in, seed);

    // Fetch + decode + project per expert. Serial fetch (mmap is free;
    // HTTP would need awaiting) but the per-expert project_rows is
    // already rayon-parallel internally.
    //
    // Result: `projections[e] = Some((d_out × k row-major, self_norm_sq))`
    // for present experts, `None` for sparse holes in the expert id
    // sequence (shouldn't happen for well-formed checkpoints, but we
    // pad with zero-similarity rather than failing).
    let mut projections: Vec<Option<(Vec<f32>, f64)>> =
        (0..n_experts as usize).map(|_| None).collect();
    for (&expert_idx, (shard, meta)) in experts {
        let nbytes = meta.file_end.saturating_sub(meta.file_start) as usize;
        let bytes = datas[*shard].fetch_range(meta.file_start, nbytes).await?;
        let w = decode_tensor_to_f32(dtype, &bytes, n_elements);
        let w_proj = crate::cka::project_rows(&w, d_out, d_in, &r, k);
        let self_sq = crate::cka::at_b_frobenius_sq(&w_proj, &w_proj, d_out, k);
        if (expert_idx as usize) < projections.len() {
            projections[expert_idx as usize] = Some((w_proj, self_sq));
        }
    }

    // Pairwise CKA over the upper triangle, parallelised. We materialise
    // pair indices into a flat vec so rayon can chunk over them; each pair
    // writes its result to two symmetric positions of `matrix` (a single
    // assignment per cell — disjoint writes, no synchronization needed).
    //
    // Use `par_iter` with an indexed output Vec held inside an UnsafeCell-
    // like wrapper? Simpler: compute into a `Vec<(i, j, score)>` then
    // materialise the heatmap in a second pass.
    let n = n_experts as usize;
    let mut pairs: Vec<(usize, usize)> = Vec::with_capacity(n * (n + 1) / 2);
    for i in 0..n {
        for j in i..n {
            pairs.push((i, j));
        }
    }
    let scores: Vec<f32> = pairs
        .par_iter()
        .map(|&(i, j)| {
            let a = projections[i].as_ref();
            let b = projections[j].as_ref();
            match (a, b) {
                (Some((ap, asq)), Some((bp, bsq))) => {
                    crate::cka::linear_cka(ap, bp, d_out, k, *asq, *bsq)
                }
                _ => 0.0,
            }
        })
        .collect();

    let mut matrix = vec![0u8; n * n];
    for (idx, &(i, j)) in pairs.iter().enumerate() {
        // CKA already in [0, 1]; scale into 0..=255.
        let v = (scores[idx].clamp(0.0, 1.0) * 255.0) as u8;
        matrix[i * n + j] = v;
        matrix[j * n + i] = v;
    }
    Ok(matrix)
}

/// Build per-tensor diff Sources from two single .safetensors files.
pub async fn build_safetensors_diff_sources(
    original: &Path,
    modified: &Path,
    is_finetune: bool,
    metric: format::DiffMetric,
) -> anyhow::Result<(Vec<Source>, u64)> {
    build_multi_safetensors_diff_sources(
        &[original.to_path_buf()],
        &[modified.to_path_buf()],
        is_finetune,
        metric,
    )
    .await
}

/// Sidecar metadata fetched alongside a safetensors source. Both fields are
/// optional — the architectural layout works without them by inferring
/// everything from tensor names + shapes; when present they let the layout
/// validate inferred layer counts and reserve canonical slots for tensors
/// that live in shards we didn't load.
#[derive(Default, Debug, Clone)]
pub struct SourceMeta {
    pub config: Option<ModelConfig>,
    pub index: Option<SafetensorsIndex>,
}

/// Best-effort opportunistic load of `config.json` and
/// `model.safetensors.index.json` for a single source. Local sources read
/// their parent directory; HF Hub sources fetch the file from the same repo;
/// other kinds return empty.
///
/// Errors are swallowed and logged at debug level — sidecar info is advisory,
/// and a missing sidecar must not break rendering.
// `try_load_source_meta` / `load_meta_for_sources` / `fetch_hf_sidecar`
// are the sidecar-fetching cluster: they load `config.json` and
// `model.safetensors.index.json` next to a source so the arch layout
// can read transformer hyperparams (num_hidden_layers, etc.) and shard
// stitching info. They're invoked by [`crate::SourceMetaSidecarHook`],
// the [`arbvis::PrepareSourcesExtension`] impl registered on the
// registry by [`crate::register_all`]. The hook runs once at the top
// of `dispatch_render`, after every `Source` has been built, and
// inserts a `SourceMeta` into each source's extensions for
// `ArchLayoutPlugin::build` to read back.
pub async fn try_load_source_meta(source: &Source) -> SourceMeta {
    match &source.kind {
        SourceKind::File(p) => {
            // For local GGUF files, the equivalent of `config.json` is
            // embedded in the binary header. Re-parse the prefix to extract
            // it; this is cheap (~1 MiB read at setup time only).
            if matches!(SourceFormat::from_path(p), Some(SourceFormat::Gguf)) {
                if let Ok(meta_size) = std::fs::metadata(p).map(|m| m.len()) {
                    let want = GGUF_HEADER_FETCH_INITIAL.min(meta_size as usize);
                    if let Ok(mut f) = File::open(p) {
                        let mut buf = vec![0u8; want];
                        if f.read_exact(&mut buf).is_ok() {
                            if let Ok(h) = format::gguf::parse_header(&buf) {
                                return SourceMeta {
                                    config: Some(ModelConfig::from_gguf_metadata(&h.metadata)),
                                    index: None,
                                };
                            }
                        }
                    }
                }
            }
            if let Some(dir) = p.parent() {
                SourceMeta {
                    config: ModelConfig::try_from_dir(dir),
                    index: SafetensorsIndex::try_from_dir(dir),
                }
            } else {
                SourceMeta::default()
            }
        }
        SourceKind::Http(spec) => {
            // For remote GGUF files, fetch the header prefix and pull
            // ModelConfig fields out of its KV table instead of looking for
            // a sibling config.json (which often doesn't exist for
            // llama.cpp-style GGUF releases).
            if matches!(
                SourceFormat::from_name(&spec.filename),
                Some(SourceFormat::Gguf)
            ) {
                let data = Data::Http {
                    repo: spec.repo.clone(),
                    filename: Arc::clone(&spec.filename),
                    revision: Arc::clone(&spec.revision),
                };
                if let Ok(buf) = data.fetch_range(0, GGUF_HEADER_FETCH_INITIAL).await {
                    if let Ok(h) = format::gguf::parse_header(&buf) {
                        return SourceMeta {
                            config: Some(ModelConfig::from_gguf_metadata(&h.metadata)),
                            index: None,
                        };
                    }
                }
                return SourceMeta::default();
            }
            // Fetch config.json and model.safetensors.index.json from the
            // same repo via a raw GET. Both files are small enough that we
            // don't bother streaming.
            let config = fetch_hf_sidecar(&spec.repo, &spec.revision, "config.json")
                .await
                .ok()
                .and_then(|b| ModelConfig::from_bytes(&b));
            let index =
                fetch_hf_sidecar(&spec.repo, &spec.revision, "model.safetensors.index.json")
                    .await
                    .ok()
                    .and_then(|b| SafetensorsIndex::from_bytes(&b));
            SourceMeta { config, index }
        }
        _ => SourceMeta::default(),
    }
}

/// Load sidecar meta for every source, de-duplicated by HF Hub repo+revision
/// (so a 4-shard model triggers one config.json fetch, not four). Returns a
/// `Vec` parallel to `sources`.
pub async fn load_meta_for_sources(sources: &[Source]) -> Vec<SourceMeta> {
    // Group sources by (repo_id, revision) for HF Hub; by parent dir for
    // local. Each group gets one fetch.
    let mut keys: Vec<Option<String>> = Vec::with_capacity(sources.len());
    for s in sources {
        let key = match &s.kind {
            SourceKind::File(p) => p.parent().map(|d| format!("local:{}", d.display())),
            SourceKind::Http(spec) => Some(format!("hf:{}@{}", spec.repo.repo_id(), spec.revision)),
            _ => None,
        };
        keys.push(key);
    }
    let mut cache: HashMap<String, SourceMeta> = HashMap::new();
    let mut result: Vec<SourceMeta> = Vec::with_capacity(sources.len());
    for (s, k) in sources.iter().zip(keys.iter()) {
        match k {
            Some(k) => {
                if let Some(m) = cache.get(k) {
                    result.push(m.clone());
                } else {
                    let m = try_load_source_meta(s).await;
                    cache.insert(k.clone(), m.clone());
                    result.push(m);
                }
            }
            None => result.push(SourceMeta::default()),
        }
    }
    result
}

/// Fetch a small sidecar file (e.g. `config.json`) from an HF Hub repo via
/// a raw GET to `{endpoint}/{repo_id}/resolve/{revision}/{filename}`.
/// Returns the raw bytes on success.
async fn fetch_hf_sidecar(
    repo: &RemoteRepo,
    revision: &str,
    filename: &str,
) -> anyhow::Result<Vec<u8>> {
    let url = format!(
        "{}/{}/resolve/{}/{}",
        hf_url::endpoint(),
        repo.repo_id(),
        revision,
        filename,
    );
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .with_context(|| format!("building reqwest client for {filename}"))?;
    let mut req = client.get(&url);
    if let Some(tok) = hf_url::read_token() {
        req = req.bearer_auth(tok);
    }
    let resp = req
        .send()
        .await
        .with_context(|| format!("fetching {url}"))?;
    let resp = resp
        .error_for_status()
        .with_context(|| format!("non-2xx for {filename}"))?;
    let bytes = resp
        .bytes()
        .await
        .with_context(|| format!("reading body of {filename}"))?;
    Ok(bytes.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::moe::{ExpertWeight as EW, FusedExpertTensor};
    use std::collections::BTreeMap;

    fn meta(shape: Vec<u64>, dtype: format::Dtype, file_start: u64) -> format::TensorMeta {
        let elems: u64 = shape.iter().product();
        format::TensorMeta {
            name: "fused".into(),
            dtype,
            file_start,
            file_end: file_start + elems * dtype.element_size() as u64,
            shape,
            packed_sidecars: None,
        }
    }

    #[test]
    fn fused_jobs_slice_gate_up_and_down_at_correct_offsets() {
        // E=2 experts, intermediate=4, hidden=4 → gate_up = [2, 8, 4],
        // down = [2, 4, 4]. F32 (4 bytes/elem) keeps the arithmetic legible.
        let mut fused: BTreeMap<(u32, FusedExpertTensor), (usize, format::TensorMeta)> =
            BTreeMap::new();
        // gate_up_proj at file offset 1000; down_proj (shard 1) at 5000.
        fused.insert(
            (0, FusedExpertTensor::GateUp),
            (0, meta(vec![2, 8, 4], format::Dtype::F32, 1000)),
        );
        fused.insert(
            (0, FusedExpertTensor::Down),
            (1, meta(vec![2, 4, 4], format::Dtype::F32, 5000)),
        );

        let (jobs, n_experts, layers) = build_fused_expert_jobs(&fused);
        assert_eq!(n_experts, 2);
        assert_eq!(layers.into_iter().collect::<Vec<_>>(), vec![0]);

        // Index the jobs by (layer, weight, expert) for assertion.
        let by_key: BTreeMap<ScalarKey, (usize, u64, u64, format::Dtype)> = jobs
            .iter()
            .map(|&(k, shard, start, len, dt)| (k, (shard, start, len, dt)))
            .collect();

        // Per expert, the gate_up block is 8*4*4 = 128 bytes; each of gate/up
        // is half (4*4*4 = 64 bytes). Expert 1's block starts 128 bytes in.
        // gate(e0): [1000, +64); up(e0): [1064, +64).
        assert_eq!(
            by_key[&(0, EW::GateProj, 0)],
            (0, 1000, 64, format::Dtype::F32)
        );
        assert_eq!(
            by_key[&(0, EW::UpProj, 0)],
            (0, 1064, 64, format::Dtype::F32)
        );
        // gate(e1): [1128, +64); up(e1): [1192, +64).
        assert_eq!(
            by_key[&(0, EW::GateProj, 1)],
            (0, 1128, 64, format::Dtype::F32)
        );
        assert_eq!(
            by_key[&(0, EW::UpProj, 1)],
            (0, 1192, 64, format::Dtype::F32)
        );
        // down block per expert is 4*4*4 = 64 bytes, shard 1.
        assert_eq!(
            by_key[&(0, EW::DownProj, 0)],
            (1, 5000, 64, format::Dtype::F32)
        );
        assert_eq!(
            by_key[&(0, EW::DownProj, 1)],
            (1, 5064, 64, format::Dtype::F32)
        );
        // Exactly 6 jobs (2 experts × {gate, up, down}).
        assert_eq!(jobs.len(), 6);
    }

    #[test]
    fn fused_jobs_skip_block_quantized_and_malformed() {
        // Block-quantized dtype can't be sub-sliced → skipped.
        let mut q: BTreeMap<(u32, FusedExpertTensor), (usize, format::TensorMeta)> =
            BTreeMap::new();
        q.insert(
            (0, FusedExpertTensor::GateUp),
            (0, meta(vec![2, 8, 4], format::Dtype::Q4K, 0)),
        );
        let (jobs, n, _) = build_fused_expert_jobs(&q);
        assert!(jobs.is_empty());
        assert_eq!(n, 0);

        // Odd dim-1 on gate_up (not 2*intermediate) → skipped.
        let mut odd: BTreeMap<(u32, FusedExpertTensor), (usize, format::TensorMeta)> =
            BTreeMap::new();
        odd.insert(
            (0, FusedExpertTensor::GateUp),
            (0, meta(vec![2, 7, 4], format::Dtype::F32, 0)),
        );
        let (jobs, _, _) = build_fused_expert_jobs(&odd);
        assert!(jobs.is_empty());

        // Wrong rank → skipped.
        let mut rank: BTreeMap<(u32, FusedExpertTensor), (usize, format::TensorMeta)> =
            BTreeMap::new();
        rank.insert(
            (0, FusedExpertTensor::Down),
            (0, meta(vec![2, 4], format::Dtype::F32, 0)),
        );
        let (jobs, _, _) = build_fused_expert_jobs(&rank);
        assert!(jobs.is_empty());
    }
}
