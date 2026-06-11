//! Tensor-aware CLI surface that flattens [`arbvis::Args`] and adds the
//! model-side flags (`--moe`, the `--probe` family, `--summary-stat`,
//! `--cka-sample`, `--finetune` / `--no-finetune`, `--diff-metric`,
//! `--layout`).
//!
//! The diff / layout flags used to live on `arbvis::Args` directly, which meant
//! `arbvis --help` advertised them even though they only do anything when
//! a tensor-aware backend is registered (and the byte-only `arbvis`
//! binary errors out at runtime if the user actually passes them). After
//! the split they're owned here, so `arbvis --help` advertises only the
//! flags it can actually run, and `modelweightvis --help` advertises
//! both halves.
//!
//! Clap's `#[command(flatten)]` makes the two structs share a single
//! `Command`, so `--diff` (defined on `arbvis::Args`) and `--finetune`
//! (defined here) can reference each other in `requires =` / `conflicts_with`
//! constraints just as if they lived on the same struct. The cross-crate
//! visibility doesn't matter to clap — `arbvis::Args` derives `Parser` /
//! `FromArgMatches`, which is all the flatten machinery needs.

use std::path::PathBuf;

use arbvis::{DiffMetric, LayoutMode, ModelOpts, ProbeOpts, ProbeSource, SummaryStat};
use clap::{Parser, ValueEnum};

/// CLI mirror of [`arbvis::DiffMetric`]. Kept separate from the core type so
/// the clap derive's variant-doc strings and enum-discoverable help text
/// don't leak into the library API.
#[derive(Clone, Copy, Debug, ValueEnum, Default)]
pub enum DiffMetricArg {
    /// Per-tensor RMS-normalized signed delta. Stable across tensors of
    /// different scale; default.
    #[default]
    Rms,
    /// Absolute delta on a log brightness scale. Honest about raw magnitudes.
    AbsLog,
    /// Ternary: identical bytes → black; any change → full saturation.
    Exact,
}

impl From<DiffMetricArg> for DiffMetric {
    fn from(a: DiffMetricArg) -> Self {
        match a {
            DiffMetricArg::Rms => DiffMetric::Rms,
            DiffMetricArg::AbsLog => DiffMetric::AbsLog,
            DiffMetricArg::Exact => DiffMetric::Exact,
        }
    }
}

/// CLI mirror of [`arbvis::SummaryStat`]. Kept separate from the core type
/// for the same reason as [`DiffMetricArg`] — the clap derive's
/// variant-doc strings shouldn't leak into the library API.
#[derive(Clone, Copy, Debug, ValueEnum, Default)]
pub enum SummaryStatArg {
    /// √(mean(x²)). Default — comparable across tensors of different scale.
    #[default]
    Rms,
    /// √(sum(x²)). Honest about total magnitude; varies with tensor size.
    Frobenius,
    /// mean(|x|). Stable, dominated by typical-magnitude entries.
    MeanAbs,
    /// Fraction of |x| < 1e-6. Surfaces dead / near-dead experts.
    Sparsity,
}

impl From<SummaryStatArg> for SummaryStat {
    fn from(a: SummaryStatArg) -> Self {
        match a {
            SummaryStatArg::Rms => SummaryStat::Rms,
            SummaryStatArg::Frobenius => SummaryStat::Frobenius,
            SummaryStatArg::MeanAbs => SummaryStat::MeanAbs,
            SummaryStatArg::Sparsity => SummaryStat::Sparsity,
        }
    }
}

/// CLI mirror of [`crate::data::MoeNorm`] — how each `--moe` summary panel's
/// per-cell scalars are scaled into the heatmap range. Kept separate from the
/// core enum so the clap derive's variant-doc help text stays out of the
/// library API (same rationale as [`SummaryStatArg`]).
#[derive(Clone, Copy, Debug, ValueEnum, Default)]
pub enum MoeNormArg {
    /// Linear `0 → panel-max`. Keeps a zero anchor; crushes contrast when
    /// values cluster near the max. The original behaviour.
    Max,
    /// Linear `panel-min → panel-max`. Stretches the actual range to reveal
    /// per-expert spread; drops the zero anchor.
    MinMax,
    /// Robust min-max: clip to the 2nd/98th percentiles, then stretch. Reveals
    /// the bulk spread while ignoring a lone outlier expert. Default.
    #[default]
    Percentile,
}

impl From<MoeNormArg> for crate::data::MoeNorm {
    fn from(a: MoeNormArg) -> Self {
        match a {
            MoeNormArg::Max => crate::data::MoeNorm::Max,
            MoeNormArg::MinMax => crate::data::MoeNorm::MinMax,
            MoeNormArg::Percentile => crate::data::MoeNorm::Percentile,
        }
    }
}

/// CLI choice for layout strategy. Mirrors [`arbvis::LayoutMode`].
#[derive(Clone, Copy, Debug, ValueEnum, Default)]
pub enum LayoutArg {
    /// Architectural layout if every input is safetensors with detectable
    /// structure; otherwise byte-Hilbert. Default.
    #[default]
    Auto,
    /// Force architectural (structure-aware) layout. Falls back to hilbert if
    /// no input is safetensors.
    Arch,
    /// Force the legacy global-Hilbert layout (1 px = 1 byte). Useful for
    /// non-safetensors inputs and regression-checking the old output.
    Hilbert,
}

impl From<LayoutArg> for LayoutMode {
    fn from(a: LayoutArg) -> Self {
        match a {
            LayoutArg::Auto => LayoutMode::Auto,
            LayoutArg::Arch => LayoutMode::Arch,
            LayoutArg::Hilbert => LayoutMode::Hilbert,
        }
    }
}

/// Tensor-aware visualization built on `arbvis`.
///
/// Flattens [`arbvis::Args`] (the byte-only CLI surface) and adds four
/// tensor-aware flags. Use [`ModelArgs::split`] to peel out the inner
/// `arbvis::Args` + `arbvis::ModelOpts` for the call into `arbvis::run`.
///
/// Named `ModelArgs` rather than `Args` because clap's `#[command(flatten)]`
/// creates an implicit `ArgGroup` keyed on the inner struct's type name.
/// Two structs both named `Args` on the same `Command` collide on that
/// group name and trip a debug-build assertion in clap.
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
// `--probe*` only make sense alongside the MoE viewer, expressed as a
// `requires = "moe"` on each probe flag.
pub struct ModelArgs {
    #[command(flatten)]
    pub arbvis: arbvis::Args,

    /// Visualize a single MoE model as a tabbed, multi-scene render. MODEL is
    /// a local path or hf:// URL. Two scenes are produced from one load and the
    /// viewer switches between them:
    ///
    /// • **Summary** — per-expert scalar heatmaps, one panel per weight
    ///   (gate_proj / up_proj / down_proj / router) as a layers × experts grid
    ///   (one cell per expert; scalar chosen by `--summary-stat`). Surfaces
    ///   outlier experts, layer-wise magnitude trends, and dead experts.
    ///
    /// • **CKA** — per-(layer, weight) expert-vs-expert linear-CKA similarity
    ///   matrices (`n_experts × n_experts`, diagonal 1.0); off-diagonal blocks
    ///   reveal redundant expert clusters. Uses Gaussian random projection on
    ///   the input axis (see `--cka-sample`).
    ///
    /// `--probe` adds a behavioral panel to each scene. Requires a tile
    /// destination (`--tiles` / `--space`) — the tab switcher is viewer-only.
    #[arg(
        long,
        value_name = "MODEL",
        conflicts_with_all = ["diff", "files", "file_list", "finetune", "no_finetune", "show_xet_xorbs"]
    )]
    pub moe: Option<PathBuf>,

    /// Force-treat the second --diff argument as a finetune of the first.
    /// In finetune mode, tensors present only on the base side are rendered
    /// as crosshatched grey (informational); anything present only on the
    /// finetune side (or with a mismatched shape) is treated as a contract
    /// violation and aborts the run. Without --finetune / --no-finetune the
    /// relation is auto-detected from the HF model card (`base_model` +
    /// `base_model_relation`) when both args are `hf://` model URLs, and
    /// defaults to non-finetune otherwise.
    #[arg(long, requires = "diff", conflicts_with = "no_finetune")]
    pub finetune: bool,

    /// Force-treat the diff as NOT a finetune (overrides auto-detection).
    /// In this mode, tensors/files present only on one side render as red
    /// (original-only) or green (modified-only) crosshatch.
    #[arg(long = "no-finetune", requires = "diff", conflicts_with = "finetune")]
    pub no_finetune: bool,

    /// How per-element tensor deltas are encoded for visualization
    /// (applies to `--diff`).
    #[arg(long, value_enum, default_value_t = DiffMetricArg::Rms)]
    pub diff_metric: DiffMetricArg,

    /// Which per-expert scalar the `--moe` summary scene computes from each
    /// FFN weight. `rms` (default) is comparable across tensors of different
    /// scale; `frobenius` is honest about total magnitude; `mean-abs` is
    /// stable and dominated by typical entries; `sparsity` surfaces dead
    /// experts.
    #[arg(long, value_enum, default_value_t = SummaryStatArg::Rms)]
    pub summary_stat: SummaryStatArg,

    /// How each `--moe` summary panel's per-cell scalars are scaled into the
    /// heatmap colour range (panels are normalised independently).
    /// `percentile` (default) is a robust min-max that clips the 2nd/98th
    /// percentiles first so one dominant or dead expert doesn't compress
    /// everyone else; `min-max` stretches `panel-min → panel-max` to reveal
    /// per-expert spread; `max` maps `0 → panel-max`, keeping a zero anchor
    /// but crushing contrast when experts cluster near the max.
    #[arg(long, value_enum, default_value_t = MoeNormArg::Percentile)]
    pub moe_norm: MoeNormArg,

    /// Random-projection dimension for the `--moe` CKA scene. Trades CKA estimation
    /// accuracy for compute: smaller is faster (per-pair cost is O(k² · d_out)),
    /// larger preserves CKA more accurately (the projection becomes lossless
    /// as k approaches d_in). Default 128 lands the full 24-layer × 3-weight
    /// Qwen MoE compute under a minute on a laptop and produces visually
    /// stable heatmaps. Range: 16..=4096.
    #[arg(long, value_name = "K", default_value_t = 128, value_parser = clap::value_parser!(u32).range(16..=4096))]
    pub cka_sample: u32,

    /// Run a routing-faithful forward pass on a probe input and add a
    /// behavioral panel to each `--moe` scene: a per-`(layer, expert)`
    /// routing-frequency heatmap in the summary scene, and per-layer
    /// `n_experts × n_experts` routing co-activation matrices in the CKA
    /// scene. Without `--probe-text` / `--probe-file` / `--probe-url`, uses a
    /// small bundled default snippet of varied prose / code / dialogue.
    /// Requires `--moe`. Supported architectures: `Qwen2MoeForCausalLM`,
    /// `MixtralForCausalLM`.
    #[arg(long, requires = "moe")]
    pub probe: bool,

    /// Override the bundled probe with literal text. Mutually exclusive with
    /// `--probe-file` and `--probe-url`. Implies `--probe`.
    #[arg(long, value_name = "TEXT", requires = "moe", conflicts_with_all = ["probe_file", "probe_url"])]
    pub probe_text: Option<String>,

    /// Override the bundled probe with the contents of a local UTF-8 text
    /// file. Mutually exclusive with `--probe-text` and `--probe-url`.
    /// Implies `--probe`.
    #[arg(long, value_name = "PATH", requires = "moe", conflicts_with_all = ["probe_text", "probe_url"])]
    pub probe_file: Option<PathBuf>,

    /// Override the bundled probe with text fetched from a URL. Accepts
    /// plain HTTPS URLs (returns the response body as UTF-8 text) or
    /// `hf://...` URLs (downloads via the HF Hub). Mutually exclusive with
    /// `--probe-text` and `--probe-file`. Implies `--probe`.
    #[arg(long, value_name = "URL", requires = "moe", conflicts_with_all = ["probe_text", "probe_file"])]
    pub probe_url: Option<String>,

    /// Layout strategy for arranging tensors on the canvas.
    ///
    /// `auto` (default): structure-aware layout when every input is
    /// safetensors and tensor names look transformer-style; otherwise the
    /// legacy global-Hilbert curve.
    ///
    /// `arch`: force structure-aware layout. Each tensor is rendered at its
    /// natural 2D element shape (1 px = 1 element); transformer blocks stack
    /// vertically with corresponding sub-tensors pixel-aligned across the
    /// stack. Falls back to hilbert if no input is safetensors.
    ///
    /// `hilbert`: force the legacy layout. 1 px = 1 byte along a global
    /// Hilbert curve over the concatenated source bytes. Reproduces the
    /// pre-architectural output for regression checks.
    #[arg(long, value_enum, default_value_t = LayoutArg::Auto)]
    pub layout: LayoutArg,
}

impl ModelArgs {
    /// Peel the flattened struct into the two halves `arbvis::run` takes:
    /// the byte-only `arbvis::Args` and the tensor-aware `ModelOpts`.
    pub fn split(self) -> (arbvis::Args, ModelOpts) {
        // The `--probe-*` flags imply `--probe`. The bare `--probe` falls
        // through to `ProbeSource::Default`.
        let probe_source = if let Some(text) = self.probe_text {
            ProbeSource::Text(text)
        } else if let Some(file) = self.probe_file {
            ProbeSource::File(file)
        } else if let Some(url) = self.probe_url {
            ProbeSource::Url(url)
        } else {
            ProbeSource::Default
        };
        // `--probe` plus any source override turns it on; without any of
        // them `probe.enabled = self.probe` (false unless the bare flag was
        // explicitly passed).
        let probe_enabled = self.probe
            || matches!(
                probe_source,
                ProbeSource::Text(_) | ProbeSource::File(_) | ProbeSource::Url(_)
            );
        let opts = ModelOpts {
            moe: self.moe,
            finetune: self.finetune,
            no_finetune: self.no_finetune,
            diff_metric: self.diff_metric.into(),
            summary_stat: self.summary_stat.into(),
            cka_sample: self.cka_sample,
            probe: ProbeOpts {
                enabled: probe_enabled,
                source: probe_source,
            },
            layout_mode: self.layout.into(),
        };
        (self.arbvis, opts)
    }
}
