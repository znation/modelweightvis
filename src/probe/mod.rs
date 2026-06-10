//! Routing-faithful forward pass on a probe input, used by `--probe`
//! to add behavioral panels (e.g. routing frequency) to `--moe-summary`.
//!
//! The reason this module exists at all rather than calling
//! `candle-transformers`: the MoE model impls there (Mixtral,
//! Qwen3-MoE, DeepSeek-V2) keep their per-layer and router internals
//! private, with the forward sealed in a monolithic method that
//! returns only the final logits. Capturing per-layer router decisions
//! would need a fork. Instead, this module reimplements just enough
//! forward path — embedding, GQA attention, RMSNorm, top-k routing
//! and expert MLPs — on top of `candle-nn` primitives, with the
//! router decisions captured between layers.
//!
//! The forward is *routing-faithful*: it runs the top-k experts the
//! router actually picks (plus the shared expert on Qwen2-MoE) so the
//! residual stream feeding layer N is the same as it would be in a
//! production inference pass. Skipping experts would give exact layer-0
//! routing but progressively biased decisions in deeper layers, so we
//! pay the expert FLOPs.
//!
//! Cost on Qwen1.5-MoE-A2.7B at ~300 probe tokens: ~600 GFLOPs total
//! — well under a minute on a laptop CPU with BLAS.

use std::path::{Path, PathBuf};

pub mod common;
pub mod mixtral;
pub mod qwen2_moe;
pub mod text;

use crate::layout::model_config::ModelConfig;

/// Architectures the routing-faithful forward supports. New entries get
/// a sibling module (`probe/{arch}.rs`) plus a branch in
/// [`detect_arch`] / [`run`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arch {
    /// `Qwen2MoeForCausalLM` — Qwen1.5-MoE family. 60-expert routed
    /// FFN + a parallel "shared expert" that runs on every token.
    /// Top-k = 4 typical.
    Qwen2Moe,
    /// `MixtralForCausalLM` — Mixtral-8x7B and friends. 8-expert
    /// routed FFN, no shared expert. Top-k = 2 typical. Optional
    /// sliding-window attention.
    Mixtral,
}

impl Arch {
    pub fn label(self) -> &'static str {
        match self {
            Arch::Qwen2Moe => "Qwen2MoeForCausalLM",
            Arch::Mixtral => "MixtralForCausalLM",
        }
    }
}

/// Map a HF `config.json`'s `architectures[0]` string to a probe arch.
/// Returns `None` when the architecture isn't supported by this module
/// — the caller produces a clear error pointing at `--probe`'s
/// supported-arch list.
pub fn detect_arch(config: &ModelConfig) -> Option<Arch> {
    let first = config.architectures.first()?;
    match first.as_str() {
        "Qwen2MoeForCausalLM" => Some(Arch::Qwen2Moe),
        "MixtralForCausalLM" => Some(Arch::Mixtral),
        _ => None,
    }
}

/// Per-(layer, expert) routing-frequency capture from one probe forward.
/// `freq[layer * n_experts + expert]` is the fraction of probe tokens
/// whose top-k routing decisions for `layer` included `expert`. Range
/// `[0, k / n_experts]` in expectation if routing were uniform; real
/// MoEs concentrate on a subset.
///
/// `coact` carries the per-layer routing *co-activation* matrix used by
/// `--moe-cka --probe`: `coact[layer * n_experts^2 + i * n_experts + j]`
/// is the fraction of probe tokens whose top-k for `layer` included
/// **both** expert `i` and `j`. It is symmetric, and the diagonal
/// `coact[.. + i * n_experts + i]` equals expert `i`'s routing frequency
/// (`freq`). Range `[0, 1]` per cell.
#[derive(Debug, Clone)]
pub struct RoutingCapture {
    pub n_layers: u32,
    pub n_experts: u32,
    pub n_tokens: u32,
    pub freq: Vec<f32>,
    pub coact: Vec<f32>,
}

/// Run the routing-faithful forward pass on `probe_text`, returning
/// per-`(layer, expert)` routing frequency.
///
/// Resolves the tokenizer from `tokenizer.json` next to the model
/// weights, encodes the probe, instantiates the architecture-specific
/// forward (loading weights via `candle-nn`'s safetensors `VarBuilder`
/// from `weight_paths`), captures router top-k decisions per layer,
/// then aggregates into per-expert frequencies.
///
/// `weight_paths` is the list of `model-XXXXX-of-NNNNN.safetensors`
/// shard paths, ordered for consistent VarBuilder loading. `model_dir`
/// is where `tokenizer.json` lives.
pub fn run(
    arch: Arch,
    model_dir: &Path,
    weight_paths: &[PathBuf],
    config: &ModelConfig,
    probe_text: &str,
) -> anyhow::Result<RoutingCapture> {
    // Tokenize the probe text using the model's own tokenizer.
    let token_ids = text::tokenize(probe_text, model_dir)?;
    if token_ids.is_empty() {
        anyhow::bail!("probe: tokenizer produced 0 tokens from the probe input");
    }
    log::info!(
        "probe: arch={} ({:?}), probe text → {} tokens, {} shard(s)",
        arch.label(),
        arch,
        token_ids.len(),
        weight_paths.len(),
    );

    match arch {
        Arch::Qwen2Moe => qwen2_moe::run(config, weight_paths, model_dir, &token_ids),
        Arch::Mixtral => mixtral::run(config, weight_paths, model_dir, &token_ids),
    }
}
