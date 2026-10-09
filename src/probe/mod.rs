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

/// `--probe` configuration: whether the routing-faithful forward pass runs,
/// and where its probe text comes from. Carried as a field on
/// [`crate::hooks::MoeSceneProvider`]. arbvis is probe-agnostic — this is a
/// modelweightvis concept.
#[derive(Clone, Debug, Default)]
pub struct ProbeOpts {
    pub enabled: bool,
    pub source: ProbeSource,
}

/// Where the probe text comes from. `Default` uses a small bundled snippet
/// (~300 tokens of varied prose / code / dialogue); the others are the
/// mutually-exclusive `--probe-text` / `--probe-file` / `--probe-url` overrides.
#[derive(Clone, Debug, Default)]
pub enum ProbeSource {
    #[default]
    Default,
    Text(String),
    File(PathBuf),
    Url(String),
}

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
    /// The HF `architectures` name this variant was detected from.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_arch_matches_supported_names() {
        for (name, want) in [
            ("Qwen2MoeForCausalLM", Some(Arch::Qwen2Moe)),
            ("MixtralForCausalLM", Some(Arch::Mixtral)),
        ] {
            let cfg = ModelConfig::from_bytes(
                format!("{{\"architectures\": [\"{name}\"]}}").as_bytes(),
            )
            .expect("parses");
            assert_eq!(detect_arch(&cfg), want);
        }
    }

    #[test]
    fn detect_arch_rejects_unsupported_or_missing() {
        let cfg =
            ModelConfig::from_bytes(br#"{"architectures": ["LlamaForCausalLM"]}"#).expect("parses");
        assert_eq!(detect_arch(&cfg), None);
        let empty = ModelConfig::from_bytes(br#"{"architectures": []}"#).expect("parses");
        assert_eq!(detect_arch(&empty), None);
    }

    #[test]
    fn arch_labels_are_the_hf_config_strings() {
        assert_eq!(Arch::Qwen2Moe.label(), "Qwen2MoeForCausalLM");
        assert_eq!(Arch::Mixtral.label(), "MixtralForCausalLM");
    }

    #[test]
    fn probe_opts_default_is_disabled_default_source() {
        let opts = ProbeOpts::default();
        assert!(!opts.enabled);
        assert!(matches!(opts.source, ProbeSource::Default));
    }

    #[test]
    fn routing_capture_invariants_hold_on_synthetic_data() {
        // The docs promise a symmetric coact matrix whose diagonal equals
        // freq; check the layout contract on a small synthetic capture.
        let (layers, experts, tokens) = (2u32, 4u32, 8u32);
        let n = (layers * experts) as usize;
        let mut freq = vec![0.0f32; n];
        let mut coact = vec![0.0f32; (layers * experts * experts) as usize];
        for l in 0..layers as usize {
            for e in 0..experts as usize {
                freq[l * experts as usize + e] = (e as f32 + 1.0) / tokens as f32;
                for j in 0..experts as usize {
                    coact[l * experts as usize * experts as usize + e * experts as usize + j] =
                        if e == j { freq[l * experts as usize + e] } else { 0.5 };
                }
            }
        }
        let cap = RoutingCapture { n_layers: layers, n_experts: experts, n_tokens: tokens, freq, coact };
        let e = cap.n_experts as usize;
        for l in 0..cap.n_layers as usize {
            for i in 0..e {
                for j in 0..e {
                    let cell = cap.coact[l * e * e + i * e + j];
                    let mirror = cap.coact[l * e * e + j * e + i];
                    assert_eq!(cell, mirror, "coact symmetric at layer {l} ({i},{j})");
                }
                assert_eq!(
                    cap.coact[l * e * e + i * e + i],
                    cap.freq[l * e + i],
                    "coact diagonal equals freq at layer {l}, expert {i}"
                );
            }
        }
    }
}
