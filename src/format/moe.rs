//! Mixture-of-Experts tensor name parsing.
//!
//! HuggingFace-style MoE checkpoints store each expert as a distinct tensor:
//!     `model.layers.{L}.mlp.experts.{E}.{gate|up|down}_proj.weight`
//!
//! This covers Mixtral, Qwen3-MoE, OLMoE, DeepSeek-V2/V3 (routed experts).
//! DeepSeek-style `mlp.shared_experts.*` are intentionally *not* matched —
//! shared experts run on every token and aren't part of the routed N×N matrix.
//!
//! GGUF fuses all experts of one layer into a single tensor named
//! `blk.{L}.ffn_{gate|up|down}_exps.weight`. Per-expert visualisation of GGUF
//! checkpoints is out of scope for v1 — `is_fused_gguf_expert` lets callers
//! detect and reject this case cleanly.
//!
//! See [`crate::format::name_map`] for the cross-format diff canonicaliser
//! (which deliberately collapses HF per-expert names to the GGUF fused form
//! for the regular `--diff` flow).
//!
//! NB: this module is the parser only. Source construction and matrix layout
//! live in [`crate::data::prepare_moe_diff_sources`] and
//! [`crate::layout::arch::ArchLayout::try_build_moe_diff`].

/// Which weight matrix of a single expert. `GateProj` / `UpProj` /
/// `DownProj` are the three per-expert FFN matrices; `Router` is the
/// layer-level router gate (`model.layers.{L}.mlp.gate.weight`) whose
/// rows are per-expert gate vectors — included so `--moe-summary` can
/// surface routing-side specialization alongside the FFN signal.
/// `Router` is not used by `--moe-diff` (the pairwise expert layout
/// only consumes the per-expert FFN slots).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[allow(clippy::enum_variant_names)]
pub enum ExpertWeight {
    GateProj,
    UpProj,
    DownProj,
    Router,
}

impl ExpertWeight {
    pub fn label(self) -> &'static str {
        match self {
            ExpertWeight::GateProj => "gate_proj",
            ExpertWeight::UpProj => "up_proj",
            ExpertWeight::DownProj => "down_proj",
            ExpertWeight::Router => "router",
        }
    }
}

/// A successfully-parsed per-expert tensor reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertRef {
    pub layer_idx: u32,
    pub expert_idx: u32,
    pub weight: ExpertWeight,
}

/// Parse an HF-style per-expert tensor name. Returns `None` for any tensor
/// that doesn't match the routed-experts pattern (top-level tensors,
/// non-MoE layers, shared experts, router gates, biases).
pub fn parse_hf_expert(name: &str) -> Option<ExpertRef> {
    // model.layers.{L}.mlp.experts.{E}.{gate|up|down}_proj.weight   (Qwen/OLMoE/…)
    // model.layers.{L}.block_sparse_moe.experts.{E}.{w1|w3|w2}.weight (classic Mixtral)
    let rest = name.strip_prefix("model.layers.")?;
    let (layer_str, rest) = rest.split_once('.')?;
    let layer_idx: u32 = layer_str.parse().ok()?;

    // The MoE block prefix is `mlp.experts.` for most HF layouts (Qwen3,
    // OLMoE, DeepSeek routed experts) and `block_sparse_moe.experts.` for the
    // classic published Mixtral layout. DeepSeek's `mlp.shared_experts.*` does
    // NOT match — those aren't routed.
    let rest = match rest.strip_prefix("mlp.experts.") {
        Some(r) => r,
        None => rest.strip_prefix("block_sparse_moe.experts.")?,
    };

    let (expert_str, leaf) = rest.split_once('.')?;
    let expert_idx: u32 = expert_str.parse().ok()?;

    // Mixtral names its SwiGLU matrices w1/w3/w2 (gate/up/down); everything
    // else uses the explicit *_proj names. Map both onto the same slots.
    let weight = match leaf {
        "gate_proj.weight" | "w1.weight" => ExpertWeight::GateProj,
        "up_proj.weight" | "w3.weight" => ExpertWeight::UpProj,
        "down_proj.weight" | "w2.weight" => ExpertWeight::DownProj,
        _ => return None,
    };

    Some(ExpertRef {
        layer_idx,
        expert_idx,
        weight,
    })
}

/// Parse an HF-style router-gate tensor name. Returns the layer index when
/// `name` matches `model.layers.{L}.mlp.gate.weight` (most HF layouts) or
/// `model.layers.{L}.block_sparse_moe.gate.weight` (classic Mixtral) — the
/// per-layer MoE router whose rows are per-expert gate vectors. Returns
/// `None` for anything else (including the per-expert `gate_proj.weight`,
/// which [`parse_hf_expert`] handles).
///
/// The router and per-expert tensors don't collide: this fn looks for the
/// layer-level `*.gate.weight` exactly, while [`parse_hf_expert`] requires an
/// `experts.{E}.…` segment (the `experts.` prefix is the discriminator).
pub fn parse_hf_router(name: &str) -> Option<u32> {
    let rest = name.strip_prefix("model.layers.")?;
    let (layer_str, rest) = rest.split_once('.')?;
    let layer_idx: u32 = layer_str.parse().ok()?;
    if rest == "mlp.gate.weight" || rest == "block_sparse_moe.gate.weight" {
        Some(layer_idx)
    } else {
        None
    }
}

/// One of the two batched fused-expert tensors in the newer `transformers`
/// MoE export, where all experts of a layer share a single parameter rather
/// than one tensor per expert.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum FusedExpertTensor {
    /// `mlp.experts.gate_up_proj`, shape `[n_experts, 2·intermediate, hidden]`.
    /// The first `intermediate` rows of dim 1 are the gate matrix, the next
    /// `intermediate` rows are the up matrix (concatenated along the output
    /// dim) — matching how [`crate::probe::mixtral`] slices it for the forward.
    GateUp,
    /// `mlp.experts.down_proj`, shape `[n_experts, hidden, intermediate]`.
    Down,
}

/// Parse a batched fused-expert tensor name from the newer `transformers`
/// MoE export. Returns the layer index and which of the two batched tensors
/// it is, for `model.layers.{L}.mlp.experts.{gate_up_proj|down_proj}` (with
/// or without a trailing `.weight`). Returns `None` for anything else.
///
/// This is the fused counterpart to [`parse_hf_expert`] (one tensor *per*
/// expert): the two never collide because [`parse_hf_expert`] requires an
/// `experts.{E}.…` numeric index, which the batched names lack. Unlike
/// [`is_fused_gguf_expert`] — which exists only to *reject* GGUF fusion —
/// this layout *is* sliceable into per-expert byte ranges, so callers use it
/// to build per-expert scalar jobs.
pub fn parse_hf_fused_expert(name: &str) -> Option<(u32, FusedExpertTensor)> {
    let rest = name.strip_prefix("model.layers.")?;
    let (layer_str, rest) = rest.split_once('.')?;
    let layer_idx: u32 = layer_str.parse().ok()?;
    let leaf = rest.strip_prefix("mlp.experts.")?;
    let kind = match leaf {
        "gate_up_proj" | "gate_up_proj.weight" => FusedExpertTensor::GateUp,
        "down_proj" | "down_proj.weight" => FusedExpertTensor::Down,
        _ => return None,
    };
    Some((layer_idx, kind))
}

/// Whether `name` is a GGUF fused-expert tensor (`ffn_{gate|up|down}_exps.weight`,
/// optionally under `blk.{L}.`). Used by callers to bail with a clear error
/// before attempting per-expert layout on a GGUF checkpoint.
pub fn is_fused_gguf_expert(name: &str) -> bool {
    let leaf = match name.strip_prefix("blk.") {
        Some(rest) => rest.split_once('.').map(|(_, x)| x).unwrap_or(rest),
        None => name,
    };
    matches!(
        leaf,
        "ffn_gate_exps.weight" | "ffn_up_exps.weight" | "ffn_down_exps.weight"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_qwen3_moe_expert() {
        let r = parse_hf_expert("model.layers.5.mlp.experts.12.gate_proj.weight").unwrap();
        assert_eq!(r.layer_idx, 5);
        assert_eq!(r.expert_idx, 12);
        assert_eq!(r.weight, ExpertWeight::GateProj);

        let r = parse_hf_expert("model.layers.0.mlp.experts.0.up_proj.weight").unwrap();
        assert_eq!(r.layer_idx, 0);
        assert_eq!(r.expert_idx, 0);
        assert_eq!(r.weight, ExpertWeight::UpProj);

        let r = parse_hf_expert("model.layers.31.mlp.experts.63.down_proj.weight").unwrap();
        assert_eq!(r.layer_idx, 31);
        assert_eq!(r.expert_idx, 63);
        assert_eq!(r.weight, ExpertWeight::DownProj);
    }

    #[test]
    fn rejects_shared_experts() {
        // DeepSeek-style shared experts run on every token — not part of
        // the routed N×N matrix.
        assert_eq!(
            parse_hf_expert("model.layers.3.mlp.shared_experts.gate_proj.weight"),
            None,
        );
        assert_eq!(
            parse_hf_expert("model.layers.3.mlp.shared_experts.0.gate_proj.weight"),
            None,
        );
    }

    #[test]
    fn rejects_router_and_norms() {
        // Router gate (`mlp.gate.weight` in HF / `ffn_gate_inp.weight` in GGUF).
        assert_eq!(parse_hf_expert("model.layers.0.mlp.gate.weight"), None);
        // Non-MoE dense MLP.
        assert_eq!(parse_hf_expert("model.layers.0.mlp.gate_proj.weight"), None,);
        // Norms / attention / top-level singletons.
        assert_eq!(
            parse_hf_expert("model.layers.0.input_layernorm.weight"),
            None,
        );
        assert_eq!(
            parse_hf_expert("model.layers.0.self_attn.q_proj.weight"),
            None,
        );
        assert_eq!(parse_hf_expert("model.embed_tokens.weight"), None);
        assert_eq!(parse_hf_expert("lm_head.weight"), None);
    }

    #[test]
    fn parses_mixtral_classic_expert() {
        // Classic published Mixtral: block_sparse_moe.experts.{E}.{w1,w3,w2}.
        // w1 = gate, w3 = up, w2 = down.
        let r = parse_hf_expert("model.layers.5.block_sparse_moe.experts.7.w1.weight").unwrap();
        assert_eq!(r.layer_idx, 5);
        assert_eq!(r.expert_idx, 7);
        assert_eq!(r.weight, ExpertWeight::GateProj);

        let r = parse_hf_expert("model.layers.0.block_sparse_moe.experts.0.w3.weight").unwrap();
        assert_eq!(r.weight, ExpertWeight::UpProj);

        let r = parse_hf_expert("model.layers.31.block_sparse_moe.experts.7.w2.weight").unwrap();
        assert_eq!(r.weight, ExpertWeight::DownProj);

        // `w1` under the `mlp.experts.` prefix is not a thing, but the leaf
        // mapping is shared — guard that a bogus leaf still rejects.
        assert_eq!(
            parse_hf_expert("model.layers.0.block_sparse_moe.experts.0.w4.weight"),
            None,
        );
    }

    #[test]
    fn rejects_non_weight_leaves() {
        assert_eq!(
            parse_hf_expert("model.layers.0.mlp.experts.0.gate_proj.bias"),
            None,
        );
        assert_eq!(
            parse_hf_expert("model.layers.0.block_sparse_moe.experts.0.w1.bias"),
            None,
        );
        // The fused batched expert tensors carry no per-expert index → no match.
        assert_eq!(
            parse_hf_expert("model.layers.0.mlp.experts.gate_up_proj"),
            None,
        );
        assert_eq!(
            parse_hf_expert("model.layers.0.mlp.experts.down_proj"),
            None,
        );
    }

    #[test]
    fn rejects_unparseable_indices() {
        assert_eq!(
            parse_hf_expert("model.layers.x.mlp.experts.0.gate_proj.weight"),
            None,
        );
        assert_eq!(
            parse_hf_expert("model.layers.0.mlp.experts.y.gate_proj.weight"),
            None,
        );
    }

    #[test]
    fn parses_router_gate() {
        assert_eq!(parse_hf_router("model.layers.0.mlp.gate.weight"), Some(0));
        assert_eq!(parse_hf_router("model.layers.31.mlp.gate.weight"), Some(31));
        // Classic Mixtral router lives under block_sparse_moe.
        assert_eq!(
            parse_hf_router("model.layers.0.block_sparse_moe.gate.weight"),
            Some(0),
        );
        assert_eq!(
            parse_hf_router("model.layers.7.block_sparse_moe.gate.weight"),
            Some(7),
        );
    }

    #[test]
    fn router_rejects_expert_tensors() {
        // The per-expert `gate_proj.weight` lives under `mlp.experts.{E}.` —
        // not the same as the layer-level `mlp.gate.weight`. The two
        // parsers must not overlap.
        assert_eq!(
            parse_hf_router("model.layers.0.mlp.experts.0.gate_proj.weight"),
            None,
        );
        // Dense MLPs (non-MoE layers in a mixed-arch model) also use
        // `mlp.gate_proj.weight` — that's not the router either.
        assert_eq!(parse_hf_router("model.layers.0.mlp.gate_proj.weight"), None);
        // Biases / norms / attention.
        assert_eq!(parse_hf_router("model.layers.0.mlp.gate.bias"), None);
        assert_eq!(parse_hf_router("model.layers.0.input_layernorm.weight"), None);
        assert_eq!(parse_hf_router("model.layers.0.self_attn.q_proj.weight"), None);
    }

    #[test]
    fn router_rejects_unparseable_indices() {
        assert_eq!(parse_hf_router("model.layers.x.mlp.gate.weight"), None);
    }

    #[test]
    fn parses_hf_fused_experts() {
        // Newer transformers export: batched per-layer expert tensors.
        assert_eq!(
            parse_hf_fused_expert("model.layers.0.mlp.experts.gate_up_proj"),
            Some((0, FusedExpertTensor::GateUp)),
        );
        assert_eq!(
            parse_hf_fused_expert("model.layers.31.mlp.experts.down_proj"),
            Some((31, FusedExpertTensor::Down)),
        );
        // Tolerate a trailing `.weight` (some exports keep it).
        assert_eq!(
            parse_hf_fused_expert("model.layers.7.mlp.experts.gate_up_proj.weight"),
            Some((7, FusedExpertTensor::GateUp)),
        );
    }

    #[test]
    fn fused_parser_does_not_collide_with_per_expert() {
        // The per-expert parser must reject the batched names…
        assert_eq!(
            parse_hf_expert("model.layers.0.mlp.experts.gate_up_proj"),
            None,
        );
        assert_eq!(parse_hf_expert("model.layers.0.mlp.experts.down_proj"), None);
        // …and the fused parser must reject the indexed per-expert names.
        assert_eq!(
            parse_hf_fused_expert("model.layers.0.mlp.experts.0.gate_proj.weight"),
            None,
        );
        assert_eq!(
            parse_hf_fused_expert("model.layers.0.block_sparse_moe.experts.0.w1.weight"),
            None,
        );
        // Router and biases are not fused-expert tensors.
        assert_eq!(parse_hf_fused_expert("model.layers.0.mlp.gate.weight"), None);
        assert_eq!(
            parse_hf_fused_expert("model.layers.0.mlp.experts.gate_up_proj_bias"),
            None,
        );
        assert_eq!(parse_hf_fused_expert("model.layers.x.mlp.experts.down_proj"), None);
    }

    #[test]
    fn detects_gguf_fused_experts() {
        assert!(is_fused_gguf_expert("blk.0.ffn_gate_exps.weight"));
        assert!(is_fused_gguf_expert("blk.31.ffn_up_exps.weight"));
        assert!(is_fused_gguf_expert("blk.15.ffn_down_exps.weight"));
        // Bare-leaf form (canonicaliser strips the `blk.{N}.` prefix
        // before lookup).
        assert!(is_fused_gguf_expert("ffn_gate_exps.weight"));
    }

    #[test]
    fn fused_detector_rejects_unrelated() {
        assert!(!is_fused_gguf_expert("blk.0.ffn_gate.weight"));
        assert!(!is_fused_gguf_expert("blk.0.attn_q.weight"));
        assert!(!is_fused_gguf_expert(
            "model.layers.0.mlp.experts.0.gate_proj.weight"
        ));
    }
}
