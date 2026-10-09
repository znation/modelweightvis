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
//! `blk.{L}.ffn_{gate|up|down}_exps.weight`. [`parse_gguf_fused_expert`] maps
//! those onto the same `(layer, ExpertWeight)` slots as the HF parsers, so the
//! `--moe` summary scene can slice them into per-expert byte ranges;
//! [`parse_gguf_router`] does the same for the per-layer router
//! `blk.{L}.ffn_gate_inp.weight`.
//!
//! See [`crate::format::name_map`] for the cross-format diff canonicaliser
//! (which deliberately collapses HF per-expert names to the GGUF fused form
//! for the regular `--diff` flow).
//!
//! NB: this module is the parser only. Source construction and panel layout
//! live in [`crate::data::build_moe_summary_sources`] /
//! [`crate::data::build_moe_cka_sources`] and the matching
//! `ArchLayout::try_build_moe_{summary,cka}` builders.

/// Which weight matrix of a single expert. `GateProj` / `UpProj` /
/// `DownProj` are the three per-expert FFN matrices; `Router` is the
/// layer-level router gate (`model.layers.{L}.mlp.gate.weight`) whose
/// rows are per-expert gate vectors — included so `--moe-summary` can
/// surface routing-side specialization alongside the FFN signal.
/// `Router` is not used by `--moe-cka` (pairwise expert CKA only
/// consumes the per-expert FFN slots).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[allow(clippy::enum_variant_names)]
pub enum ExpertWeight {
    GateProj,
    UpProj,
    DownProj,
    Router,
}

impl ExpertWeight {
    /// The canonical HF module-name suffix for this expert weight.
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
    let (layer_idx, rest) = hf_layer_leaf(name)?;

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
    let (layer_idx, rest) = hf_layer_leaf(name)?;
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
    let (layer_idx, rest) = hf_layer_leaf(name)?;
    let leaf = rest.strip_prefix("mlp.experts.")?;
    let kind = match leaf {
        "gate_up_proj" | "gate_up_proj.weight" => FusedExpertTensor::GateUp,
        "down_proj" | "down_proj.weight" => FusedExpertTensor::Down,
        _ => return None,
    };
    Some((layer_idx, kind))
}

/// Parse a GGUF fused-expert tensor name. Returns the layer index and which
/// expert weight the tensor holds, for `blk.{L}.ffn_{gate|up|down}_exps.weight`.
/// The bare-leaf form (no `blk.{L}.` prefix, as produced by the diff
/// canonicaliser's prefix-stripping) maps to layer 0. Returns `None` for
/// anything else — including the per-tensor (non-fused) `ffn_{gate|up|down}.weight`,
/// which lacks the `_exps` suffix, and non-expert GGUF tensors.
///
/// This is the GGUF counterpart to [`parse_hf_expert`] and [`parse_hf_fused_expert`];
/// the three never collide because the GGUF names use `blk.{N}.ffn_*` addressing
/// and the `_exps` suffix, while the HF names require `model.layers.{N}.mlp.experts.…`.
pub fn parse_gguf_fused_expert(name: &str) -> Option<(u32, ExpertWeight)> {
    let (layer_idx, leaf) = gguf_layer_leaf(name)?;
    let weight = match leaf {
        "ffn_gate_exps.weight" => ExpertWeight::GateProj,
        "ffn_up_exps.weight" => ExpertWeight::UpProj,
        "ffn_down_exps.weight" => ExpertWeight::DownProj,
        _ => return None,
    };
    Some((layer_idx, weight))
}

/// Parse a GGUF router-gate tensor name. Returns the layer index for
/// `blk.{L}.ffn_gate_inp.weight` (the per-layer MoE router whose rows are
/// per-expert gate vectors). The bare-leaf form maps to layer 0, as in
/// [`parse_gguf_fused_expert`]. Returns `None` for anything else — including
/// the per-expert `ffn_gate.weight`/`ffn_gate_exps.weight` (different leaves,
/// handled by other parsers or ignored) and attention tensors.
pub fn parse_gguf_router(name: &str) -> Option<u32> {
    let (layer_idx, leaf) = gguf_layer_leaf(name)?;
    (leaf == "ffn_gate_inp.weight").then_some(layer_idx)
}

/// Split a GGUF tensor name into `(layer_idx, leaf)`. `blk.{L}.{leaf}` yields
/// the parsed layer index; a bare leaf (no `blk.` prefix) yields layer 0.
/// Returns `None` for a `blk.` prefix whose layer segment isn't numeric.
fn gguf_layer_leaf(name: &str) -> Option<(u32, &str)> {
    match name.strip_prefix("blk.") {
        Some(rest) => {
            let (l, leaf) = rest.split_once('.')?;
            Some((l.parse().ok()?, leaf))
        }
        None => Some((0, name)),
    }
}

/// Split an HF tensor name into `(layer_idx, rest)` at the layer segment:
/// `model.layers.{L}.{rest}` with a numeric `{L}`. Used by the three HF MoE
/// parsers ([`parse_hf_expert`], [`parse_hf_router`],
/// [`parse_hf_fused_expert`]) — the HF counterpart to [`gguf_layer_leaf`].
/// Returns `None` for anything without the `model.layers.` prefix or a
/// non-numeric layer segment.
fn hf_layer_leaf(name: &str) -> Option<(u32, &str)> {
    let rest = name.strip_prefix("model.layers.")?;
    let (l, leaf) = rest.split_once('.')?;
    Some((l.parse().ok()?, leaf))
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
        assert_eq!(
            parse_hf_router("model.layers.0.input_layernorm.weight"),
            None
        );
        assert_eq!(
            parse_hf_router("model.layers.0.self_attn.q_proj.weight"),
            None
        );
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
        assert_eq!(
            parse_hf_expert("model.layers.0.mlp.experts.down_proj"),
            None
        );
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
        assert_eq!(
            parse_hf_fused_expert("model.layers.0.mlp.gate.weight"),
            None
        );
        assert_eq!(
            parse_hf_fused_expert("model.layers.0.mlp.experts.gate_up_proj_bias"),
            None,
        );
        assert_eq!(
            parse_hf_fused_expert("model.layers.x.mlp.experts.down_proj"),
            None
        );
    }

    #[test]
    fn parses_gguf_fused_experts() {
        let (l, w) = parse_gguf_fused_expert("blk.0.ffn_gate_exps.weight").unwrap();
        assert_eq!((l, w), (0, ExpertWeight::GateProj));
        let (l, w) = parse_gguf_fused_expert("blk.31.ffn_up_exps.weight").unwrap();
        assert_eq!((l, w), (31, ExpertWeight::UpProj));
        let (l, w) = parse_gguf_fused_expert("blk.15.ffn_down_exps.weight").unwrap();
        assert_eq!((l, w), (15, ExpertWeight::DownProj));
        // Bare-leaf form (canonicaliser strips the `blk.{N}.` prefix
        // before lookup) maps to layer 0.
        assert_eq!(
            parse_gguf_fused_expert("ffn_gate_exps.weight"),
            Some((0, ExpertWeight::GateProj))
        );
    }

    #[test]
    fn parses_gguf_router() {
        assert_eq!(parse_gguf_router("blk.7.ffn_gate_inp.weight"), Some(7));
        assert_eq!(parse_gguf_router("ffn_gate_inp.weight"), Some(0));
        // Per-tensor (non-fused) and attention tensors never match.
        assert_eq!(parse_gguf_router("blk.7.ffn_gate.weight"), None);
        assert_eq!(parse_gguf_router("blk.7.attn_q.weight"), None);
        // Non-numeric layer segment is rejected.
        assert_eq!(parse_gguf_router("blk.x.ffn_gate_inp.weight"), None);
    }

    #[test]
    fn gguf_parsers_reject_unrelated_and_dont_collide_with_hf() {
        for name in [
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
            "blk.0.ffn_down.weight",
            "blk.0.ffn_gate_exps",       // missing .weight suffix
            "blk.0.attn_q.weight",
            "blk.0.ffn_gate_inp_ff.weight",
            "model.layers.0.mlp.experts.0.gate_proj.weight",
            "token_embd.weight",
        ] {
            assert_eq!(parse_gguf_fused_expert(name), None, "{name}");
            assert_eq!(parse_gguf_router(name), None, "{name}");
        }
        // HF-style names parse under the HF parsers, not the GGUF ones, and
        // vice versa: the two families never cross-match.
        let hf = "model.layers.3.mlp.experts.1.gate_proj.weight";
        assert!(parse_hf_expert(hf).is_some());
        assert_eq!(parse_gguf_fused_expert(hf), None);
        let gguf = "blk.3.ffn_gate_exps.weight";
        assert!(parse_gguf_fused_expert(gguf).is_some());
        assert_eq!(parse_hf_expert(gguf), None);
        assert_eq!(parse_hf_router(gguf), None);
        assert_eq!(parse_hf_router("blk.3.ffn_gate_inp.weight"), None);
        assert_eq!(parse_hf_fused_expert(gguf), None);
    }
}
