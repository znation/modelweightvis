//! Routing-faithful forward pass for `Qwen2MoeForCausalLM`.
//!
//! Implements just enough of the architecture to drive the per-layer
//! routing decisions on a probe input. Loads safetensors weights via
//! `candle-nn`'s `VarBuilder` and walks the standard pre-norm
//! transformer block:
//!
//! ```text
//! x = embed_tokens[input_ids]
//! for layer L in 0..N:
//!     # Attention (with QKV biases — Qwen's quirk)
//!     residual = x
//!     x = rms_norm(x, layers[L].input_layernorm)
//!     attn_out = gqa_attention(x, layers[L].self_attn)
//!     x = residual + attn_out
//!
//!     # MoE
//!     residual = x
//!     x = rms_norm(x, layers[L].post_attention_layernorm)
//!     router_logits = x @ gate.T             # [B, S, n_experts]
//!     (topk_w, topk_i) = top_k(softmax(router_logits), top_k)
//!     # CAPTURE topk_i for routing-frequency aggregation
//!     moe_out = sum_k topk_w[..,k] * experts[topk_i[..,k]](x)
//!     shared_out = sigmoid(shared_gate(x)) * shared_expert(x)
//!     x = residual + moe_out + shared_out
//! ```
//!
//! The final norm + `lm_head` projection are skipped — we don't care
//! about token-level logits, only the per-layer routing decisions
//! recorded along the way.

use std::path::Path;
use std::path::PathBuf;

use anyhow::Context;
use candle_core::{DType, Device, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

use crate::layout::model_config::ModelConfig;
use crate::probe::common::{
    accumulate_coactivation, dispatch_experts, renormalize_topk, rms_norm, topk_per_row,
    GqaAttention, RotaryEmbedding, SwiGluExpert,
};
use crate::probe::RoutingCapture;

/// Hyperparameters extracted from `config.json` and frozen per run.
#[derive(Debug, Clone)]
struct Cfg {
    vocab_size: usize,
    hidden_size: usize,
    n_layers: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    n_experts: usize,
    top_k: usize,
    moe_intermediate_size: usize,
    shared_intermediate_size: usize,
    rms_norm_eps: f64,
    rope_theta: f64,
    max_pos: usize,
    /// When true, re-normalise `topk_weights` to sum to 1 along the
    /// top-k axis. Qwen1.5-MoE has this false (per its `config.json`);
    /// other Qwen-MoE-shape models may flip it.
    norm_topk_prob: bool,
}

impl Cfg {
    fn from_config(config: &ModelConfig, norm_topk_prob: bool) -> anyhow::Result<Self> {
        let hidden_size = config
            .hidden_size
            .context("Qwen2-MoE config missing hidden_size")? as usize;
        let n_heads = config
            .num_attention_heads
            .context("Qwen2-MoE config missing num_attention_heads")?
            as usize;
        // Qwen omits head_dim from config; derive from hidden_size / n_heads.
        let head_dim = config
            .head_dim
            .map(|d| d as usize)
            .unwrap_or(hidden_size / n_heads);
        let n_kv_heads = config
            .num_key_value_heads
            .map(|n| n as usize)
            .unwrap_or(n_heads);
        let n_experts = config
            .n_experts()
            .context("Qwen2-MoE config missing num_experts")? as usize;
        let top_k = config
            .num_experts_per_tok
            .context("Qwen2-MoE config missing num_experts_per_tok")? as usize;
        let moe_intermediate_size = config
            .moe_intermediate_size
            .context("Qwen2-MoE config missing moe_intermediate_size")?
            as usize;
        let shared_intermediate_size = config
            .shared_expert_intermediate_size
            .or(config.intermediate_size)
            .context("Qwen2-MoE config missing shared_expert_intermediate_size")?
            as usize;
        Ok(Self {
            vocab_size: config.vocab_size.unwrap_or(0) as usize,
            hidden_size,
            n_layers: config
                .num_hidden_layers
                .context("Qwen2-MoE config missing num_hidden_layers")?
                as usize,
            n_heads,
            n_kv_heads,
            head_dim,
            n_experts,
            top_k,
            moe_intermediate_size,
            shared_intermediate_size,
            rms_norm_eps: config.rms_norm_eps.unwrap_or(1e-6),
            rope_theta: config.rope_theta.unwrap_or(1_000_000.0),
            max_pos: config.max_position_embeddings.unwrap_or(8192) as usize,
            norm_topk_prob,
        })
    }
}

/// Public entry point. Loads the model weights from `weight_paths`,
/// runs the routing-faithful forward on `token_ids`, and returns the
/// per-`(layer, expert)` routing-frequency capture.
pub fn run(
    config: &ModelConfig,
    weight_paths: &[PathBuf],
    _model_dir: &Path,
    token_ids: &[u32],
) -> anyhow::Result<RoutingCapture> {
    let cfg = Cfg::from_config(config, /* norm_topk_prob */ false)?;
    let device = Device::Cpu;

    // Cap the probe at `max_pos` to avoid RoPE/attention issues. Real
    // probes are ~300 tokens; this is just a safety guard.
    let n_tokens = token_ids.len().min(cfg.max_pos);
    let input_ids = Tensor::from_slice(&token_ids[..n_tokens], (1, n_tokens), &device)?;

    // candle-nn VarBuilder over the model shards. The model was saved
    // in bf16, but candle's CPU matmul doesn't accept bf16 operands so
    // we load as f32. Naively materialising all layers up front blows
    // past the memory budget (~50 GB of routed-expert weights alone on
    // Qwen1.5-MoE-A2.7B in f32). Instead, we load one layer at a time
    // inside the forward loop — peak working set is bounded by a single
    // layer's f32 weights (~2 GB) plus the embedding table and activations.
    let dtype = DType::F32;
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(weight_paths, dtype, &device)
            .context("Qwen2-MoE: opening safetensors shards via VarBuilder")?
    };

    let embed_tokens =
        candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;
    let rope = RotaryEmbedding::new(cfg.head_dim, cfg.max_pos, cfg.rope_theta, &device)?;

    let mut hidden = embed_tokens.forward(&input_ids)?;

    // Per-layer routing-decision counts. `counts[layer * n_experts + expert]`
    // increments each time `expert` appears in top-k for any token in that
    // layer. Divide by `n_tokens` at the end for frequencies.
    let mut counts = vec![0u32; cfg.n_layers * cfg.n_experts];
    // Per-layer routing co-occurrence counts (`--moe-cka --probe`).
    let mut coact_counts = vec![0u32; cfg.n_layers * cfg.n_experts * cfg.n_experts];

    let layers_vb = vb.pp("model.layers");
    let pb = indicatif::ProgressBar::new(cfg.n_layers as u64);
    pb.set_message("probe: layer forward");
    for layer_idx in 0..cfg.n_layers {
        // Materialise this layer's weights on entry; they drop at the end
        // of the iteration so the next layer doesn't pile on top.
        let layer = Layer::load(&layers_vb.pp(layer_idx), &cfg)?;
        // --- Attention ----------------------------------------------------
        let residual = hidden.clone();
        let x = rms_norm(&hidden, &layer.input_layernorm, cfg.rms_norm_eps)?;
        let attn_out = layer.self_attn.forward(&x, &rope)?;
        hidden = (&residual + attn_out)?;

        // --- MoE ----------------------------------------------------------
        let residual = hidden.clone();
        let x = rms_norm(&hidden, &layer.post_attention_layernorm, cfg.rms_norm_eps)?;

        // Router: [B, S, H] @ [H, E] → [B, S, E]
        let router_logits = layer.gate.forward(&x)?;
        let router_probs_f32 =
            candle_nn::ops::softmax_last_dim(&router_logits.to_dtype(DType::F32)?)?;

        // Top-k: candle has `topk_last_dim` via sort or manual reduce. We
        // do it in plain Rust on a CPU copy — it's tiny (B * S * n_experts
        // ≈ 1 * 300 * 60 = 18k floats) and lets us also tally counts
        // here without an extra device→host trip.
        let probs_host: Vec<f32> = router_probs_f32.flatten_all()?.to_vec1::<f32>()?;
        let (topk_weights_host, topk_indices_host) =
            topk_per_row(&probs_host, n_tokens, cfg.n_experts, cfg.top_k);

        // Tally counts for this layer.
        let layer_off = layer_idx * cfg.n_experts;
        for &e in &topk_indices_host {
            counts[layer_off + e as usize] += 1;
        }
        accumulate_coactivation(
            &mut coact_counts,
            &topk_indices_host,
            n_tokens,
            cfg.top_k,
            cfg.n_experts,
            layer_idx,
        );

        // Optionally renormalize top-k weights to sum to 1 along the k
        // axis. Qwen1.5-MoE has this off.
        let topk_weights_renorm = if cfg.norm_topk_prob {
            renormalize_topk(&topk_weights_host, cfg.top_k)
        } else {
            topk_weights_host.clone()
        };

        // Expert dispatch — group tokens by chosen expert, run that
        // expert's SwiGLU FFN on the (small) gather, scatter weighted
        // outputs back into `moe_out`.
        let moe_out = dispatch_experts(
            &x,
            &layer.experts,
            &topk_indices_host,
            &topk_weights_renorm,
            cfg.top_k,
            cfg.n_experts,
            n_tokens,
            cfg.hidden_size,
            &device,
            dtype,
        )?;

        // Shared expert: runs on every token, gated by a sigmoid scalar.
        let shared_act = layer.shared_expert.forward(&x)?;
        let shared_gate_logits = layer.shared_expert_gate.forward(&x)?;
        let shared_gate_sig = candle_nn::ops::sigmoid(&shared_gate_logits)?;
        let shared_out = shared_act.broadcast_mul(&shared_gate_sig)?;

        hidden = ((&residual + moe_out)? + shared_out)?;

        pb.inc(1);
    }
    pb.finish_and_clear();

    // Frequencies — counts / n_tokens.
    let freq: Vec<f32> = counts.iter().map(|&c| c as f32 / n_tokens as f32).collect();
    let coact: Vec<f32> = coact_counts
        .iter()
        .map(|&c| c as f32 / n_tokens as f32)
        .collect();

    Ok(RoutingCapture {
        n_layers: cfg.n_layers as u32,
        n_experts: cfg.n_experts as u32,
        n_tokens: n_tokens as u32,
        freq,
        coact,
    })
}

// ============================================================================
// Per-layer modules
// ============================================================================

struct Layer {
    input_layernorm: Tensor,
    self_attn: GqaAttention,
    post_attention_layernorm: Tensor,
    gate: Linear,
    experts: Vec<SwiGluExpert>,
    shared_expert: SwiGluExpert,
    shared_expert_gate: Linear,
}

impl Layer {
    fn load(vb: &VarBuilder, cfg: &Cfg) -> anyhow::Result<Self> {
        let input_layernorm = vb.pp("input_layernorm").get(cfg.hidden_size, "weight")?;
        let post_attention_layernorm = vb
            .pp("post_attention_layernorm")
            .get(cfg.hidden_size, "weight")?;
        // Qwen has biases on Q/K/V but not on O.
        let self_attn = GqaAttention::load(
            &vb.pp("self_attn"),
            cfg.hidden_size,
            cfg.n_heads,
            cfg.n_kv_heads,
            cfg.head_dim,
            /* qkv_bias */ true,
        )?;
        let mlp_vb = vb.pp("mlp");
        let gate = candle_nn::linear_no_bias(cfg.hidden_size, cfg.n_experts, mlp_vb.pp("gate"))?;
        let mut experts = Vec::with_capacity(cfg.n_experts);
        for e in 0..cfg.n_experts {
            experts.push(SwiGluExpert::load(
                &mlp_vb.pp(format!("experts.{e}")),
                cfg.hidden_size,
                cfg.moe_intermediate_size,
                "gate_proj",
                "up_proj",
                "down_proj",
            )?);
        }
        let shared_expert = SwiGluExpert::load(
            &mlp_vb.pp("shared_expert"),
            cfg.hidden_size,
            cfg.shared_intermediate_size,
            "gate_proj",
            "up_proj",
            "down_proj",
        )?;
        // `shared_expert_gate` is a Linear with output dim 1 — a per-token
        // scalar that gates the shared-expert contribution.
        let shared_expert_gate =
            candle_nn::linear_no_bias(cfg.hidden_size, 1, mlp_vb.pp("shared_expert_gate"))?;
        Ok(Self {
            input_layernorm,
            self_attn,
            post_attention_layernorm,
            gate,
            experts,
            shared_expert,
            shared_expert_gate,
        })
    }
}
