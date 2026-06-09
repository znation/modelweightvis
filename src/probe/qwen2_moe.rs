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
use crate::probe::common::{causal_sdpa, repeat_kv, rms_norm, RotaryEmbedding};
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
            .context("Qwen2-MoE config missing num_attention_heads")? as usize;
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
            .context("Qwen2-MoE config missing num_experts")?
            as usize;
        let top_k = config
            .num_experts_per_tok
            .context("Qwen2-MoE config missing num_experts_per_tok")?
            as usize;
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
    // we load as f32. Memory hit is real (~2× the on-disk size — for
    // Qwen1.5-MoE-A2.7B that's ~28 GB in working set) but tractable on
    // 32 GB machines; the alternative (cast-on-use per matmul) duplicates
    // activation buffers per layer and isn't actually cheaper in practice.
    let dtype = DType::F32;
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(weight_paths, dtype, &device)
            .context("Qwen2-MoE: opening safetensors shards via VarBuilder")?
    };

    let model = Model::load(&vb, &cfg)?;
    let rope = RotaryEmbedding::new(cfg.head_dim, cfg.max_pos, cfg.rope_theta, &device)?;

    let mut hidden = model.embed_tokens.forward(&input_ids)?;

    // Per-layer routing-decision counts. `counts[layer * n_experts + expert]`
    // increments each time `expert` appears in top-k for any token in that
    // layer. Divide by `n_tokens` at the end for frequencies.
    let mut counts = vec![0u32; cfg.n_layers * cfg.n_experts];

    let pb = indicatif::ProgressBar::new(cfg.n_layers as u64);
    pb.set_message("probe: layer forward");
    for (layer_idx, layer) in model.layers.iter().enumerate() {
        // --- Attention ----------------------------------------------------
        let residual = hidden.clone();
        let x = rms_norm(&hidden, &layer.input_layernorm, cfg.rms_norm_eps)?;
        let attn_out = layer.self_attn.forward(&x, &rope, &cfg)?;
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
    let freq: Vec<f32> = counts
        .iter()
        .map(|&c| c as f32 / n_tokens as f32)
        .collect();

    Ok(RoutingCapture {
        n_layers: cfg.n_layers as u32,
        n_experts: cfg.n_experts as u32,
        n_tokens: n_tokens as u32,
        freq,
    })
}

// ============================================================================
// Per-layer modules
// ============================================================================

struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
}

impl Attention {
    fn load(vb: &VarBuilder, cfg: &Cfg) -> anyhow::Result<Self> {
        let h = cfg.hidden_size;
        let nh = cfg.n_heads;
        let nkv = cfg.n_kv_heads;
        let d = cfg.head_dim;
        // Qwen has biases on Q/K/V but NOT on O.
        Ok(Self {
            q_proj: candle_nn::linear(h, nh * d, vb.pp("q_proj"))?,
            k_proj: candle_nn::linear(h, nkv * d, vb.pp("k_proj"))?,
            v_proj: candle_nn::linear(h, nkv * d, vb.pp("v_proj"))?,
            o_proj: candle_nn::linear_no_bias(nh * d, h, vb.pp("o_proj"))?,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        rope: &RotaryEmbedding,
        cfg: &Cfg,
    ) -> anyhow::Result<Tensor> {
        let (b, s, _h) = x.dims3()?;
        let q = self.q_proj.forward(x)?; // [B, S, n_heads * head_dim]
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;
        let q = q
            .reshape((b, s, cfg.n_heads, cfg.head_dim))?
            .transpose(1, 2)?; // [B, n_heads, S, head_dim]
        let k = k
            .reshape((b, s, cfg.n_kv_heads, cfg.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, s, cfg.n_kv_heads, cfg.head_dim))?
            .transpose(1, 2)?;
        let q = rope.apply(&q)?;
        let k = rope.apply(&k)?;
        let n_rep = cfg.n_heads / cfg.n_kv_heads;
        let k = repeat_kv(&k, n_rep)?;
        let v = repeat_kv(&v, n_rep)?;
        let attn = causal_sdpa(&q, &k, &v)?; // [B, n_heads, S, head_dim]
        let attn = attn
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, s, cfg.n_heads * cfg.head_dim))?;
        Ok(self.o_proj.forward(&attn)?)
    }
}

/// SwiGLU FFN: `down(silu(gate(x)) * up(x))`. The shape across Qwen2-MoE
/// experts: routed experts have `intermediate = moe_intermediate_size`;
/// the shared expert uses `shared_intermediate_size`. The expert struct
/// itself is shape-agnostic — sizes come from `VarBuilder` weight loads.
struct SwiGluFfn {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl SwiGluFfn {
    fn load(vb: &VarBuilder, hidden_size: usize, intermediate: usize) -> anyhow::Result<Self> {
        Ok(Self {
            gate_proj: candle_nn::linear_no_bias(hidden_size, intermediate, vb.pp("gate_proj"))?,
            up_proj: candle_nn::linear_no_bias(hidden_size, intermediate, vb.pp("up_proj"))?,
            down_proj: candle_nn::linear_no_bias(intermediate, hidden_size, vb.pp("down_proj"))?,
        })
    }
}

impl Module for SwiGluFfn {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let act = candle_nn::ops::silu(&gate)?;
        let mlp = (act * up)?;
        self.down_proj.forward(&mlp)
    }
}

struct Layer {
    input_layernorm: Tensor,
    self_attn: Attention,
    post_attention_layernorm: Tensor,
    gate: Linear,
    experts: Vec<SwiGluFfn>,
    shared_expert: SwiGluFfn,
    shared_expert_gate: Linear,
}

impl Layer {
    fn load(vb: &VarBuilder, cfg: &Cfg) -> anyhow::Result<Self> {
        let input_layernorm = vb
            .pp("input_layernorm")
            .get(cfg.hidden_size, "weight")?;
        let post_attention_layernorm = vb
            .pp("post_attention_layernorm")
            .get(cfg.hidden_size, "weight")?;
        let self_attn = Attention::load(&vb.pp("self_attn"), cfg)?;
        let mlp_vb = vb.pp("mlp");
        let gate = candle_nn::linear_no_bias(cfg.hidden_size, cfg.n_experts, mlp_vb.pp("gate"))?;
        let mut experts = Vec::with_capacity(cfg.n_experts);
        for e in 0..cfg.n_experts {
            experts.push(SwiGluFfn::load(
                &mlp_vb.pp(format!("experts.{e}")),
                cfg.hidden_size,
                cfg.moe_intermediate_size,
            )?);
        }
        let shared_expert = SwiGluFfn::load(
            &mlp_vb.pp("shared_expert"),
            cfg.hidden_size,
            cfg.shared_intermediate_size,
        )?;
        // `shared_expert_gate` is a Linear with output dim 1 — a per-token
        // scalar that gates the shared-expert contribution.
        let shared_expert_gate = candle_nn::linear_no_bias(
            cfg.hidden_size,
            1,
            mlp_vb.pp("shared_expert_gate"),
        )?;
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

struct Model {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<Layer>,
}

impl Model {
    fn load(vb: &VarBuilder, cfg: &Cfg) -> anyhow::Result<Self> {
        let embed_tokens = candle_nn::embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            vb.pp("model.embed_tokens"),
        )?;
        let mut layers = Vec::with_capacity(cfg.n_layers);
        let layers_vb = vb.pp("model.layers");
        for layer_idx in 0..cfg.n_layers {
            layers.push(Layer::load(&layers_vb.pp(layer_idx), cfg)?);
        }
        Ok(Self {
            embed_tokens,
            layers,
        })
    }
}

// ============================================================================
// Top-k helpers (CPU-side, on the small router-probability tensor)
// ============================================================================

/// Pick top-k entries per row of an `[n_rows, n_cols]` row-major matrix.
/// Returns `(weights, indices)`, each `[n_rows * k]` row-major. `weights`
/// preserves the original probability values; if the caller wants them
/// renormalised, they re-divide afterwards.
fn topk_per_row(
    probs: &[f32],
    n_rows: usize,
    n_cols: usize,
    k: usize,
) -> (Vec<f32>, Vec<u32>) {
    let mut weights = Vec::with_capacity(n_rows * k);
    let mut indices = Vec::with_capacity(n_rows * k);
    let mut scratch: Vec<(f32, u32)> = Vec::with_capacity(n_cols);
    for r in 0..n_rows {
        let row = &probs[r * n_cols..(r + 1) * n_cols];
        scratch.clear();
        for (i, &p) in row.iter().enumerate() {
            scratch.push((p, i as u32));
        }
        // Partial sort: descending by probability. For small n_cols (~60
        // for Qwen MoE) this is fine without nth-element.
        scratch.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        for &(p, idx) in scratch.iter().take(k) {
            weights.push(p);
            indices.push(idx);
        }
    }
    (weights, indices)
}

/// In-place: renormalise each row's k weights to sum to 1.
fn renormalize_topk(weights: &[f32], k: usize) -> Vec<f32> {
    let mut out = weights.to_vec();
    for chunk in out.chunks_exact_mut(k) {
        let s: f32 = chunk.iter().sum();
        if s > 0.0 {
            for w in chunk.iter_mut() {
                *w /= s;
            }
        }
    }
    out
}

// ============================================================================
// Expert dispatch
// ============================================================================

/// Run each expert's FFN on the tokens that selected it (according to
/// `topk_indices`), and scatter weighted outputs back into `moe_out`.
///
/// Implementation: for each expert E in 0..n_experts, collect the
/// `(token_idx, weight)` pairs where E appears in any top-k slot,
/// gather those tokens into a `[N_E, H]` tensor, run the expert FFN,
/// then scatter the result weighted back.
///
/// `x` is `[1, n_tokens, hidden_size]` (we operate on a single probe
/// batch). `topk_indices` / `topk_weights` are `[n_tokens * top_k]`.
#[allow(clippy::too_many_arguments)]
fn dispatch_experts(
    x: &Tensor,
    experts: &[SwiGluFfn],
    topk_indices: &[u32],
    topk_weights: &[f32],
    top_k: usize,
    n_experts: usize,
    n_tokens: usize,
    hidden_size: usize,
    device: &Device,
    dtype: DType,
) -> anyhow::Result<Tensor> {
    let mut moe_out = Tensor::zeros((1, n_tokens, hidden_size), dtype, device)?;

    // Squeeze the batch dim to make gather/scatter simpler.
    let x_flat = x.reshape((n_tokens, hidden_size))?; // [N, H]

    for e in 0..n_experts {
        // Find token indices that picked expert e, plus their assigned weights.
        let mut token_ids: Vec<u32> = Vec::new();
        let mut token_weights: Vec<f32> = Vec::new();
        for t in 0..n_tokens {
            for k_slot in 0..top_k {
                let idx = topk_indices[t * top_k + k_slot];
                if idx as usize == e {
                    token_ids.push(t as u32);
                    token_weights.push(topk_weights[t * top_k + k_slot]);
                }
            }
        }
        if token_ids.is_empty() {
            continue;
        }

        // Gather x[token_ids, :] → [n_e, H]
        let n_e = token_ids.len();
        let ids_t = Tensor::from_vec(token_ids, n_e, device)?;
        let x_gathered = x_flat.index_select(&ids_t, 0)?; // [n_e, H]

        // Run expert FFN.
        let expert_out = experts[e].forward(&x_gathered)?; // [n_e, H]

        // Weighted scatter: moe_out[t, :] += w * expert_out[i, :]
        let weights_t = Tensor::from_vec(token_weights, (n_e, 1), device)?.to_dtype(dtype)?;
        let weighted = expert_out.broadcast_mul(&weights_t)?; // [n_e, H]

        // candle doesn't ship a scatter-add over rows by index; we do it
        // via index_add. moe_out has shape [1, N, H]; squeeze, scatter, unsqueeze.
        let acc = moe_out.squeeze(0)?; // [N, H]
        let acc = acc.index_add(&ids_t, &weighted, 0)?;
        moe_out = acc.unsqueeze(0)?;
    }

    Ok(moe_out)
}

