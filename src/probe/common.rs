//! Shared transformer primitives used by every `--probe` arch module.
//!
//! Wraps `candle-nn` / `candle-core` ops in the specific shapes the
//! MoE forward passes need: per-token RMSNorm, RoPE rotation, and a
//! grouped-query attention helper that handles the head ≠ kv-head
//! fan-out shared by Qwen2-MoE and Mixtral.

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{Linear, Module, VarBuilder};

/// Standard transformer RMSNorm:
///   `y = x / sqrt(mean(x², -1) + eps) * weight`
/// Applied per-token along the last dim. `weight` is `[hidden_size]`
/// and gets broadcast across the leading dims.
///
/// We re-implement here (rather than reuse `candle_nn::rms_norm`)
/// only because we want a function that's transparent for activation
/// capture — same body, but easy to instrument inline.
pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    // Compute in f32 for numerical stability, then cast back to the
    // input dtype. Matches transformers' default behavior.
    let x_dtype = x.dtype();
    let x_f32 = x.to_dtype(DType::F32)?;
    let variance = x_f32.sqr()?.mean_keepdim(D::Minus1)?;
    let x_norm = x_f32.broadcast_div(&(variance + eps)?.sqrt()?)?;
    let x_back = x_norm.to_dtype(x_dtype)?;
    x_back.broadcast_mul(weight)
}

/// Precomputed cosine / sine tables for rotary position embeddings.
/// The standard RoPE construction: for each position `m` and each
/// pair of dimensions `(2i, 2i+1)`, the angle is
/// `m / theta^(2i / head_dim)`. We precompute `cos` and `sin` of
/// these angles for every (position, dim-pair) up to `max_pos`.
pub struct RotaryEmbedding {
    /// `[max_pos, head_dim/2]`, f32.
    pub cos: Tensor,
    /// `[max_pos, head_dim/2]`, f32.
    pub sin: Tensor,
}

impl RotaryEmbedding {
    /// Build cosine and sine tables for `max_pos` positions over
    /// `head_dim`. `head_dim` must be even.
    pub fn new(head_dim: usize, max_pos: usize, theta: f64, device: &Device) -> Result<Self> {
        assert_eq!(head_dim % 2, 0, "RoPE: head_dim must be even");
        let half = head_dim / 2;
        // freqs[i] = 1 / theta^(2i / head_dim)  for i in 0..half
        let freqs: Vec<f32> = (0..half)
            .map(|i| {
                let exp = (2 * i) as f64 / head_dim as f64;
                (theta.powf(exp)).recip() as f32
            })
            .collect();
        let freqs = Tensor::from_vec(freqs, half, device)?;
        let positions: Vec<f32> = (0..max_pos).map(|p| p as f32).collect();
        let positions = Tensor::from_vec(positions, max_pos, device)?;
        // angles[m, i] = positions[m] * freqs[i]
        let angles = positions
            .unsqueeze(1)?
            .broadcast_mul(&freqs.unsqueeze(0)?)?;
        let cos = angles.cos()?;
        let sin = angles.sin()?;
        Ok(Self { cos, sin })
    }

    /// Apply RoPE to a `q` / `k` tensor of shape `[batch, n_heads, seq, head_dim]`.
    /// Slices the precomputed cos/sin tables down to `seq` rows
    /// starting at position 0 (the probe is a single contiguous sequence
    /// — no KV cache, no position offset).
    pub fn apply(&self, x: &Tensor) -> Result<Tensor> {
        let (_b, _h, seq, head_dim) = x.dims4()?;
        let cos = self.cos.narrow(0, 0, seq)?; // [seq, head_dim/2]
        let sin = self.sin.narrow(0, 0, seq)?;
        // x has shape [B, H, S, D]. We split D into (D/2) interleaved
        // pairs along the last dim. Rather than interleave-then-unpair
        // (the original RoPE), HF's impl rotates the *halves*: split
        // x along D into x1, x2 (each D/2), then output is
        //   (x1*cos - x2*sin, x1*sin + x2*cos)
        // along D. This is what candle-transformers and HF transformers
        // implement; we match it so loaded weights line up.
        let half = head_dim / 2;
        let x1 = x.narrow(D::Minus1, 0, half)?;
        let x2 = x.narrow(D::Minus1, half, half)?;
        // cos / sin: [S, D/2] → broadcast to [B, H, S, D/2].
        let cos_b = cos.unsqueeze(0)?.unsqueeze(0)?; // [1, 1, S, D/2]
        let sin_b = sin.unsqueeze(0)?.unsqueeze(0)?;
        let rotated_1 = (x1.broadcast_mul(&cos_b)? - x2.broadcast_mul(&sin_b)?)?;
        let rotated_2 = (x1.broadcast_mul(&sin_b)? + x2.broadcast_mul(&cos_b)?)?;
        Tensor::cat(&[&rotated_1, &rotated_2], D::Minus1)
    }
}

/// Repeat the kv-head dim to match the q-head dim — the GQA "broadcast"
/// of key/value across the query heads that share them. `kv` is
/// `[batch, n_kv_heads, seq, head_dim]`; returns
/// `[batch, n_heads, seq, head_dim]` by repeating each kv-head
/// `n_heads / n_kv_heads` times.
pub fn repeat_kv(kv: &Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(kv.clone());
    }
    let (b, n_kv, s, d) = kv.dims4()?;
    let expanded = kv
        .unsqueeze(2)? // [b, n_kv, 1, s, d]
        .expand((b, n_kv, n_rep, s, d))?
        .reshape((b, n_kv * n_rep, s, d))?;
    Ok(expanded)
}

/// Scaled dot-product attention with a causal mask.
///
/// `q` / `k` / `v` are `[batch, n_heads, seq, head_dim]` (with `q` and
/// `k` having had RoPE applied, and `k` / `v` already expanded across
/// the GQA group via [`repeat_kv`]).
///
/// Causal: token at position `i` attends to positions `0..=i`.
pub fn causal_sdpa(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let (_b, _h, seq, head_dim) = q.dims4()?;
    let scale = (head_dim as f64).powf(-0.5) as f32;
    // scores: q @ k^T → [B, H, S, S]
    let scores = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;
    let scaled = (scores * scale as f64)?;
    // Causal mask: upper triangle (j > i) set to -inf. Build once per
    // shape; for the probe we're called per-layer so reallocating is
    // a small relative cost. If perf matters, lift this to the caller.
    let mask = causal_mask(seq, q.dtype(), q.device())?;
    let masked = scaled.broadcast_add(&mask)?;
    let probs = candle_nn::ops::softmax_last_dim(&masked)?;
    probs.matmul(v)
}

/// `[seq, seq]` causal mask: `0` on and below the diagonal, `-inf`
/// above. Cast to `dtype`. Used by [`causal_sdpa`].
fn causal_mask(seq: usize, dtype: DType, device: &Device) -> Result<Tensor> {
    let mut data = vec![0.0f32; seq * seq];
    for i in 0..seq {
        for j in (i + 1)..seq {
            data[i * seq + j] = f32::NEG_INFINITY;
        }
    }
    let m = Tensor::from_vec(data, (seq, seq), device)?;
    m.to_dtype(dtype)
}

// ============================================================================
// Grouped-query attention (shared across arches)
// ============================================================================

/// A single transformer self-attention block with grouped-query attention,
/// RoPE, and a causal mask. Shared by every probe arch — the only on-disk
/// difference between Qwen2-MoE and Mixtral/Llama here is whether Q/K/V carry
/// a bias (Qwen2 does; Mixtral/Llama don't), captured by `qkv_bias` at load.
/// The output projection is always bias-free. KV cache is disabled — the probe
/// is a single full-sequence forward.
pub struct GqaAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
}

impl GqaAttention {
    /// Load Q/K/V/O projections under `vb` (expects `q_proj`/`k_proj`/`v_proj`/
    /// `o_proj` submodules). `qkv_bias` toggles the Q/K/V bias terms.
    pub fn load(
        vb: &VarBuilder,
        hidden: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        qkv_bias: bool,
    ) -> Result<Self> {
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let (q_proj, k_proj, v_proj) = if qkv_bias {
            (
                candle_nn::linear(hidden, q_dim, vb.pp("q_proj"))?,
                candle_nn::linear(hidden, kv_dim, vb.pp("k_proj"))?,
                candle_nn::linear(hidden, kv_dim, vb.pp("v_proj"))?,
            )
        } else {
            (
                candle_nn::linear_no_bias(hidden, q_dim, vb.pp("q_proj"))?,
                candle_nn::linear_no_bias(hidden, kv_dim, vb.pp("k_proj"))?,
                candle_nn::linear_no_bias(hidden, kv_dim, vb.pp("v_proj"))?,
            )
        };
        let o_proj = candle_nn::linear_no_bias(q_dim, hidden, vb.pp("o_proj"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            n_heads,
            n_kv_heads,
            head_dim,
        })
    }

    /// Full-sequence causal attention over `x` of shape `[B, S, hidden]`.
    pub fn forward(&self, x: &Tensor, rope: &RotaryEmbedding) -> Result<Tensor> {
        let (b, s, _h) = x.dims3()?;
        let q = self.q_proj.forward(x)?; // [B, S, n_heads * head_dim]
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;
        let q = q
            .reshape((b, s, self.n_heads, self.head_dim))?
            .transpose(1, 2)?; // [B, n_heads, S, head_dim]
        let k = k
            .reshape((b, s, self.n_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, s, self.n_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let q = rope.apply(&q)?;
        let k = rope.apply(&k)?;
        let n_rep = self.n_heads / self.n_kv_heads;
        let k = repeat_kv(&k, n_rep)?;
        let v = repeat_kv(&v, n_rep)?;
        let attn = causal_sdpa(&q, &k, &v)?; // [B, n_heads, S, head_dim]
        let attn =
            attn.transpose(1, 2)?
                .contiguous()?
                .reshape((b, s, self.n_heads * self.head_dim))?;
        self.o_proj.forward(&attn)
    }
}

// ============================================================================
// MoE expert FFN + dispatch (shared across arches)
// ============================================================================

/// A single SwiGLU expert FFN: `down(silu(gate(x)) * up(x))`.
///
/// This shape is shared by every MoE arch the probe supports — Qwen2-MoE's
/// routed and shared experts and Mixtral's experts are all SwiGLU; only the
/// on-disk weight *names* differ (Qwen: `gate_proj`/`up_proj`/`down_proj`;
/// classic Mixtral: `w1`/`w3`/`w2`). The struct is name-agnostic — load it via
/// [`SwiGluExpert::load`] when each projection is its own named tensor, or via
/// [`SwiGluExpert::from_linears`] when the projections are sliced out of a
/// fused/batched expert tensor (the newer Mixtral checkpoint layout).
pub struct SwiGluExpert {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl SwiGluExpert {
    /// Load from three individually-named weights under `vb`. `gate_name` /
    /// `up_name` / `down_name` are the sub-paths (e.g. `"gate_proj"`,
    /// `"up_proj"`, `"down_proj"` for Qwen; `"w1"`, `"w3"`, `"w2"` for classic
    /// Mixtral). All three are bias-free.
    pub fn load(
        vb: &VarBuilder,
        hidden: usize,
        intermediate: usize,
        gate_name: &str,
        up_name: &str,
        down_name: &str,
    ) -> Result<Self> {
        Ok(Self {
            gate: candle_nn::linear_no_bias(hidden, intermediate, vb.pp(gate_name))?,
            up: candle_nn::linear_no_bias(hidden, intermediate, vb.pp(up_name))?,
            down: candle_nn::linear_no_bias(intermediate, hidden, vb.pp(down_name))?,
        })
    }

    /// Build directly from prepared `Linear`s. Used when the projections come
    /// from slices of a batched expert tensor rather than named submodules.
    pub fn from_linears(gate: Linear, up: Linear, down: Linear) -> Self {
        Self { gate, up, down }
    }
}

impl Module for SwiGluExpert {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate.forward(x)?;
        let up = self.up.forward(x)?;
        let act = candle_nn::ops::silu(&gate)?;
        let mlp = (act * up)?;
        self.down.forward(&mlp)
    }
}

/// Pick the top-`k` entries per row of an `[n_rows, n_cols]` row-major matrix.
/// Returns `(weights, indices)`, each `[n_rows * k]` row-major and ordered by
/// descending value within each row. `weights` preserves the original
/// probability values; renormalise with [`renormalize_topk`] if the arch
/// requires it.
///
/// Done in plain Rust on a host copy of the router probabilities — the matrix
/// is tiny (`n_rows * n_cols` ≈ a few thousand floats) and this lets the caller
/// tally routing counts without an extra device→host round-trip.
pub fn topk_per_row(probs: &[f32], n_rows: usize, n_cols: usize, k: usize) -> (Vec<f32>, Vec<u32>) {
    let mut weights = Vec::with_capacity(n_rows * k);
    let mut indices = Vec::with_capacity(n_rows * k);
    let mut scratch: Vec<(f32, u32)> = Vec::with_capacity(n_cols);
    for r in 0..n_rows {
        let row = &probs[r * n_cols..(r + 1) * n_cols];
        scratch.clear();
        for (i, &p) in row.iter().enumerate() {
            scratch.push((p, i as u32));
        }
        // Partial sort, descending by probability. For the small n_cols MoEs
        // have (8–128 experts) this is fine without an nth-element selection.
        scratch.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        for &(p, idx) in scratch.iter().take(k) {
            weights.push(p);
            indices.push(idx);
        }
    }
    (weights, indices)
}

/// Accumulate one layer's routing co-occurrence into a shared
/// `n_layers * n_experts^2` count buffer. `topk_indices` is `[n_tokens * top_k]`
/// row-major — the same buffer the caller already feeds to [`dispatch_experts`]
/// and tallies routing-frequency from.
///
/// For each token, for every *unordered* pair `(a, b)` of distinct experts in
/// its top-k slots we bump `[a][b]` and `[b][a]`; the diagonal `[a][a]` is
/// bumped once per selected expert, so the diagonal reproduces the per-expert
/// routing-frequency counts exactly (its sum over experts is `n_tokens * top_k`).
/// `topk_per_row` yields distinct indices per row, so the off-diagonal pairs
/// never collide with the diagonal.
///
/// Cheap: `O(n_tokens * top_k^2)` host-side integer increments.
pub fn accumulate_coactivation(
    coact_counts: &mut [u32],
    topk_indices: &[u32],
    n_tokens: usize,
    top_k: usize,
    n_experts: usize,
    layer_idx: usize,
) {
    let base = layer_idx * n_experts * n_experts;
    for t in 0..n_tokens {
        let slots = &topk_indices[t * top_k..(t + 1) * top_k];
        for (si, &a) in slots.iter().enumerate() {
            let a = a as usize;
            // Diagonal: counts how many tokens selected expert `a` (== freq count).
            coact_counts[base + a * n_experts + a] += 1;
            for &b in &slots[si + 1..] {
                let b = b as usize;
                coact_counts[base + a * n_experts + b] += 1;
                coact_counts[base + b * n_experts + a] += 1;
            }
        }
    }
}

/// Renormalise each row's `k` weights to sum to 1. Mixtral always does this;
/// Qwen1.5-MoE leaves it off (per its `norm_topk_prob`).
pub fn renormalize_topk(weights: &[f32], k: usize) -> Vec<f32> {
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

/// Run each expert's FFN on the tokens that selected it, and scatter the
/// weighted outputs back into a `[1, n_tokens, hidden_size]` accumulator.
///
/// For each expert `E`, collect the `(token_idx, weight)` pairs where `E`
/// appears in any of the token's top-k slots, gather those tokens into a
/// `[n_e, H]` batch, run the expert once, then index-add the weighted result
/// back. `topk_indices` / `topk_weights` are `[n_tokens * top_k]` row-major.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_experts(
    x: &Tensor,
    experts: &[SwiGluExpert],
    topk_indices: &[u32],
    topk_weights: &[f32],
    top_k: usize,
    n_experts: usize,
    n_tokens: usize,
    hidden_size: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let mut moe_out = Tensor::zeros((1, n_tokens, hidden_size), dtype, device)?;

    // Squeeze the batch dim to make gather/scatter simpler.
    let x_flat = x.reshape((n_tokens, hidden_size))?; // [N, H]

    for (e, expert) in experts.iter().enumerate().take(n_experts) {
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

        // Gather x[token_ids, :] → [n_e, H], run the expert FFN once.
        let n_e = token_ids.len();
        let ids_t = Tensor::from_vec(token_ids, n_e, device)?;
        let x_gathered = x_flat.index_select(&ids_t, 0)?; // [n_e, H]
        let expert_out = expert.forward(&x_gathered)?; // [n_e, H]

        // Weighted scatter: moe_out[t, :] += w * expert_out[i, :]. candle has
        // no row-indexed scatter-add, so we squeeze → index_add → unsqueeze.
        let weights_t = Tensor::from_vec(token_weights, (n_e, 1), device)?.to_dtype(dtype)?;
        let weighted = expert_out.broadcast_mul(&weights_t)?; // [n_e, H]
        let acc = moe_out.squeeze(0)?; // [N, H]
        let acc = acc.index_add(&ids_t, &weighted, 0)?;
        moe_out = acc.unsqueeze(0)?;
    }

    Ok(moe_out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coactivation_symmetric_diagonal_is_freq() {
        // 3 experts, top_k = 2, 3 tokens, single layer.
        // Token top-k picks: [0,1], [0,2], [0,1].
        let idx = vec![0u32, 1, 0, 2, 0, 1];
        let (n_tokens, top_k, e) = (3usize, 2usize, 3usize);
        let mut c = vec![0u32; e * e];
        accumulate_coactivation(&mut c, &idx, n_tokens, top_k, e, 0);
        let at = |i: usize, j: usize| c[i * e + j];

        // Symmetric.
        for i in 0..e {
            for j in 0..e {
                assert_eq!(at(i, j), at(j, i), "asymmetry at ({i},{j})");
            }
        }
        // Diagonal == per-expert selection count: e0 in all 3, e1 in 2, e2 in 1.
        assert_eq!(at(0, 0), 3);
        assert_eq!(at(1, 1), 2);
        assert_eq!(at(2, 2), 1);
        // Off-diagonal co-counts: (0,1) in tokens 0 & 2 → 2; (0,2) in token 1 → 1;
        // (1,2) never co-occur → 0.
        assert_eq!(at(0, 1), 2);
        assert_eq!(at(0, 2), 1);
        assert_eq!(at(1, 2), 0);
        // Diagonal mass == n_tokens * top_k.
        let diag: u32 = (0..e).map(|i| at(i, i)).sum();
        assert_eq!(diag, (n_tokens * top_k) as u32);
    }

    #[test]
    fn coactivation_writes_into_correct_layer_block() {
        // Two layers, 2 experts, top_k = 2, 1 token picking [0,1] in layer 1 only.
        let (n_tokens, top_k, e) = (1usize, 2usize, 2usize);
        let idx = vec![0u32, 1];
        let mut c = vec![0u32; 2 * e * e];
        accumulate_coactivation(&mut c, &idx, n_tokens, top_k, e, 1);
        // Layer 0 block untouched.
        assert!(c[0..e * e].iter().all(|&v| v == 0));
        // Layer 1 block: the lone token co-activates both experts, so every
        // cell (diagonal + off-diagonal) of the 2×2 block is 1.
        let base = e * e;
        assert!(c[base..base + e * e].iter().all(|&v| v == 1));
    }
}
