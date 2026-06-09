//! Shared transformer primitives used by every `--probe` arch module.
//!
//! Wraps `candle-nn` / `candle-core` ops in the specific shapes the
//! MoE forward passes need: per-token RMSNorm, RoPE rotation, and a
//! grouped-query attention helper that handles the head ≠ kv-head
//! fan-out shared by Qwen2-MoE and Mixtral.

use candle_core::{DType, Device, Result, Tensor, D};

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
    pub fn new(
        head_dim: usize,
        max_pos: usize,
        theta: f64,
        device: &Device,
    ) -> Result<Self> {
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
        let cos_b = cos
            .unsqueeze(0)?
            .unsqueeze(0)?; // [1, 1, S, D/2]
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
