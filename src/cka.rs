//! Linear Centered Kernel Alignment (CKA) for comparing MoE expert
//! weight matrices that share their input dim.
//!
//! Reference: Kornblith et al. 2019, "Similarity of Neural Network
//! Representations Revisited". The variant used here is the "feature-
//! space" form of linear CKA — equivalent to the standard
//! representation-space CKA but expressed in terms of weight matrices,
//! which lets us compute it from static checkpoints without inference:
//!
//! ```text
//!     CKA(W_a, W_b) = ‖W_a^T W_b‖_F^2 / (‖W_a^T W_a‖_F · ‖W_b^T W_b‖_F)
//! ```
//!
//! `W_a` and `W_b` must share their input dimension (the column count
//! in HF safetensors layout). Output dims can differ. Result is in
//! `[0, 1]`: `1.0` for identical (up to scale) experts, `0.0` for
//! experts whose row-space spans are orthogonal. Diagonal entries
//! `(i, i)` are always `1.0`.
//!
//! ## Random projection
//!
//! Exact CKA on a single pair costs `O(d_in² · d_out)` — dominated by
//! the `W_a^T W_b` matmul. For Qwen1.5-MoE-A2.7B (d_in=2048,
//! d_out=1408) a single layer's pairwise grid (1830 pairs) is already
//! 11 TFLOPs of dense f32 matmul; the full 24-layer × 3-weight panel
//! set is ~800 TFLOPs.
//!
//! We sketch each weight matrix's *input* axis with a Gaussian random
//! projection of dimension `k` (CLI knob `--cka-sample`, default 128).
//! For `W` of shape `(d_out, d_in)` and `R` of shape `(k, d_in)` with
//! entries `~ N(0, 1/k)`, the projected matrix `W_proj = W · R^T` has
//! shape `(d_out, k)`. The identity `E[R^T R] = I_{d_in}` makes the
//! projected Frobenius norm an unbiased estimator of the unprojected
//! one (Johnson-Lindenstrauss applied to matrix inner products).
//!
//! Per-pair cost drops to `O(k² · d_out)`; precompute amortises the
//! one-time projection cost across all pairs of one expert group.

use rayon::prelude::*;

/// Gaussian random matrix of shape `(k, d)`, row-major, deterministically
/// seeded. Entries are sampled from `N(0, 1/k)` via the Box-Muller
/// transform over a splitmix64 PRNG — pure-Rust and self-contained, no
/// dep on `rand`.
///
/// The `1/k` variance means `E[R^T R] = I_d`, which makes projected
/// CKA an unbiased estimate of unprojected CKA.
pub fn gaussian_projection(k: usize, d: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15);
    let std = (1.0_f32 / k as f32).sqrt();
    let mut out = vec![0.0f32; k * d];
    // Box-Muller produces samples in pairs (z1, z2) — fill two slots
    // per outer iteration.
    let mut idx = 0;
    while idx + 1 < out.len() {
        let u1 = next_uniform(&mut state).max(f32::MIN_POSITIVE);
        let u2 = next_uniform(&mut state);
        let r = (-2.0_f32 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        out[idx] = std * r * theta.cos();
        out[idx + 1] = std * r * theta.sin();
        idx += 2;
    }
    if idx < out.len() {
        let u1 = next_uniform(&mut state).max(f32::MIN_POSITIVE);
        let u2 = next_uniform(&mut state);
        let r = (-2.0_f32 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        out[idx] = std * r * theta.cos();
    }
    out
}

/// splitmix64 step — fast deterministic 64-bit PRNG, no external state.
#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Uniform `[0, 1)` f32 from the PRNG.
#[inline]
fn next_uniform(state: &mut u64) -> f32 {
    // High 24 bits → f32 mantissa, giving exact uniform `[0, 1)`.
    let bits = splitmix64(state);
    ((bits >> 40) as f32) / (1u32 << 24) as f32
}

/// Project the rows of `w` (shape `rows × cols`, row-major) through
/// `r` (shape `k × cols`, row-major). Result has shape `rows × k`,
/// row-major: `out[i, j] = dot(w[i, :], r[j, :])`.
///
/// One straight cache-friendly triple loop. The inner loop streams a
/// row of `w` and one row of `r` together — both contiguous — so
/// auto-vectorisation handles the dot product. Outer parallelism is
/// over rows of `w` via rayon.
pub fn project_rows(w: &[f32], rows: usize, cols: usize, r: &[f32], k: usize) -> Vec<f32> {
    assert_eq!(w.len(), rows * cols, "w shape mismatch");
    assert_eq!(r.len(), k * cols, "r shape mismatch");
    let mut out = vec![0.0f32; rows * k];
    out.par_chunks_mut(k)
        .zip(w.par_chunks(cols))
        .for_each(|(out_row, w_row)| {
            for j in 0..k {
                let r_row = &r[j * cols..(j + 1) * cols];
                let mut s = 0.0f32;
                for c in 0..cols {
                    s += w_row[c] * r_row[c];
                }
                out_row[j] = s;
            }
        });
    out
}

/// Squared Frobenius norm of `A^T B` for two matrices `A`, `B` of
/// shape `(rows × k)`, row-major. Returns `∑_{ij} (A^T B)[i, j]^2`
/// in `f64` so accumulation error doesn't bias the result for many
/// rows.
///
/// Computes `G = A^T B` (shape `k × k`) first, then `‖G‖_F^2`. The
/// outer loop is over rows of `A` and `B` so the inner two loops
/// each iterate over `k` contiguous elements — cache-friendly.
pub fn at_b_frobenius_sq(a: &[f32], b: &[f32], rows: usize, k: usize) -> f64 {
    assert_eq!(a.len(), rows * k);
    assert_eq!(b.len(), rows * k);
    // G[i, j] = ∑_m A[m, i] * B[m, j]. Build G in row-major (k × k).
    let mut g = vec![0.0f32; k * k];
    for m in 0..rows {
        let a_row = &a[m * k..(m + 1) * k];
        let b_row = &b[m * k..(m + 1) * k];
        for i in 0..k {
            let a_mi = a_row[i];
            let g_row = &mut g[i * k..(i + 1) * k];
            for j in 0..k {
                g_row[j] += a_mi * b_row[j];
            }
        }
    }
    g.iter().map(|&x| (x as f64) * (x as f64)).sum()
}

/// Linear CKA between two row-projected weight matrices.
///
/// `a` and `b` are both `(rows × k)`, row-major, projections of the
/// same input axis (via the same `R` from [`gaussian_projection`]).
/// `a_self_sq` is `‖A^T A‖_F^2`, precomputed via [`at_b_frobenius_sq`]
/// once per expert; ditto `b_self_sq`.
///
/// Result is clamped to `[0, 1]` — tiny negative values from f64
/// rounding around `0` get flushed to `0`, and the upper bound from
/// the Cauchy-Schwarz inequality means values above `1` are also just
/// rounding artefacts.
pub fn linear_cka(
    a: &[f32],
    b: &[f32],
    rows: usize,
    k: usize,
    a_self_sq: f64,
    b_self_sq: f64,
) -> f32 {
    let num = at_b_frobenius_sq(a, b, rows, k);
    let denom = (a_self_sq * b_self_sq).sqrt();
    if denom <= 1e-30 {
        return 0.0;
    }
    (num / denom).clamp(0.0, 1.0) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a simple test matrix `rows × cols` with deterministic
    /// values — `m[i, j] = (i * cols + j) as f32 / total` so entries
    /// lie in `[0, 1]` and CKA computations stay numerically tame.
    fn ramp_matrix(rows: usize, cols: usize) -> Vec<f32> {
        let n = (rows * cols) as f32;
        (0..rows * cols).map(|k| k as f32 / n).collect()
    }

    #[test]
    fn gaussian_projection_is_deterministic() {
        let a = gaussian_projection(8, 16, 42);
        let b = gaussian_projection(8, 16, 42);
        assert_eq!(a, b);
        let c = gaussian_projection(8, 16, 43);
        assert_ne!(a, c);
    }

    #[test]
    fn gaussian_projection_has_expected_variance() {
        // E[R[i, j]^2] = 1/k. Sample many entries and confirm the
        // empirical mean of squares lands near 1/k.
        let k = 32;
        let d = 1024;
        let r = gaussian_projection(k, d, 7);
        let mean_sq: f64 = r.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / r.len() as f64;
        let expected = 1.0 / k as f64;
        // 1/sqrt(N) noise floor with N = k*d = 32768 → ~0.005 relative.
        assert!(
            (mean_sq - expected).abs() < 0.1 * expected,
            "mean of squares {mean_sq}, expected ~{expected}",
        );
    }

    #[test]
    fn identical_experts_give_cka_one() {
        // Two copies of the same weight matrix → CKA = 1.0 exactly
        // (independent of the projection — the projected matrices
        // are also identical so the ratio is 1).
        let rows = 16;
        let cols = 24;
        let w = ramp_matrix(rows, cols);
        let k = 8;
        let r = gaussian_projection(k, cols, 1);
        let w_proj = project_rows(&w, rows, cols, &r, k);
        let self_sq = at_b_frobenius_sq(&w_proj, &w_proj, rows, k);
        let cka = linear_cka(&w_proj, &w_proj, rows, k, self_sq, self_sq);
        assert!((cka - 1.0).abs() < 1e-5, "CKA should be 1.0, got {cka}");
    }

    #[test]
    fn scaled_experts_give_cka_one() {
        // CKA is scale-invariant: W and 7·W have CKA = 1 (because the
        // scale factor cancels out of the ratio numerator/denominator).
        let rows = 16;
        let cols = 24;
        let w_a = ramp_matrix(rows, cols);
        let w_b: Vec<f32> = w_a.iter().map(|v| v * 7.0).collect();
        let k = 8;
        let r = gaussian_projection(k, cols, 2);
        let a_proj = project_rows(&w_a, rows, cols, &r, k);
        let b_proj = project_rows(&w_b, rows, cols, &r, k);
        let a_sq = at_b_frobenius_sq(&a_proj, &a_proj, rows, k);
        let b_sq = at_b_frobenius_sq(&b_proj, &b_proj, rows, k);
        let cka = linear_cka(&a_proj, &b_proj, rows, k, a_sq, b_sq);
        assert!(
            (cka - 1.0).abs() < 1e-4,
            "scaled CKA should be 1.0, got {cka}"
        );
    }

    #[test]
    fn orthogonal_experts_give_low_cka() {
        // For (W_a^T W_b)[i, j] = ∑_k W_a[k, i] · W_b[k, j] to be zero
        // everywhere we need, for every k, either W_a[k, :] = 0 or
        // W_b[k, :] = 0. Construct that by putting W_a's support on the
        // first half of output rows and W_b's on the second half.
        let rows = 8;
        let cols = 16;
        let mut w_a = vec![0.0f32; rows * cols];
        let mut w_b = vec![0.0f32; rows * cols];
        for i in 0..rows / 2 {
            for j in 0..cols {
                w_a[i * cols + j] = ((i + j) as f32 + 1.0).cos();
            }
        }
        for i in rows / 2..rows {
            for j in 0..cols {
                w_b[i * cols + j] = ((i + j) as f32 + 1.0).sin();
            }
        }
        let k = 32;
        let r = gaussian_projection(k, cols, 3);
        let a_proj = project_rows(&w_a, rows, cols, &r, k);
        let b_proj = project_rows(&w_b, rows, cols, &r, k);
        let a_sq = at_b_frobenius_sq(&a_proj, &a_proj, rows, k);
        let b_sq = at_b_frobenius_sq(&b_proj, &b_proj, rows, k);
        let cka = linear_cka(&a_proj, &b_proj, rows, k, a_sq, b_sq);
        // Exact CKA = 0; projected estimate ~ small (projection-noise).
        assert!(cka < 0.1, "orthogonal CKA should be near 0, got {cka}");
    }

    #[test]
    fn projection_preserves_cka_approximately() {
        // Pick two arbitrary matrices, compute CKA at high k (close to
        // exact) and low k (more noise). The two values should agree
        // to within a few percent — the projection is unbiased and
        // its variance decreases as k grows.
        let rows = 32;
        let cols = 64;
        let w_a = ramp_matrix(rows, cols);
        let w_b: Vec<f32> = (0..rows * cols).map(|k| (k as f32 * 0.07).sin()).collect();

        let cka_for_k = |k: usize| {
            let r = gaussian_projection(k, cols, 42);
            let a_proj = project_rows(&w_a, rows, cols, &r, k);
            let b_proj = project_rows(&w_b, rows, cols, &r, k);
            let a_sq = at_b_frobenius_sq(&a_proj, &a_proj, rows, k);
            let b_sq = at_b_frobenius_sq(&b_proj, &b_proj, rows, k);
            linear_cka(&a_proj, &b_proj, rows, k, a_sq, b_sq)
        };
        let cka_high = cka_for_k(cols); // k = d_in: minimal projection loss
        let cka_low = cka_for_k(8);
        assert!(
            (cka_high - cka_low).abs() < 0.15,
            "CKA estimates should be close across projection dims: high={cka_high} low={cka_low}",
        );
    }

    #[test]
    fn cka_is_symmetric() {
        let rows = 12;
        let cols = 20;
        let w_a = ramp_matrix(rows, cols);
        let w_b: Vec<f32> = (0..rows * cols).map(|k| (k as f32).cos()).collect();
        let k = 16;
        let r = gaussian_projection(k, cols, 5);
        let a_proj = project_rows(&w_a, rows, cols, &r, k);
        let b_proj = project_rows(&w_b, rows, cols, &r, k);
        let a_sq = at_b_frobenius_sq(&a_proj, &a_proj, rows, k);
        let b_sq = at_b_frobenius_sq(&b_proj, &b_proj, rows, k);
        let cka_ab = linear_cka(&a_proj, &b_proj, rows, k, a_sq, b_sq);
        let cka_ba = linear_cka(&b_proj, &a_proj, rows, k, b_sq, a_sq);
        assert!(
            (cka_ab - cka_ba).abs() < 1e-6,
            "CKA(A, B) = CKA(B, A), got {cka_ab} vs {cka_ba}",
        );
    }

    #[test]
    fn zero_self_norm_clamps_to_zero() {
        // Degenerate edge case: an expert with all-zero weights. The
        // self-norm is 0, the ratio's denom is 0 — should return 0
        // (no signal), not NaN.
        let rows = 4;
        let k = 4;
        let a = vec![0.0f32; rows * k];
        let b: Vec<f32> = (0..rows * k).map(|i| i as f32).collect();
        let b_sq = at_b_frobenius_sq(&b, &b, rows, k);
        let cka = linear_cka(&a, &b, rows, k, 0.0, b_sq);
        assert_eq!(cka, 0.0);
    }
}
