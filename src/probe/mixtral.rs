//! Routing-faithful forward pass for `MixtralForCausalLM`.
//!
//! Mirrors [`crate::probe::qwen2_moe`] but for Mixtral's MoE shape, which
//! differs in three ways: there is **no shared expert**, top-k is typically 2
//! (vs Qwen's 4), and the top-k routing weights are **always renormalised** to
//! sum to 1. Attention is fully bias-free (Qwen carries Q/K/V biases). The
//! transformer block is otherwise the standard pre-norm shape, reusing the
//! shared primitives in [`crate::probe::common`]:
//!
//! ```text
//! x = embed_tokens[input_ids]
//! for layer L in 0..N:
//!     residual = x
//!     x = rms_norm(x, layers[L].input_layernorm)
//!     x = residual + gqa_attention(x)                 # bias-free QKVO
//!
//!     residual = x
//!     x = rms_norm(x, layers[L].post_attention_layernorm)
//!     router_logits = x @ gate.T                      # [B, S, n_experts]
//!     (topk_w, topk_i) = top_k(softmax(router_logits), top_k)
//!     # CAPTURE topk_i for routing-frequency aggregation
//!     topk_w = renormalize(topk_w)                    # Mixtral always does this
//!     x = residual + sum_k topk_w * experts[topk_i](x)
//! ```
//!
//! ## Two on-disk weight layouts
//!
//! Mixtral checkpoints come in two flavours, both handled here (detected per
//! layer via the presence of the router tensor):
//!
//! - **Classic** — the originally-published `mistralai/Mixtral-8x7B-*` layout:
//!   router at `block_sparse_moe.gate`, experts as individual submodules
//!   `block_sparse_moe.experts.{e}.{w1,w2,w3}` (`w1` = gate, `w3` = up,
//!   `w2` = down).
//! - **Fused** — the newer `transformers` MoE layout: router at `mlp.gate`,
//!   experts as two batched tensors `mlp.experts.gate_up_proj`
//!   (`[n_experts, 2*intermediate, hidden]`, gate then up concatenated along
//!   the output dim) and `mlp.experts.down_proj` (`[n_experts, hidden,
//!   intermediate]`). We slice these into per-expert `Linear`s at load.

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
    /// Per-expert FFN intermediate dim. Mixtral spells this `intermediate_size`
    /// (unlike Qwen2-MoE, which uses `moe_intermediate_size` for routed
    /// experts and reserves `intermediate_size` for the shared expert).
    intermediate_size: usize,
    rms_norm_eps: f64,
    rope_theta: f64,
    max_pos: usize,
}

impl Cfg {
    fn from_config(config: &ModelConfig) -> anyhow::Result<Self> {
        let hidden_size = config
            .hidden_size
            .context("Mixtral config missing hidden_size")? as usize;
        let n_heads = config
            .num_attention_heads
            .context("Mixtral config missing num_attention_heads")? as usize;
        // Mixtral may omit head_dim from config; derive from hidden / n_heads.
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
            .context("Mixtral config missing num_local_experts")? as usize;
        let top_k = config
            .num_experts_per_tok
            .context("Mixtral config missing num_experts_per_tok")? as usize;
        let intermediate_size = config
            .intermediate_size
            .context("Mixtral config missing intermediate_size")?
            as usize;
        Ok(Self {
            vocab_size: config.vocab_size.unwrap_or(0) as usize,
            hidden_size,
            n_layers: config
                .num_hidden_layers
                .context("Mixtral config missing num_hidden_layers")?
                as usize,
            n_heads,
            n_kv_heads,
            head_dim,
            n_experts,
            top_k,
            intermediate_size,
            rms_norm_eps: config.rms_norm_eps.unwrap_or(1e-5),
            rope_theta: config.rope_theta.unwrap_or(1_000_000.0),
            max_pos: config.max_position_embeddings.unwrap_or(32768) as usize,
        })
    }
}

/// Public entry point. Loads the model weights from `weight_paths`, runs the
/// routing-faithful forward on `token_ids`, and returns the per-`(layer,
/// expert)` routing-frequency capture. Weights are streamed one layer at a
/// time (see [`crate::probe::qwen2_moe::run`] for the memory rationale).
pub fn run(
    config: &ModelConfig,
    weight_paths: &[PathBuf],
    _model_dir: &Path,
    token_ids: &[u32],
) -> anyhow::Result<RoutingCapture> {
    let cfg = Cfg::from_config(config)?;
    let device = Device::Cpu;

    let n_tokens = token_ids.len().min(cfg.max_pos);
    let input_ids = Tensor::from_slice(&token_ids[..n_tokens], (1, n_tokens), &device)?;

    // candle's CPU matmul rejects bf16 operands, so load f32. Stream one layer
    // at a time inside the loop to bound peak memory at a single layer's f32
    // weights (the `Layer` value drops at the end of each iteration).
    let dtype = DType::F32;
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(weight_paths, dtype, &device)
            .context("Mixtral: opening safetensors shards via VarBuilder")?
    };

    let embed_tokens =
        candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;
    let rope = RotaryEmbedding::new(cfg.head_dim, cfg.max_pos, cfg.rope_theta, &device)?;

    let mut hidden = embed_tokens.forward(&input_ids)?;

    // Per-layer routing-decision counts; divided by n_tokens at the end.
    let mut counts = vec![0u32; cfg.n_layers * cfg.n_experts];
    // Per-layer routing co-occurrence counts (`--moe-cka --probe`).
    let mut coact_counts = vec![0u32; cfg.n_layers * cfg.n_experts * cfg.n_experts];

    let layers_vb = vb.pp("model.layers");
    let pb = indicatif::ProgressBar::new(cfg.n_layers as u64);
    pb.set_message("probe: layer forward");
    for layer_idx in 0..cfg.n_layers {
        let layer = Layer::load(&layers_vb.pp(layer_idx), &cfg)?;

        // --- Attention (bias-free QKVO) -----------------------------------
        let residual = hidden.clone();
        let x = rms_norm(&hidden, &layer.input_layernorm, cfg.rms_norm_eps)?;
        let attn_out = layer.self_attn.forward(&x, &rope)?;
        hidden = (&residual + attn_out)?;

        // --- MoE (no shared expert) ---------------------------------------
        let residual = hidden.clone();
        let x = rms_norm(&hidden, &layer.post_attention_layernorm, cfg.rms_norm_eps)?;

        // Router: [B, S, H] @ [H, E] → [B, S, E]
        let router_logits = layer.gate.forward(&x)?;
        let router_probs_f32 =
            candle_nn::ops::softmax_last_dim(&router_logits.to_dtype(DType::F32)?)?;
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

        // Mixtral always renormalises the top-k routing weights to sum to 1.
        let topk_weights_renorm = renormalize_topk(&topk_weights_host, cfg.top_k);

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

        hidden = (&residual + moe_out)?;

        pb.inc(1);
    }
    pb.finish_and_clear();

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
// Per-layer module
// ============================================================================

struct Layer {
    input_layernorm: Tensor,
    self_attn: GqaAttention,
    post_attention_layernorm: Tensor,
    gate: Linear,
    experts: Vec<SwiGluExpert>,
}

impl Layer {
    fn load(vb: &VarBuilder, cfg: &Cfg) -> anyhow::Result<Self> {
        let input_layernorm = vb.pp("input_layernorm").get(cfg.hidden_size, "weight")?;
        let post_attention_layernorm = vb
            .pp("post_attention_layernorm")
            .get(cfg.hidden_size, "weight")?;
        // Mixtral attention is fully bias-free.
        let self_attn = GqaAttention::load(
            &vb.pp("self_attn"),
            cfg.hidden_size,
            cfg.n_heads,
            cfg.n_kv_heads,
            cfg.head_dim,
            /* qkv_bias */ false,
        )?;

        // Detect the MoE weight layout (see module docs).
        let (gate, experts) = if vb.contains_tensor("block_sparse_moe.gate.weight") {
            load_classic_moe(vb, cfg)?
        } else if vb.contains_tensor("mlp.experts.gate_up_proj") {
            load_fused_moe(vb, cfg)?
        } else {
            anyhow::bail!(
                "Mixtral: MoE layer has neither classic `block_sparse_moe.*` nor fused \
                 `mlp.experts.gate_up_proj` weights — unrecognised checkpoint layout"
            );
        };

        Ok(Self {
            input_layernorm,
            self_attn,
            post_attention_layernorm,
            gate,
            experts,
        })
    }
}

/// Load the classic published-Mixtral MoE layout: router at
/// `block_sparse_moe.gate`, experts as `block_sparse_moe.experts.{e}.{w1,w2,w3}`
/// submodules (`w1` = gate, `w3` = up, `w2` = down).
fn load_classic_moe(vb: &VarBuilder, cfg: &Cfg) -> anyhow::Result<(Linear, Vec<SwiGluExpert>)> {
    let moe_vb = vb.pp("block_sparse_moe");
    let gate = candle_nn::linear_no_bias(cfg.hidden_size, cfg.n_experts, moe_vb.pp("gate"))?;
    let mut experts = Vec::with_capacity(cfg.n_experts);
    for e in 0..cfg.n_experts {
        experts.push(SwiGluExpert::load(
            &moe_vb.pp(format!("experts.{e}")),
            cfg.hidden_size,
            cfg.intermediate_size,
            "w1", // gate
            "w3", // up
            "w2", // down
        )?);
    }
    Ok((gate, experts))
}

/// Load the newer fused `transformers` MoE layout: router at `mlp.gate`, and
/// two batched expert tensors under `mlp.experts`. `gate_up_proj` is
/// `[n_experts, 2*intermediate, hidden]` (gate rows then up rows along the
/// output dim); `down_proj` is `[n_experts, hidden, intermediate]`. We slice
/// per-expert weights out into individual bias-free `Linear`s.
fn load_fused_moe(vb: &VarBuilder, cfg: &Cfg) -> anyhow::Result<(Linear, Vec<SwiGluExpert>)> {
    let moe_vb = vb.pp("mlp");
    let gate = candle_nn::linear_no_bias(cfg.hidden_size, cfg.n_experts, moe_vb.pp("gate"))?;

    let experts_vb = moe_vb.pp("experts");
    let h = cfg.hidden_size;
    let inter = cfg.intermediate_size;
    let n = cfg.n_experts;

    let gate_up = experts_vb
        .get((n, 2 * inter, h), "gate_up_proj")
        .context("Mixtral fused MoE: loading mlp.experts.gate_up_proj")?;
    let down = experts_vb
        .get((n, h, inter), "down_proj")
        .context("Mixtral fused MoE: loading mlp.experts.down_proj")?;

    let mut experts = Vec::with_capacity(n);
    for e in 0..n {
        let gu = gate_up.get(e)?; // [2*inter, hidden]
        let gate_w = gu.narrow(0, 0, inter)?.contiguous()?; // [inter, hidden]
        let up_w = gu.narrow(0, inter, inter)?.contiguous()?; // [inter, hidden]
        let down_w = down.get(e)?.contiguous()?; // [hidden, inter]
        experts.push(SwiGluExpert::from_linears(
            Linear::new(gate_w, None),
            Linear::new(up_w, None),
            Linear::new(down_w, None),
        ));
    }
    Ok((gate, experts))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mixtral_config() -> ModelConfig {
        ModelConfig {
            architectures: vec!["MixtralForCausalLM".to_string()],
            num_hidden_layers: Some(32),
            hidden_size: Some(4096),
            num_attention_heads: Some(32),
            num_key_value_heads: Some(8),
            intermediate_size: Some(14336),
            vocab_size: Some(32000),
            num_local_experts: Some(8),
            num_experts_per_tok: Some(2),
            rms_norm_eps: Some(1e-5),
            rope_theta: Some(1_000_000.0),
            max_position_embeddings: Some(32768),
            ..Default::default()
        }
    }

    #[test]
    fn cfg_reads_mixtral_naming() {
        let cfg = Cfg::from_config(&mixtral_config()).expect("cfg parses");
        // Experts come from num_local_experts (Mixtral name), not num_experts.
        assert_eq!(cfg.n_experts, 8);
        assert_eq!(cfg.top_k, 2);
        // Expert FFN dim is intermediate_size (no moe_intermediate_size on Mixtral).
        assert_eq!(cfg.intermediate_size, 14336);
        // head_dim derived from hidden / n_heads when absent.
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.n_kv_heads, 8);
    }

    #[test]
    fn cfg_derives_head_dim_for_tiny() {
        // The tiny-random fixture omits head_dim and uses small dims.
        let mut c = mixtral_config();
        c.hidden_size = Some(64);
        c.num_attention_heads = Some(4);
        c.num_key_value_heads = Some(2);
        c.intermediate_size = Some(128);
        c.num_local_experts = Some(4);
        c.head_dim = None;
        let cfg = Cfg::from_config(&c).expect("cfg parses");
        assert_eq!(cfg.head_dim, 16); // 64 / 4
        assert_eq!(cfg.n_experts, 4);
    }

    #[test]
    fn cfg_missing_experts_errors() {
        let mut c = mixtral_config();
        c.num_local_experts = None;
        c.num_experts = None;
        assert!(Cfg::from_config(&c).is_err());
    }

    /// End-to-end forward against a real (if tiny) `MixtralForCausalLM`
    /// checkpoint. Ignored by default — set `MIXTRAL_TINY_DIR` to a local
    /// model directory (config.json + tokenizer.json + safetensors shards),
    /// e.g. the `hf-internal-testing/tiny-random-MixtralForCausalLM` fixture,
    /// then run with `cargo test --lib -- --ignored mixtral`.
    ///
    /// Validates the routing-faithful forward + the fused/classic expert
    /// loader: every token contributes exactly `top_k` routing decisions, so
    /// the per-layer routing-frequency mass must sum to `top_k`.
    #[test]
    #[ignore = "requires a local MixtralForCausalLM checkpoint; set MIXTRAL_TINY_DIR"]
    fn forward_on_local_checkpoint() {
        let dir = match std::env::var("MIXTRAL_TINY_DIR") {
            Ok(d) => std::path::PathBuf::from(d),
            Err(_) => return,
        };
        let config = crate::layout::model_config::ModelConfig::try_from_dir(&dir)
            .expect("config.json present and parseable");
        let mut shards: Vec<std::path::PathBuf> = std::fs::read_dir(&dir)
            .expect("read model dir")
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("safetensors"))
            .collect();
        shards.sort();
        assert!(
            !shards.is_empty(),
            "no safetensors shards in {}",
            dir.display()
        );

        let cap = crate::probe::run(
            crate::probe::Arch::Mixtral,
            &dir,
            &shards,
            &config,
            "The quick brown fox jumps over the lazy dog.",
        )
        .expect("Mixtral forward runs");

        assert_eq!(cap.n_layers, config.num_hidden_layers.unwrap());
        assert_eq!(cap.n_experts, config.n_experts().unwrap());
        assert!(cap.n_tokens > 0, "no tokens captured");

        // Each token routes to exactly top_k experts → per-layer freq mass = top_k.
        let k = config.num_experts_per_tok.unwrap() as f32;
        let ne = cap.n_experts as usize;
        for l in 0..cap.n_layers as usize {
            let mass: f32 = cap.freq[l * ne..(l + 1) * ne].iter().sum();
            assert!(
                (mass - k).abs() < 1e-3,
                "layer {l}: routing-freq mass {mass} != top_k {k}",
            );
        }

        // Co-activation matrix: right size, symmetric, and its diagonal equals
        // the per-expert routing frequency (a token co-activates an expert with
        // itself exactly when it selects it).
        assert_eq!(cap.coact.len(), cap.n_layers as usize * ne * ne);
        for l in 0..cap.n_layers as usize {
            let block = &cap.coact[l * ne * ne..(l + 1) * ne * ne];
            for i in 0..ne {
                for j in 0..ne {
                    assert!(
                        (block[i * ne + j] - block[j * ne + i]).abs() < 1e-6,
                        "layer {l}: co-activation asymmetry at ({i},{j})",
                    );
                }
                let diag = block[i * ne + i];
                let freq = cap.freq[l * ne + i];
                assert!(
                    (diag - freq).abs() < 1e-6,
                    "layer {l} expert {i}: co-activation diagonal {diag} != freq {freq}",
                );
            }
        }
    }

    /// Validate the classic `block_sparse_moe` loader against the fused loader:
    /// rewrite the fused tiny fixture into the classic on-disk layout (same
    /// weight *values*, different names/shapes) and confirm both produce a
    /// bit-identical routing capture. This is the only local check of the
    /// classic path — published Mixtral-8x7B (~90 GB) is infeasible to run
    /// here. Set `MIXTRAL_TINY_DIR` to the fused fixture and run with
    /// `cargo test --lib -- --ignored mixtral`.
    #[test]
    #[ignore = "requires MIXTRAL_TINY_DIR (fused tiny checkpoint)"]
    fn fused_and_classic_loaders_agree() {
        let dir = match std::env::var("MIXTRAL_TINY_DIR") {
            Ok(d) => std::path::PathBuf::from(d),
            Err(_) => return,
        };
        let config = crate::layout::model_config::ModelConfig::try_from_dir(&dir)
            .expect("config.json present and parseable");
        let text = "The quick brown fox jumps over the lazy dog.";

        let fused_shards = sorted_safetensors(&dir);
        let cap_fused = crate::probe::run(
            crate::probe::Arch::Mixtral,
            &dir,
            &fused_shards,
            &config,
            text,
        )
        .expect("fused forward runs");

        // Rewrite the same weights into the classic layout in a temp dir.
        let classic_dir = std::env::temp_dir().join("mwv-mixtral-classic-fixture");
        convert_fused_to_classic(&dir, &classic_dir, &config);
        let classic_shards = sorted_safetensors(&classic_dir);
        let cap_classic = crate::probe::run(
            crate::probe::Arch::Mixtral,
            &classic_dir,
            &classic_shards,
            &config,
            text,
        )
        .expect("classic forward runs");

        assert_eq!(cap_fused.n_layers, cap_classic.n_layers);
        assert_eq!(cap_fused.n_experts, cap_classic.n_experts);
        assert_eq!(
            cap_fused.freq, cap_classic.freq,
            "fused vs classic loaders disagree on routing",
        );
    }

    fn sorted_safetensors(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
        let mut v: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
            .expect("read dir")
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("safetensors"))
            .collect();
        v.sort();
        v
    }

    /// Rewrite a fused-layout Mixtral checkpoint (`mlp.gate` +
    /// `mlp.experts.gate_up_proj`/`down_proj`) into the classic published
    /// layout (`block_sparse_moe.gate` + `block_sparse_moe.experts.{e}.w1/w3/w2`)
    /// with identical weight values. Used only by the test above.
    fn convert_fused_to_classic(src: &std::path::Path, dst: &std::path::Path, cfg: &ModelConfig) {
        use std::collections::HashMap;
        let device = Device::Cpu;
        let inter = cfg.intermediate_size.unwrap() as usize;

        let mut all: HashMap<String, Tensor> = HashMap::new();
        for shard in sorted_safetensors(src) {
            for (k, v) in candle_core::safetensors::load(&shard, &device).expect("load shard") {
                all.insert(k, v);
            }
        }

        let mut out: HashMap<String, Tensor> = HashMap::new();
        for (name, t) in all {
            if let Some(prefix) = name.strip_suffix(".mlp.gate.weight") {
                out.insert(format!("{prefix}.block_sparse_moe.gate.weight"), t);
            } else if let Some(prefix) = name.strip_suffix(".mlp.experts.gate_up_proj") {
                let e_n = t.dim(0).unwrap();
                for e in 0..e_n {
                    let gu = t.get(e).unwrap(); // [2*inter, hidden]
                    let w1 = gu.narrow(0, 0, inter).unwrap().contiguous().unwrap();
                    let w3 = gu.narrow(0, inter, inter).unwrap().contiguous().unwrap();
                    out.insert(
                        format!("{prefix}.block_sparse_moe.experts.{e}.w1.weight"),
                        w1,
                    );
                    out.insert(
                        format!("{prefix}.block_sparse_moe.experts.{e}.w3.weight"),
                        w3,
                    );
                }
            } else if let Some(prefix) = name.strip_suffix(".mlp.experts.down_proj") {
                let e_n = t.dim(0).unwrap();
                for e in 0..e_n {
                    let w2 = t.get(e).unwrap().contiguous().unwrap(); // [hidden, inter]
                    out.insert(
                        format!("{prefix}.block_sparse_moe.experts.{e}.w2.weight"),
                        w2,
                    );
                }
            } else {
                out.insert(name, t);
            }
        }

        std::fs::create_dir_all(dst).unwrap();
        candle_core::safetensors::save(&out, dst.join("model.safetensors")).unwrap();
        std::fs::copy(src.join("tokenizer.json"), dst.join("tokenizer.json")).unwrap();
        std::fs::copy(src.join("config.json"), dst.join("config.json")).unwrap();
    }
}
