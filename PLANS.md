# Plans

Planned features, written by the plan loop and implemented by the feature loop.
Each plan: goal, approach, files touched, acceptance criteria. Move finished plans to Done.

## Planned

### Support GGUF fused-expert tensors in `--moe` Summary scene

**Goal.** `--moe` currently hard-errors on GGUF MoE checkpoints (`ffn_{gate|up|down}_exps.weight`, e.g. Mixtral/Qwen GGUF quants) — see the bail in `open_moe_model_sources` (`src/data.rs`, the `is_fused_gguf_expert` rejection block, currently commented "GGUF fused-expert rejection — not yet supported by either scene"). The Summary scene should render these checkpoints like every other fused-layout export (HF batched `gate_up_proj` already works).

**Approach.**
1. `src/format/moe.rs`: add `parse_gguf_fused_expert(name) -> Option<(u32, ExpertWeight)>` for `blk.{L}.ffn_{gate|up|down}_exps.weight` (and the bare, no-`blk.` prefix form), mapping gate/up/down onto the existing `ExpertWeight` variants. Add `parse_gguf_router(name) -> Option<u32>` for `blk.{L}.ffn_gate_inp.weight` (also matched without the `blk.` prefix). Unit-test both, including non-collision with `parse_hf_expert` / `parse_hf_router` / plain `ffn_{gate|up|down}.weight` (per-tensor, not `_exps`) names, in the style of the existing `detects_gguf_fused_experts` test.
2. `src/data.rs`, `build_moe_summary_sources`: after the header scan, collect GGUF fused tensors into the existing `fused` path via a new `build_gguf_fused_expert_jobs` helper alongside `build_fused_expert_jobs`. GGUF fused tensors are `[n_experts, …, …]` row-major, so expert `e` occupies the contiguous sub-range `[file_start + e*stride, …)` with `stride = (file_end - file_start) / shape[0]`; for block-quantized dtypes blocks run along the last dim, so expert slices stay block-aligned and the existing `scalar_from_buf` → `format::*_from_buf` dequant path applies unchanged. Derive `n_experts` per layer from `shape[0]` (feeds the existing `fused_n_experts` max). Route the per-layer GGUF router (`ffn_gate_inp.weight`, shape `[n_experts, hidden]`) through the existing per-row router-slicing path.
3. Narrow the rejection: `open_moe_model_sources` stops bailing on `is_fused_gguf_expert`; the CKA scene keeps declining GGUF fused checkpoints with its existing warning-and-skip behavior (per-expert CKA on fused tensors is a separate follow-up, not this plan).

**Files touched.** `src/format/moe.rs` (parsers + tests), `src/data.rs` (summary job building, bail removal, CKA-skip guard).

**Acceptance criteria.**
- `parse_gguf_fused_expert` / `parse_gguf_router` unit tests pass, including rejection of non-fused GGUF tensor names and non-collision with HF parsers.
- A synthetic GGUF MoE file (or a unit-level test of `build_gguf_fused_expert_jobs` over hand-built `TensorMeta`s with `shape = [E, h, f]`) yields per-expert byte sub-ranges whose union covers the tensor exactly, with no off-by-one on the last expert.
- The `--moe` Summary scene no longer bails on a GGUF checkpoint containing `ffn_gate_exps.weight`; the CKA scene logs its skip warning instead of failing the run.
- `cargo test` passes; no change to Summary output for HF per-expert and HF batched-fused checkpoints (existing tests stay green).

_Found by plan loop 2026-10-09._


## Done

_None yet._
