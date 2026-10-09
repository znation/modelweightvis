# modelweightvis — project brief

## Initial prompt

<!-- tumwater:prompt:start -->
Tensor-format-aware visualization for ML model weights, built on [arbvis](https://github.com/znation/arbvis). Renders `.safetensors` / `.gguf` / PyTorch `.bin` / `.pth` / `.pt` checkpoints at each tensor's natural element shape — 1 px = 1 element — and stacks transformer blocks vertically so corresponding sub-tensors (`q_proj`, `gate_proj`, etc.) line up across every layer. Block-to-block changes — quantization steps, finetune deltas, dead heads — appear as horizontal bands.

**For non-tensor files** (binaries, JSON, anything else), use [**arbvis**](https://github.com/znation/arbvis) directly. modelweightvis is a thin crate that adds tensor awareness on top of arbvis: it registers `FormatPlugin` / `LayoutPlugin` / `DiffSourceBuilder` impls and CLI dispatch hooks against arbvis's registry, then hands the actual rendering, Hub I/O, tile pyramid, and Space deploy off to arbvis. The `modelweightvis` binary inherits arbvis's full CLI surface — `--out`, `--space`, `--3d`, `--stream`, `--show-xet-xorbs`, etc. — so you don't need to use both. See [Relationship to arbvis](#relationship-to-arbvis) below for the architectural picture.
<!-- tumwater:prompt:end -->

## Summary

modelweightvis is a thin Rust crate on top of [arbvis](https://github.com/znation/arbvis): it registers tensor-format plugins (safetensors, GGUF, PyTorch pickle), architectural / MoE layouts, and tensor-aware diff providers against arbvis's registry, and hands rendering, Hub I/O, tiling, and Space deploy back to arbvis.

## Status

<!-- tumwater:status:start -->
Working: arch layout (2D + `--3d`), `--diff` with `--diff-metric` / finetune detection, `--moe` summary + CKA scenes with optional `--probe`, dtype-aware coloring; full usage in [README.md](README.md).

Open work is tracked in [PLANS.md](PLANS.md) (plans), [BUGS.md](BUGS.md) (bugs), and [QUESTIONS.md](QUESTIONS.md) (questions for a human).
<!-- tumwater:status:end -->

## Usage

```sh
cargo install --path .
modelweightvis hf://meta-llama/Llama-3.2-1B --out ./out   # then open out/index.html
```

All flags (`--diff`, `--moe`, `--3d`, `--out`, `--space`, …) and architecture notes: [README.md](README.md).
