# modelweightvis

Tensor-format-aware visualization for ML model weights, built on [arbvis](https://github.com/znation/arbvis). Renders `.safetensors` / `.gguf` / PyTorch `.bin` / `.pth` / `.pt` checkpoints at each tensor's natural element shape — 1 px = 1 element — and stacks transformer blocks vertically so corresponding sub-tensors (`q_proj`, `gate_proj`, etc.) line up across every layer. Block-to-block changes — quantization steps, finetune deltas, dead heads — appear as horizontal bands.

**For non-tensor files** (binaries, JSON, anything else), use [**arbvis**](https://github.com/znation/arbvis) directly. modelweightvis is a thin crate that adds tensor awareness on top of arbvis: it registers `FormatPlugin` / `LayoutPlugin` / `DiffSourceBuilder` impls and CLI dispatch hooks against arbvis's registry, then hands the actual rendering, Hub I/O, tile pyramid, and Space deploy off to arbvis. The `modelweightvis` binary inherits arbvis's full CLI surface — `--tiles`, `--space`, `--stream`, `--show-xet-xorbs`, etc. — so you don't need to use both. See [Relationship to arbvis](#relationship-to-arbvis) below for the architectural picture.

## Quick start

```sh
modelweightvis hf://meta-llama/Llama-3.2-1B --tiles ./out
# then open out/index.html in a browser
```

`hf://` inputs are fetched directly — no manual download. The output is a [Leaflet.js](https://leafletjs.com/) tile pyramid you can zoom across; at maximum zoom, one pixel is one tensor element.

## What modelweightvis adds

### Architectural layout

When every input is safetensors with transformer-style tensor names, modelweightvis renders each tensor at its 2D element shape and stacks transformer blocks vertically. Corresponding sub-tensors (e.g. `q_proj` across every layer) are pixel-aligned, so block-to-block changes line up as horizontal bands.

Override with `--layout auto|arch|hilbert`:
- `auto` (default) — architectural if every input is safetensors with detectable transformer structure; otherwise byte-Hilbert.
- `arch` — force architectural. Falls back to byte-Hilbert if no input is safetensors.
- `hilbert` — force the byte-only Hilbert layout (1 px = 1 byte) for regression checks against arbvis output.

### Tensor format parsing

- **safetensors** (`.safetensors`) — single file or sharded index. Header is range-fetched for `hf://` inputs.
- **GGUF** (`.gguf`) — quantized weights are dequantized for diffing.
- **PyTorch pickle** (`.bin` / `.pth` / `.pt`) — parsed without invoking `__reduce__` / `find_class`, so loading untrusted headers is safe. Remote pickle isn't supported (the zip end-of-central-directory lives at the file tail and can't be prefix-fetched).

Diffs match tensors by canonical name across formats, so a GGUF checkpoint diffs cleanly against the corresponding safetensors release.

### Dtype-aware element coloring

Tensor elements (not just raw bytes) are colored according to dtype:
- Float values are mapped through a perceptually-uniform brightness scale honoring sign and magnitude.
- Integer-quant elements (GGUF `Q4_K`, `Q8_0`, etc.) are dequantized first, then colored on the float scale.
- Padding regions and unused shard slots render as a recognizable non-pure-black so they're distinguishable from real zero-valued elements.

This applies in both the normal arch render and the `--show-xet-xorbs` xet-coloring path inherited from arbvis (hue from xorb ID, intensity from element value instead of raw byte).

## Comparing two models: `--diff`

```sh
modelweightvis --diff hf://meta-llama/Llama-3.2-1B hf://meta-llama/Llama-3.2-1B-Instruct --tiles ./out
```

Per-tensor element-wise diff between two checkpoints (local files, directories, or `hf://` URLs). Each pixel encodes a signed delta: **black** for identical, **green** for values that grew, **red** for values that shrank, **white** for non-finite results.

### Diff metric (`--diff-metric`)

- `rms` (default) — per-tensor RMS-normalized signed delta. Stable across tensors of wildly different scale.
- `abs-log` — absolute delta on a log brightness scale. Honest about raw magnitudes.
- `exact` — ternary: identical bytes → black; any change → full saturation.

### Finetune mode (`--finetune` / `--no-finetune`)

When both arguments are `hf://` model URLs, modelweightvis auto-detects whether the second is declared as a finetune of the first via the HF model card (`base_model` + `base_model_relation`). In finetune mode, tensors present only on the base side render as grey crosshatch (informational); anything new on the finetune side or with a mismatched shape aborts the run. Pass `--finetune` to force the relation on, `--no-finetune` to force it off.

Non-tensor files in a `--diff` between repos or directories (READMEs, tokenizer configs, etc.) fall back to arbvis's plain-byte / JSON-aware diff path.

## MoE expert-vs-expert matrix: `--moe-diff`

```sh
modelweightvis --moe-diff hf://mistralai/Mixtral-8x7B-v0.1 --tiles ./out
```

Renders an N×N grid showing the element-wise diff of every expert pair within a single MoE model. Each cell stacks `gate_proj`, `up_proj`, and `down_proj` horizontally; only the upper triangle and diagonal are drawn (the raw diff is antisymmetric).

Supports HF-style per-expert safetensors layouts: Mixtral, Qwen3-MoE, OLMoE, and DeepSeek routed experts. GGUF fused-expert tensors are not yet supported.

## Inherited from arbvis

modelweightvis inherits arbvis's full CLI surface. The output destinations, Hub I/O, and viewer-side flags work the same on tensor-aware renders:

- `--tiles DIR` — zoomable Leaflet tile pyramid (recommended).
- `--output FILE` — single PNG (capped at 4096×4096; for full resolution use `--tiles`).
- `--space OWNER/REPO` — deploy a Docker Space serving the Leaflet viewer.
- `--stream` — keep `hf://` inputs remote and push tiles to the Hub as they're produced.
- `--show-xet-xorbs` — color regions by xorb ID for xet-backed inputs (hue per xorb).
- `--tile-format avif|png`, `--regen-html DIR`, `--title TEXT`, `-l/--file-list FILE`, and `hf://` output for both `--output` and `--tiles`.

See the [arbvis README](https://github.com/znation/arbvis#readme) for the full reference on all of these.

## Relationship to arbvis

[arbvis](https://github.com/znation/arbvis) is the byte-only foundation: Hilbert layout, byte coloring, JSON-aware diff, Hub I/O, tile pyramid, Space deploy, xet xorb path, streaming. It has no knowledge of tensors or model formats — `.safetensors` and `.gguf` get the same byte-Hilbert treatment as any other binary.

modelweightvis extends arbvis through its plugin / hook surface — no fork, no patch:

- `FormatPlugin` impls (`SafetensorsFormatPlugin`, `GgufFormatPlugin`, `PickleFormatPlugin`) parse each format's header and stuff a `ModelInfo` (tensors + dtype color ranges) into the source's `extensions` map.
- `LayoutPlugin` impls (`ArchLayoutPlugin`, `MoeDiffLayoutPlugin`) register the architectural canvas and the MoE-diff matrix layout; arbvis's plugin-iteration `select_layout` picks them by priority.
- `LeafLoader` + `LeafRenderer` pair (`ArchRegionsLoader`, `ArchRegionsRenderer`) drive per-tensor tile rendering at element granularity.
- `DiffSourceBuilder` (`TensorDiffBuilder`) handles tensor-aware `--diff` at priority above arbvis's JSON / plain-byte fallbacks.
- Option-slot hooks (`MoeDiffPrep`, `RepoDiffPrep`, `DirectoryTensorDiffPrep`, `FinetuneDetect`, `SingleImageArchHook`, `PrepareSourcesExtension`) tap arbvis's CLI dispatch points so `--moe-diff`, repo-level `--diff`, single-image arch render, and HF model-card finetune detection slot in cleanly.

The `modelweightvis` binary itself is tiny — it builds an `arbvis::Registry::with_defaults()`, calls `modelweightvis::register_all(&mut registry)`, and hands off to `arbvis::run`. Same renderer, same Hub I/O, same tile pyramid; the tensor awareness comes entirely through the registered plugins.

**Which to use:**
- **modelweightvis** — for `.safetensors` / `.gguf` / `.bin` model checkpoints, architectural transformer layout, `--moe-diff`, `--diff-metric`, `--finetune` / `--no-finetune`, `--layout`, dtype-aware coloring. Inherits arbvis's full CLI surface.
- **arbvis** — for non-model binaries (any file format), JSON/JSONL diffs, plain-byte diffs, the xet xorb path on arbitrary content. Smaller dependency footprint (no `candle-core` / `regex` / `zip` / `half`).

## Building

Requires Rust (stable).

```sh
cargo build --release
./target/release/modelweightvis <model.safetensors> --tiles ./output
```

Or install into your `PATH`:

```sh
cargo install --path .
```

`Cargo.toml` pins arbvis to a specific commit on [znation/arbvis](https://github.com/znation/arbvis) via `arbvis = { git = "…", rev = "…" }`. Cargo fetches and builds it on first build; no sibling checkout needed. Bump the `rev` to pick up new arbvis changes.

### Local arbvis co-development

If you're editing arbvis and want modelweightvis to build against your local checkout instead of the pinned git rev, create a gitignored `.cargo/config.toml` next to this repo's `Cargo.toml`:

```toml
# .cargo/config.toml — not committed; per-developer override.
[patch."https://github.com/znation/arbvis"]
arbvis = { path = "../arbvis" }
```

Layout:

```
~/your-code/
├── arbvis/            # https://github.com/znation/arbvis
└── modelweightvis/    # this repo (with .cargo/config.toml above)
```

`cargo build` will now resolve arbvis from `../arbvis` instead of GitHub. Remove the file (or the `[patch]` block) to switch back to the pinned rev.

## Credits

Built on [arbvis](https://github.com/znation/arbvis) for everything not tensor-aware. Tensor parsing leans on [candle-core](https://crates.io/crates/candle-core) (GGUF dequantization, pickle reading), [half](https://crates.io/crates/half) (FP16 / BF16), [zip](https://crates.io/crates/zip) (PyTorch `.bin` zip-entry resolution), and [regex](https://crates.io/crates/regex) (transformer-style tensor-name classification).
