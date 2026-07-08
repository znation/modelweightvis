# modelweightvis

Tensor-format-aware visualization for ML model weights, built on [arbvis](https://github.com/znation/arbvis). Renders `.safetensors` / `.gguf` / PyTorch `.bin` / `.pth` / `.pt` checkpoints at each tensor's natural element shape — 1 px = 1 element — and stacks transformer blocks vertically so corresponding sub-tensors (`q_proj`, `gate_proj`, etc.) line up across every layer. Block-to-block changes — quantization steps, finetune deltas, dead heads — appear as horizontal bands.

**For non-tensor files** (binaries, JSON, anything else), use [**arbvis**](https://github.com/znation/arbvis) directly. modelweightvis is a thin crate that adds tensor awareness on top of arbvis: it registers `FormatPlugin` / `LayoutPlugin` / `DiffSourceBuilder` impls and CLI dispatch hooks against arbvis's registry, then hands the actual rendering, Hub I/O, tile pyramid, and Space deploy off to arbvis. The `modelweightvis` binary inherits arbvis's full CLI surface — `--out`, `--space`, `--3d`, `--stream`, `--show-xet-xorbs`, etc. — so you don't need to use both. See [Relationship to arbvis](#relationship-to-arbvis) below for the architectural picture.

## Quick start

```sh
modelweightvis hf://meta-llama/Llama-3.2-1B --out ./out
# then open out/index.html in a browser
```

`hf://` inputs are fetched directly — no manual download. The output is a [Leaflet.js](https://leafletjs.com/) tile pyramid you can zoom across; at maximum zoom, one pixel is one tensor element.

## What modelweightvis adds

### Architectural layout

When every input is safetensors with transformer-style tensor names, modelweightvis renders each tensor at its 2D element shape and stacks transformer blocks vertically. Corresponding sub-tensors (e.g. `q_proj` across every layer) are pixel-aligned, so block-to-block changes line up as horizontal bands.

Override with `--layout arch|hilbert`:
- `arch` (default) — force architectural, and **abort if it can't be built**. A model file that matched a format plugin but couldn't be parsed (e.g. an IQ-quant GGUF candle can't decode) has no tensor metadata, so the run errors out instead of silently degrading to a byte view — modelweightvis is for tensor files; use [arbvis](https://github.com/znation/arbvis) directly for arbitrary binaries. Non-tensor siblings (`config.json`, `tokenizer.json`, README) are ignored, so a full repo still renders. Applies to the 2D render, the `--3d` arch volume, and `--diff`; `--moe` renders its own scene layouts.
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
modelweightvis --diff hf://meta-llama/Llama-3.2-1B hf://meta-llama/Llama-3.2-1B-Instruct --out ./out
```

Per-tensor element-wise diff between two checkpoints (local files, directories, or `hf://` URLs). Each pixel encodes a signed delta: **black** for identical, **green** for values that grew, **red** for values that shrank, **white** for non-finite results.

### Diff metric (`--diff-metric`)

- `rms` (default) — per-tensor RMS-normalized signed delta. Stable across tensors of wildly different scale.
- `abs-log` — absolute delta on a log brightness scale. Honest about raw magnitudes.
- `exact` — ternary: identical bytes → black; any change → full saturation.

### Finetune mode (`--finetune` / `--no-finetune`)

When both arguments are `hf://` model URLs, modelweightvis auto-detects whether the second is declared as a finetune of the first via the HF model card (`base_model` + `base_model_relation`). In finetune mode, tensors present only on the base side render as grey crosshatch (informational); anything new on the finetune side or with a mismatched shape aborts the run. Pass `--finetune` to force the relation on, `--no-finetune` to force it off.

Non-tensor files in a `--diff` between repos or directories (READMEs, tokenizer configs, etc.) fall back to arbvis's plain-byte / JSON-aware diff path.

## MoE viewer: `--moe`

```sh
modelweightvis --moe hf://Qwen/Qwen1.5-MoE-A2.7B --out ./out
# then open out/index.html — toggle the Summary / CKA tabs (top-right)
```

`--moe` loads the model **once** and renders two complementary lenses as separate, tab-switchable scenes in the viewer (the tab switcher is the Leaflet base-layer control top-right). It renders the 2D tabbed viewer (`--out` or `--space`); it's incompatible with `--3d`, whose volume bundle can't host the tabs.

### Summary scene

One heatmap panel per FFN weight (`gate_proj`, `up_proj`, `down_proj`) plus the router gate, side by side. Each panel is a `layers × experts` grid with one colored cell per expert, so dead experts, per-layer magnitude trends, and outliers pop out at a glance. The scalar shown per cell is chosen by `--summary-stat`:

- `rms` (default) — √(mean(x²)), comparable across tensors of different scale.
- `frobenius` — √(sum(x²)), honest about total magnitude.
- `mean-abs` — mean(|x|), stable and dominated by typical entries.
- `sparsity` — fraction of near-zero entries, surfacing dead / near-dead experts.

Supports HF per-expert safetensors (Qwen-MoE, OLMoE, DeepSeek routed experts), the classic Mixtral `block_sparse_moe` layout, and the newer fused `transformers` export (batched `mlp.experts.gate_up_proj` / `down_proj`). GGUF fused-expert tensors are not yet supported.

### CKA scene

One `n_experts × n_experts` linear-CKA similarity heatmap per `(layer, weight)`: the diagonal is 1.0 (every expert is self-identical) and bright off-diagonal blocks reveal redundant expert clusters. Uses Gaussian random projection on the input axis (`--cka-sample`, default 128) to keep a 60-expert × 24-layer model tractable. The CKA scene needs **per-expert** tensors — on a fused-layout checkpoint it's skipped (with a warning) and only the Summary scene renders.

### Routing probe: `--probe`

```sh
modelweightvis --moe /path/to/Qwen1.5-MoE --probe --out ./out
```

Adds one behavioral panel to **each** scene from a routing-faithful forward pass over a probe input — the one signal that reflects which experts the router actually fires on real tokens, not just static weights:

- **Summary** gets a per-`(layer, expert)` **routing-frequency** panel.
- **CKA** gets a per-layer `n_experts × n_experts` **co-activation** grid: cell `(i, j)` is the fraction of probe tokens whose router top-k fired **both** expert `i` and `j` (diagonal `(i, i)` = expert `i`'s own frequency). CKA asks "which experts have similar *weights*"; co-activation asks "which experts actually *fire together*." Each layer's panel is per-panel normalized.

Override the bundled probe text with `--probe-text "…"`, `--probe-file <path>`, or `--probe-url <https|hf://…>`. The probe input must be a local directory (resolve `hf://` repos with `hf download …` first). Supported architectures: `Qwen2MoeForCausalLM`, `MixtralForCausalLM`. Probe failures are non-fatal — the static scenes still render.

## Inherited from arbvis

modelweightvis inherits arbvis's full CLI surface. The output destinations, Hub I/O, and viewer-side flags work the same on tensor-aware renders:

- `--out DIR` — write the viewer bundle here: a zoomable Leaflet tile pyramid (2D), or the Three.js volume bundle under `--3d`. Accepts a local dir or an `hf://` URL.
- `--3d` (with `--grid N`) — emit a Three.js volume viewer instead of the 2D pyramid, rendering the structure-aware arch layout as a voxel cube (transformer blocks stacked along Z, per-tensor magnitude baked to RGB). Above a small coarse cap the structured volume **streams by default** — a compact coarse grid ships in `volume.bin` while full detail is served on demand from range-fetched bricks — so large-model bundles stay small instead of shipping a dense multi-GB volume.
- `--volume-res N` (with `--3d`) — build the sparse brick volume at a higher virtual resolution than `--grid` (power of two, 8–2048). Only occupied bricks are stored and the viewer ray-guides them into a bounded GPU cache on demand, so VRAM tracks the visible working set rather than the data's total size — letting the volume far exceed the dense grid for sparse model weights. The same streaming applies to the default structured path above the coarse cap, not just when `--volume-res` is set.
- `--space OWNER/REPO` — deploy a Docker Space serving the viewer (works for both 2D and `--3d`).
- `--stream` — keep `hf://` inputs remote and push tiles to the Hub as they're produced.
- `--show-xet-xorbs` — color regions by xorb ID for xet-backed inputs (hue per xorb).
- `--tile-format avif|png`, `--regen-html DIR`, `--title TEXT`, `-l/--file-list FILE`.

See the [arbvis README](https://github.com/znation/arbvis#readme) for the full reference on all of these.

## Relationship to arbvis

[arbvis](https://github.com/znation/arbvis) is the byte-only foundation: Hilbert layout, byte coloring, JSON-aware diff, Hub I/O, tile pyramid, Space deploy, xet xorb path, streaming. It has no knowledge of tensors or model formats — `.safetensors` and `.gguf` get the same byte-Hilbert treatment as any other binary.

modelweightvis extends arbvis through its plugin / hook surface — no fork, no patch:

- `FormatPlugin` impls (`SafetensorsFormatPlugin`, `GgufFormatPlugin`, `PickleFormatPlugin`) parse each format's header and stuff a `ModelInfo` (tensors + dtype color ranges) into the source's `extensions` map.
- `LayoutPlugin` impls (`ArchLayoutPlugin`, `MoeSummaryLayoutPlugin`, `MoeCkaLayoutPlugin`) register the architectural canvas and the MoE summary / CKA panel layouts; arbvis's plugin-iteration `select_layout` picks them by priority.
- `LeafLoader` + `LeafRenderer` pair (`ArchRegionsLoader`, `ArchRegionsRenderer`) drive per-tensor tile rendering at element granularity.
- `DiffSourceBuilder` (`TensorDiffBuilder`) handles tensor-aware file-pair `--diff` at priority above arbvis's JSON / plain-byte fallbacks.
- `SourceProvider` impls (`MoeSceneProvider`, `RepoDiffProvider`, `TensorDiffProvider`) turn an invocation into render sources — `--moe`, a repo-level `hf://` `--diff`, and a directory `--diff` — each registered above arbvis's byte-diff / normal-bytes built-ins. Finetune detection (HF model card) is resolved inside the diff providers.
- `PrepareSourcesExtension` (`SourceMetaSidecarHook`) fetches `config.json` / index sidecars after the sources are built, enriching each with transformer hyperparameters the arch layout reads back.

The `modelweightvis` binary itself is tiny — it builds an `arbvis::Registry::with_defaults()`, calls `modelweightvis::register_all(&mut registry, &args)` (which also wires the parsed CLI flags into the providers and the registry's layout mode), and hands off to `arbvis::run`. Same renderer, same Hub I/O, same tile pyramid; the tensor awareness comes entirely through the registered plugins.

**Which to use:**
- **modelweightvis** — for `.safetensors` / `.gguf` / `.bin` model checkpoints, architectural transformer layout, `--moe` (tabbed summary + CKA scenes) / `--probe`, `--diff-metric`, `--finetune` / `--no-finetune`, `--layout`, dtype-aware coloring. Inherits arbvis's full CLI surface.
- **arbvis** — for non-model binaries (any file format), JSON/JSONL diffs, plain-byte diffs, the xet xorb path on arbitrary content. Smaller dependency footprint (no `candle-core` / `regex` / `zip` / `half`).

## Building

Requires Rust (stable).

```sh
cargo build --release
./target/release/modelweightvis <model.safetensors> --out ./output
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
