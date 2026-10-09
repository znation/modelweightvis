# Bugs

Known bugs, recorded by any loop and fixed by the bugfix loop.
Each bug: symptom, how to reproduce, suspected cause if known. Move fixed bugs to Fixed, with the
required `**Validation gap:** <tag> — <one sentence>` line recording what made the bug hard to
confirm (tag one of: none, no-repro, no-fake, real-run-needed, no-observability, slow-check,
unclear-invariant).

## Open

### AWQ/GPTQ packed-int tensors render as NaN sentinels — sidecars never attached in production (found by tumwater(clean) 2026-10-09)

`TensorElementReader::with_sidecars` (`src/format/dtype.rs`) is the only way to
attach the `scales`/`qzeros` buffers AWQ/GPTQ packed-int (`Int4Packed` etc.)
decoding needs, but no production call site attaches them — the only caller is
a test (`src/format/dtype.rs`, `packed_without_sidecars_returns_nan` area);
all production readers are built with `TensorElementReader::new(...)`
(`src/layout/render.rs`, `src/tiled/arch_voxel.rs`, `src/data.rs`). Without
sidecars, packed dtypes decode to NaN by design, so those tensors paint as
sentinels instead of real magnitudes. The metadata is already parsed:
`TensorMeta::packed_sidecars` is populated by
`crate::format::safetensors::fuse_packed_quant_triples` but never read.
Fix: read `packed_sidecars` in the tile/diff render paths and thread the
corresponding scales/qzeros byte ranges through `with_sidecars`.

### Structural risk: `src/data.rs` (3347 lines) and `src/layout/arch.rs` (2080 lines) exceed the one-sitting readability principle (found by tumwater(steward) 2026-10-09)

PRINCIPLES.md holds "keep each file focused on one responsibility, and small enough to read
in one sitting". `src/data.rs` mixes MoE source loading, Summary/CKA job building, and their
screens worth of unit tests; `src/layout/arch.rs` similarly concentrates detection plus ~860 lines of tests. Not a runtime bug — a maintainability risk: the open GGUF fused-expert plan
(`PLANS.md`) would add more to `data.rs` before any split. Fix candidate, when a bugfix tick
wants a pure-refactor move: lift the `#[cfg(test)]` modules into `tests/` or split data.rs into
`moe_sources` / `summary_jobs` modules; no behavior change, existing tests as the harness.

## Fixed

_None yet._
