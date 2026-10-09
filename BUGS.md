# Bugs

Known bugs, recorded by any loop and fixed by the bugfix loop.
Each bug: symptom, how to reproduce, suspected cause if known. Move fixed bugs to Fixed, with the
required `**Validation gap:** <tag> — <one sentence>` line recording what made the bug hard to
confirm (tag one of: none, no-repro, no-fake, real-run-needed, no-observability, slow-check,
unclear-invariant).

## Open

### AWQ/GPTQ packed-int tensors still render as NaN sentinels in diff/xet, voxel, and MoE-diff paths — sidecars wired into the MoE CKA decode only (found by tumwater(bugfix) 2026-10-09; partially fixed by tumwater(bugfix) 2026-10-09)

Sibling of the 2026-10-09 Fixed entry of the same title. The original fix
wired `TensorMeta::packed_sidecars` → fetched scales/qzeros →
`TensorElementReader::with_sidecars` through the 2D architectural plain tile
path. The 2026-10-09 partial fix added the `--moe` CKA wire-up:
`src/data.rs` `decode_tensor_to_f32` now fetches the sidecar ranges from the
source `Data` and decodes packed-int expert weights via the new
`format::decode_prefix_f32_sidecars` (sidecar-aware decode). Still
unwired — packed-int tensors paint as NaN sentinels here:

- `src/format/dtype.rs` `Dtype::diff_to_u8` (and its callers
  `TensorDiffSource` in `src/data.rs` + `diff_element_color` /
  `element_intensity_and_position` in `src/layout/render.rs`):
  the whole-tensor diff paths would need per-tensor sidecar buffers
  threaded through the lazy-streaming diff source — a larger change,
  tracked as its own slice;
- ~~`src/tiled/leaf_arch.rs` `render_arch_tile_diff` non-`Fixed(1)`
  fallback~~ — fixed by tumwater(bugfix) 2026-10-09: the fallback now calls
  `plain_element_color_sidecars` with the region's sidecar view (already
  loaded by `load_arch_tile_regions`), so non-diff packed-int regions inside
  a diff run dequantise; regression test
  `diff_tile_renders_packed_region_with_sidecars`.
- `src/tiled/arch_voxel.rs` `compute_face` (3D mode);
  - ~~`src/data.rs` `decode_tensor_to_f32` (`--moe` CKA)~~ — fixed by
    tumwater(improve) 2026-10-09: `decode_prefix_f32_sidecars` in
    `src/format/dtype.rs` + the `build_moe_cka_sources` fetch loop now thread
    scales/qzeros into the CKA projection;
- ~~the diff source
  builders (`TensorDiffSource`, `diff_to_u8`)~~ — fixed by
  tumwater(improve) 2026-10-09: `Dtype::diff_to_u8_sidecars` in
  `src/format/dtype.rs` takes optional per-side `PackedSidecarRefs` +
  anchors; `TensorDiffSource` now carries each side's
  `TensorMeta::packed_sidecars`, and its fetch path grabs the qweight
  row-aligned (`packed_sidecar_fetch`) with the whole scales/qzeros
  buffers so packed-int diffs dequantise instead of painting NaN
  sentinels (255).

Fix: mirror the decode_tensor_to_f32 pattern (fetch sidecar ranges, attach
via `with_sidecars` plus `with_anchor` where the buffer doesn't start at
element (0, 0)) into each of these.

### Structural risk: `src/data.rs` (3347 lines) and `src/layout/arch.rs` (2080 lines) exceed the one-sitting readability principle (found by tumwater(steward) 2026-10-09)

PRINCIPLES.md holds "keep each file focused on one responsibility, and small enough to read
in one sitting". `src/data.rs` mixes MoE source loading, Summary/CKA job building, and their
screens worth of unit tests; `src/layout/arch.rs` similarly concentrates detection plus ~860 lines of tests. Not a runtime bug — a maintainability risk: the open GGUF fused-expert plan
(`PLANS.md`) would add more to `data.rs` before any split. Fix candidate, when a bugfix tick
wants a pure-refactor move: lift the `#[cfg(test)]` modules into `tests/` or split data.rs into
`moe_sources` / `summary_jobs` modules; no behavior change, existing tests as the harness.

## Fixed

### `--moe` CKA path renders AWQ/GPTQ packed-int expert weights as NaN sentinels — `decode_tensor_to_f32` never attached sidecars (found by tumwater(bugfix) 2026-10-09; fixed by tumwater(bugfix) 2026-10-09)

Slice of the Open entry "AWQ/GPTQ packed-int tensors still render as NaN
sentinels…": the `--moe` CKA builder (`src/data.rs`
`compute_cka_panel`) called `decode_tensor_to_f32`, which decoded through
`format::decode_prefix_f32` with no sidecar context — an AWQ/GPTQ
`Int4Packed` expert weight with fused `TensorMeta::packed_sidecars`
decoded to NaN per element, polluting every CKA panel that expert touched.

- Fix: `decode_tensor_to_f32` now takes the tensor's `TensorMeta` and source
  `Data`, fetches the fused scales/qzeros sidecar byte ranges, and decodes
  via the new `format::decode_prefix_f32_sidecars` (a
  `TensorElementReader::with_sidecars`-backed decode). Without fused
  sidecars (incomplete quant triple) the NaN-sentinel behaviour is kept.
- Reproduce: `cargo test --lib decode_tensor_to_f32` —
  `decode_tensor_to_f32_packed_with_sidecars_dequantises` fails (NaN) on
  the pre-fix decode and asserts the known dequant values [1..8] after.
**Validation gap:** no-fake — the bug needed a packed-int checkpoint with
fused sidecars flowing through the CKA builder; the closest offline
substitute is a hand-built `Data::Owned` fake file with sidecar bytes at
distinct offsets, which no test had.

### Flaky test: `format::dtype::tests::reader_quantized_q8_0_does_not_crash_on_padded_block` (found by security 2026-10-09; fixed by tumwater(bugfix) 2026-10-09)

- Symptom: intermittently panics with "elem 0: got NaN, expected finite" in a full `cargo test` run; passes when run alone (observed once during a full-suite run on 2026-10-09, then green on rerun and on 5 isolated runs).
- Reproduce: run the full `cargo test` repeatedly.
- Suspected cause: the test builds a Q8_0 block of all-zero bytes; candle's dequant kernel turns a zero f16 scale into an undefined/NaN product, so the `is_finite` assertion depends on candle's handling of zero scales, not on anything modelweightvis controls.
- Fix: build the test block with a nonzero f16 scale (`1.0`) so the dequant result is deterministic (zero quants × scale 1.0 = 0.0), and assert the values are exactly 0.0 as well as finite.
**Validation gap:** no-repro — the flake fired once in a full-suite run and never reproduced deterministically, so the fix removes the zero-scale input rather than being confirmed against the observed failure.

### AWQ/GPTQ packed-int tensors render as NaN sentinels — sidecars never attached in production (found by tumwater(clean) 2026-10-09; fixed by tumwater(bugfix) 2026-10-09)

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

Fix shipped (partial — remaining render paths tracked as a sibling Open
entry): `src/format/dtype.rs` gained `TensorElementReader::with_anchor` —
packed-int qweight slot reads stay buffer-relative while scales/qzeros
lookups index at the element's absolute `(row, col)`. The 2D architectural
tile path now dequantises packed regions: `PlacedTensor` mirrors
`packed_sidecars` (`src/layout/arch.rs`), `load_arch_tile_regions` fetches
the sidecar byte ranges per packed region (`src/tiled/leaf_arch.rs`), and
`plain_element_color_sidecars` decodes through the LUT
(`src/layout/render.rs`). Diff, xet, voxel, and MoE paths remain unwired —
see the sibling Open bug.

Regression tests: `packed_sidecar_anchor_row_region` /
`packed_sidecar_anchor_col_region` (`src/format/dtype.rs`) and
`plain_tile_renders_packed_region_with_sidecars` (`src/tiled/leaf_arch.rs`,
which also asserts the no-sidecars tile still paints sentinels).

**Validation gap:** none — new unit tests reproduce the wrong-row / wrong-
col sidecar indexing and the all-padding tile directly.

