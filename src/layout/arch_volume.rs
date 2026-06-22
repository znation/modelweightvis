//! Structure-aware 3D (`--3d`) volume layout: the depth-stacked analog of
//! [`crate::layout::arch::ArchLayout`].
//!
//! arbvis's 3D path renders a [`arbvis::VolumeShape`] — a list of per-tensor
//! [`arbvis::VolumeEntity`] boxes in a bounded voxel cube — instead of laying
//! raw bytes on a blind 3D Hilbert curve. We build that shape by **reusing the
//! 2D arch layout wholesale**: [`ArchLayout::try_build`] already classifies
//! tensors into transformer blocks and lays each layer out as a canonical-slot
//! arrangement (so `q_proj` sits at the same in-layer position in every block).
//! We then project that 2D placement into the cube with one change of axis:
//!
//! - **Z (depth) = transformer layer.** Each block becomes a Z-slab; top-level
//!   tensors (embeddings, `lm_head`, final norms) cap the front.
//! - **X/Y = the in-layer canonical-slot position**, mapped onto the cube face.
//!   Because every layer shares that arrangement, a given sub-tensor forms a
//!   column through Z — so cross-layer change reads as variation along depth.
//!
//! The matching [`crate::tiled::arch_voxel`] `VoxelRenderer` colors each box by
//! aggregated per-element magnitude. Selection mirrors [`ArchLayoutPlugin`]:
//! eligible when sources carry `ModelInfo` and `--layout` isn't `hilbert`.
//!
//! [`ArchLayoutPlugin`]: crate::layout::ArchLayoutPlugin

use std::any::Any;

use arbvis::{
    LayoutBuildCtx, Source, VolumeEntity, VolumeLabel, VolumeShape, VolumeShapePlugin, VoxelBox,
};

use crate::data::SourceMeta;
use crate::layout::arch::ArchLayout;
use crate::tiled::arch_voxel::ArchVoxelExtra;

/// A placed tensor, pre-computed at build time. Kept Clone-able (plain data) so
/// [`ArchVolume::entities`] can materialize the arbvis entity list — whose
/// `extra: Box<dyn Any>` isn't itself Clone — fresh on demand.
#[derive(Clone)]
struct EntityDesc {
    source_idx: usize,
    byte_start: u64,
    byte_len: u64,
    bbox: VoxelBox,
    extra: ArchVoxelExtra,
    /// Display name + coarse group for the viewer's click-to-pick manifest.
    name: String,
    group: String,
}

/// A structure-aware 3D volume layout (the `"arch"` [`VolumeShape`]).
pub struct ArchVolume {
    side: u32,
    descs: Vec<EntityDesc>,
    focus: ([f32; 3], f32),
}

impl VolumeShape for ArchVolume {
    fn id(&self) -> &'static str {
        "arch"
    }
    fn grid_side(&self) -> u32 {
        self.side
    }
    fn entities(&self) -> Option<Vec<VolumeEntity>> {
        Some(
            self.descs
                .iter()
                .map(|d| VolumeEntity {
                    source_idx: d.source_idx,
                    byte_start: d.byte_start,
                    byte_len: d.byte_len,
                    bbox: d.bbox,
                    renderer_id: "arch",
                    extra: Box::new(d.extra.clone()),
                })
                .collect(),
        )
    }
    fn focus(&self) -> Option<([f32; 3], f32)> {
        Some(self.focus)
    }
    fn manifest(&self) -> Vec<VolumeLabel> {
        self.descs
            .iter()
            .map(|d| VolumeLabel {
                name: d.name.clone(),
                group: d.group.clone(),
                bbox: d.bbox,
            })
            .collect()
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl ArchVolume {
    /// Project a built [`ArchLayout`] into a `side³` cube. Returns `None` when
    /// the layout has no transformer blocks (a flat single slab isn't worth a
    /// structured 3D render — let arbvis's byte-Hilbert floor handle it).
    fn from_arch(
        arch: &ArchLayout,
        sources: &[Source],
        cumulative_offsets: &[u64],
        side: u32,
    ) -> Option<Self> {
        if arch.layer_bounds.is_empty() || side == 0 {
            return None;
        }

        // Layer ordinals: distinct block indices in ascending order, plus a
        // per-layer canvas origin. A uniform in-layer extent (max over blocks)
        // keeps a sub-path's slot mapping to the same face region in every
        // layer even when some blocks are narrower (missing/padded tensors).
        let mut layers: Vec<u32> = arch.layer_bounds.iter().map(|b| b.layer_idx).collect();
        layers.sort_unstable();
        layers.dedup();
        let ordinal = |idx: u32| layers.iter().position(|&l| l == idx).unwrap_or(0);
        let origin = |idx: u32| -> (u32, u32) {
            arch.layer_bounds
                .iter()
                .find(|b| b.layer_idx == idx)
                .map(|b| (b.canvas_x, b.canvas_y))
                .unwrap_or((0, 0))
        };
        let layer_w = arch
            .layer_bounds
            .iter()
            .map(|b| b.width)
            .max()
            .unwrap_or(1)
            .max(1);
        let layer_h = arch
            .layer_bounds
            .iter()
            .map(|b| b.height)
            .max()
            .unwrap_or(1)
            .max(1);

        // Top-level tensors (embeddings / lm_head / final norm) share a front
        // cap; map them relative to their own bounding box.
        let top: Vec<&_> = arch
            .tensors
            .iter()
            .filter(|t| t.layer_idx.is_none())
            .collect();
        let has_top = !top.is_empty();
        let (top_ox, top_oy, top_w, top_h) = if has_top {
            let minx = top.iter().map(|t| t.canvas_x).min().unwrap_or(0);
            let miny = top.iter().map(|t| t.canvas_y).min().unwrap_or(0);
            let maxx = top.iter().map(|t| t.canvas_x + t.disp_w).max().unwrap_or(1);
            let maxy = top.iter().map(|t| t.canvas_y + t.disp_h).max().unwrap_or(1);
            (minx, miny, (maxx - minx).max(1), (maxy - miny).max(1))
        } else {
            (0, 0, 1, 1)
        };

        // Z-groups: a front cap for top-level (ordinal 0) then one per layer.
        let z_off = if has_top { 1u32 } else { 0 };
        let groups = layers.len() as u32 + z_off;

        let z_span = |g: u32| -> (u32, u32) {
            let z0 = g * side / groups;
            let z1 = ((g + 1) * side / groups).max(z0 + 1).min(side);
            (z0, z1.max(z0 + 1).min(side))
        };

        let mut descs: Vec<EntityDesc> = Vec::with_capacity(arch.tensors.len());
        for t in &arch.tensors {
            if t.tensor_rows == 0 || t.tensor_cols == 0 {
                continue; // reserved canonical slot with no loaded tensor
            }
            let (ox, oy, lw, lh, g) = match t.layer_idx {
                Some(l) => {
                    let (ox, oy) = origin(l);
                    (ox, oy, layer_w, layer_h, z_off + ordinal(l) as u32)
                }
                None => (top_ox, top_oy, top_w, top_h, 0),
            };
            let (z0, z1) = z_span(g);

            // In-layer rect → cube face.
            let lx = t.canvas_x.saturating_sub(ox) as u64;
            let ly = t.canvas_y.saturating_sub(oy) as u64;
            let x0 = (lx * side as u64 / lw as u64).min(side as u64 - 1) as u32;
            let y0 = (ly * side as u64 / lh as u64).min(side as u64 - 1) as u32;
            let x1 = (((lx + t.disp_w as u64) * side as u64 / lw as u64).max(x0 as u64 + 1))
                .min(side as u64) as u32;
            let y1 = (((ly + t.disp_h as u64) * side as u64 / lh as u64).max(y0 as u64 + 1))
                .min(side as u64) as u32;

            // Byte span within the source. `bytes_per_row * rows` is exact for
            // fixed dtypes and the canonical row stride for block-quant; clamp
            // to the source so a quant rounding never over-reads.
            let local_start = t
                .tensor_byte_start
                .saturating_sub(cumulative_offsets.get(t.source_idx).copied().unwrap_or(0));
            let src_size = sources.get(t.source_idx).map(|s| s.byte_size).unwrap_or(0);
            let row_bytes = t.dtype.stride().bytes_per_row(t.tensor_cols);
            let byte_len = row_bytes
                .saturating_mul(t.tensor_rows)
                .min(src_size.saturating_sub(local_start));
            if byte_len == 0 {
                continue;
            }

            let group = match t.layer_idx {
                Some(l) => format!("layer {l}"),
                None => "top-level".to_string(),
            };
            descs.push(EntityDesc {
                source_idx: t.source_idx,
                byte_start: local_start,
                byte_len,
                bbox: VoxelBox {
                    x0,
                    y0,
                    z0,
                    x1,
                    y1,
                    z1,
                },
                extra: ArchVoxelExtra {
                    dtype: t.dtype,
                    rows: t.tensor_rows,
                    cols: t.tensor_cols,
                },
                name: t.name.clone(),
                group,
            });
        }

        if descs.is_empty() {
            return None;
        }
        let focus = focus_of(&descs, side);
        Some(Self { side, descs, focus })
    }
}

/// Cube-space framing for the union of all entity boxes (voxel `v` on an axis →
/// `(v + 0.5)/side - 0.5`, matching arbvis's shader). Independent of magnitude,
/// so empty/padded slabs don't throw off the frame.
fn focus_of(descs: &[EntityDesc], side: u32) -> ([f32; 3], f32) {
    let mut lo = [u32::MAX; 3];
    let mut hi = [0u32; 3];
    for d in descs {
        let b = d.bbox;
        for (a, (mn, mx)) in [(b.x0, b.x1), (b.y0, b.y1), (b.z0, b.z1)]
            .into_iter()
            .enumerate()
        {
            lo[a] = lo[a].min(mn);
            hi[a] = hi[a].max(mx);
        }
    }
    let s = side as f32;
    let to_cube = |v: f32| (v + 0.5) / s - 0.5;
    let mut center = [0f32; 3];
    let mut radius = 0f32;
    for a in 0..3 {
        let c0 = to_cube(lo[a] as f32);
        let c1 = to_cube(hi[a].saturating_sub(1) as f32);
        center[a] = (c0 + c1) * 0.5;
        radius = radius.max((center[a] - c0).max(c1 - center[a]));
    }
    (center, radius.max(0.02))
}

/// 3D volume plugin — the volume analog of [`crate::layout::ArchLayoutPlugin`].
/// Eligible on the same conditions (sources carry `ModelInfo`, `--layout` isn't
/// `hilbert`); builds the 2D arch layout and projects it into the cube.
pub struct ArchVolumePlugin;

impl ArchVolumePlugin {
    fn eligible(ctx: &LayoutBuildCtx<'_>) -> bool {
        // Same tensor-aware eligibility as the 2D arch layout: at least one
        // tensor-format source, and every tensor-format source parsed
        // (non-tensor siblings ignored). Under the forced+strict default a
        // failed parse aborts instead of byte-falling-back.
        crate::layout::arch_eligible(ctx)
    }
}

impl VolumeShapePlugin for ArchVolumePlugin {
    fn id(&self) -> &'static str {
        "arch"
    }
    fn priority(&self) -> i32 {
        100
    }
    fn applicable(&self, ctx: &LayoutBuildCtx<'_>) -> bool {
        Self::eligible(ctx)
    }
    fn build(&self, ctx: &LayoutBuildCtx<'_>) -> Option<Box<dyn VolumeShape>> {
        // Sidecar metas, exactly as `ArchLayoutPlugin::build` pulls them.
        let metas: Vec<SourceMeta> = ctx
            .sources
            .iter()
            .map(|s| {
                s.extensions
                    .get::<SourceMeta>()
                    .cloned()
                    .unwrap_or_default()
            })
            .collect();
        let arch = ArchLayout::try_build(ctx.sources, ctx.cumulative_offsets, &metas)?;
        let vol = ArchVolume::from_arch(&arch, ctx.sources, ctx.cumulative_offsets, ctx.grid_side)?;
        Some(Box::new(vol))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeSet, HashMap};

    use arbvis::{Extensions, SourceKind};

    use crate::format::{Dtype, ModelInfo, SourceFormat, TensorMeta};

    fn tensor(name: &str, rows: u64, cols: u64, start: u64) -> TensorMeta {
        TensorMeta {
            name: name.to_string(),
            dtype: Dtype::F32,
            shape: vec![rows, cols],
            file_start: start,
            file_end: start + rows * cols * 4,
            packed_sidecars: None,
        }
    }

    fn source(tensors: Vec<TensorMeta>) -> Source {
        let total = tensors.iter().map(|t| t.file_end).max().unwrap_or(0);
        let mut ext = Extensions::default();
        ext.insert(ModelInfo {
            format: SourceFormat::Safetensors,
            tensors,
            color_ranges: Vec::new(),
        });
        Source {
            file_idx: 0,
            kind: SourceKind::Buffered(vec![0u8; total as usize]),
            byte_size: total,
            name_override: None,
            xet_terms: None,
            extensions: ext,
        }
    }

    /// Two transformer layers, each with q_proj + gate_proj, must project into
    /// distinct Z-slabs (layer = depth) while a shared sub-path keeps the same
    /// X/Y face box across layers (the canonical-slot alignment carried over
    /// from the 2D layout).
    #[test]
    fn stacks_layers_along_z_with_aligned_subpaths() {
        let mut tensors = Vec::new();
        let mut off = 0u64;
        for l in 0..2u64 {
            for sp in ["self_attn.q_proj.weight", "mlp.gate_proj.weight"] {
                let t = tensor(&format!("model.layers.{l}.{sp}"), 16, 16, off);
                off = t.file_end;
                tensors.push(t);
            }
        }
        let sources = vec![source(tensors)];
        let cum = vec![0u64];
        let metas = vec![SourceMeta::default()];

        let arch = ArchLayout::try_build(&sources, &cum, &metas).expect("arch layout builds");
        let vol = ArchVolume::from_arch(&arch, &sources, &cum, 64).expect("arch volume builds");
        let ents = vol.entities().expect("entities");
        assert!(!ents.is_empty());

        // At least two distinct Z-slabs (one per layer).
        let zs: BTreeSet<(u32, u32)> = ents.iter().map(|e| (e.bbox.z0, e.bbox.z1)).collect();
        assert!(
            zs.len() >= 2,
            "layers should occupy distinct Z slabs, got {zs:?}"
        );

        // A sub-path aligns across layers: same face box, different Z.
        let mut by_xy: HashMap<(u32, u32, u32, u32), Vec<u32>> = HashMap::new();
        for e in &ents {
            by_xy
                .entry((e.bbox.x0, e.bbox.y0, e.bbox.x1, e.bbox.y1))
                .or_default()
                .push(e.bbox.z0);
        }
        assert!(
            by_xy
                .values()
                .any(|z| z.len() >= 2 && z.iter().collect::<BTreeSet<_>>().len() >= 2),
            "a sub-path should share an X/Y box across layers but sit at different Z"
        );

        // Byte spans stay within the source.
        for e in &ents {
            assert!(e.byte_start + e.byte_len <= sources[0].byte_size);
        }
    }
}
