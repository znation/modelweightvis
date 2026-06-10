//! Tensor-aware layout module: the architectural canvas, MoE summary / CKA
//! layouts, transformer-name classification, dtype-aware element colorizers,
//! and the matching `LayoutPlugin` impls.

pub mod arch;
pub mod bin_pack;
pub mod model_config;
pub mod name_tree;
pub mod render;

use std::any::Any;

use arbvis::{CanvasGeom, LayoutBuildCtx, LayoutMode, LayoutPlugin, LayoutShape};

use crate::data::{MoeCkaPanel, MoeCkaProbePanel, MoeSummaryPanel, SourceMeta};
use crate::format::{Dtype, ModelInfo};
pub use arch::ArchLayout;

// Constant arbvis-side; redefining here to avoid pulling in arbvis::tiled.
const TILE: u32 = 512;

impl LayoutShape for ArchLayout {
    fn id(&self) -> &'static str {
        "arch"
    }
    fn canvas_geom(&self) -> CanvasGeom {
        // Use the unpadded content extent for the viewer's world bounds, so
        // `map.fitBounds(...)` in the leaflet HTML zooms onto the matrix
        // instead of the next-pow2-padded tile grid. The padded canvas is
        // typically much larger than the placed-tensor extent — most starkly
        // for the tall MoE layouts (e.g. the CKA grid stacks many per-layer
        // panels, so a content extent well short of a power-of-two tile count
        // gets padded up to the next pow2 on each axis). Without this override
        // the initial view fits the padded canvas and the content becomes a
        // tiny strip in a sea of empty padding tiles.
        //
        // Tile coords still operate on the padded grid (`width_tiles`,
        // `height_tiles` unchanged) — the pyramid accumulator needs a
        // power-of-two tile count to drain. Padding tiles outside the world
        // bounds are still generated (they render as empty padding); leaflet
        // simply doesn't fetch the ones fully outside the bounds.
        //
        // The JS label-coord conversion (`canvas_x = lng * WIDTH / WORLD_W`)
        // stays consistent: both halves of the ratio derive from the same
        // content extent, so a tensor at `canvas_x = content_w` maps to
        // `lng = world_w` (the right edge of the world bounds).
        let two_pow_mz = 1u32 << self.max_zoom;
        let world_w = (self.content_w / two_pow_mz.max(1)).max(1);
        let world_h = (self.content_h / two_pow_mz.max(1)).max(1);
        CanvasGeom {
            kh: 0,
            width_tiles: self.width_tiles,
            height_tiles: self.height_tiles,
            world_w,
            world_h,
            width: self.content_w,
            height: self.content_h,
            max_zoom: self.max_zoom,
            total_tiles: self.total_tiles,
            square_pixels: 1,
            total: self.content_w as u64 * self.content_h as u64,
        }
    }
    fn detail_depth(&self) -> u32 {
        self.detail_depth
    }
    fn layout_entities(&self) -> Option<Vec<arbvis::FileEntity>> {
        let mut ents: Vec<arbvis::FileEntity> = Vec::with_capacity(self.tensors.len());
        for t in &self.tensors {
            let w = t.disp_w;
            let h = t.disp_h;
            let x0 = t.canvas_x;
            let y0 = t.canvas_y;
            let x1 = x0.saturating_add(w);
            let y1 = y0.saturating_add(h);
            let segments = vec![
                (x0, y0, x1, y0),
                (x1, y0, x1, y1),
                (x0, y1, x1, y1),
                (x0, y0, x0, y1),
            ];
            let cx = x0 + (x1 - x0) / 2;
            let cy = y0 + (y1 - y0) / 2;
            ents.push(arbvis::FileEntity {
                name: t.name.clone(),
                pixel_x: cx,
                pixel_y: cy,
                hue: t.hue,
                byte_size: t
                    .tensor_rows
                    .saturating_mul(t.tensor_cols)
                    .saturating_mul(t.dtype.element_size() as u64),
                bbox: (x0, y0, x1, y1),
                segments,
            });
        }
        Some(ents)
    }
    fn detail_coords(&self, zoom: u32) -> Vec<(u32, u32)> {
        use std::collections::BTreeSet;
        let level = zoom.saturating_sub(self.max_zoom);
        let f = 1u64 << level;
        let t_sz = TILE as u64;
        let mut set: BTreeSet<(u32, u32)> = BTreeSet::new();
        for t in &self.tensors {
            if arch::detail_depth_for_scale(t.scale) < level {
                continue;
            }
            let x0 = t.canvas_x as u64 * f;
            let y0 = t.canvas_y as u64 * f;
            let x1 = x0 + t.disp_w as u64 * f;
            let y1 = y0 + t.disp_h as u64 * f;
            let tx0 = (x0 / t_sz) as u32;
            let ty0 = (y0 / t_sz) as u32;
            let tx1 = ((x1 - 1) / t_sz) as u32;
            let ty1 = ((y1 - 1) / t_sz) as u32;
            for ty in ty0..=ty1 {
                for tx in tx0..=tx1 {
                    set.insert((tx, ty));
                }
            }
        }
        set.into_iter().collect()
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// One contiguous (row-major) element range of one tensor that overlaps a
/// tile. The tile renderer fetches `byte_start..byte_end` from the source
/// and decodes elements at the natural dtype stride.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct TileRegion {
    pub source_idx: usize,
    pub tensor_id: usize,
    pub dtype: Dtype,
    pub tensor_rows: u64,
    pub tensor_cols: u64,
    pub row_first: u64,
    pub row_last_exclusive: u64,
    pub col_first: u64,
    pub col_last_exclusive: u64,
    pub tensor_byte_start: u64,
    pub footprint_w: u64,
    pub footprint_h: u64,
    pub samp_x0: u64,
    pub samp_y0: u64,
    pub tile_x0: u32,
    pub tile_y0: u32,
    pub tile_x1: u32,
    pub tile_y1: u32,
}

/// Architectural plugin — applies when sources carry safetensors metadata
/// and `--layout` doesn't force hilbert. Build returns `None` if no
/// transformer-style structure is detectable.
pub struct ArchLayoutPlugin;

impl ArchLayoutPlugin {
    /// In non-diff mode every source must be safetensors (otherwise the user
    /// has explicitly mixed in non-tensor inputs they'd expect to see). In
    /// diff mode it's enough that any source carries safetensors info: the
    /// typical case is a model-repo diff where the tensor sources are the
    /// point and tokenizer/config diffs are incidental.
    fn eligible(ctx: &LayoutBuildCtx<'_>) -> bool {
        if matches!(ctx.mode, LayoutMode::Hilbert) {
            return false;
        }
        let all = !ctx.sources.is_empty()
            && ctx
                .sources
                .iter()
                .all(|s| s.extensions.get::<ModelInfo>().is_some());
        let any = ctx
            .sources
            .iter()
            .any(|s| s.extensions.get::<ModelInfo>().is_some());
        if ctx.diff_mode {
            any
        } else {
            all
        }
    }
}

impl LayoutPlugin for ArchLayoutPlugin {
    fn id(&self) -> &'static str {
        "arch"
    }
    fn priority(&self) -> i32 {
        100
    }
    fn applicable(&self, ctx: &LayoutBuildCtx<'_>) -> bool {
        Self::eligible(ctx)
    }
    fn build(&self, ctx: &LayoutBuildCtx<'_>) -> Option<Box<dyn LayoutShape>> {
        // Sidecar metas (config.json / safetensors index) are populated
        // out-of-band by [`crate::SourceMetaSidecarHook`] (registered as a
        // `PrepareSourcesExtension` on the registry) which fetches the
        // siblings once per repo/parent-dir and inserts a `SourceMeta` into
        // each `Source.extensions`. Pull them back out as a parallel slice
        // (one per source, defaulting to empty when the hook didn't run or
        // the source kind is exotic).
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
        // Diff-mode info note: surface tensor sources that don't carry
        // safetensors info (e.g. tokenizer.json file diffs) — they won't
        // appear on the arch canvas.
        if ctx.diff_mode {
            let all = !ctx.sources.is_empty()
                && ctx
                    .sources
                    .iter()
                    .all(|s| s.extensions.get::<ModelInfo>().is_some());
            if !all {
                let skipped = ctx
                    .sources
                    .iter()
                    .filter(|s| s.extensions.get::<ModelInfo>().is_none())
                    .count();
                log::info!(
                    "arch layout: {skipped} non-safetensors diff source(s) will not appear on the arch canvas (file-level diffs are only rendered in --layout hilbert)"
                );
            }
        }
        Some(Box::new(arch))
    }
}

/// MoE-summary plugin — applies when any source carries a `MoeSummaryPanel`
/// tag (set by [`crate::data::build_moe_summary_sources`]).
///
/// Can't collide with [`MoeCkaLayoutPlugin`]: under `--moe` the two lenses are
/// emitted as separate `arbvis::SceneTag` scenes, which arbvis partitions into
/// independent tile pyramids *before* layout selection — so each plugin only
/// ever sees its own scene's sources (and they key off different per-source
/// extension tags besides).
pub struct MoeSummaryLayoutPlugin;

impl LayoutPlugin for MoeSummaryLayoutPlugin {
    fn id(&self) -> &'static str {
        "moe-summary"
    }
    fn priority(&self) -> i32 {
        200
    }
    fn applicable(&self, ctx: &LayoutBuildCtx<'_>) -> bool {
        if matches!(ctx.mode, LayoutMode::Hilbert) {
            return false;
        }
        ctx.sources
            .iter()
            .any(|s| s.extensions.get::<MoeSummaryPanel>().is_some())
    }
    fn build(&self, ctx: &LayoutBuildCtx<'_>) -> Option<Box<dyn LayoutShape>> {
        ArchLayout::try_build_moe_summary(ctx.sources, ctx.cumulative_offsets)
            .map(|l| Box::new(l) as Box<dyn LayoutShape>)
    }
}

/// MoE-CKA plugin — applies when any source carries a `MoeCkaPanel` or
/// `MoeCkaProbePanel` tag (set by [`crate::data::build_moe_cka_sources`]).
/// Lives in its own `arbvis::SceneTag` scene, so it can't collide with the
/// summary plugin (see [`MoeSummaryLayoutPlugin`]).
pub struct MoeCkaLayoutPlugin;

impl LayoutPlugin for MoeCkaLayoutPlugin {
    fn id(&self) -> &'static str {
        "moe-cka"
    }
    fn priority(&self) -> i32 {
        200
    }
    fn applicable(&self, ctx: &LayoutBuildCtx<'_>) -> bool {
        if matches!(ctx.mode, LayoutMode::Hilbert) {
            return false;
        }
        ctx.sources.iter().any(|s| {
            s.extensions.get::<MoeCkaPanel>().is_some()
                || s.extensions.get::<MoeCkaProbePanel>().is_some()
        })
    }
    fn build(&self, ctx: &LayoutBuildCtx<'_>) -> Option<Box<dyn LayoutShape>> {
        ArchLayout::try_build_moe_cka(ctx.sources, ctx.cumulative_offsets)
            .map(|l| Box::new(l) as Box<dyn LayoutShape>)
    }
}
