//! The `"arch"` [`arbvis::VoxelRenderer`]: the 3D analog of
//! [`crate::tiled::leaf_arch`]'s per-tensor tile renderer.
//!
//! A bounded voxel cube can't hold one voxel per element — many elements fall
//! into each voxel — so the 3D renderer **aggregates**, decoding the elements a
//! voxel covers by dtype via [`TensorElementReader`] (bounded sampling) and
//! baking a final RGB straight into the voxel (arbvis's `color_mode: "rgb"`). A
//! tensor is one Z-slab; its 2D image is extruded through the slab's depth, so
//! the 3D structure comes from *different layers at different Z* — see
//! [`crate::layout::arch_volume`].
//!
//! Two modes, picked from [`arbvis::VoxelRenderCtx::diff_mode`]:
//! - **plain** — mean `|value|` per voxel, normalized per-tensor, through the
//!   perceptual [`crate::colormap::CIVIDIS_LUT`] magnitude ramp.
//! - **diff** — the source is `Dtype::U8` signed-delta codes (127 = no change,
//!   above 127 = increased, below = decreased, 255 = non-finite; see
//!   [`crate::format::Dtype::diff_to_u8`]). Per voxel we mean the signed delta
//!   and color it through arbvis's signed-diff LUT, with opacity = magnitude of
//!   the delta — so unchanged regions vanish and only the changed weights light
//!   up (green grew / red shrank), matching the 2D diff viewer.

use std::collections::HashMap;
use std::ops::Range;
use std::sync::{Arc, Mutex};

use arbvis::{VoxelBox, VoxelCell, VoxelGridMut, VoxelRenderCtx, VoxelRenderer};

use crate::colormap::CIVIDIS_LUT;
use crate::format::{Dtype, TensorElementReader};

/// Per-entity payload carried on [`arbvis::VolumeEntity::extra`], set by
/// [`crate::layout::arch_volume`] and downcast here. Plain data so it's cheap
/// to clone when the shape materializes its entity list.
#[derive(Clone)]
pub struct ArchVoxelExtra {
    pub dtype: Dtype,
    /// Element grid (rows × cols) of the tensor this entity renders.
    pub rows: u64,
    pub cols: u64,
}

/// Upper bound on element decodes per voxel. Keeps the single build-time pass
/// `O(occupied_voxels × budget)` — bounded by the cube, independent of model
/// size — the 3D analog of the byte path's run-aggregation.
const PER_VOXEL_SAMPLES: u64 = 48;

/// The element sub-rectangle a face voxel `(fx, fy)` covers, as
/// `(row0, row1, col0, col1)` (half-open). The tensor's `rows × cols` element
/// grid is spread across the `bw × bh` face.
fn voxel_rect(fx: u64, fy: u64, bw: u64, bh: u64, rows: u64, cols: u64) -> (u64, u64, u64, u64) {
    let r0 = fy * rows / bh;
    let r1 = ((fy + 1) * rows / bh).max(r0 + 1);
    let c0 = fx * cols / bw;
    let c1 = ((fx + 1) * cols / bw).max(c0 + 1);
    (r0, r1, c0, c1)
}

/// Identifies a tensor entity within one build for the streamed-slab face
/// cache: `(source_idx, byte_start, byte_len, diff_mode)`. The byte span is
/// unique per entity and `diff_mode` picks the color path; a fresh renderer
/// (hence a fresh, empty cache) is constructed per run, so keys never collide
/// across models.
type FaceKey = (usize, u64, u64, bool);

/// The `"arch"` voxel renderer.
///
/// Output is a pure function of the entity, but it carries a small per-entity
/// **face cache** used only by the streamed slab path
/// ([`render_window`](VoxelRenderer::render_window)). A tensor's 2D face is
/// z-invariant — it extrudes unchanged through its whole Z-slab — yet arbvis
/// bricks the volume in `BRICK`-aligned Z-slabs and dispatches `render_window`
/// once per slab an entity's bbox intersects. At the default `--grid 2048` a
/// layer slab is ~32 voxels deep and `BRICK` is 8, so each entity is dispatched
/// ~4×; the default `render_window` would recompute the face — the expensive
/// decode+aggregate step — every time. Caching it decodes each face exactly
/// once and reuses it across the slabs, then drops the entry on the last slab
/// that touches the entity so live faces never exceed the current slab's set
/// (mirroring arbvis's own fetch-once byte residency, keeping build RAM
/// bounded by one slab rather than the whole volume).
#[derive(Default)]
pub struct ArchVoxelRenderer {
    faces: Mutex<HashMap<FaceKey, Arc<Vec<Option<VoxelCell>>>>>,
}

impl ArchVoxelRenderer {
    pub fn new() -> Self {
        Self::default()
    }

    /// The entity's face box dimensions `(bw, bh)`, or `None` when the box or
    /// element grid is degenerate (nothing to render).
    fn face_dims(ctx: &VoxelRenderCtx<'_>, ex: &ArchVoxelExtra) -> Option<(u64, u64)> {
        let bb = ctx.entity.bbox;
        let bw = bb.x1.saturating_sub(bb.x0) as u64;
        let bh = bb.y1.saturating_sub(bb.y0) as u64;
        if bw == 0 || bh == 0 || bb.z1 <= bb.z0 || ex.rows == 0 || ex.cols == 0 {
            return None;
        }
        Some((bw, bh))
    }

    /// Decode + aggregate the entity's z-invariant 2D face (magnitude or diff).
    ///
    /// One reader over the whole entity span (arbvis fetched it). For fixed
    /// dtypes `element(k)` is a direct read; for block-quant it dequantizes with
    /// an internal block cache. Out-of-range indices return NaN, so a
    /// slightly-off byte span degrades to fewer samples, never a panic.
    fn compute_face(
        ctx: &VoxelRenderCtx<'_>,
        ex: &ArchVoxelExtra,
        bw: u64,
        bh: u64,
    ) -> Vec<Option<VoxelCell>> {
        let mut reader = TensorElementReader::new(ex.dtype, ctx.bytes);
        if ctx.diff_mode {
            diff_face(&mut reader, ex, bw, bh)
        } else {
            magnitude_face(&mut reader, ex, bw, bh)
        }
    }
}

impl VoxelRenderer for ArchVoxelRenderer {
    fn id(&self) -> &'static str {
        "arch"
    }

    fn render(&self, ctx: &VoxelRenderCtx<'_>, grid: &mut VoxelGridMut<'_>) {
        let Some(ex) = ctx.entity.extra.downcast_ref::<ArchVoxelExtra>() else {
            return;
        };
        let Some((bw, bh)) = Self::face_dims(ctx, ex) else {
            return;
        };
        // Dense (non-streamed) path: one call per entity, so decode inline —
        // the cache would only ever hold and immediately drop a single face.
        let bb = ctx.entity.bbox;
        let face = Self::compute_face(ctx, ex, bw, bh);
        extrude_face(grid, bb, bw, bh, &face, bb.z0, bb.z1);
    }

    fn render_window(
        &self,
        ctx: &VoxelRenderCtx<'_>,
        grid: &mut VoxelGridMut<'_>,
        z_range: Range<u32>,
    ) {
        let Some(ex) = ctx.entity.extra.downcast_ref::<ArchVoxelExtra>() else {
            return;
        };
        let Some((bw, bh)) = Self::face_dims(ctx, ex) else {
            return;
        };
        let bb = ctx.entity.bbox;

        // The face is z-invariant, so decode it on the first slab that touches
        // this entity and reuse it for the rest instead of re-decoding per slab.
        let key: FaceKey = (
            ctx.entity.source_idx,
            ctx.entity.byte_start,
            ctx.entity.byte_len,
            ctx.diff_mode,
        );
        let cached = self.faces.lock().unwrap().get(&key).cloned();
        let face = match cached {
            Some(f) => f,
            None => {
                let f = Arc::new(Self::compute_face(ctx, ex, bw, bh));
                self.faces.lock().unwrap().insert(key, f.clone());
                f
            }
        };

        // Extrude only the planes in both the entity's slab and this window;
        // arbvis drops puts outside the window anyway, but clamping here avoids
        // the wasted iterations.
        let z0 = bb.z0.max(z_range.start);
        let z1 = bb.z1.min(z_range.end);
        extrude_face(grid, bb, bw, bh, &face, z0, z1);

        // The slabs advance front-to-back, so once the window reaches this
        // entity's last plane it is done — drop its face to keep the cache
        // bounded to the entities intersecting the current slab.
        if z_range.end >= bb.z1 {
            self.faces.lock().unwrap().remove(&key);
        }
    }
}

/// Extrude a computed 2D `face` (indexed `fy * bw + fx`) through the z-planes
/// `[z0, z1)` of `grid`, writing each non-empty cell down its column. Callers
/// pass the entity's full `[bb.z0, bb.z1)` span (dense path) or a clamped slab
/// window (streamed path); out-of-box puts are dropped by the grid.
fn extrude_face(
    grid: &mut VoxelGridMut<'_>,
    bb: VoxelBox,
    bw: u64,
    bh: u64,
    face: &[Option<VoxelCell>],
    z0: u32,
    z1: u32,
) {
    for fy in 0..bh {
        for fx in 0..bw {
            let Some(cell) = face[(fy * bw + fx) as usize] else {
                continue;
            };
            let x = bb.x0 + fx as u32;
            let y = bb.y0 + fy as u32;
            for z in z0..z1 {
                grid.put(x, y, z, cell);
            }
        }
    }
}

/// Plain mode: per voxel mean `|value|`, normalized by the tensor's own max so
/// each tensor uses its full dynamic range, through the CIVIDIS magnitude ramp.
/// `a` (= the normalized byte) doubles as opacity, so near-zero weights fade.
fn magnitude_face(
    reader: &mut TensorElementReader<'_>,
    ex: &ArchVoxelExtra,
    bw: u64,
    bh: u64,
) -> Vec<Option<VoxelCell>> {
    let mut means = vec![0f32; (bw * bh) as usize];
    let mut max_mean = 0f32;
    for fy in 0..bh {
        for fx in 0..bw {
            let (r0, r1, c0, c1) = voxel_rect(fx, fy, bw, bh, ex.rows, ex.cols);
            let total = (r1 - r0) * (c1 - c0);
            let stride = (total / PER_VOXEL_SAMPLES).max(1);
            let (mut sum, mut n) = (0f64, 0u64);
            let mut t = 0u64;
            while t < total {
                let r = r0 + t / (c1 - c0);
                let c = c0 + t % (c1 - c0);
                let v = reader.element((r * ex.cols + c) as usize);
                if v.is_finite() {
                    sum += v.abs() as f64;
                    n += 1;
                }
                t += stride;
            }
            let m = if n > 0 { (sum / n as f64) as f32 } else { 0.0 };
            means[(fy * bw + fx) as usize] = m;
            if m > max_mean {
                max_mean = m;
            }
        }
    }
    if max_mean <= 0.0 {
        return vec![None; (bw * bh) as usize]; // all-zero tensor → empty box
    }
    let inv = 1.0 / max_mean;
    means
        .iter()
        .map(|&m| {
            if m <= 0.0 {
                return None;
            }
            let byte = ((m * inv).clamp(0.0, 1.0) * 255.0).round() as u8;
            let c = CIVIDIS_LUT[byte as usize].0;
            Some(VoxelCell {
                r: c[0],
                g: c[1],
                b: c[2],
                a: byte,
            })
        })
        .collect()
}

/// Diff mode: the source is `Dtype::U8` signed-delta codes. Per voxel we mean
/// the signed delta `(code - 127) / 127 ∈ [-1, 1]`, color it through arbvis's
/// signed-diff LUT (the same table the 2D viewer uses), and set opacity to |Δ|
/// — so unchanged voxels (mean ≈ 127) are transparent and the cube shows only
/// where, and how, the weights moved. A voxel sampling only non-finite codes
/// (255) bakes opaque white.
fn diff_face(
    reader: &mut TensorElementReader<'_>,
    ex: &ArchVoxelExtra,
    bw: u64,
    bh: u64,
) -> Vec<Option<VoxelCell>> {
    let lut = arbvis::color::build_diff_signed_lut();
    let mut out = Vec::with_capacity((bw * bh) as usize);
    for fy in 0..bh {
        for fx in 0..bw {
            let (r0, r1, c0, c1) = voxel_rect(fx, fy, bw, bh, ex.rows, ex.cols);
            let total = (r1 - r0) * (c1 - c0);
            let stride = (total / PER_VOXEL_SAMPLES).max(1);
            let (mut sum_signed, mut n, mut nonfinite) = (0f64, 0u64, 0u64);
            let mut t = 0u64;
            while t < total {
                let r = r0 + t / (c1 - c0);
                let c = c0 + t % (c1 - c0);
                let code = reader.element((r * ex.cols + c) as usize);
                if code.is_finite() {
                    let code = code as i32;
                    if code >= 255 {
                        nonfinite += 1; // diff_to_u8's non-finite sentinel
                    } else {
                        sum_signed += (code - 127) as f64 / 127.0;
                        n += 1;
                    }
                }
                t += stride;
            }
            let cell = if n == 0 {
                // Only non-finite (or nothing) sampled.
                nonfinite.gt(&0).then(|| {
                    let c = lut[255].0;
                    VoxelCell {
                        r: c[0],
                        g: c[1],
                        b: c[2],
                        a: 255,
                    }
                })
            } else {
                let mean = (sum_signed / n as f64).clamp(-1.0, 1.0);
                let a = (mean.abs() * 255.0).round() as u8;
                if a == 0 {
                    None // no net change → transparent
                } else {
                    let code = (127.0 + mean * 127.0).round().clamp(0.0, 254.0) as usize;
                    let c = lut[code].0;
                    Some(VoxelCell {
                        r: c[0],
                        g: c[1],
                        b: c[2],
                        a,
                    })
                }
            };
            out.push(cell);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use arbvis::{VolumeEntity, VoxelBox};

    /// Diff mode: a 4-wide face over U8 delta codes — column 0 unchanged (127),
    /// 1 decreased (60), 2 increased (200), 3 non-finite (255) — bakes
    /// transparent / red / green / white respectively.
    #[test]
    fn diff_codes_color_by_sign_and_vanish_when_unchanged() {
        let (rows, cols) = (4u64, 4u64);
        let codes = [127u8, 60, 200, 255];
        let mut bytes = vec![0u8; (rows * cols) as usize];
        for r in 0..rows {
            for c in 0..cols {
                bytes[(r * cols + c) as usize] = codes[c as usize];
            }
        }
        let ent = VolumeEntity {
            source_idx: 0,
            byte_start: 0,
            byte_len: bytes.len() as u64,
            // 4 face voxels wide, 1 tall, 1 deep → one column of codes each.
            bbox: VoxelBox {
                x0: 0,
                y0: 0,
                z0: 0,
                x1: 4,
                y1: 1,
                z1: 1,
            },
            renderer_id: "arch",
            extra: Box::new(ArchVoxelExtra {
                dtype: Dtype::U8,
                rows,
                cols,
            }),
        };
        let side = 8u32;
        let mut cells = vec![VoxelCell::default(); (side as usize).pow(3)];
        {
            let mut grid = VoxelGridMut::new(&mut cells, [side; 3]);
            ArchVoxelRenderer::new().render(
                &VoxelRenderCtx {
                    entity: &ent,
                    bytes: &bytes,
                    extent: [side; 3],
                    diff_mode: true,
                },
                &mut grid,
            );
        }
        // z=0, y=0 row ⇒ linear index == x.
        let v = |x: usize| cells[x];
        assert_eq!(v(0).a, 0, "unchanged (127) → transparent");
        assert!(v(1).a > 0 && v(1).r > 0 && v(1).g == 0, "decrease → red");
        assert!(v(2).a > 0 && v(2).g > 0 && v(2).r == 0, "increase → green");
        assert_eq!(
            (v(3).r, v(3).g, v(3).b, v(3).a),
            (255, 255, 255, 255),
            "non-finite (255) → opaque white"
        );
    }

    /// The streamed slab path drives `render_window` once per BRICK-deep Z-slab
    /// an entity crosses; the face cache + windowed extrusion must reproduce a
    /// single full `render()` byte-for-byte, and the cache must drain once the
    /// slab front passes the entity (so build RAM stays bounded to one slab).
    #[test]
    fn windowed_render_matches_full_render_and_evicts() {
        let (rows, cols) = (8u64, 8u64);
        // Non-uniform bytes so the face varies across x/y (a flat face would
        // pass even if the windowing were wrong).
        let bytes: Vec<u8> = (0..(rows * cols) as usize)
            .map(|i| (i as u8).wrapping_mul(7).wrapping_add(3))
            .collect();
        let side = 16u32;
        // z-slab 11 voxels deep — deeper than BRICK (8) — so the streamed path
        // dispatches this entity across two slabs.
        let ent = VolumeEntity {
            source_idx: 0,
            byte_start: 0,
            byte_len: bytes.len() as u64,
            bbox: VoxelBox {
                x0: 1,
                y0: 2,
                z0: 3,
                x1: 6,
                y1: 7,
                z1: 14,
            },
            renderer_id: "arch",
            extra: Box::new(ArchVoxelExtra {
                dtype: Dtype::U8,
                rows,
                cols,
            }),
        };
        let ctx = || VoxelRenderCtx {
            entity: &ent,
            bytes: &bytes,
            extent: [side; 3],
            diff_mode: false,
        };
        let tuples = |cells: &[VoxelCell]| -> Vec<(u8, u8, u8, u8)> {
            cells.iter().map(|c| (c.r, c.g, c.b, c.a)).collect()
        };

        // Reference: one full render().
        let mut full = vec![VoxelCell::default(); (side as usize).pow(3)];
        {
            let mut g = VoxelGridMut::new(&mut full, [side; 3]);
            ArchVoxelRenderer::new().render(&ctx(), &mut g);
        }

        // Windowed: the same entity fed through BRICK-deep slabs, exactly as the
        // streamed structured build path drives it.
        let r = ArchVoxelRenderer::new();
        let brick = 8u32;
        let mut win = vec![VoxelCell::default(); (side as usize).pow(3)];
        let mut z0 = 0u32;
        while z0 < side {
            let z1 = (z0 + brick).min(side);
            if ent.bbox.z0 < z1 && ent.bbox.z1 > z0 {
                let mut g = VoxelGridMut::new(&mut win, [side; 3]);
                r.render_window(&ctx(), &mut g, z0..z1);
            }
            z0 = z1;
        }

        assert_eq!(
            tuples(&full),
            tuples(&win),
            "windowed slab render must match the full render"
        );
        assert!(
            r.faces.lock().unwrap().is_empty(),
            "face cache must drain once the slab front passes the entity"
        );
    }
}
