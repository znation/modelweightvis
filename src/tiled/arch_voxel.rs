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

use arbvis::{VoxelCell, VoxelGridMut, VoxelRenderCtx, VoxelRenderer};

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

pub struct ArchVoxelRenderer;

impl VoxelRenderer for ArchVoxelRenderer {
    fn id(&self) -> &'static str {
        "arch"
    }

    fn render(&self, ctx: &VoxelRenderCtx<'_>, grid: &mut VoxelGridMut<'_>) {
        let Some(ex) = ctx.entity.extra.downcast_ref::<ArchVoxelExtra>() else {
            return;
        };
        let bb = ctx.entity.bbox;
        let bw = bb.x1.saturating_sub(bb.x0) as u64;
        let bh = bb.y1.saturating_sub(bb.y0) as u64;
        if bw == 0 || bh == 0 || bb.z1 <= bb.z0 || ex.rows == 0 || ex.cols == 0 {
            return;
        }

        // One reader over the whole entity span (arbvis fetched it). For fixed
        // dtypes `element(k)` is a direct read; for block-quant it dequantizes
        // with an internal block cache. Out-of-range indices return NaN, so a
        // slightly-off byte span degrades to fewer samples, never a panic.
        let mut reader = TensorElementReader::new(ex.dtype, ctx.bytes);

        let face = if ctx.diff_mode {
            diff_face(&mut reader, ex, bw, bh)
        } else {
            magnitude_face(&mut reader, ex, bw, bh)
        };

        // Extrude the 2D face through every Z plane of the slab.
        for fy in 0..bh {
            for fx in 0..bw {
                let Some(cell) = face[(fy * bw + fx) as usize] else {
                    continue;
                };
                let x = bb.x0 + fx as u32;
                let y = bb.y0 + fy as u32;
                for z in bb.z0..bb.z1 {
                    grid.put(x, y, z, cell);
                }
            }
        }
    }

    // Element count — splits the streamed point-octree budget across tensors so
    // bigger tensors get proportionally more points.
    fn point_weight(&self, ctx: &VoxelRenderCtx<'_>) -> u64 {
        ctx.entity
            .extra
            .downcast_ref::<ArchVoxelExtra>()
            .map(|ex| ex.rows.saturating_mul(ex.cols))
            .unwrap_or(0)
    }

    // Emit per-element points so the Points view drills toward one point per
    // weight on zoom (the LOD octree streams them). A tensor is one flat sheet
    // at mid-slab — the volume mode extrudes it through depth, but the exact
    // points place each element on the face: column → x, row → y (matching the
    // voxel face mapping). Coloring mirrors the voxel renderer.
    fn render_points(
        &self,
        ctx: &VoxelRenderCtx<'_>,
        budget: u64,
        emit: &mut dyn FnMut([f32; 3], [u8; 4]),
    ) {
        let Some(ex) = ctx.entity.extra.downcast_ref::<ArchVoxelExtra>() else {
            return;
        };
        let (rows, cols) = (ex.rows, ex.cols);
        let total = rows.saturating_mul(cols);
        if total == 0 || budget == 0 {
            return;
        }
        let stride = (total / budget).max(1) as usize;
        let mut reader = TensorElementReader::new(ex.dtype, ctx.bytes);
        let pos = |k: usize| -> [f32; 3] {
            let r = (k as u64 / cols) as f32;
            let c = (k as u64 % cols) as f32;
            [(c + 0.5) / cols as f32, (r + 0.5) / rows as f32, 0.5]
        };

        if ctx.diff_mode {
            // U8 signed-delta codes (127 = unchanged, 255 = non-finite), like
            // diff_face: color by sign, opacity = |Δ|, unchanged weights vanish.
            let lut = arbvis::color::build_diff_signed_lut();
            let mut k = 0usize;
            while (k as u64) < total {
                let code = reader.element(k);
                if code.is_finite() {
                    let code = code as i32;
                    if code >= 255 {
                        let c = lut[255].0;
                        emit(pos(k), [c[0], c[1], c[2], 255]);
                    } else {
                        let signed = (code - 127) as f32 / 127.0;
                        let a = (signed.abs() * 255.0).round() as u8;
                        if a > 0 {
                            let idx =
                                (127.0 + signed * 127.0).round().clamp(0.0, 254.0) as usize;
                            let c = lut[idx].0;
                            emit(pos(k), [c[0], c[1], c[2], a]);
                        }
                    }
                }
                k += stride;
            }
        } else {
            // Plain mode: |value| normalized by the tensor's own max, through
            // the CIVIDIS magnitude ramp; opacity = normalized magnitude.
            let mut maxv = 0f32;
            let mut k = 0usize;
            while (k as u64) < total {
                let v = reader.element(k);
                if v.is_finite() {
                    maxv = maxv.max(v.abs());
                }
                k += stride;
            }
            if maxv <= 0.0 {
                return;
            }
            let inv = 1.0 / maxv;
            let mut k = 0usize;
            while (k as u64) < total {
                let v = reader.element(k);
                if v.is_finite() {
                    let byte = ((v.abs() * inv).clamp(0.0, 1.0) * 255.0).round() as u8;
                    if byte > 0 {
                        let c = CIVIDIS_LUT[byte as usize].0;
                        emit(pos(k), [c[0], c[1], c[2], byte]);
                    }
                }
                k += stride;
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
            ArchVoxelRenderer.render(
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
}
