//! The `"arch"` [`arbvis::VoxelRenderer`]: the 3D analog of
//! [`crate::tiled::leaf_arch`]'s per-tensor tile renderer.
//!
//! Where the 2D arch renderer colors one canvas pixel per element by the
//! element's *literal* value (Stairwell byte LUT), a bounded voxel cube can't
//! hold one voxel per element — many elements fall into each voxel. So the 3D
//! renderer **aggregates magnitude**: per voxel it samples the elements it
//! covers, decodes them by dtype via [`TensorElementReader`], reduces to a mean
//! `|value|`, normalizes per-tensor, and bakes a perceptual
//! [`crate::colormap::CIVIDIS_LUT`] color straight into the voxel (arbvis's
//! `color_mode: "rgb"`). A tensor is one Z-slab; its 2D magnitude image is
//! extruded through the slab's depth, so the 3D structure comes from *different
//! layers at different Z* — see [`crate::layout::arch_volume`].

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

        // Pass 1: per face-voxel mean |magnitude|; track the max for per-tensor
        // normalization (each tensor uses its own full dynamic range).
        let mut means = vec![0f32; (bw * bh) as usize];
        let mut max_mean = 0f32;
        for fy in 0..bh {
            let r0 = fy * ex.rows / bh;
            let r1 = ((fy + 1) * ex.rows / bh).max(r0 + 1);
            for fx in 0..bw {
                let c0 = fx * ex.cols / bw;
                let c1 = ((fx + 1) * ex.cols / bw).max(c0 + 1);
                let rr = r1 - r0;
                let cc = c1 - c0;
                let total = rr * cc;
                let stride = (total / PER_VOXEL_SAMPLES).max(1);
                let (mut sum, mut n) = (0f64, 0u64);
                let mut t = 0u64;
                while t < total {
                    let r = r0 + t / cc;
                    let c = c0 + t % cc;
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
            return; // all-zero tensor → leave its box empty (transparent)
        }
        let inv = 1.0 / max_mean;

        // Pass 2: normalize → CIVIDIS → bake. The tensor's 2D image is extruded
        // through every Z plane of its slab (a < 1 makes near-zero weights faint
        // rather than empty, so a dead column reads as a hole).
        for fy in 0..bh {
            for fx in 0..bw {
                let m = means[(fy * bw + fx) as usize];
                if m <= 0.0 {
                    continue;
                }
                let byte = ((m * inv).clamp(0.0, 1.0) * 255.0).round() as u8;
                let c = CIVIDIS_LUT[byte as usize].0;
                let cell = VoxelCell {
                    r: c[0],
                    g: c[1],
                    b: c[2],
                    a: byte,
                };
                let x = bb.x0 + fx as u32;
                let y = bb.y0 + fy as u32;
                for z in bb.z0..bb.z1 {
                    grid.put(x, y, z, cell);
                }
            }
        }
    }
}
