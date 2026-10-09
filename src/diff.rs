//! Tensor-aware `DiffSourceBuilder` impl for a local file pair. Delegates to
//! `crate::data::build_safetensors_diff_sources`.

use std::path::Path;

use arbvis::{DiffBuildCtx, DiffSourceBuilder, Source};

use crate::data::build_safetensors_diff_sources;
use crate::format::{DiffMetric, SourceFormat};

/// Tensor-aware diff (safetensors / GGUF) for a local file pair. Applies when
/// both paths look like a recognised model-format file. Carries its own
/// `diff_metric` (arbvis's `DiffBuildCtx` no longer plumbs a metric — it's a
/// modelweightvis concept), set at registration from the `--diff-metric` flag.
pub struct TensorDiffBuilder {
    pub diff_metric: DiffMetric,
}

#[async_trait::async_trait]
impl DiffSourceBuilder for TensorDiffBuilder {
    fn id(&self) -> &'static str {
        "tensor"
    }
    fn priority(&self) -> i32 {
        300
    }
    async fn try_build(
        &self,
        ctx: &DiffBuildCtx<'_>,
    ) -> anyhow::Result<Option<(Vec<Source>, u64)>> {
        let is_st = |p: &Path| -> bool { SourceFormat::from_path(p).is_some() };
        if !(is_st(ctx.original) && is_st(ctx.modified)) {
            return Ok(None);
        }
        let out = build_safetensors_diff_sources(
            ctx.original,
            ctx.modified,
            ctx.is_finetune,
            self.diff_metric,
        )
        .await?;
        Ok(Some(out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    /// Write a minimal one-tensor F32 safetensors file: 8-byte LE header
    /// length, JSON header, then 2x2 f32 data.
    fn write_safetensors(path: &Path, values: [f32; 4]) {
        let json = format!(
            "{{\"w\":{{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[0,16]}}}}"
        );
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(json.len() as u64).to_le_bytes());
        bytes.extend_from_slice(json.as_bytes());
        for v in values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(path, bytes).unwrap();
    }

    #[tokio::test]
    async fn try_build_skips_non_model_paths() {
        let dir = TempDir::new().unwrap();
        let a = dir.path().join("a.txt");
        let b = dir.path().join("b.json");
        std::fs::write(&a, "x").unwrap();
        std::fs::write(&b, "{}\n").unwrap();
        let ctx = DiffBuildCtx {
            original: &a,
            modified: &b,
            is_finetune: false,
        };
        let builder = TensorDiffBuilder {
            diff_metric: DiffMetric::default(),
        };
        assert_eq!(builder.id(), "tensor");
        assert_eq!(builder.priority(), 300);
        assert!(builder.try_build(&ctx).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn try_build_skips_when_only_one_side_is_a_model_file() {
        let dir = TempDir::new().unwrap();
        let st = dir.path().join("a.safetensors");
        let txt = dir.path().join("b.txt");
        write_safetensors(&st, [0.0; 4]);
        std::fs::write(&txt, "x").unwrap();
        let ctx = DiffBuildCtx {
            original: &st,
            modified: &txt,
            is_finetune: false,
        };
        let builder = TensorDiffBuilder {
            diff_metric: DiffMetric::default(),
        };
        assert!(builder.try_build(&ctx).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn try_build_builds_sources_for_safetensors_pair() {
        let dir = TempDir::new().unwrap();
        let a = dir.path().join("a.safetensors");
        let b = dir.path().join("b.safetensors");
        write_safetensors(&a, [1.0, 1.0, 1.0, 1.0]);
        write_safetensors(&b, [1.5, 0.0, -1.0, 2.0]);
        let ctx = DiffBuildCtx {
            original: &a,
            modified: &b,
            is_finetune: true,
        };
        let builder = TensorDiffBuilder {
            diff_metric: DiffMetric::default(),
        };
        let (sources, bytes) = builder.try_build(&ctx).await.unwrap().unwrap();
        assert_eq!(sources.len(), 1, "one diff source for the single tensor");
        assert!(bytes > 0, "expected a nonzero byte size, got {bytes}");
    }
}
