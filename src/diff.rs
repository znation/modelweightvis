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
