//! Option-slot hook impls on top of arbvis's [`Registry`].
//!
//! These wrap the heavy tensor-aware helpers in [`crate::data`] and
//! [`crate::finetune`] into the trait objects arbvis::run dispatches
//! through. Each is a thin glue layer: argument shuffling, error
//! re-contextualisation, no logic of its own. The actual work lives in
//! `crate::data::*`.
//!
//! Wired up by [`crate::register_all`].

use std::borrow::Cow;
use std::path::{Path, PathBuf};

use anyhow::Context;
use arbvis::hf_url;
use arbvis::{
    DestKind, LayoutShape, PrepareSourcesExtension, RenderHints, SingleImageRenderer, Source,
    SourceCtx, SourceKind, SourceProvider,
};
use async_trait::async_trait;

use crate::data::{
    build_multi_safetensors_diff_sources, collect_files_recursive, load_meta_for_sources,
    prepare_diff_sources_from_http, prepare_moe_scenes_sources, MoeNorm,
};
use crate::finetune::FinetuneForce;
use crate::format::{DiffMetric, SourceFormat, SummaryStat};
use crate::probe::ProbeOpts;

/// `--moe <model>` source provider (priority 400). Loads the model once and
/// builds two scenes — a per-expert scalar "summary" (panels tagged
/// `MoeSummaryPanel`, read by [`crate::MoeSummaryLayoutPlugin`]) and an N×N
/// "cka" similarity grid (panels tagged `MoeCkaPanel`, read by
/// [`crate::MoeCkaLayoutPlugin`]) — each stamped with an `arbvis::SceneTag` so
/// the tiler renders a tab switcher.
///
/// Carries its lens config (summary stat, normalization, CKA sample, probe) as
/// fields, set from the CLI flags by [`crate::register_all`]. Registered only
/// when `--moe` was passed, so [`applicable`](SourceProvider::applicable) can
/// simply check "no diff, no positional inputs" without shadowing the normal
/// byte path of a bare invocation.
pub struct MoeSceneProvider {
    pub target: PathBuf,
    pub stat: SummaryStat,
    pub norm: MoeNorm,
    pub cka_sample: u32,
    pub probe: ProbeOpts,
}

#[async_trait(?Send)]
impl SourceProvider for MoeSceneProvider {
    fn id(&self) -> &'static str {
        "moe-scenes"
    }
    fn priority(&self) -> i32 {
        400
    }
    fn applicable(&self, ctx: &SourceCtx<'_>) -> bool {
        ctx.diff.is_none() && ctx.inputs.is_empty()
    }
    async fn prepare(
        &self,
        ctx: &SourceCtx<'_>,
    ) -> anyhow::Result<(Vec<Source>, u64, RenderHints)> {
        // The MoE viewer is a tabbed, multi-scene render; the tab switcher only
        // exists in the interactive Leaflet viewer, so a single-PNG / window
        // destination can't represent it.
        if ctx.dest_kind != DestKind::Tiles {
            anyhow::bail!(
                "--moe renders a tabbed multi-scene viewer and needs a tile destination; \
                 pass --tiles <DIR> (or --space <OWNER/REPO>)"
            );
        }
        let input = self.target.to_string_lossy().into_owned();
        let (sources, total) = prepare_moe_scenes_sources(
            &input,
            self.stat,
            self.norm,
            self.cka_sample,
            ctx.stream,
            &self.probe,
        )
        .await
        .with_context(|| format!("--moe {input}"))?;
        let hints = RenderHints {
            diff_mode: false,
            title_suffix: Cow::Borrowed("moe"),
            show_xet_xorbs: false,
            inputs: vec![input],
        };
        Ok((sources, total, hints))
    }
}

/// Repo-level `--diff hf://… hf://…` provider (priority 300). Lists both repos
/// over the HF API and lazily diffs safetensors shards over HTTP range requests
/// (small non-safetensors siblings are eagerly byte-diffed) via
/// [`prepare_diff_sources_from_http`]. Resolves the finetune relation itself
/// (see [`crate::finetune`]).
pub struct RepoDiffProvider {
    pub diff_metric: DiffMetric,
    pub finetune: FinetuneForce,
}

#[async_trait(?Send)]
impl SourceProvider for RepoDiffProvider {
    fn id(&self) -> &'static str {
        "repo-diff"
    }
    fn priority(&self) -> i32 {
        300
    }
    fn applicable(&self, ctx: &SourceCtx<'_>) -> bool {
        ctx.diff.as_ref().is_some_and(|d| {
            hf_url::is_repo_level(d.original).unwrap_or(false)
                && hf_url::is_repo_level(d.modified).unwrap_or(false)
        })
    }
    async fn prepare(
        &self,
        ctx: &SourceCtx<'_>,
    ) -> anyhow::Result<(Vec<Source>, u64, RenderHints)> {
        let d = ctx
            .diff
            .as_ref()
            .expect("repo-diff applies only when --diff is set");
        let is_finetune = crate::finetune::resolve(self.finetune, d.original, d.modified).await;
        let (orig_specs, mod_specs) = tokio::try_join!(
            async {
                hf_url::list_repo_as_http_specs(d.original)
                    .await
                    .with_context(|| format!("listing files in {}", d.original))
            },
            async {
                hf_url::list_repo_as_http_specs(d.modified)
                    .await
                    .with_context(|| format!("listing files in {}", d.modified))
            },
        )?;
        let (sources, total) = prepare_diff_sources_from_http(
            &orig_specs,
            &mod_specs,
            is_finetune,
            self.diff_metric,
            ctx.stream,
        )
        .await?;
        let hints = RenderHints {
            diff_mode: true,
            title_suffix: Cow::Borrowed("diff"),
            show_xet_xorbs: false,
            inputs: vec![d.original.to_string(), d.modified.to_string()],
        };
        Ok((sources, total, hints))
    }
}

/// Local directory `--diff <dir> <dir>` provider (priority 250). Diffs the
/// tensor files (matched across shards by tensor name) via
/// [`build_multi_safetensors_diff_sources`], then hands the non-tensor
/// remainder to arbvis's [`arbvis::byte_directory_diff`] (which renders
/// crosshatched unmatched / size-mismatched siblings).
pub struct TensorDiffProvider {
    pub diff_metric: DiffMetric,
    pub finetune: FinetuneForce,
}

#[async_trait(?Send)]
impl SourceProvider for TensorDiffProvider {
    fn id(&self) -> &'static str {
        "tensor-diff"
    }
    fn priority(&self) -> i32 {
        250
    }
    fn applicable(&self, ctx: &SourceCtx<'_>) -> bool {
        ctx.diff
            .as_ref()
            .is_some_and(|d| Path::new(d.original).is_dir() && Path::new(d.modified).is_dir())
    }
    async fn prepare(
        &self,
        ctx: &SourceCtx<'_>,
    ) -> anyhow::Result<(Vec<Source>, u64, RenderHints)> {
        let d = ctx
            .diff
            .as_ref()
            .expect("tensor-diff applies only when --diff is set");
        let orig = Path::new(d.original);
        let mod_ = Path::new(d.modified);
        let is_finetune = crate::finetune::resolve(self.finetune, d.original, d.modified).await;

        let is_tensor = |p: &Path| SourceFormat::from_path(p).is_some();
        let orig_tensor: Vec<PathBuf> = collect_files_recursive(orig)
            .into_iter()
            .filter(|p| is_tensor(p))
            .collect();
        let mod_tensor: Vec<PathBuf> = collect_files_recursive(mod_)
            .into_iter()
            .filter(|p| is_tensor(p))
            .collect();

        // Tensor files first (their own 0-based `file_idx`), matched across
        // shards by tensor name.
        let mut sources = Vec::new();
        let mut total = 0u64;
        if !orig_tensor.is_empty() || !mod_tensor.is_empty() {
            match build_multi_safetensors_diff_sources(
                &orig_tensor,
                &mod_tensor,
                is_finetune,
                self.diff_metric,
            )
            .await
            {
                Ok((tensor_sources, bytes)) => {
                    sources.extend(tensor_sources);
                    total += bytes;
                }
                Err(e) => log::warn!("tensor-aware directory diff failed: {e} — skipping"),
            }
        }

        // Non-tensor remainder: byte-diff by relative path, skipping the tensor
        // files we just handled. Offset their `file_idx` past the tensor block.
        let (mut byte_sources, byte_total) =
            arbvis::byte_directory_diff(orig, mod_, is_finetune, &is_tensor)?;
        let base_idx = sources.len();
        for s in &mut byte_sources {
            s.file_idx += base_idx;
        }
        sources.extend(byte_sources);
        total += byte_total;

        if sources.is_empty() {
            anyhow::bail!("--diff: no matching file pairs found between the two directories");
        }
        let hints = RenderHints {
            diff_mode: true,
            title_suffix: Cow::Borrowed("diff"),
            show_xet_xorbs: false,
            inputs: vec![d.original.to_string(), d.modified.to_string()],
        };
        Ok((sources, total, hints))
    }
}

/// Cross-source sidecar enrichment hook. Runs after `prepare_sources` /
/// `prepare_sources_from_specs` has built every `Source` and per-source
/// `FormatPlugin::populate_*` has stuffed each source's own `ModelInfo`
/// into its `extensions`. We then opportunistically fetch `config.json`
/// and `model.safetensors.index.json` alongside every source (deduped by
/// HF repo + revision or by parent directory) and insert a `SourceMeta`
/// into each source's extensions. [`crate::ArchLayoutPlugin`] reads it
/// back to validate transformer hyperparameters and reserve canonical
/// slots for tensors that live in shards we didn't load.
///
/// Errors from the sidecar fetches are swallowed inside
/// [`crate::data::try_load_source_meta`] — sidecar info is advisory and a
/// missing sidecar must not break rendering — so this hook can't fail.
pub struct SourceMetaSidecarHook;

#[async_trait(?Send)]
impl PrepareSourcesExtension for SourceMetaSidecarHook {
    async fn enrich(&self, sources: &mut [Source]) -> anyhow::Result<()> {
        let metas = load_meta_for_sources(sources).await;
        for (s, m) in sources.iter_mut().zip(metas) {
            s.extensions.insert(m);
        }
        Ok(())
    }
}

/// Single-image renderer for the `"arch"` layout. arbvis's
/// `single::run_single` looks this up by layout id and invokes it when
/// [`SingleImageRenderer::applicable`] is true. We delegate to the model-side
/// single-image renderer that lifts the per-tensor color buffers from the
/// `ArchLayout` and paints them via the dtype-aware element colorizer.
pub struct ArchSingleImageHook;

impl SingleImageRenderer for ArchSingleImageHook {
    fn id(&self) -> &'static str {
        "arch"
    }

    fn applicable(&self, sources: &[Source], diff_mode: bool, show_xet_xorbs: bool) -> bool {
        // The arch single-image renderer is synchronous and only handles local
        // (mmap'd / owned) data — it can't block a worker on per-pixel HTTP
        // fetches — and has no diff / xet-xorb drawing path. arbvis falls back
        // to byte-Hilbert when this returns false.
        !diff_mode
            && !show_xet_xorbs
            && sources.iter().all(|s| {
                matches!(
                    s.kind,
                    SourceKind::File(_) | SourceKind::Buffered(_) | SourceKind::Diff { .. }
                )
            })
    }

    fn render(
        &self,
        files: &[PathBuf],
        output: Option<PathBuf>,
        sources: &[Source],
        layout: &dyn LayoutShape,
    ) -> anyhow::Result<()> {
        crate::single_arch::run_single_arch(files, output, sources, layout)
    }
}
