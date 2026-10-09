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
use arbvis::{PrepareSourcesExtension, RenderHints, Source, SourceCtx, SourceProvider};
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
        // exists in the interactive 2D Leaflet viewer. The `--3d` volume bundle
        // lays every byte along one Hilbert curve with no notion of scenes, so
        // it can't represent the summary / CKA lenses.
        if ctx.three_d {
            anyhow::bail!(
                "--moe renders a tabbed multi-scene 2D viewer and is incompatible with --3d; \
                 drop --3d to render the MoE scenes"
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

#[cfg(test)]
mod tests {
    use super::*;
    use arbvis::{DestKind, DiffPair, Registry};
    use tempfile::TempDir;

    fn ctx<'a>(
        inputs: &'a [PathBuf],
        diff: Option<(&'a str, &'a str)>,
        three_d: bool,
        stream: bool,
        registry: &'a Registry,
    ) -> SourceCtx<'a> {
        SourceCtx {
            inputs,
            diff: diff.map(|(original, modified)| DiffPair { original, modified }),
            dest_kind: DestKind::Bundle,
            three_d,
            stream,
            show_xet_xorbs: false,
            registry,
        }
    }

    fn with_ctx<T>(
        inputs: &[PathBuf],
        diff: Option<(&str, &str)>,
        three_d: bool,
        stream: bool,
        f: impl FnOnce(&SourceCtx<'_>) -> T,
    ) -> T {
        let registry = Registry::default();
        f(&ctx(inputs, diff, three_d, stream, &registry))
    }

    fn moe_provider() -> MoeSceneProvider {
        MoeSceneProvider {
            target: PathBuf::from("hf://org/model"),
            stat: SummaryStat::default(),
            norm: MoeNorm::default(),
            cka_sample: 1024,
            probe: ProbeOpts::default(),
        }
    }

    // --- MoeSceneProvider: gating + metadata -------------------------------

    #[test]
    fn moe_provider_id_priority_and_applicable_gate() {
        let p = moe_provider();
        assert_eq!(p.id(), "moe-scenes");
        assert_eq!(p.priority(), 400);
        // Applies to a bare `--moe <model>`: no positional inputs, no --diff.
        let empty: [PathBuf; 0] = [];
        let files = [PathBuf::from("model.safetensors")];
        with_ctx(&empty, None, false, false, |c| assert!(p.applicable(c)));
        // A positional input or --diff means a different provider must win.
        with_ctx(&files, None, false, false, |c| assert!(!p.applicable(c)));
        with_ctx(&empty, Some(("hf://a/b", "hf://c/d")), false, false, |c| assert!(
            !p.applicable(c)
        ));
    }

    #[tokio::test]
    async fn moe_provider_rejects_three_d() {
        let p = moe_provider();
        let empty: [PathBuf; 0] = [];
        let res = {
            let registry = Registry::default();
            p.prepare(&ctx(&empty, None, true, false, &registry)).await
        };
        let err = match res {
            Ok(_) => panic!("--moe + --3d must bail"),
            Err(e) => e,
        };
        let msg = format!("{err:#}");
        assert!(msg.contains("--3d"), "error should mention --3d: {msg}");
        assert!(msg.contains("drop --3d"), "error should say how to fix: {msg}");
    }

    // --- RepoDiffProvider: gating + metadata -------------------------------

    #[test]
    fn repo_diff_provider_gates_on_repo_level_urls() {
        let p = RepoDiffProvider {
            diff_metric: DiffMetric::default(),
            finetune: FinetuneForce::Off,
        };
        assert_eq!(p.id(), "repo-diff");
        assert_eq!(p.priority(), 300);
        let empty: [PathBuf; 0] = [];
        // Both sides repo-level HF URLs → applies.
        with_ctx(&empty, Some(("hf://a/b", "hf://c/d")), false, false, |c| assert!(
            p.applicable(c)
        ));
        // One side a file path → not repo-level.
        with_ctx(
            &empty,
            Some(("hf://a/b/file.safetensors", "hf://c/d")),
            false,
            false,
            |c| assert!(!p.applicable(c)),
        );
        // No HF URLs at all (local paths) → does not apply.
        with_ctx(&empty, Some(("/tmp/a", "/tmp/b")), false, false, |c| assert!(
            !p.applicable(c)
        ));
        // No --diff at all → does not apply.
        with_ctx(&empty, None, false, false, |c| assert!(!p.applicable(c)));
    }

    // --- TensorDiffProvider: gating + prepare ------------------------------

    #[test]
    fn tensor_diff_provider_gates_on_two_existing_dirs() {
        let p = TensorDiffProvider {
            diff_metric: DiffMetric::default(),
            finetune: FinetuneForce::Off,
        };
        assert_eq!(p.id(), "tensor-diff");
        assert_eq!(p.priority(), 250);
        let empty: [PathBuf; 0] = [];
        let dir = TempDir::new().unwrap();
        let a = dir.path().join("a");
        let b = dir.path().join("b");
        std::fs::create_dir(&a).unwrap();
        std::fs::create_dir(&b).unwrap();
        let (oa, ob) = (a.to_string_lossy().into_owned(), b.to_string_lossy().into_owned());
        with_ctx(&empty, Some((&oa, &ob)), false, false, |c| assert!(
            p.applicable(c)
        ));
        // One side missing / a file → does not apply.
        with_ctx(&empty, Some((&oa, "/nonexistent-dir-zz")), false, false, |c| assert!(
            !p.applicable(c)
        ));
        // Repo-level URLs → repo-diff's job, not tensor-diff's.
        with_ctx(&empty, Some(("hf://a/b", "hf://c/d")), false, false, |c| assert!(
            !p.applicable(c)
        ));
        // No --diff → does not apply.
        with_ctx(&empty, None, false, false, |c| assert!(!p.applicable(c)));
    }

    #[tokio::test]
    async fn tensor_diff_prepare_bails_on_empty_dirs() {
        let p = TensorDiffProvider {
            diff_metric: DiffMetric::default(),
            finetune: FinetuneForce::On, // no network: skip auto-detection
        };
        let dir = TempDir::new().unwrap();
        let a = dir.path().join("a");
        let b = dir.path().join("b");
        std::fs::create_dir(&a).unwrap();
        std::fs::create_dir(&b).unwrap();
        let (oa, ob) = (a.to_string_lossy().into_owned(), b.to_string_lossy().into_owned());
        let empty: [PathBuf; 0] = [];
        let res = {
            let registry = Registry::default();
            p.prepare(&ctx(&empty, Some((&oa, &ob)), false, false, &registry)).await
        };
        let err = match res {
            Ok(_) => panic!("empty dirs must bail"),
            Err(e) => e,
        };
        assert!(format!("{err:#}").contains("no matching file pairs"));
    }

    fn write_bytes(p: &Path, data: &[u8]) {
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, data).unwrap();
    }

    /// Minimal valid safetensors file: 8-byte LE header size + JSON header
    /// declaring one F32 tensor + its data.
    fn tiny_safetensors() -> Vec<u8> {
        let header = br#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header);
        out.extend_from_slice(&1.0f32.to_le_bytes());
        out
    }

    #[tokio::test]
    async fn tensor_diff_prepare_offsets_byte_sources_past_tensor_sources() {
        let p = TensorDiffProvider {
            diff_metric: DiffMetric::default(),
            finetune: FinetuneForce::On,
        };
        let dir = TempDir::new().unwrap();
        let a = dir.path().join("a");
        let b = dir.path().join("b");
        let st = tiny_safetensors();
        write_bytes(&a.join("w.safetensors"), &st);
        write_bytes(&b.join("w.safetensors"), &st);
        write_bytes(&a.join("notes.txt"), b"hello");
        write_bytes(&b.join("notes.txt"), b"hello");
        let (oa, ob) = (a.to_string_lossy().into_owned(), b.to_string_lossy().into_owned());
        let empty: [PathBuf; 0] = [];
        let registry = Registry::default();
        let (sources, _total, hints) = p
            .prepare(&ctx(&empty, Some((&oa, &ob)), false, false, &registry))
            .await
            .expect("mixed tensor+byte diff should succeed");
        // Both the tensor pair and the byte pair must be represented.
        let tensor = sources
            .iter()
            .filter(|s| matches!(s.kind, arbvis::SourceKind::Custom(_)))
            .count();
        assert!(tensor > 0, "safetensors pair should build a tensor source");
        assert!(sources.iter().any(|s| matches!(s.kind, arbvis::SourceKind::Diff { .. })),
            "byte pair should build a byte diff source");
        // Tensor sources come first (0-based file_idx); the byte remainder's
        // file_idx is re-based past the tensor block.
        assert_eq!(sources[0].file_idx, 0);
        for s in sources.iter().skip(tensor) {
            assert_eq!(s.file_idx, tensor, "byte source file_idx must be offset past the tensor block");
        }
        // Hints: diff mode, "diff" suffix, no xet, both dirs as inputs.
        assert!(hints.diff_mode);
        assert_eq!(hints.title_suffix, "diff");
        assert!(!hints.show_xet_xorbs);
        assert_eq!(hints.inputs, vec![oa.clone(), ob.clone()]);
    }
}

