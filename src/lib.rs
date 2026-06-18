//! Tensor-format-aware visualization built on `arbvis`.
//!
//! Owns every model-aware piece of the workspace:
//! - `format/` — safetensors / GGUF / PyTorch-pickle header parsers, dtype
//!   tables, MoE expert decoding.
//! - `layout/` — architectural canvas, MoE summary / CKA layouts, transformer
//!   name classification, dtype-aware element colorizers.
//! - `tiled/` — per-tensor tile load and the dtype-aware tile renderers.
//! - `data` — `TensorDiffSource`, multi-shard safetensors diff helpers, MoE
//!   summary / CKA source prep, `SourceMeta`.
//! - `finetune` — HF model-card finetune auto-detection.
//! - Plugin impls (`ArchLayoutPlugin`, `MoeSummaryLayoutPlugin`,
//!   `MoeCkaLayoutPlugin`, `TensorDiffBuilder`, `ArchRegionsLoader`,
//!   `ArchRegionsRenderer`, the `FormatPlugin` family, the
//!   `SourceMetaSidecarHook`, and the
//!   `MoeSceneProvider`/`RepoDiffProvider`/`TensorDiffProvider` source
//!   providers) are registered on a registry via [`register_all`].
//!
//! The `modelweightvis` binary builds `arbvis::Registry::with_defaults()`,
//! calls `register_all(&mut registry, &model_args)`, and hands off to
//! `arbvis::run`.

#![allow(clippy::too_many_arguments, clippy::type_complexity)]

mod args;
mod cka;
mod colormap;
mod data;
mod diff;
mod finetune;
mod format;
mod format_plugin;
mod hooks;
mod layout;
mod probe;
mod tiled;

pub use args::{DiffMetricArg, LayoutArg, ModelArgs, MoeNormArg};
pub use data::MoeNorm;
pub use diff::TensorDiffBuilder;
pub use format_plugin::{GgufFormatPlugin, PickleFormatPlugin, SafetensorsFormatPlugin};
pub use hooks::{MoeSceneProvider, RepoDiffProvider, SourceMetaSidecarHook, TensorDiffProvider};
pub use layout::{ArchLayoutPlugin, ArchVolumePlugin, MoeCkaLayoutPlugin, MoeSummaryLayoutPlugin};
pub use tiled::{ArchRegionsLoader, ArchRegionsRenderer, ArchVoxelRenderer};

use std::sync::Arc;

use arbvis::Registry;

use crate::finetune::FinetuneForce;
use crate::format::DiffMetric;

/// Register every tensor-aware plugin on `registry` and wire the parsed CLI
/// flags (`args`) into it.
///
/// Populates the Vec slots (`formats`, `layouts`, `diffs`, `providers`), the
/// id-keyed maps (`leaf`'s loader+renderer pair for the `"arch"` layout), the
/// `prepare_sources_extension` hook, the `layout_mode`, and the viewer
/// branding. After this returns, `arbvis::run` handles every CLI shape the
/// model-aware crate supports: `--moe`, repo-level `--diff`,
/// directory-safetensors `--diff`, file-pair tensor `--diff`, and
/// FormatPlugin-driven `ModelInfo` population.
pub fn register_all(registry: &mut Registry, args: &ModelArgs) {
    // --- Static plugins (independent of the parsed flags) ---

    // Per-format header parsers — first plugin that detects a path wins. Stuff
    // `ModelInfo` into `Source.extensions` so the arch layout / arch tile
    // loader / renderer pick it up downstream.
    registry.formats.push(Arc::new(SafetensorsFormatPlugin));
    registry.formats.push(Arc::new(GgufFormatPlugin));
    registry.formats.push(Arc::new(PickleFormatPlugin));

    // Architectural + MoE summary + MoE CKA layouts. `select_layout` sorts by
    // `priority()` descending. The MoE plugins can't collide — they look for
    // different per-source extension tags (`MoeSummaryPanel` vs `MoeCkaPanel`),
    // set by different CLI dispatches.
    registry.layouts.push(Arc::new(ArchLayoutPlugin));
    registry.layouts.push(Arc::new(MoeSummaryLayoutPlugin));
    registry.layouts.push(Arc::new(MoeCkaLayoutPlugin));

    // Tile loader+renderer pair for the `"arch"` layout id.
    registry.leaf.register_loader(Arc::new(ArchRegionsLoader));
    registry
        .leaf
        .register_renderer(Arc::new(ArchRegionsRenderer));

    // 3D (`--3d`) counterparts: a structure-aware volume layout (transformer
    // blocks stacked along Z) and the matching voxel renderer that bakes
    // per-tensor magnitude into the cube. Both keyed on the same `"arch"` id;
    // `select_volume_shape` picks the plugin over arbvis's byte-Hilbert floor
    // when `--layout` isn't `hilbert`.
    registry.volume_shapes.push(Arc::new(ArchVolumePlugin));
    registry
        .voxel
        .register_renderer(Arc::new(ArchVoxelRenderer));

    // Cross-source sidecar enrichment. Runs once per render after every source
    // is built. Fetches `config.json` / `model.safetensors.index.json` next to
    // each source and inserts a `SourceMeta` extension that `ArchLayoutPlugin`
    // reads back for transformer-aware grouping.
    registry.prepare_sources_extension = Some(Arc::new(SourceMetaSidecarHook));

    // Rebrand the viewer arbvis generates: title fallback ("modelweightvis
    // moe" / " diff" / plain) + the info-panel title link + leaflet
    // attribution all point at modelweightvis instead of arbvis.
    registry.branding = arbvis::Branding::new(
        "modelweightvis",
        "https://github.com/znation/modelweightvis",
    );

    // --- Wired from the parsed CLI flags ---
    registry.layout_mode = args.layout.into();
    // `--strict-layout`: turn a forced-layout fallback into a hard error
    // (honored by both `select_layout` and `select_volume_shape`).
    registry.strict_layout = args.strict_layout;
    let diff_metric: DiffMetric = args.diff_metric.into();
    let finetune = FinetuneForce::from_flags(args.finetune, args.no_finetune);

    // Tensor-aware file-pair diff builder (priority 300, consulted by arbvis's
    // byte-diff provider for a local `.safetensors` / `.gguf` file pair).
    registry
        .diffs
        .push(Arc::new(TensorDiffBuilder { diff_metric }));

    // Source providers, all higher priority than arbvis's byte built-ins. The
    // MoE provider is registered ONLY when `--moe` is present so its
    // `applicable` ("no diff, no inputs") can't shadow a bare stdin invocation.
    if let Some(moe) = &args.moe {
        registry.providers.push(Arc::new(MoeSceneProvider {
            target: moe.clone(),
            stat: args.summary_stat.into(),
            norm: args.moe_norm.into(),
            cka_sample: args.cka_sample,
            probe: args.probe_opts(),
        }));
    }
    registry.providers.push(Arc::new(RepoDiffProvider {
        diff_metric,
        finetune,
    }));
    registry.providers.push(Arc::new(TensorDiffProvider {
        diff_metric,
        finetune,
    }));
}
