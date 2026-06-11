//! Thin binary entrypoint for `modelweightvis`.
//!
//! Diverges from the byte-only `arbvis` binary on two axes:
//! - Uses `modelweightvis::ModelArgs` (clap-flatten of `arbvis::Args` + the
//!   tensor-aware flags `--moe`, `--probe`, `--finetune` / `--no-finetune`,
//!   `--diff-metric`, `--layout`) instead of `arbvis::Args` directly.
//! - Registers tensor-aware plugins (format parsers, arch + MoE summary / CKA
//!   layouts, tensor-diff source builder, arch leaf loader+renderer,
//!   `SourceMetaSidecarHook`, and the option-slot hooks) on top of the
//!   default registry via `register_all`.

use clap::Parser;

fn main() -> anyhow::Result<()> {
    let rt = arbvis::init()?;
    // Bind to a named local so the run-flag stays alive for the process
    // lifetime (drop stops the monitor). See `perf_monitor::spawn_if_enabled`.
    let _perf_monitor_stop = arbvis::perf_monitor_spawn_if_enabled();
    let model_args = modelweightvis::ModelArgs::parse();
    // `--moe-norm` can't ride through arbvis's `ModelOpts`/`MoeScenesPrep`
    // (it predates the option), so pull it off before `split` and hand it to
    // `register_all`, which bakes it into the MoE prep hook.
    let moe_norm: modelweightvis::MoeNorm = model_args.moe_norm.into();
    let (args, opts) = model_args.split();
    let mut registry = arbvis::Registry::with_defaults();
    modelweightvis::register_all(&mut registry, moe_norm);
    rt.block_on(arbvis::run(args, opts, registry))
}
