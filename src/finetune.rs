//! HF model-card "is mod a finetune of orig" auto-detection.
//!
//! Finetune is a model-card concept arbvis has no notion of, so the diff
//! providers ([`crate::hooks::RepoDiffProvider`] / [`crate::hooks::TensorDiffProvider`])
//! resolve it here — honoring a forced `--finetune` / `--no-finetune`, else
//! auto-detecting via [`detect_relation`] (`cardData.base_model` /
//! `base_model_relation`, produced by the HF training scripts).

use arbvis::hf_url::{self, RepoKind};

/// The user's `--finetune` / `--no-finetune` choice, before auto-detection.
#[derive(Clone, Copy, Debug)]
pub enum FinetuneForce {
    /// `--finetune`: treat the diff as a finetune unconditionally.
    On,
    /// `--no-finetune`: treat the diff as NOT a finetune unconditionally.
    Off,
    /// Neither flag: auto-detect from the HF model card, defaulting to off.
    Auto,
}

impl FinetuneForce {
    /// Build from the two mutually-exclusive CLI flags.
    pub fn from_flags(finetune: bool, no_finetune: bool) -> Self {
        if finetune {
            Self::On
        } else if no_finetune {
            Self::Off
        } else {
            Self::Auto
        }
    }
}

/// Resolve whether a `--diff` is a finetune: honor a forced choice, else
/// auto-detect via the HF model card (defaulting to off). Logs the decision.
pub async fn resolve(force: FinetuneForce, orig_url: &str, mod_url: &str) -> bool {
    match force {
        FinetuneForce::On => {
            log::info!("--diff finetune mode: forced on by --finetune");
            true
        }
        FinetuneForce::Off => {
            log::info!("--diff finetune mode: forced off by --no-finetune");
            false
        }
        FinetuneForce::Auto => match detect_relation(orig_url, mod_url).await {
            Some(true) => {
                log::info!(
                    "--diff finetune mode: auto-detected ON ({mod_url} declares {orig_url} as its base in its HF model card)"
                );
                true
            }
            Some(false) => {
                log::info!(
                    "--diff finetune mode: auto-detected OFF ({mod_url} does not declare {orig_url} as a finetune base)"
                );
                false
            }
            None => {
                log::info!(
                    "--diff finetune mode: auto-detect skipped (not both hf:// model URLs, or API lookup failed); defaulting to OFF — pass --finetune to override"
                );
                false
            }
        },
    }
}

/// Auto-detect whether `mod_url` is a HuggingFace-declared finetune of
/// `orig_url`, by reading the modified-side model card metadata via the HF
/// Hub API.
///
/// Returns:
/// - `Some(true)` when both args are model URLs **and** the modified side's
///   `cardData.base_model` (string or list) includes the original side's
///   repo id **and** `cardData.base_model_relation` is `finetune` (or
///   unspecified — HF's convention is that relation defaults to finetune
///   when omitted but a base is declared).
/// - `Some(false)` when both args are HF model URLs but the metadata
///   doesn't establish a finetune relation in the orig→mod direction.
/// - `None` when detection isn't applicable: either side isn't an `hf://`
///   model URL, or the API call failed (network error, private repo
///   without token, 404, etc.). Caller falls back to its own default.
pub async fn detect_relation(orig_url: &str, mod_url: &str) -> Option<bool> {
    let orig_hf = hf_url::parse(orig_url).ok()?;
    let mod_hf = hf_url::parse(mod_url).ok()?;
    // The finetune relation is only meaningful between two model repos.
    if !matches!(orig_hf.kind, RepoKind::Model) || !matches!(mod_hf.kind, RepoKind::Model) {
        return None;
    }

    let info = match hf_url::fetch_model_card(&mod_hf.repo_id).await {
        Ok(v) => v,
        Err(e) => {
            log::debug!(
                "finetune auto-detect: HF API lookup for {} failed: {e:#}",
                mod_hf.repo_id
            );
            return None;
        }
    };

    let cd = info.get("cardData")?;
    let base_match = match cd.get("base_model") {
        Some(serde_json::Value::String(s)) => repo_ids_equal(s, &orig_hf.repo_id),
        Some(serde_json::Value::Array(a)) => a.iter().any(|v| {
            v.as_str()
                .map(|s| repo_ids_equal(s, &orig_hf.repo_id))
                .unwrap_or(false)
        }),
        _ => return Some(false),
    };
    if !base_match {
        return Some(false);
    }
    // Per the HF model-card spec the relation defaults to "finetune" when a
    // base is declared but no explicit relation is given. Anything else
    // ("quantized", "merge", "adapter", "other", ...) does *not* satisfy the
    // finetune contract.
    let relation = cd
        .get("base_model_relation")
        .and_then(|v| v.as_str())
        .unwrap_or("finetune");
    Some(relation.eq_ignore_ascii_case("finetune"))
}

/// `owner/name` comparison that tolerates case differences in the owner /
/// name segments (HF treats these as case-insensitive).
fn repo_ids_equal(a: &str, b: &str) -> bool {
    a.eq_ignore_ascii_case(b)
}
