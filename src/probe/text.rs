//! Probe-input resolution and tokenization for `--probe`.
//!
//! Resolves [`arbvis::ProbeSource`] to a UTF-8 string (one of: a
//! bundled default snippet, a literal `--probe-text` string, a local
//! file via `--probe-file`, or an HF / HTTPS URL via `--probe-url`),
//! then tokenizes it through the model's own `tokenizer.json`.

use std::path::Path;

use anyhow::Context;
use arbvis::ProbeSource;
use tokenizers::Tokenizer;

/// Embedded default probe corpus — ~300 tokens of varied prose, code,
/// math, and multilingual text. Diverse on purpose so the router sees
/// a representative slice rather than one narrow distribution.
const DEFAULT_PROBE_TEXT: &str = include_str!("probe_default.txt");

/// Resolve `source` to a UTF-8 string, fetching from disk or the
/// network as needed. Async because URL-backed sources hit the
/// network; the other variants short-circuit synchronously.
pub async fn resolve(source: &ProbeSource) -> anyhow::Result<String> {
    match source {
        ProbeSource::Default => Ok(DEFAULT_PROBE_TEXT.to_string()),
        ProbeSource::Text(s) => Ok(s.clone()),
        ProbeSource::File(path) => {
            std::fs::read_to_string(path)
                .with_context(|| format!("--probe-file: reading {}", path.display()))
        }
        ProbeSource::Url(url) => fetch_url(url).await,
    }
}

/// Fetch `url` and return its body as a UTF-8 string. Accepts plain
/// HTTPS URLs (e.g. a raw text file on a CDN) or `hf://...` URLs
/// (resolved through the existing `arbvis::hf_url` machinery —
/// downloads via `hf` CLI to the HF cache, then reads from disk).
async fn fetch_url(url: &str) -> anyhow::Result<String> {
    if url.starts_with("hf://") {
        let resolved = arbvis::hf_url::resolve(Path::new(url))
            .await
            .with_context(|| format!("--probe-url: resolving {url}"))?;
        std::fs::read_to_string(&resolved)
            .with_context(|| format!("--probe-url: reading {}", resolved.display()))
    } else {
        let resp = reqwest::get(url)
            .await
            .with_context(|| format!("--probe-url: GET {url}"))?;
        let status = resp.status();
        if !status.is_success() {
            anyhow::bail!("--probe-url: {url} returned HTTP {status}");
        }
        resp.text()
            .await
            .with_context(|| format!("--probe-url: decoding response from {url} as UTF-8"))
    }
}

/// Tokenize `text` with the tokenizer at `<model_dir>/tokenizer.json`.
/// Returns the list of token IDs (no special BOS/EOS unless the
/// tokenizer's own config adds them).
pub fn tokenize(text: &str, model_dir: &Path) -> anyhow::Result<Vec<u32>> {
    let tokenizer_path = model_dir.join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| {
        anyhow::anyhow!(
            "--probe: loading tokenizer from {}: {e}",
            tokenizer_path.display(),
        )
    })?;
    let encoding = tokenizer
        .encode(text, false)
        .map_err(|e| anyhow::anyhow!("--probe: tokenizer.encode failed: {e}"))?;
    Ok(encoding.get_ids().to_vec())
}
