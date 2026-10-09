//! Probe-input resolution and tokenization for `--probe`.
//!
//! Resolves [`arbvis::ProbeSource`] to a UTF-8 string (one of: a
//! bundled default snippet, a literal `--probe-text` string, a local
//! file via `--probe-file`, or an HF / HTTPS URL via `--probe-url`),
//! then tokenizes it through the model's own `tokenizer.json`.

use std::path::Path;

use anyhow::Context;
use tokenizers::Tokenizer;

use crate::probe::ProbeSource;

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
        ProbeSource::File(path) => std::fs::read_to_string(path)
            .with_context(|| format!("--probe-file: reading {}", path.display())),
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

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn write_tokenizer(dir: &std::path::Path) {
        // A minimal but valid tokenizer.json: a tiny BPE vocab so encode
        // is deterministic without depending on an HF download.
        let json = r#"{
            "version": "1.0",
            "model": {
                "type": "BPE",
                "vocab": {"h": 0, "e": 1, "l": 2, "o": 3, " ": 4,
                          "he": 5, "llo": 6, " hell": 7},
                "merges": []
            },
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": null,
            "post_processor": null,
            "decoder": null
        }"#;
        std::fs::write(dir.join("tokenizer.json"), json).expect("write tokenizer");
    }

    #[tokio::test]
    async fn resolve_default_returns_bundled_snippet() {
        let text = resolve(&ProbeSource::Default).await.expect("default resolves");
        assert!(!text.trim().is_empty());
    }

    #[tokio::test]
    async fn resolve_text_is_passthrough() {
        let text = resolve(&ProbeSource::Text("hello world".to_string()))
            .await
            .expect("text resolves");
        assert_eq!(text, "hello world");
    }

    #[tokio::test]
    async fn resolve_file_reads_from_disk() {
        let dir = tempdir().expect("tempdir");
        let p = dir.path().join("probe.txt");
        std::fs::write(&p, "from disk").expect("write");
        let text = resolve(&ProbeSource::File(p)).await.expect("file resolves");
        assert_eq!(text, "from disk");
    }

    #[tokio::test]
    async fn resolve_file_missing_errors_with_context() {
        let dir = tempdir().expect("tempdir");
        let p = dir.path().join("absent.txt");
        let err = resolve(&ProbeSource::File(p.clone()))
            .await
            .expect_err("missing file errors");
        assert!(err.to_string().contains(&p.to_string_lossy().to_string()));
    }

    #[tokio::test]
    async fn resolve_bad_url_errors() {
        // Port 1 on loopback refuses connections; reqwest maps it to an
        // error either way, which resolve must surface.
        assert!(resolve(&ProbeSource::Url("http://127.0.0.1:1/nope".to_string()))
            .await
            .is_err());
    }

    #[test]
    fn tokenize_returns_ids_from_model_dir_tokenizer() {
        let dir = tempdir().expect("tempdir");
        write_tokenizer(dir.path());
        let ids = tokenize("hello", dir.path()).expect("tokenize");
        assert!(!ids.is_empty());
        // Deterministic: same input → same ids.
        assert_eq!(ids, tokenize("hello", dir.path()).expect("again"));
    }

    #[test]
    fn tokenize_missing_tokenizer_file_errors() {
        let dir = tempdir().expect("tempdir");
        let err = tokenize("hi", dir.path()).expect_err("no tokenizer.json");
        assert!(err.to_string().contains("tokenizer.json"));
    }
}
