//! [`arbvis::FormatPlugin`] impls — one per supported model file format.
//!
//! Each plugin claims its file extension (`detects_path`), reads the
//! header (sync from a `Path` in `populate_local`, async from a `Data`
//! handle in `populate_remote`), and stuffs a [`ModelInfo`] into the
//! source's [`arbvis::Extensions`] map. Downstream the architectural
//! layout / arch tile loader / renderer read it back via
//! `extensions.get::<ModelInfo>()`.
//!
//! Failures are non-fatal: arbvis's `prepare_sources` logs the plugin
//! error and continues with no `ModelInfo` populated — the file then
//! falls through to the byte-Hilbert path the same way an `.iso` would.

use arbvis::{Data, Extensions, FormatPlugin};
use futures::future::BoxFuture;
use std::path::Path;

use crate::data::{load_model_info, load_model_info_async};
use crate::format::{ModelInfo, SourceFormat};

/// `.safetensors` header parser — produces a [`ModelInfo`] with one
/// [`crate::format::TensorMeta`] per tensor entry.
pub struct SafetensorsFormatPlugin;

impl FormatPlugin for SafetensorsFormatPlugin {
    fn id(&self) -> &'static str {
        "safetensors"
    }
    fn detects_path(&self, path: &Path) -> bool {
        matches!(
            SourceFormat::from_path(path),
            Some(SourceFormat::Safetensors)
        )
    }
    fn populate_local(
        &self,
        path: &Path,
        file_size: u64,
        exts: &mut Extensions,
    ) -> anyhow::Result<()> {
        let info: ModelInfo = load_model_info(path, file_size, SourceFormat::Safetensors)?;
        exts.insert(info);
        Ok(())
    }
    fn populate_remote<'a>(
        &'a self,
        data: &'a Data,
        byte_size: u64,
        exts: &'a mut Extensions,
    ) -> BoxFuture<'a, anyhow::Result<()>> {
        Box::pin(async move {
            let info = load_model_info_async(data, byte_size, SourceFormat::Safetensors).await?;
            exts.insert(info);
            Ok(())
        })
    }
}

/// `.gguf` header parser — produces a [`ModelInfo`] with one
/// [`crate::format::TensorMeta`] per tensor and pre-built color ranges
/// for the metadata + tensor regions.
pub struct GgufFormatPlugin;

impl FormatPlugin for GgufFormatPlugin {
    fn id(&self) -> &'static str {
        "gguf"
    }
    fn detects_path(&self, path: &Path) -> bool {
        matches!(SourceFormat::from_path(path), Some(SourceFormat::Gguf))
    }
    fn populate_local(
        &self,
        path: &Path,
        file_size: u64,
        exts: &mut Extensions,
    ) -> anyhow::Result<()> {
        let info = load_model_info(path, file_size, SourceFormat::Gguf)?;
        exts.insert(info);
        Ok(())
    }
    fn populate_remote<'a>(
        &'a self,
        data: &'a Data,
        byte_size: u64,
        exts: &'a mut Extensions,
    ) -> BoxFuture<'a, anyhow::Result<()>> {
        Box::pin(async move {
            let info = load_model_info_async(data, byte_size, SourceFormat::Gguf).await?;
            exts.insert(info);
            Ok(())
        })
    }
}

/// PyTorch pickle (`.bin` / `.pth` / `.pt`) header parser.
///
/// Local: full `candle_core::pickle` parse of the zip-packed opcode stream.
/// Remote: errors — pickle's zip end-of-central-directory record lives at
/// the END of the file, so a head-prefix range fetch can't parse it. The
/// caller treats remote pickle as plain bytes; downloading first
/// re-enables the local path.
pub struct PickleFormatPlugin;

impl FormatPlugin for PickleFormatPlugin {
    fn id(&self) -> &'static str {
        "pickle"
    }
    fn detects_path(&self, path: &Path) -> bool {
        matches!(SourceFormat::from_path(path), Some(SourceFormat::Pickle))
    }
    fn populate_local(
        &self,
        path: &Path,
        file_size: u64,
        exts: &mut Extensions,
    ) -> anyhow::Result<()> {
        let info = load_model_info(path, file_size, SourceFormat::Pickle)?;
        exts.insert(info);
        Ok(())
    }
    fn populate_remote<'a>(
        &'a self,
        _data: &'a Data,
        _byte_size: u64,
        _exts: &'a mut Extensions,
    ) -> BoxFuture<'a, anyhow::Result<()>> {
        Box::pin(async move {
            anyhow::bail!("pickle: remote header fetch not yet supported — download the file first")
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    /// Minimal valid safetensors file: 8-byte LE header size + JSON header
    /// declaring one F32 tensor + its data (same fixture shape as hooks.rs).
    fn tiny_safetensors() -> Vec<u8> {
        let header = br#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header);
        out.extend_from_slice(&1.0f32.to_le_bytes());
        out
    }

    /// Minimal valid GGUF v2 stream: magic, version 2, zero tensors, one
    /// string KV (same fixture shape as format/gguf.rs).
    fn synthetic_no_tensor_gguf() -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&0x46554747u32.to_le_bytes());
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());
        let key = b"general.architecture";
        bytes.extend_from_slice(&(key.len() as u64).to_le_bytes());
        bytes.extend_from_slice(key);
        bytes.extend_from_slice(&8u32.to_le_bytes());
        let val = b"llama";
        bytes.extend_from_slice(&(val.len() as u64).to_le_bytes());
        bytes.extend_from_slice(val);
        while bytes.len() % 32 != 0 {
            bytes.push(0);
        }
        bytes
    }

    fn write_file(dir: &TempDir, name: &str, data: &[u8]) -> std::path::PathBuf {
        let p = dir.path().join(name);
        std::fs::write(&p, data).unwrap();
        p
    }

    #[test]
    fn ids_are_distinct_and_plugin_detection_partitions_formats() {
        assert_eq!(SafetensorsFormatPlugin.id(), "safetensors");
        assert_eq!(GgufFormatPlugin.id(), "gguf");
        assert_eq!(PickleFormatPlugin.id(), "pickle");

        let cases = [
            ("model.safetensors", true, false, false),
            ("MODEL.SAFETENSORS", true, false, false),
            ("model.gguf", false, true, false),
            ("pytorch_model.bin", false, false, true),
            ("model.pth", false, false, true),
            ("model.pt", false, false, true),
            ("notes.txt", false, false, false),
            ("model.safetensors.bin", false, false, true),
        ];
        for (name, st, gguf, pickle) in cases {
            let p = Path::new(name);
            assert_eq!(SafetensorsFormatPlugin.detects_path(p), st, "{name}");
            assert_eq!(GgufFormatPlugin.detects_path(p), gguf, "{name}");
            assert_eq!(PickleFormatPlugin.detects_path(p), pickle, "{name}");
        }
    }

    #[test]
    fn safetensors_populate_local_inserts_model_info() {
        let dir = TempDir::new().unwrap();
        let path = write_file(&dir, "w.safetensors", &tiny_safetensors());
        let mut exts = Extensions::default();
        let size = std::fs::metadata(&path).unwrap().len();
        SafetensorsFormatPlugin
            .populate_local(&path, size, &mut exts)
            .expect("populates");
        let info = exts.get::<ModelInfo>().expect("ModelInfo inserted");
        assert_eq!(info.format, SourceFormat::Safetensors);
        assert_eq!(info.tensors.len(), 1);
        assert_eq!(info.tensors[0].name, "w");
    }

    #[test]
    fn gguf_populate_local_inserts_model_info() {
        let dir = TempDir::new().unwrap();
        let path = write_file(&dir, "model.gguf", &synthetic_no_tensor_gguf());
        let mut exts = Extensions::default();
        let size = std::fs::metadata(&path).unwrap().len();
        GgufFormatPlugin
            .populate_local(&path, size, &mut exts)
            .expect("populates");
        let info = exts.get::<ModelInfo>().expect("ModelInfo inserted");
        assert_eq!(info.format, SourceFormat::Gguf);
        assert!(info.tensors.is_empty());
    }

    #[test]
    fn safetensors_populate_local_fails_on_garbage() {
        let dir = TempDir::new().unwrap();
        let path = write_file(&dir, "bad.safetensors", b"not a header");
        let mut exts = Extensions::default();
        // Non-fatal by design: the error must surface (no partial insert).
        assert!(SafetensorsFormatPlugin
            .populate_local(&path, 12, &mut exts)
            .is_err());
        assert!(exts.get::<ModelInfo>().is_none());
    }

    #[tokio::test]
    async fn safetensors_populate_remote_uses_owned_data() {
        let data = Data::Owned(tiny_safetensors());
        let size = data.len() as u64;
        let mut exts = Extensions::default();
        SafetensorsFormatPlugin
            .populate_remote(&data, size, &mut exts)
            .await
            .expect("populates");
        let info = exts.get::<ModelInfo>().expect("ModelInfo inserted");
        assert_eq!(info.format, SourceFormat::Safetensors);
        assert_eq!(info.tensors.len(), 1);
    }

    #[tokio::test]
    async fn pickle_populate_remote_is_unsupported() {
        let data = Data::Owned(b"nothing useful".to_vec());
        let mut exts = Extensions::default();
        let err = PickleFormatPlugin
            .populate_remote(&data, 14, &mut exts)
            .await
            .expect_err("remote pickle must fail");
        assert!(err.to_string().contains("not yet supported"), "{err}");
        assert!(exts.get::<ModelInfo>().is_none());
    }
}
