//! # Importer framework
//!
//! Format-agnostic import plumbing: the [`VectorImporter`] trait that the SVG
//! and DXF adapters implement, size/emptiness guards, and [`sniff_and_import`]
//! which picks the first adapter that claims the file. Adapters convert
//! everything into the canonical millimetre [`VectorDocument`] from
//! `laser-core`; unit ambiguity is reported, not guessed (see
//! `laser-core::units`).

use laser_core::{ImportResult, LengthUnit};
use serde::{Deserialize, Serialize};

/// Import failures, each with a stable machine-readable code for the UI.
#[derive(Debug, thiserror::Error)]
pub enum ImportError {
    #[error("{0}")]
    Failed(String),
    #[error("unsupported file type")]
    UnsupportedType,
    #[error("file exceeds the import size limit")]
    TooLarge,
    #[error("empty file")]
    Empty,
}

impl ImportError {
    pub fn code(&self) -> &'static str {
        match self {
            Self::Failed(_) => "import_failed",
            Self::UnsupportedType => "unsupported_type",
            Self::TooLarge => "too_large",
            Self::Empty => "empty",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImportOptions {
    pub file_name: String,
    pub assumed_unit: Option<LengthUnit>,
    pub max_bytes: usize,
}

impl ImportOptions {
    pub fn new(file_name: impl Into<String>) -> Self {
        Self {
            file_name: file_name.into(),
            assumed_unit: None,
            max_bytes: 20 * 1024 * 1024,
        }
    }
}

pub trait VectorImporter: Send + Sync {
    fn can_import(&self, file_name: &str, sniffed_bytes: &[u8]) -> bool;
    fn import(&self, input: &[u8], options: &ImportOptions) -> Result<ImportResult, ImportError>;
}

pub fn sniff_and_import(
    importers: &[&dyn VectorImporter],
    file_name: &str,
    bytes: &[u8],
    mut options: ImportOptions,
) -> Result<ImportResult, ImportError> {
    if bytes.is_empty() {
        return Err(ImportError::Empty);
    }
    if bytes.len() > options.max_bytes {
        return Err(ImportError::TooLarge);
    }
    options.file_name = file_name.to_string();
    let sniffed = &bytes[..bytes.len().min(256)];
    let importer = importers
        .iter()
        .copied()
        .find(|imp| imp.can_import(file_name, sniffed))
        .ok_or(ImportError::UnsupportedType)?;
    importer.import(bytes, &options)
}

pub fn extension_is(file_name: &str, ext: &str) -> bool {
    file_name
        .rsplit('.')
        .next()
        .map(|e| e.eq_ignore_ascii_case(ext))
        .unwrap_or(false)
}
