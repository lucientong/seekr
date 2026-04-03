//! Unified error types for Seekr.
//!
//! Uses `thiserror` to define structured error enums for each module,
//! with a top-level `SeekrError` that aggregates all module errors.

use std::path::PathBuf;

/// Top-level error type for Seekr.
#[derive(Debug, thiserror::Error)]
pub enum SeekrError {
    #[error("Scanner error: {0}")]
    Scanner(#[from] ScannerError),

    #[error("Parser error: {0}")]
    Parser(#[from] ParserError),

    #[error("Embedder error: {0}")]
    Embedder(#[from] EmbedderError),

    #[error("Index error: {0}")]
    Index(#[from] IndexError),

    #[error("Search error: {0}")]
    Search(#[from] SearchError),

    #[error("Server error: {0}")]
    Server(#[from] ServerError),

    #[error("Config error: {0}")]
    Config(#[from] ConfigError),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Errors from the file scanner module.
#[derive(Debug, thiserror::Error)]
pub enum ScannerError {
    #[error("Failed to walk directory '{path}': {source}")]
    WalkError {
        path: PathBuf,
        source: std::io::Error,
    },

    #[error("File filter error: {0}")]
    FilterError(String),

    #[error("File watcher error: {0}")]
    WatchError(String),
}

/// Errors from the code parser module.
#[derive(Debug, thiserror::Error)]
pub enum ParserError {
    #[error("Unsupported language: {0}")]
    UnsupportedLanguage(String),

    #[error("Failed to parse file '{path}': {reason}")]
    ParseFailed { path: PathBuf, reason: String },

    #[error("Chunking error: {0}")]
    ChunkError(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Errors from the embedding engine module.
#[derive(Debug, thiserror::Error)]
pub enum EmbedderError {
    #[error("Model not found at '{0}'")]
    ModelNotFound(PathBuf),

    #[error("Model download failed: {0}")]
    DownloadFailed(String),

    #[error("Model checksum mismatch: expected {expected}, got {actual}")]
    ChecksumMismatch { expected: String, actual: String },

    #[error("ONNX runtime error: {0}")]
    OnnxError(String),

    #[error("Embedding dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Errors from the index engine module.
#[derive(Debug, thiserror::Error)]
pub enum IndexError {
    #[error("Index not found at '{0}'")]
    NotFound(PathBuf),

    #[error("Index version mismatch: file version {file_version}, expected {expected_version}")]
    VersionMismatch {
        file_version: u32,
        expected_version: u32,
    },

    #[error("Corrupted index: {0}")]
    Corrupted(String),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Errors from the search module.
#[derive(Debug, thiserror::Error)]
pub enum SearchError {
    #[error("Invalid regex pattern: {0}")]
    InvalidRegex(String),

    #[error("Invalid AST pattern: {0}")]
    InvalidAstPattern(String),

    #[error("Index error: {0}")]
    Index(#[from] IndexError),

    #[error("Embedder error: {0}")]
    Embedder(#[from] EmbedderError),
}

/// Errors from the server module.
#[derive(Debug, thiserror::Error)]
pub enum ServerError {
    #[error("Failed to bind address '{address}': {source}")]
    BindFailed {
        address: String,
        source: std::io::Error,
    },

    #[error("MCP protocol error: {0}")]
    McpError(String),

    #[error("Internal error: {0}")]
    Internal(String),
}

/// Errors from the configuration module.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("Config file parse error: {0}")]
    ParseError(String),

    #[error("Invalid config value for '{key}': {reason}")]
    InvalidValue { key: String, reason: String },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Convenience type alias for Seekr results.
pub type Result<T> = std::result::Result<T, SeekrError>;
