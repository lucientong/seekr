//! # Seekr
//!
//! A semantic code search engine, smarter than grep.
//!
//! Seekr supports three search modes:
//! - **Text regex**: High-performance regular expression matching
//! - **Semantic vector**: ONNX-based local embedding + HNSW KNN search
//! - **AST pattern**: Function signature pattern matching via Tree-sitter
//!
//! 100% local execution — no data leaves your machine.

pub mod config;
pub mod error;

// Core modules — will be implemented in subsequent phases
pub mod scanner;
pub mod parser;
pub mod embedder;
pub mod index;
pub mod search;
pub mod server;

/// Current index format version.
/// Increment this when the on-disk index format changes.
pub const INDEX_VERSION: u32 = 1;

/// Seekr version string (from Cargo.toml).
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
