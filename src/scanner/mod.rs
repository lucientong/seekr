//! File scanner module.
//!
//! Responsible for parallel file tree traversal, file filtering,
//! and file system watching for incremental updates.

pub mod filter;
pub mod walker;
pub mod watcher;

use std::path::PathBuf;
use std::time::SystemTime;

/// A single scanned file entry.
#[derive(Debug, Clone)]
pub struct ScanEntry {
    /// Absolute path to the file.
    pub path: PathBuf,

    /// File size in bytes.
    pub size: u64,

    /// Last modification time.
    pub modified: Option<SystemTime>,
}

/// Result of scanning a directory tree.
#[derive(Debug)]
pub struct ScanResult {
    /// List of scanned file entries.
    pub entries: Vec<ScanEntry>,

    /// Number of files skipped by filters.
    pub skipped: usize,

    /// Total scan duration.
    pub duration: std::time::Duration,
}
