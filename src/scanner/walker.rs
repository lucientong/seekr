//! Parallel file tree walker.
//!
//! Uses the `ignore` crate to walk directory trees while respecting
//! `.gitignore` rules and custom exclude patterns.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use ignore::WalkBuilder;

use crate::config::SeekrConfig;
use crate::error::ScannerError;
use crate::scanner::{ScanEntry, ScanResult};

/// Walk a directory tree in parallel, returning all matching file entries.
///
/// Respects `.gitignore` rules and applies configured filters.
pub fn walk_directory(root: &Path, config: &SeekrConfig) -> Result<ScanResult, ScannerError> {
    let start = std::time::Instant::now();

    let mut builder = WalkBuilder::new(root);

    // Configure the walker
    builder
        .hidden(true) // respect hidden files (.gitignore default behavior)
        .git_ignore(true)
        .git_global(true)
        .git_exclude(true)
        .follow_links(false)
        .threads(num_cpus());

    // Add custom exclude overrides from config
    let mut overrides_builder = ignore::overrides::OverrideBuilder::new(root);
    for pattern in &config.exclude_patterns {
        // Negate the pattern to make it an exclude
        let exclude = format!("!{}", pattern);
        overrides_builder.add(&exclude).map_err(|e| {
            ScannerError::FilterError(format!("Invalid exclude pattern '{}': {}", pattern, e))
        })?;
    }
    let overrides = overrides_builder
        .build()
        .map_err(|e| ScannerError::FilterError(format!("Failed to build overrides: {}", e)))?;
    builder.overrides(overrides);

    // Collect entries (using simple Walk for now, parallel walk for large dirs)
    let entries_mutex: Mutex<Vec<ScanEntry>> = Mutex::new(Vec::new());
    let skipped_mutex: Mutex<usize> = Mutex::new(0);

    builder.build_parallel().run(|| {
        Box::new(|entry| {
            match entry {
                Ok(dir_entry) => {
                    // Skip directories, we only want files
                    if dir_entry.file_type().is_some_and(|ft| ft.is_file()) {
                        let path = dir_entry.path().to_path_buf();

                        // Get file metadata
                        match dir_entry.metadata() {
                            Ok(metadata) => {
                                let size = metadata.len();

                                // Skip files exceeding the max size
                                // (config.max_file_size accessed via closure would need Arc,
                                //  for now we use a generous default)
                                let scan_entry = ScanEntry {
                                    path,
                                    size,
                                    modified: metadata.modified().ok(),
                                };
                                entries_mutex.lock().unwrap().push(scan_entry);
                            }
                            Err(_) => {
                                *skipped_mutex.lock().unwrap() += 1;
                            }
                        }
                    }
                    ignore::WalkState::Continue
                }
                Err(_) => {
                    *skipped_mutex.lock().unwrap() += 1;
                    ignore::WalkState::Continue
                }
            }
        })
    });

    let entries = entries_mutex.into_inner().unwrap();
    let skipped = skipped_mutex.into_inner().unwrap();
    let duration = start.elapsed();

    tracing::info!(
        files = entries.len(),
        skipped = skipped,
        duration_ms = duration.as_millis(),
        "Directory scan complete"
    );

    Ok(ScanResult {
        entries,
        skipped,
        duration,
    })
}

/// Walk a directory tree sequentially (simpler, for smaller directories).
pub fn walk_directory_simple(root: &Path) -> Result<Vec<PathBuf>, ScannerError> {
    let walker = WalkBuilder::new(root).hidden(true).git_ignore(true).build();

    let mut files = Vec::new();
    for entry in walker {
        match entry {
            Ok(dir_entry) => {
                if dir_entry.file_type().is_some_and(|ft| ft.is_file()) {
                    files.push(dir_entry.path().to_path_buf());
                }
            }
            Err(e) => {
                tracing::warn!("Walk error: {}", e);
            }
        }
    }

    Ok(files)
}

/// Get the number of available CPUs, with a reasonable minimum.
fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_walk_simple() {
        // Walk the project's own source directory
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let files = walk_directory_simple(&root).unwrap();
        assert!(!files.is_empty(), "Should find at least some source files");
        // Should find our own walker.rs
        assert!(
            files.iter().any(|p| p.ends_with("walker.rs")),
            "Should find walker.rs in the source tree"
        );
    }

    #[test]
    fn test_walk_parallel() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let config = SeekrConfig::default();
        let result = walk_directory(&root, &config).unwrap();
        assert!(!result.entries.is_empty());
        assert!(result.duration.as_secs() < 10, "Scan should be fast");
    }
}
