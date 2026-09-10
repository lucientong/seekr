//! Incremental index updates.
//!
//! Detects file changes via mtime + content blake3 hash comparison.
//! Only re-processes changed files' chunks, avoiding full rebuild.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use serde::{Deserialize, Serialize};

use crate::error::IndexError;

/// One immutable file snapshot used for change detection, parsing and state.
#[derive(Debug)]
pub struct FileSnapshot {
    pub path: PathBuf,
    pub content: Vec<u8>,
    pub mtime: SystemTime,
    pub content_hash: String,
}

/// State of a previously indexed file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileState {
    /// Last modification time.
    pub mtime: SystemTime,

    /// Blake3 hash of file content.
    pub content_hash: String,

    /// Chunk IDs produced from this file.
    pub chunk_ids: Vec<u64>,
}

/// The incremental state, tracking which files have been indexed.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct IncrementalState {
    /// Map from file path to its last indexed state.
    pub files: HashMap<PathBuf, FileState>,
}

/// Classification of file changes.
#[derive(Debug)]
pub struct FileChanges {
    /// Files that are new or have been modified.
    pub changed: Vec<FileSnapshot>,

    /// Files that have been deleted since last index.
    pub deleted: Vec<PathBuf>,

    /// Files that are unchanged.
    pub unchanged: Vec<PathBuf>,
}

impl IncrementalState {
    /// Load incremental state from disk.
    pub fn load(path: &Path) -> Result<Self, IndexError> {
        if !path.exists() {
            return Ok(Self::default());
        }

        let data = std::fs::read(path)?;
        serde_json::from_slice(&data).map_err(|e| {
            IndexError::Serialization(format!("Failed to load incremental state: {}", e))
        })
    }

    /// Save incremental state to disk.
    pub fn save(&self, path: &Path) -> Result<(), IndexError> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let data = serde_json::to_vec_pretty(self)
            .map_err(|e| IndexError::Serialization(e.to_string()))?;
        crate::index::atomic::atomic_write(path, &data)
    }

    /// Detect changes between the current file system state and the last index.
    pub fn detect_changes(&self, current_files: &[PathBuf]) -> Result<FileChanges, IndexError> {
        let mut changed = Vec::new();
        let mut unchanged = Vec::new();

        let current_set: std::collections::HashSet<&PathBuf> = current_files.iter().collect();

        for file in current_files {
            let content = std::fs::read(file)?;
            let metadata = std::fs::metadata(file)?;
            let mtime = metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH);
            let content_hash = blake3::hash(&content).to_hex().to_string();
            if self
                .files
                .get(file)
                .is_some_and(|previous| previous.content_hash == content_hash)
            {
                unchanged.push(file.clone());
            } else {
                changed.push(FileSnapshot {
                    path: file.clone(),
                    content,
                    mtime,
                    content_hash,
                });
            }
        }

        // Detect deleted files
        let deleted: Vec<PathBuf> = self
            .files
            .keys()
            .filter(|f| !current_set.contains(f))
            .cloned()
            .collect();

        Ok(FileChanges {
            changed,
            deleted,
            unchanged,
        })
    }

    /// Update the state for a file that has been indexed.
    pub fn update_file(&mut self, path: PathBuf, content: &[u8], chunk_ids: Vec<u64>) {
        let hash = blake3::hash(content).to_hex().to_string();
        let mtime = std::fs::metadata(&path)
            .and_then(|m| m.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH);
        self.update_snapshot(path, mtime, hash, chunk_ids);
    }

    /// Update state from the exact immutable snapshot that was indexed.
    pub fn update_snapshot(
        &mut self,
        path: PathBuf,
        mtime: SystemTime,
        content_hash: String,
        chunk_ids: Vec<u64>,
    ) {
        self.files.insert(
            path,
            FileState {
                mtime,
                content_hash,
                chunk_ids,
            },
        );
    }

    /// Remove a file from the incremental state.
    pub fn remove_file(&mut self, path: &Path) -> Option<FileState> {
        self.files.remove(path)
    }

    /// Get chunk IDs associated with a file.
    pub fn chunk_ids_for_file(&self, path: &Path) -> Vec<u64> {
        self.files
            .get(path)
            .map(|state| state.chunk_ids.clone())
            .unwrap_or_default()
    }

    /// Get all chunk IDs from deleted files.
    pub fn chunk_ids_to_remove(&self, deleted_files: &[PathBuf]) -> Vec<u64> {
        deleted_files
            .iter()
            .flat_map(|path| self.chunk_ids_for_file(path))
            .collect()
    }

    /// Merge changes: remove deleted file entries, return IDs to remove from index.
    pub fn apply_deletions(&mut self, deleted_files: &[PathBuf]) -> Vec<u64> {
        let mut removed_ids = Vec::new();
        for path in deleted_files {
            if let Some(state) = self.remove_file(path) {
                removed_ids.extend(state.chunk_ids);
            }
        }
        removed_ids
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_incremental_state_save_load() {
        let dir = tempfile::tempdir().unwrap();
        let state_path = dir.path().join("state.json");

        let mut state = IncrementalState::default();
        state.update_file(
            PathBuf::from("/test/file.rs"),
            b"fn main() {}",
            vec![1, 2, 3],
        );

        state.save(&state_path).unwrap();

        let loaded = IncrementalState::load(&state_path).unwrap();
        assert_eq!(loaded.files.len(), 1);
        assert!(loaded.files.contains_key(&PathBuf::from("/test/file.rs")));
    }

    #[test]
    fn test_detect_new_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("new.rs");
        std::fs::write(&path, "fn new() {}").unwrap();
        let state = IncrementalState::default();
        let changes = state.detect_changes(std::slice::from_ref(&path)).unwrap();
        assert_eq!(changes.changed.len(), 1);
        assert_eq!(changes.changed[0].path, path);
        assert!(changes.deleted.is_empty());
        assert!(changes.unchanged.is_empty());
    }

    #[test]
    fn content_change_is_detected_even_when_mtime_is_unchanged() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("same-mtime.rs");
        std::fs::write(&path, "fn before() {}").unwrap();
        let mtime = std::fs::metadata(&path).unwrap().modified().unwrap();
        let mut state = IncrementalState::default();
        state.update_snapshot(
            path.clone(),
            mtime,
            blake3::hash(b"fn before() {}").to_hex().to_string(),
            vec![1],
        );

        std::fs::write(&path, "fn after() {}").unwrap();
        // The detector is content-authoritative, independent of mtime.
        let changes = state.detect_changes(&[path]).unwrap();
        assert_eq!(changes.changed.len(), 1);
    }
}
