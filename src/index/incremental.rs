//! Incremental index updates.
//!
//! Detects file changes via mtime + content blake3 hash comparison.
//! Only re-processes changed files' chunks, avoiding full rebuild.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use serde::{Deserialize, Serialize};

use crate::error::IndexError;

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
    pub changed: Vec<PathBuf>,

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
        serde_json::from_slice(&data)
            .map_err(|e| IndexError::Serialization(format!("Failed to load incremental state: {}", e)))
    }

    /// Save incremental state to disk.
    pub fn save(&self, path: &Path) -> Result<(), IndexError> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let data = serde_json::to_vec_pretty(self)
            .map_err(|e| IndexError::Serialization(e.to_string()))?;
        std::fs::write(path, data)?;
        Ok(())
    }

    /// Detect changes between the current file system state and the last index.
    pub fn detect_changes(&self, current_files: &[PathBuf]) -> FileChanges {
        let mut changed = Vec::new();
        let mut unchanged = Vec::new();

        let current_set: std::collections::HashSet<&PathBuf> = current_files.iter().collect();

        for file in current_files {
            if let Some(prev_state) = self.files.get(file) {
                // Check if file has changed
                let mtime = std::fs::metadata(file)
                    .and_then(|m| m.modified())
                    .ok();

                if mtime != Some(prev_state.mtime) {
                    // Mtime changed, verify with content hash
                    if let Ok(content) = std::fs::read(file) {
                        let hash = blake3::hash(&content).to_hex().to_string();
                        if hash != prev_state.content_hash {
                            changed.push(file.clone());
                        } else {
                            unchanged.push(file.clone());
                        }
                    } else {
                        changed.push(file.clone());
                    }
                } else {
                    unchanged.push(file.clone());
                }
            } else {
                // New file
                changed.push(file.clone());
            }
        }

        // Detect deleted files
        let deleted: Vec<PathBuf> = self
            .files
            .keys()
            .filter(|f| !current_set.contains(f))
            .cloned()
            .collect();

        FileChanges {
            changed,
            deleted,
            unchanged,
        }
    }

    /// Update the state for a file that has been indexed.
    pub fn update_file(&mut self, path: PathBuf, content: &[u8], chunk_ids: Vec<u64>) {
        let hash = blake3::hash(content).to_hex().to_string();
        let mtime = std::fs::metadata(&path)
            .and_then(|m| m.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH);

        self.files.insert(
            path,
            FileState {
                mtime,
                content_hash: hash,
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
        let state = IncrementalState::default();
        let changes = state.detect_changes(&[PathBuf::from("/new/file.rs")]);
        assert_eq!(changes.changed.len(), 1);
        assert!(changes.deleted.is_empty());
        assert!(changes.unchanged.is_empty());
    }
}
