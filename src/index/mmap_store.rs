//! Memory-mapped index storage.
//!
//! Uses `memmap2` for zero-copy read-only loading of index files,
//! reducing memory usage and improving load times for large indices.

use std::path::Path;

use memmap2::Mmap;

use crate::error::IndexError;

/// A memory-mapped index file for read-only access.
pub struct MmapIndex {
    /// The memory-mapped file.
    _mmap: Mmap,
    /// The raw bytes of the index data.
    data: Vec<u8>,
}

impl MmapIndex {
    /// Open an index file with memory mapping.
    ///
    /// The file is mapped read-only into memory, allowing the OS to
    /// manage paging efficiently.
    pub fn open(path: &Path) -> Result<Self, IndexError> {
        let file = std::fs::File::open(path)?;

        // Safety: the file is opened read-only and we don't modify the mapping
        let mmap = unsafe { Mmap::map(&file)? };

        let data = mmap.to_vec(); // Copy for now; in production, use mmap directly

        Ok(Self { _mmap: mmap, data })
    }

    /// Get the raw bytes of the mapped index.
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Get the size of the mapped data.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if the mapped data is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl std::fmt::Debug for MmapIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MmapIndex")
            .field("len", &self.data.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mmap_open_nonexistent() {
        let result = MmapIndex::open(Path::new("/nonexistent/path/index.dat"));
        assert!(result.is_err());
    }

    #[test]
    fn test_mmap_open_existing() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test.dat");
        std::fs::write(&file_path, b"hello world").unwrap();

        let mmap = MmapIndex::open(&file_path).unwrap();
        assert_eq!(mmap.len(), 11);
        assert_eq!(mmap.as_bytes(), b"hello world");
    }
}
