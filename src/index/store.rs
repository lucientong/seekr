//! Core index storage.
//!
//! Manages a vector index (HNSW + fallback brute-force KNN) for semantic search
//! and an inverted text index for keyword search. Provides build, query, save,
//! and load operations.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use crate::INDEX_VERSION;
use crate::error::IndexError;
use crate::index::{IndexEntry, SearchHit};
use crate::parser::CodeChunk;

// ============================================================
// HNSW Point wrapper
// ============================================================

/// Wrapper around `Vec<f32>` implementing `instant_distance::Point` for HNSW.
///
/// Uses cosine distance (1 - cosine_similarity) as the distance metric,
/// which is appropriate for L2-normalized embedding vectors.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct EmbeddingPoint(Vec<f32>);

#[derive(serde::Serialize, serde::Deserialize)]
struct HnswSidecar {
    version: u32,
    vector_fingerprint: [u8; 32],
    hnsw: instant_distance::HnswMap<EmbeddingPoint, u64>,
}

#[derive(serde::Serialize)]
struct HnswSidecarRef<'a> {
    version: u32,
    vector_fingerprint: [u8; 32],
    hnsw: &'a instant_distance::HnswMap<EmbeddingPoint, u64>,
}

const HNSW_SIDECAR_VERSION: u32 = 1;
const HNSW_FILENAME: &str = "hnsw.bin";

impl instant_distance::Point for EmbeddingPoint {
    fn distance(&self, other: &Self) -> f32 {
        // For L2-normalized vectors, cosine_similarity = dot product.
        // Distance = 1 - similarity (lower is closer).
        let dot: f32 = self.0.iter().zip(other.0.iter()).map(|(a, b)| a * b).sum();
        1.0 - dot
    }
}

/// The main index structure holding both vector and text indices.
///
/// Uses HNSW (Hierarchical Navigable Small Worlds) for fast approximate
/// nearest neighbor search, with brute-force KNN as fallback.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct SeekrIndex {
    /// Index format version for compatibility checks.
    pub version: u32,

    /// Vector index: chunk_id -> embedding vector.
    pub vectors: HashMap<u64, Vec<f32>>,

    /// Inverted text index: token -> list of (chunk_id, frequency).
    pub inverted_index: HashMap<String, Vec<(u64, u32)>>,

    /// Metadata: chunk_id -> stored chunk data.
    pub chunks: HashMap<u64, CodeChunk>,

    /// Embedding dimension.
    pub embedding_dim: usize,

    /// Total number of indexed chunks.
    pub chunk_count: usize,

    /// HNSW index loaded or built only when semantic search needs it.
    #[serde(skip)]
    hnsw: OnceLock<instant_distance::HnswMap<EmbeddingPoint, u64>>,

    #[serde(skip)]
    index_dir: Option<PathBuf>,
}

impl std::fmt::Debug for SeekrIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SeekrIndex")
            .field("version", &self.version)
            .field("embedding_dim", &self.embedding_dim)
            .field("chunk_count", &self.chunk_count)
            .field("vectors_len", &self.vectors.len())
            .field("hnsw", &self.hnsw.get().map(|_| "Some(<HnswMap>)"))
            .finish()
    }
}

impl SeekrIndex {
    /// Create a new empty index.
    pub fn new(embedding_dim: usize) -> Self {
        Self {
            version: INDEX_VERSION,
            vectors: HashMap::new(),
            inverted_index: HashMap::new(),
            chunks: HashMap::new(),
            embedding_dim,
            chunk_count: 0,
            hnsw: OnceLock::new(),
            index_dir: None,
        }
    }

    /// Add an entry to the index.
    pub fn add_entry(&mut self, entry: IndexEntry, chunk: CodeChunk) {
        self.try_add_entry(entry, chunk)
            .expect("content-addressed chunk ID collision");
    }

    /// Add an entry while reporting the vanishingly rare hash collision.
    pub fn try_add_entry(&mut self, entry: IndexEntry, chunk: CodeChunk) -> Result<(), IndexError> {
        self.hnsw.take();
        let chunk_id = entry.chunk_id;

        if let Some(existing) = self.chunks.get(&chunk_id) {
            let same_identity = existing.file_path == chunk.file_path
                && existing.kind == chunk.kind
                && existing.name == chunk.name
                && existing.body == chunk.body;
            if !same_identity {
                return Err(IndexError::ChunkIdCollision {
                    chunk_id,
                    existing_path: existing.file_path.clone(),
                    incoming_path: chunk.file_path.clone(),
                });
            }
            self.remove_chunk(chunk_id);
        }

        // Add to vector index
        self.vectors.insert(chunk_id, entry.embedding);

        // Add to inverted text index
        for token in &entry.text_tokens {
            let posting_list = self.inverted_index.entry(token.clone()).or_default();
            if let Some(existing) = posting_list.iter_mut().find(|(id, _)| *id == chunk_id) {
                existing.1 += 1;
            } else {
                posting_list.push((chunk_id, 1));
            }
        }

        // Store chunk metadata
        self.chunks.insert(chunk_id, chunk);
        self.chunk_count = self.chunks.len();
        Ok(())
    }

    /// Remove a chunk from the index by ID.
    ///
    /// Removes the chunk from vectors, inverted index, and metadata.
    pub fn remove_chunk(&mut self, chunk_id: u64) {
        self.hnsw.take();
        // Remove from vector index
        self.vectors.remove(&chunk_id);

        // Remove from inverted text index
        self.inverted_index.retain(|_token, posting_list| {
            posting_list.retain(|(id, _)| *id != chunk_id);
            !posting_list.is_empty()
        });

        // Remove from chunk metadata
        self.chunks.remove(&chunk_id);
        self.chunk_count = self.chunks.len();
    }

    /// Remove multiple chunks by their IDs.
    pub fn remove_chunks(&mut self, chunk_ids: &[u64]) {
        for &chunk_id in chunk_ids {
            self.remove_chunk(chunk_id);
        }
    }

    /// Build the index from chunks and their embeddings.
    ///
    /// Also builds the HNSW graph for fast approximate nearest neighbor search.
    pub fn build_from(chunks: &[CodeChunk], embeddings: &[Vec<f32>], embedding_dim: usize) -> Self {
        Self::try_build_from(chunks, embeddings, embedding_dim)
            .expect("content-addressed chunk ID collision")
    }

    /// Build an index and return a structured error if chunk IDs collide.
    pub fn try_build_from(
        chunks: &[CodeChunk],
        embeddings: &[Vec<f32>],
        embedding_dim: usize,
    ) -> Result<Self, IndexError> {
        let mut index = Self::new(embedding_dim);

        for (chunk, embedding) in chunks.iter().zip(embeddings.iter()) {
            let text_tokens = tokenize_for_index(&chunk.body);

            let entry = IndexEntry {
                chunk_id: chunk.id,
                embedding: embedding.clone(),
                text_tokens,
            };

            index.try_add_entry(entry, chunk.clone())?;
        }

        // Build HNSW graph from all vectors
        index.rebuild_hnsw();

        Ok(index)
    }

    /// Rebuild the HNSW graph from the current vectors HashMap.
    ///
    /// Called after build_from(), load(), or after incremental updates.
    pub fn rebuild_hnsw(&mut self) {
        self.hnsw.take();
        if self.vectors.is_empty() {
            return;
        }

        let hnsw_map = self.build_hnsw();
        let _ = self.hnsw.set(hnsw_map);

        tracing::debug!(chunks = self.vectors.len(), "HNSW graph built");
    }

    /// Perform a vector similarity search.
    ///
    /// Uses HNSW approximate nearest neighbor search when available (O(log n)),
    /// falling back to brute-force KNN (O(n*d)) for older indexes or when
    /// HNSW is not built.
    ///
    /// Returns the top-k most similar chunks by cosine similarity.
    pub fn search_vector(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        score_threshold: f32,
    ) -> Vec<SearchHit> {
        if let Some(hnsw) = self.ensure_hnsw() {
            // Fast path: HNSW approximate nearest neighbor search
            self.search_vector_hnsw(hnsw, query_embedding, top_k, score_threshold)
        } else {
            // Fallback: brute-force KNN (for backward compatibility or small indexes)
            self.search_vector_brute_force(query_embedding, top_k, score_threshold)
        }
    }

    fn ensure_hnsw(&self) -> Option<&instant_distance::HnswMap<EmbeddingPoint, u64>> {
        if self.vectors.is_empty() {
            return None;
        }
        Some(self.hnsw.get_or_init(|| {
            if let Some(index_dir) = &self.index_dir {
                match self.load_hnsw_sidecar(index_dir) {
                    Ok(Some(hnsw)) => {
                        tracing::debug!("HNSW sidecar loaded");
                        return hnsw;
                    }
                    Ok(None) => {}
                    Err(error) => {
                        tracing::warn!(%error, "Ignoring invalid HNSW sidecar");
                    }
                }
            }

            let hnsw = self.build_hnsw();
            if let Some(index_dir) = &self.index_dir {
                if let Err(error) = self.save_hnsw_sidecar(index_dir, &hnsw) {
                    tracing::warn!(%error, "Failed to persist rebuilt HNSW sidecar");
                }
            }
            hnsw
        }))
    }

    fn build_hnsw(&self) -> instant_distance::HnswMap<EmbeddingPoint, u64> {
        let mut entries: Vec<_> = self.vectors.iter().collect();
        entries.sort_unstable_by_key(|(chunk_id, _)| **chunk_id);
        let points = entries
            .iter()
            .map(|(_, embedding)| EmbeddingPoint((*embedding).clone()))
            .collect();
        let values = entries.iter().map(|(chunk_id, _)| **chunk_id).collect();
        instant_distance::Builder::default().build(points, values)
    }

    /// HNSW-based vector search (O(log n) per query).
    fn search_vector_hnsw(
        &self,
        hnsw: &instant_distance::HnswMap<EmbeddingPoint, u64>,
        query_embedding: &[f32],
        top_k: usize,
        score_threshold: f32,
    ) -> Vec<SearchHit> {
        let query_point = EmbeddingPoint(query_embedding.to_vec());
        let mut search = instant_distance::Search::default();

        let results: Vec<SearchHit> = hnsw
            .search(&query_point, &mut search)
            .take(top_k)
            .filter_map(|item| {
                let chunk_id = *item.value;
                // Convert distance back to similarity: similarity = 1 - distance
                let score = 1.0 - item.distance;
                if score >= score_threshold {
                    Some(SearchHit { chunk_id, score })
                } else {
                    None
                }
            })
            .collect();

        results
    }

    /// Brute-force KNN vector search (O(n*d) per query).
    ///
    /// Used as fallback when HNSW index is not available.
    fn search_vector_brute_force(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        score_threshold: f32,
    ) -> Vec<SearchHit> {
        let mut scores: Vec<(u64, f32)> = self
            .vectors
            .iter()
            .map(|(&chunk_id, embedding)| {
                let score = cosine_similarity(query_embedding, embedding);
                (chunk_id, score)
            })
            .filter(|(_, score)| *score >= score_threshold)
            .collect();

        // Sort by score descending
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scores
            .into_iter()
            .take(top_k)
            .map(|(chunk_id, score)| SearchHit { chunk_id, score })
            .collect()
    }

    /// Perform a text search using the inverted index.
    ///
    /// Returns chunks that contain the query tokens, scored by TF.
    pub fn search_text(&self, query: &str, top_k: usize) -> Vec<SearchHit> {
        let query_tokens = tokenize_for_index(query);

        if query_tokens.is_empty() {
            return Vec::new();
        }

        // Accumulate scores for each chunk
        let mut scores: HashMap<u64, f32> = HashMap::new();

        for token in &query_tokens {
            if let Some(posting_list) = self.inverted_index.get(token) {
                for &(chunk_id, frequency) in posting_list {
                    *scores.entry(chunk_id).or_default() += frequency as f32;
                }
            }
        }

        // Normalize by number of query tokens
        let num_tokens = query_tokens.len() as f32;
        let mut results: Vec<(u64, f32)> = scores
            .into_iter()
            .map(|(id, score)| (id, score / num_tokens))
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        results
            .into_iter()
            .take(top_k)
            .map(|(chunk_id, score)| SearchHit { chunk_id, score })
            .collect()
    }

    /// Get a chunk by its ID.
    pub fn get_chunk(&self, chunk_id: u64) -> Option<&CodeChunk> {
        self.chunks.get(&chunk_id)
    }

    /// Save the index to a directory.
    ///
    /// Uses bincode serialization and an atomic replace (v3 format).
    pub fn save(&self, dir: &Path) -> Result<(), IndexError> {
        std::fs::create_dir_all(dir)?;

        let index_path = dir.join("index.bin");
        let data =
            bincode::serialize(self).map_err(|e| IndexError::Serialization(e.to_string()))?;
        crate::index::atomic::atomic_write(&index_path, &data)?;
        if let Some(hnsw) = self.hnsw.get() {
            self.save_hnsw_sidecar(dir, hnsw)?;
        }

        // Remove old JSON index if present (migration from v1)
        let old_json_path = dir.join("index.json");
        if old_json_path.exists() {
            let _ = std::fs::remove_file(&old_json_path);
        }

        tracing::info!(
            chunks = self.chunk_count,
            path = %dir.display(),
            "Index saved (bincode v3)"
        );

        Ok(())
    }

    fn save_hnsw_sidecar(
        &self,
        dir: &Path,
        hnsw: &instant_distance::HnswMap<EmbeddingPoint, u64>,
    ) -> Result<(), IndexError> {
        let sidecar = HnswSidecarRef {
            version: HNSW_SIDECAR_VERSION,
            vector_fingerprint: self.vector_fingerprint(),
            hnsw,
        };
        let data =
            bincode::serialize(&sidecar).map_err(|e| IndexError::Serialization(e.to_string()))?;
        crate::index::atomic::atomic_write(&dir.join(HNSW_FILENAME), &data)
    }

    fn load_hnsw_sidecar(
        &self,
        dir: &Path,
    ) -> Result<Option<instant_distance::HnswMap<EmbeddingPoint, u64>>, IndexError> {
        let path = dir.join(HNSW_FILENAME);
        if !path.exists() {
            return Ok(None);
        }
        let data = std::fs::read(path)?;
        let sidecar: HnswSidecar =
            bincode::deserialize(&data).map_err(|e| IndexError::Serialization(e.to_string()))?;
        if sidecar.version != HNSW_SIDECAR_VERSION
            || sidecar.vector_fingerprint != self.vector_fingerprint()
        {
            return Ok(None);
        }
        Ok(Some(sidecar.hnsw))
    }

    fn vector_fingerprint(&self) -> [u8; 32] {
        let mut entries: Vec<_> = self.vectors.iter().collect();
        entries.sort_unstable_by_key(|(chunk_id, _)| **chunk_id);
        let mut hasher = blake3::Hasher::new();
        hasher.update(&(self.embedding_dim as u64).to_le_bytes());
        hasher.update(&(entries.len() as u64).to_le_bytes());
        for (chunk_id, embedding) in entries {
            hasher.update(&chunk_id.to_le_bytes());
            hasher.update(&(embedding.len() as u64).to_le_bytes());
            for value in embedding {
                hasher.update(&value.to_bits().to_le_bytes());
            }
        }
        *hasher.finalize().as_bytes()
    }

    /// Load an index from a directory.
    ///
    /// Tries bincode format first, then falls back to JSON (v1) for migration.
    /// After loading, rebuilds the HNSW graph from the vectors for fast search.
    pub fn load(dir: &Path) -> Result<Self, IndexError> {
        let bin_path = dir.join("index.bin");
        let json_path = dir.join("index.json");

        let mut index: SeekrIndex = if bin_path.exists() {
            // v2/v3: bincode format
            let data = std::fs::read(&bin_path)?;
            bincode::deserialize(&data).map_err(|e| IndexError::Serialization(e.to_string()))?
        } else if json_path.exists() {
            // v1: JSON format (backward compatibility)
            let data = std::fs::read(&json_path)?;
            serde_json::from_slice(&data).map_err(|e| IndexError::Serialization(e.to_string()))?
        } else {
            return Err(IndexError::NotFound(bin_path));
        };

        // Version check
        if index.version != INDEX_VERSION {
            return Err(IndexError::VersionMismatch {
                file_version: index.version,
                expected_version: INDEX_VERSION,
            });
        }

        index.index_dir = Some(dir.to_path_buf());

        tracing::info!(
            chunks = index.chunk_count,
            path = %dir.display(),
            "Index loaded (HNSW deferred)"
        );

        Ok(index)
    }
}

/// Compute cosine similarity between two vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

/// Simple tokenization for the inverted text index.
///
/// Splits on whitespace and punctuation, lowercases, filters short tokens.
fn tokenize_for_index(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric() && c != '_')
        .map(|s| s.to_lowercase())
        .filter(|s| s.len() >= 2)
        .collect()
}

/// Public wrapper for `tokenize_for_index` — used by incremental indexing.
pub fn tokenize_for_index_pub(text: &str) -> Vec<String> {
    tokenize_for_index(text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::ChunkKind;
    use std::path::PathBuf;

    fn make_test_chunk(id: u64, name: &str, body: &str) -> CodeChunk {
        CodeChunk {
            id,
            file_path: PathBuf::from("test.rs"),
            language: "rust".to_string(),
            kind: ChunkKind::Function,
            name: Some(name.to_string()),
            signature: None,
            doc_comment: None,
            body: body.to_string(),
            byte_range: 0..body.len(),
            line_range: 0..1,
        }
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&a, &b)).abs() < 0.01);

        let c = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &c) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_build_and_search_text() {
        let chunks = vec![
            make_test_chunk(
                1,
                "authenticate",
                "fn authenticate(user: &str, password: &str) -> Result<Token, Error>",
            ),
            make_test_chunk(2, "calculate", "fn calculate_total(items: &[Item]) -> f64"),
        ];
        let embeddings = vec![vec![0.1; 8], vec![0.2; 8]];

        let index = SeekrIndex::build_from(&chunks, &embeddings, 8);

        assert_eq!(index.chunk_count, 2);

        // Text search
        let results = index.search_text("authenticate user password", 10);
        assert!(!results.is_empty());
        assert_eq!(results[0].chunk_id, 1);
    }

    #[test]
    fn test_build_and_search_vector() {
        let chunks = vec![
            make_test_chunk(1, "foo", "fn foo()"),
            make_test_chunk(2, "bar", "fn bar()"),
        ];
        let embeddings = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];

        let index = SeekrIndex::build_from(&chunks, &embeddings, 3);

        // Search for something similar to chunk 1
        let query = vec![0.9, 0.1, 0.0];
        let results = index.search_vector(&query, 2, 0.0);
        assert!(!results.is_empty());
        assert_eq!(
            results[0].chunk_id, 1,
            "Should find the most similar chunk first"
        );
    }

    #[test]
    fn test_save_and_load() {
        let chunks = vec![make_test_chunk(1, "test", "fn test() {}")];
        let embeddings = vec![vec![0.5; 4]];
        let index = SeekrIndex::build_from(&chunks, &embeddings, 4);

        let dir = tempfile::tempdir().unwrap();
        index.save(dir.path()).unwrap();
        index.save(dir.path()).unwrap();
        assert!(dir.path().join(HNSW_FILENAME).exists());

        let loaded = SeekrIndex::load(dir.path()).unwrap();
        assert_eq!(loaded.chunk_count, 1);
        assert_eq!(loaded.version, INDEX_VERSION);
        assert!(loaded.hnsw.get().is_none());
        assert!(!loaded.search_text("test", 10).is_empty());
        assert!(
            loaded.hnsw.get().is_none(),
            "text search must not load HNSW"
        );
        assert!(!loaded.search_vector(&[0.5; 4], 1, 0.0).is_empty());
        assert!(loaded.hnsw.get().is_some());
        assert!(std::fs::read_dir(dir.path()).unwrap().all(|entry| {
            !entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .contains(".tmp-")
        }));
    }

    #[test]
    fn test_tokenize_for_index() {
        let tokens = tokenize_for_index("fn authenticate_user(username: &str) -> Result<String>");
        assert!(tokens.contains(&"fn".to_string()));
        assert!(tokens.contains(&"authenticate_user".to_string()));
        assert!(tokens.contains(&"username".to_string()));
        assert!(tokens.contains(&"result".to_string()));
        assert!(tokens.contains(&"string".to_string()));
    }

    #[test]
    fn test_add_entry_rejects_chunk_id_collision_without_overwrite() {
        let mut index = SeekrIndex::new(2);
        let first = make_test_chunk(7, "first", "fn first() {}");
        index.add_entry(
            IndexEntry {
                chunk_id: 7,
                embedding: vec![1.0, 0.0],
                text_tokens: vec!["first".to_string()],
            },
            first.clone(),
        );

        let mut collision = make_test_chunk(7, "second", "fn second() {}");
        collision.file_path = PathBuf::from("other.rs");
        let result = index.try_add_entry(
            IndexEntry {
                chunk_id: 7,
                embedding: vec![0.0, 1.0],
                text_tokens: vec!["second".to_string()],
            },
            collision,
        );

        assert!(matches!(result, Err(IndexError::ChunkIdCollision { .. })));
        assert_eq!(index.get_chunk(7).unwrap().name.as_deref(), Some("first"));
        assert_eq!(index.vectors[&7], vec![1.0, 0.0]);
    }
}
