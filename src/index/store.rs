//! Core index storage.
//!
//! Manages a simple vector index for semantic search and an inverted text
//! index for keyword search. Provides build, query, save, and load operations.

use std::collections::HashMap;
use std::path::Path;

use crate::error::IndexError;
use crate::index::{IndexEntry, SearchHit};
use crate::parser::CodeChunk;
use crate::INDEX_VERSION;

/// The main index structure holding both vector and text indices.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
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
        }
    }

    /// Add an entry to the index.
    pub fn add_entry(&mut self, entry: IndexEntry, chunk: CodeChunk) {
        let chunk_id = entry.chunk_id;

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
    }

    /// Remove a chunk from the index by ID.
    ///
    /// Removes the chunk from vectors, inverted index, and metadata.
    pub fn remove_chunk(&mut self, chunk_id: u64) {
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
    pub fn build_from(
        chunks: &[CodeChunk],
        embeddings: &[Vec<f32>],
        embedding_dim: usize,
    ) -> Self {
        let mut index = Self::new(embedding_dim);

        for (chunk, embedding) in chunks.iter().zip(embeddings.iter()) {
            let text_tokens = tokenize_for_index(&chunk.body);

            let entry = IndexEntry {
                chunk_id: chunk.id,
                embedding: embedding.clone(),
                text_tokens,
            };

            index.add_entry(entry, chunk.clone());
        }

        index
    }

    /// Perform a vector similarity search (brute-force KNN).
    ///
    /// Returns the top-k most similar chunks by cosine similarity.
    pub fn search_vector(
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
    pub fn save(&self, dir: &Path) -> Result<(), IndexError> {
        std::fs::create_dir_all(dir)?;

        let index_path = dir.join("index.json");
        let data = serde_json::to_vec(self)
            .map_err(|e| IndexError::Serialization(e.to_string()))?;
        std::fs::write(&index_path, data)?;

        tracing::info!(
            chunks = self.chunk_count,
            path = %dir.display(),
            "Index saved"
        );

        Ok(())
    }

    /// Load an index from a directory.
    pub fn load(dir: &Path) -> Result<Self, IndexError> {
        let index_path = dir.join("index.json");

        if !index_path.exists() {
            return Err(IndexError::NotFound(index_path));
        }

        let data = std::fs::read(&index_path)?;
        let index: SeekrIndex = serde_json::from_slice(&data)
            .map_err(|e| IndexError::Serialization(e.to_string()))?;

        // Version check
        if index.version != INDEX_VERSION {
            return Err(IndexError::VersionMismatch {
                file_version: index.version,
                expected_version: INDEX_VERSION,
            });
        }

        tracing::info!(
            chunks = index.chunk_count,
            path = %dir.display(),
            "Index loaded"
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
            make_test_chunk(1, "authenticate", "fn authenticate(user: &str, password: &str) -> Result<Token, Error>"),
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
        assert_eq!(results[0].chunk_id, 1, "Should find the most similar chunk first");
    }

    #[test]
    fn test_save_and_load() {
        let chunks = vec![make_test_chunk(1, "test", "fn test() {}")];
        let embeddings = vec![vec![0.5; 4]];
        let index = SeekrIndex::build_from(&chunks, &embeddings, 4);

        let dir = tempfile::tempdir().unwrap();
        index.save(dir.path()).unwrap();

        let loaded = SeekrIndex::load(dir.path()).unwrap();
        assert_eq!(loaded.chunk_count, 1);
        assert_eq!(loaded.version, INDEX_VERSION);
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
}
