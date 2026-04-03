//! Index engine module.
//!
//! Manages HNSW vector index for semantic search and inverted text index
//! for keyword search. Supports mmap-based zero-copy loading and
//! incremental updates.

pub mod incremental;
pub mod mmap_store;
pub mod store;

/// An entry in the search index.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct IndexEntry {
    /// Unique chunk ID corresponding to a CodeChunk.
    pub chunk_id: u64,

    /// Embedding vector.
    pub embedding: Vec<f32>,

    /// Tokenized text for inverted index.
    pub text_tokens: Vec<String>,
}

/// A search result from the index.
#[derive(Debug, Clone)]
pub struct SearchHit {
    /// Chunk ID of the matched entry.
    pub chunk_id: u64,

    /// Relevance score.
    pub score: f32,
}
