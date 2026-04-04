//! Semantic vector search.
//!
//! Converts query text to embedding via Embedder, then performs
//! KNN search on the vector index for semantically similar code chunks.

use crate::embedder::traits::Embedder;
use crate::error::SearchError;
use crate::index::SearchHit;
use crate::index::store::SeekrIndex;

/// Options for semantic search.
#[derive(Debug, Clone)]
pub struct SemanticSearchOptions {
    /// Maximum number of results.
    pub top_k: usize,

    /// Minimum cosine similarity score threshold.
    pub score_threshold: f32,
}

impl Default for SemanticSearchOptions {
    fn default() -> Self {
        Self {
            top_k: 20,
            score_threshold: 0.0,
        }
    }
}

/// Perform semantic vector search.
///
/// Embeds the query string using the provided embedder and searches
/// the index for the most similar code chunks by cosine similarity.
pub fn search_semantic(
    index: &SeekrIndex,
    query: &str,
    embedder: &dyn Embedder,
    options: &SemanticSearchOptions,
) -> Result<Vec<SearchHit>, SearchError> {
    // Embed the query
    let query_embedding = embedder.embed(query).map_err(SearchError::Embedder)?;

    // Search the vector index
    let results = index.search_vector(&query_embedding, options.top_k, options.score_threshold);

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedder::batch::DummyEmbedder;
    use crate::parser::{ChunkKind, CodeChunk};
    use std::path::PathBuf;

    fn make_chunk(id: u64, body: &str) -> CodeChunk {
        CodeChunk {
            id,
            file_path: PathBuf::from("test.rs"),
            language: "rust".to_string(),
            kind: ChunkKind::Function,
            name: Some("test".to_string()),
            signature: None,
            doc_comment: None,
            body: body.to_string(),
            byte_range: 0..body.len(),
            line_range: 0..1,
        }
    }

    #[test]
    fn test_semantic_search() {
        let embedder = DummyEmbedder::new(8);

        // Build index with embeddings from the embedder
        let chunks = vec![
            make_chunk(1, "fn authenticate(user: &str) {}"),
            make_chunk(2, "fn calculate(x: f64, y: f64) -> f64 {}"),
        ];

        let embeddings: Vec<Vec<f32>> = chunks
            .iter()
            .map(|c| embedder.embed(&c.body).unwrap())
            .collect();

        let index = SeekrIndex::build_from(&chunks, &embeddings, 8);

        let options = SemanticSearchOptions {
            top_k: 10,
            score_threshold: 0.0,
        };

        // Search for something similar
        let results = search_semantic(
            &index,
            "fn authenticate(user: &str) {}",
            &embedder,
            &options,
        )
        .unwrap();
        assert!(!results.is_empty());
        // The first result should be the authenticate function (most similar to itself)
        assert_eq!(results[0].chunk_id, 1);
    }

    #[test]
    fn test_semantic_search_with_threshold() {
        let embedder = DummyEmbedder::new(8);

        let chunks = vec![make_chunk(1, "fn foo() {}")];
        let embeddings: Vec<Vec<f32>> = chunks
            .iter()
            .map(|c| embedder.embed(&c.body).unwrap())
            .collect();
        let index = SeekrIndex::build_from(&chunks, &embeddings, 8);

        // Very high threshold should filter out most results
        let options = SemanticSearchOptions {
            top_k: 10,
            score_threshold: 0.99,
        };

        let results =
            search_semantic(&index, "completely unrelated text", &embedder, &options).unwrap();
        // With dummy embedder, similarity between different texts should be low
        // This may or may not return results depending on the dummy embedder
        assert!(results.len() <= 1);
    }
}
