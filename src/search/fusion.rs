//! Multi-source result fusion.
//!
//! Implements Reciprocal Rank Fusion (RRF) to combine results from
//! text search and semantic search into a unified ranked list.
//!
//! Formula: `score = sum(1 / (k + rank_i))` where k defaults to 60.

use std::collections::HashMap;

use crate::index::SearchHit;
use crate::search::ast_pattern::AstMatch;
use crate::search::text::TextMatch;

/// Fused search result combining scores from multiple sources.
#[derive(Debug, Clone)]
pub struct FusedResult {
    /// The chunk ID.
    pub chunk_id: u64,

    /// The RRF fusion score.
    pub fused_score: f32,

    /// Original score from text search (if any).
    pub text_score: Option<f32>,

    /// Original score from semantic search (if any).
    pub semantic_score: Option<f32>,

    /// Original score from AST pattern search (if any).
    pub ast_score: Option<f32>,

    /// Matched line numbers from text search (propagated for display).
    pub matched_lines: Vec<usize>,
}

/// Perform Reciprocal Rank Fusion (RRF) on multiple result lists.
///
/// Combines text search and semantic search results into a single
/// ranked list using the RRF formula: `score = sum(1 / (k + rank))`.
///
/// # Arguments
/// * `text_results` - Results from text regex search (in rank order).
/// * `semantic_results` - Results from semantic vector search (in rank order).
/// * `k` - RRF parameter controlling rank discount (default: 60).
/// * `top_k` - Maximum number of fused results to return.
pub fn rrf_fuse(
    text_results: &[TextMatch],
    semantic_results: &[SearchHit],
    k: u32,
    top_k: usize,
) -> Vec<FusedResult> {
    let mut scores: HashMap<u64, FusedResult> = HashMap::new();

    // Process text search results
    for (rank, result) in text_results.iter().enumerate() {
        let rrf_score = 1.0 / (k as f32 + rank as f32 + 1.0);

        scores
            .entry(result.chunk_id)
            .and_modify(|e| {
                e.fused_score += rrf_score;
                e.text_score = Some(result.score);
                e.matched_lines = result.matched_lines.clone();
            })
            .or_insert(FusedResult {
                chunk_id: result.chunk_id,
                fused_score: rrf_score,
                text_score: Some(result.score),
                semantic_score: None,
                ast_score: None,
                matched_lines: result.matched_lines.clone(),
            });
    }

    // Process semantic search results
    for (rank, result) in semantic_results.iter().enumerate() {
        let rrf_score = 1.0 / (k as f32 + rank as f32 + 1.0);

        scores
            .entry(result.chunk_id)
            .and_modify(|e| {
                e.fused_score += rrf_score;
                e.semantic_score = Some(result.score);
            })
            .or_insert(FusedResult {
                chunk_id: result.chunk_id,
                fused_score: rrf_score,
                text_score: None,
                semantic_score: Some(result.score),
                ast_score: None,
                matched_lines: Vec::new(),
            });
    }

    // Sort by fused score descending
    let mut fused: Vec<FusedResult> = scores.into_values().collect();
    fused.sort_by(|a, b| {
        b.fused_score
            .partial_cmp(&a.fused_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Truncate to top-k
    fused.truncate(top_k);

    fused
}

/// Perform three-way RRF fusion across text, semantic, and AST results.
///
/// Extends the basic two-way RRF to include AST pattern search results.
pub fn rrf_fuse_three(
    text_results: &[TextMatch],
    semantic_results: &[SearchHit],
    ast_results: &[AstMatch],
    k: u32,
    top_k: usize,
) -> Vec<FusedResult> {
    let mut scores: HashMap<u64, FusedResult> = HashMap::new();

    // Process text search results
    for (rank, result) in text_results.iter().enumerate() {
        let rrf_score = 1.0 / (k as f32 + rank as f32 + 1.0);

        scores
            .entry(result.chunk_id)
            .and_modify(|e| {
                e.fused_score += rrf_score;
                e.text_score = Some(result.score);
                e.matched_lines = result.matched_lines.clone();
            })
            .or_insert(FusedResult {
                chunk_id: result.chunk_id,
                fused_score: rrf_score,
                text_score: Some(result.score),
                semantic_score: None,
                ast_score: None,
                matched_lines: result.matched_lines.clone(),
            });
    }

    // Process semantic search results
    for (rank, result) in semantic_results.iter().enumerate() {
        let rrf_score = 1.0 / (k as f32 + rank as f32 + 1.0);

        scores
            .entry(result.chunk_id)
            .and_modify(|e| {
                e.fused_score += rrf_score;
                e.semantic_score = Some(result.score);
            })
            .or_insert(FusedResult {
                chunk_id: result.chunk_id,
                fused_score: rrf_score,
                text_score: None,
                semantic_score: Some(result.score),
                ast_score: None,
                matched_lines: Vec::new(),
            });
    }

    // Process AST pattern search results
    for (rank, result) in ast_results.iter().enumerate() {
        let rrf_score = 1.0 / (k as f32 + rank as f32 + 1.0);

        scores
            .entry(result.chunk_id)
            .and_modify(|e| {
                e.fused_score += rrf_score;
                e.ast_score = Some(result.score);
            })
            .or_insert(FusedResult {
                chunk_id: result.chunk_id,
                fused_score: rrf_score,
                text_score: None,
                semantic_score: None,
                ast_score: Some(result.score),
                matched_lines: Vec::new(),
            });
    }

    // Sort by fused score descending
    let mut fused: Vec<FusedResult> = scores.into_values().collect();
    fused.sort_by(|a, b| {
        b.fused_score
            .partial_cmp(&a.fused_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    fused.truncate(top_k);
    fused
}

/// Fuse only semantic results (used when text search is not applicable).
pub fn fuse_semantic_only(
    semantic_results: &[SearchHit],
    top_k: usize,
) -> Vec<FusedResult> {
    semantic_results
        .iter()
        .take(top_k)
        .map(|r| FusedResult {
            chunk_id: r.chunk_id,
            fused_score: r.score,
            text_score: None,
            semantic_score: Some(r.score),
            ast_score: None,
            matched_lines: Vec::new(),
        })
        .collect()
}

/// Fuse only text results (used when semantic search is not applicable).
pub fn fuse_text_only(
    text_results: &[TextMatch],
    top_k: usize,
) -> Vec<FusedResult> {
    text_results
        .iter()
        .take(top_k)
        .map(|r| FusedResult {
            chunk_id: r.chunk_id,
            fused_score: r.score,
            text_score: Some(r.score),
            semantic_score: None,
            ast_score: None,
            matched_lines: r.matched_lines.clone(),
        })
        .collect()
}

/// Fuse only AST pattern search results.
pub fn fuse_ast_only(
    ast_results: &[AstMatch],
    top_k: usize,
) -> Vec<FusedResult> {
    ast_results
        .iter()
        .take(top_k)
        .map(|r| FusedResult {
            chunk_id: r.chunk_id,
            fused_score: r.score,
            text_score: None,
            semantic_score: None,
            ast_score: Some(r.score),
            matched_lines: Vec::new(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_text_matches(chunk_ids: &[u64]) -> Vec<TextMatch> {
        chunk_ids
            .iter()
            .enumerate()
            .map(|(i, &id)| TextMatch {
                chunk_id: id,
                matched_lines: vec![0],
                score: (chunk_ids.len() - i) as f32,
            })
            .collect()
    }

    fn make_semantic_hits(chunk_ids: &[u64]) -> Vec<SearchHit> {
        chunk_ids
            .iter()
            .enumerate()
            .map(|(i, &id)| SearchHit {
                chunk_id: id,
                score: 1.0 - (i as f32 * 0.1),
            })
            .collect()
    }

    #[test]
    fn test_rrf_basic_fusion() {
        // Text: [1, 2, 3]
        // Semantic: [2, 3, 4]
        let text = make_text_matches(&[1, 2, 3]);
        let semantic = make_semantic_hits(&[2, 3, 4]);

        let fused = rrf_fuse(&text, &semantic, 60, 10);

        // Chunk 2 appears in both lists, should have highest fused score
        assert!(!fused.is_empty());

        let chunk_2 = fused.iter().find(|r| r.chunk_id == 2).unwrap();
        let chunk_1 = fused.iter().find(|r| r.chunk_id == 1).unwrap();

        // Chunk 2 is in both, chunk 1 is in text only
        assert!(
            chunk_2.fused_score > chunk_1.fused_score,
            "Chunk appearing in both lists should rank higher"
        );
    }

    #[test]
    fn test_rrf_preserves_all_unique_results() {
        let text = make_text_matches(&[1, 2]);
        let semantic = make_semantic_hits(&[3, 4]);

        let fused = rrf_fuse(&text, &semantic, 60, 10);
        assert_eq!(fused.len(), 4, "All unique chunks should be in results");
    }

    #[test]
    fn test_rrf_top_k_truncation() {
        let text = make_text_matches(&[1, 2, 3, 4, 5]);
        let semantic = make_semantic_hits(&[6, 7, 8, 9, 10]);

        let fused = rrf_fuse(&text, &semantic, 60, 3);
        assert_eq!(fused.len(), 3, "Should respect top-k");
    }

    #[test]
    fn test_rrf_empty_inputs() {
        let fused = rrf_fuse(&[], &[], 60, 10);
        assert!(fused.is_empty());
    }

    #[test]
    fn test_fuse_semantic_only() {
        let semantic = make_semantic_hits(&[1, 2, 3]);
        let fused = fuse_semantic_only(&semantic, 2);
        assert_eq!(fused.len(), 2);
        assert!(fused[0].text_score.is_none());
        assert!(fused[0].semantic_score.is_some());
    }

    #[test]
    fn test_fuse_text_only() {
        let text = make_text_matches(&[1, 2, 3]);
        let fused = fuse_text_only(&text, 2);
        assert_eq!(fused.len(), 2);
        assert!(fused[0].text_score.is_some());
        assert!(fused[0].semantic_score.is_none());
        assert!(fused[0].ast_score.is_none());
    }

    fn make_ast_matches(chunk_ids: &[u64]) -> Vec<AstMatch> {
        chunk_ids
            .iter()
            .enumerate()
            .map(|(i, &id)| AstMatch {
                chunk_id: id,
                score: 1.0 - (i as f32 * 0.1),
            })
            .collect()
    }

    #[test]
    fn test_fuse_ast_only() {
        let ast = make_ast_matches(&[1, 2, 3]);
        let fused = fuse_ast_only(&ast, 2);
        assert_eq!(fused.len(), 2);
        assert!(fused[0].text_score.is_none());
        assert!(fused[0].semantic_score.is_none());
        assert!(fused[0].ast_score.is_some());
    }

    #[test]
    fn test_rrf_three_way_fusion() {
        let text = make_text_matches(&[1, 2]);
        let semantic = make_semantic_hits(&[2, 3]);
        let ast = make_ast_matches(&[3, 4]);

        let fused = rrf_fuse_three(&text, &semantic, &ast, 60, 10);

        // All 4 unique chunks should appear
        assert_eq!(fused.len(), 4);

        // Chunk 2 appears in text + semantic, chunk 3 in semantic + ast
        let chunk_2 = fused.iter().find(|r| r.chunk_id == 2).unwrap();
        let chunk_3 = fused.iter().find(|r| r.chunk_id == 3).unwrap();
        let chunk_1 = fused.iter().find(|r| r.chunk_id == 1).unwrap();
        let chunk_4 = fused.iter().find(|r| r.chunk_id == 4).unwrap();

        // Chunks appearing in 2 lists should rank higher than those in 1
        assert!(chunk_2.fused_score > chunk_1.fused_score);
        assert!(chunk_3.fused_score > chunk_4.fused_score);
    }
}
