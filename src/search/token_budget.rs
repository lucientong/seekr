//! Conservative token-budgeting utilities for agent-oriented search results.

use crate::search::SearchResult;

/// Estimate the number of LLM tokens needed to serialize one result.
///
/// This intentionally uses one token per three JSON bytes. It is an estimate,
/// not a tokenizer-specific exact count, and errs on the conservative side for
/// typical source code while also accounting for non-ASCII content.
pub fn estimate_search_result_tokens(result: &SearchResult) -> usize {
    let serialized_bytes = serde_json::to_vec(result)
        .map(|serialized| serialized.len())
        .unwrap_or_else(|_| result.chunk.body.len());
    serialized_bytes.div_ceil(3).max(1)
}

/// Estimate the total token count of a result set.
pub fn estimate_search_results_tokens(results: &[SearchResult]) -> usize {
    results.iter().map(estimate_search_result_tokens).sum()
}

/// Greedily retain ranked results that fit within an estimated token budget.
///
/// Oversized results are skipped so that a single large chunk does not prevent
/// smaller, lower-ranked results from providing useful context.
pub fn apply_token_budget(
    results: Vec<SearchResult>,
    max_tokens: Option<usize>,
) -> Vec<SearchResult> {
    let Some(max_tokens) = max_tokens else {
        return results;
    };
    let mut remaining = max_tokens;
    results
        .into_iter()
        .filter(|result| {
            let estimated = estimate_search_result_tokens(result);
            if estimated > remaining {
                return false;
            }
            remaining -= estimated;
            true
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::parser::{ChunkKind, CodeChunk};
    use crate::search::SearchMode;

    fn result(id: u64, body: &str) -> SearchResult {
        SearchResult {
            chunk: CodeChunk {
                id,
                file_path: PathBuf::from(format!("src/{id}.rs")),
                language: "rust".to_string(),
                kind: ChunkKind::Function,
                name: Some(format!("function_{id}")),
                signature: None,
                doc_comment: None,
                body: body.to_string(),
                byte_range: 0..body.len(),
                line_range: 0..1,
            },
            score: 1.0,
            source: SearchMode::Text,
            matched_lines: Vec::new(),
        }
    }

    #[test]
    fn estimate_grows_with_result_content() {
        assert!(
            estimate_search_result_tokens(&result(1, &"x".repeat(300)))
                > estimate_search_result_tokens(&result(2, "x"))
        );
    }

    #[test]
    fn budget_skips_oversized_results_and_preserves_rank_order() {
        let small = result(2, "small");
        let budget = estimate_search_result_tokens(&small);
        let selected = apply_token_budget(vec![result(1, &"x".repeat(3_000)), small], Some(budget));

        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].chunk.id, 2);
    }

    #[test]
    fn missing_budget_keeps_all_results() {
        let selected = apply_token_budget(vec![result(1, "one"), result(2, "two")], None);
        assert_eq!(selected.len(), 2);
    }
}
