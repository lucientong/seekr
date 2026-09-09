//! Unified search application service.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;

use crate::config::SearchConfig;
use crate::embedder::traits::Embedder;
use crate::error::SearchError;
use crate::index::store::SeekrIndex;
use crate::search::ast_pattern::{looks_like_ast_pattern, search_ast_pattern};
use crate::search::fusion::{
    FusedResult, fuse_ast_only, fuse_semantic_only, fuse_text_only, rrf_fuse, rrf_fuse_three,
};
use crate::search::semantic::{SemanticSearchOptions, search_semantic};
use crate::search::text::{TextMatch, TextSearchOptions, search_text_regex};
use crate::search::token_budget::apply_token_budget;
use crate::search::{SearchMode, SearchResult};

#[derive(Debug, Clone)]
pub struct SearchOptions {
    pub top_k: usize,
    pub max_results: Option<usize>,
    pub max_tokens: Option<usize>,
    pub path_prefix: Option<PathBuf>,
    pub languages: Vec<String>,
    pub excluded_chunk_ids: HashSet<u64>,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            top_k: 20,
            max_results: None,
            max_tokens: None,
            path_prefix: None,
            languages: Vec::new(),
            excluded_chunk_ids: HashSet::new(),
        }
    }
}

/// Executes all search modes with one shared configuration and embedder.
pub struct SearchEngine {
    config: SearchConfig,
    embedder: Option<Arc<dyn Embedder>>,
}

impl SearchEngine {
    pub fn new(config: SearchConfig, embedder: Option<Arc<dyn Embedder>>) -> Self {
        Self { config, embedder }
    }

    pub fn search(
        &self,
        index: &SeekrIndex,
        query: &str,
        mode: SearchMode,
        options: &SearchOptions,
    ) -> Result<Vec<SearchResult>, SearchError> {
        let has_filters = options.path_prefix.is_some() || !options.languages.is_empty();
        let candidate_limit = if has_filters {
            index.chunk_count()
        } else if !options.excluded_chunk_ids.is_empty() {
            options
                .top_k
                .saturating_add(options.excluded_chunk_ids.len())
                .min(index.chunk_count())
        } else {
            options.top_k
        };
        let fused = self.search_fused(index, query, &mode, candidate_limit)?;
        let languages: HashSet<String> = options
            .languages
            .iter()
            .map(|language| language.to_lowercase())
            .collect();

        let output_limit = options
            .max_results
            .unwrap_or(options.top_k)
            .min(options.top_k);
        let results = fused
            .into_iter()
            .filter_map(|result| {
                let chunk = index.get_chunk(result.chunk_id)?;
                if options
                    .path_prefix
                    .as_ref()
                    .is_some_and(|prefix| !chunk.file_path.starts_with(prefix))
                    || (!languages.is_empty()
                        && !languages.contains(&chunk.language.to_lowercase()))
                {
                    return None;
                }
                if options.excluded_chunk_ids.contains(&chunk.id) {
                    return None;
                }
                Some(SearchResult {
                    chunk: chunk.clone(),
                    score: result.fused_score,
                    source: mode.clone(),
                    matched_lines: result.matched_lines,
                })
            })
            .take(output_limit)
            .collect();
        Ok(apply_token_budget(results, options.max_tokens))
    }

    fn search_fused(
        &self,
        index: &SeekrIndex,
        query: &str,
        mode: &SearchMode,
        top_k: usize,
    ) -> Result<Vec<FusedResult>, SearchError> {
        match mode {
            SearchMode::Text => {
                let results = search_text_regex(index, query, &self.text_options(top_k))?;
                Ok(fuse_text_only(&results, top_k))
            }
            SearchMode::Semantic => {
                let results = search_semantic(
                    index,
                    query,
                    self.require_embedder()?,
                    &self.semantic_options(top_k),
                )?;
                Ok(fuse_semantic_only(&results, top_k))
            }
            SearchMode::Ast => {
                let results = search_ast_pattern(index, query, top_k)?;
                Ok(fuse_ast_only(&results, top_k))
            }
            SearchMode::Hybrid => {
                let text_results: Vec<TextMatch> = index
                    .search_bm25(query, top_k)
                    .into_iter()
                    .map(|hit| TextMatch {
                        chunk_id: hit.chunk_id,
                        matched_lines: Vec::new(),
                        score: hit.score,
                    })
                    .collect();
                let semantic_results = search_semantic(
                    index,
                    query,
                    self.require_embedder()?,
                    &self.semantic_options(top_k),
                )?;
                let ast_results = if looks_like_ast_pattern(query) {
                    search_ast_pattern(index, query, top_k).unwrap_or_default()
                } else {
                    Vec::new()
                };

                if ast_results.is_empty() {
                    Ok(rrf_fuse(
                        &text_results,
                        &semantic_results,
                        self.config.rrf_k,
                        top_k,
                    ))
                } else {
                    Ok(rrf_fuse_three(
                        &text_results,
                        &semantic_results,
                        &ast_results,
                        self.config.rrf_k,
                        top_k,
                    ))
                }
            }
        }
    }

    fn require_embedder(&self) -> Result<&dyn Embedder, SearchError> {
        self.embedder
            .as_deref()
            .ok_or(SearchError::EmbedderUnavailable)
    }

    fn text_options(&self, top_k: usize) -> TextSearchOptions {
        TextSearchOptions {
            case_sensitive: false,
            context_lines: self.config.context_lines,
            top_k,
        }
    }

    fn semantic_options(&self, top_k: usize) -> SemanticSearchOptions {
        SemanticSearchOptions {
            top_k,
            score_threshold: self.config.score_threshold,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedder::batch::DummyEmbedder;
    use crate::index::store::SeekrIndex;
    use crate::parser::{ChunkKind, CodeChunk};

    fn chunk(id: u64, path: &str, language: &str, body: &str) -> CodeChunk {
        CodeChunk {
            id,
            file_path: PathBuf::from(path),
            language: language.to_string(),
            kind: ChunkKind::Function,
            name: Some(format!("function_{id}")),
            signature: None,
            doc_comment: None,
            body: body.to_string(),
            byte_range: 0..body.len(),
            line_range: 0..1,
        }
    }

    #[test]
    fn filters_by_path_and_language_after_ranking() {
        let chunks = vec![
            chunk(1, "src/a.rs", "rust", "fn shared() {}"),
            chunk(2, "tests/a.py", "python", "def shared(): pass"),
        ];
        let index = SeekrIndex::build_from(&chunks, &[vec![1.0; 8], vec![1.0; 8]], 8);
        let engine = SearchEngine::new(
            SearchConfig::default(),
            Some(Arc::new(DummyEmbedder::new(8))),
        );
        let results = engine
            .search(
                &index,
                "shared",
                SearchMode::Text,
                &SearchOptions {
                    top_k: 10,
                    max_results: None,
                    max_tokens: None,
                    path_prefix: Some(PathBuf::from("src")),
                    languages: vec!["rust".to_string()],
                    excluded_chunk_ids: HashSet::new(),
                },
            )
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].chunk.id, 1);
    }

    #[test]
    fn hybrid_uses_bm25_for_natural_language_queries() {
        let chunks = vec![
            chunk(1, "src/auth.rs", "rust", "fn verify_password_hash() {}"),
            chunk(2, "src/cache.rs", "rust", "fn clear_cache() {}"),
        ];
        let index = SeekrIndex::build_from(&chunks, &[vec![1.0; 8], vec![0.0; 8]], 8);
        let engine = SearchEngine::new(
            SearchConfig::default(),
            Some(Arc::new(DummyEmbedder::new(8))),
        );

        let results = engine
            .search(
                &index,
                "verify password [hash",
                SearchMode::Hybrid,
                &SearchOptions {
                    top_k: 10,
                    ..SearchOptions::default()
                },
            )
            .expect("BM25 hybrid queries must not be parsed as regular expressions");

        assert!(results.iter().any(|result| result.chunk.id == 1));
    }

    #[test]
    fn applies_result_and_token_limits_after_ranking() {
        use crate::search::token_budget::estimate_search_result_tokens;

        let chunks = vec![
            chunk(1, "src/a.rs", "rust", "fn shared() {}"),
            chunk(2, "src/b.rs", "rust", "fn shared() {}"),
        ];
        let index = SeekrIndex::build_from(&chunks, &[vec![1.0; 8], vec![1.0; 8]], 8);
        let engine = SearchEngine::new(SearchConfig::default(), None);
        let baseline = engine
            .search(
                &index,
                "shared",
                SearchMode::Text,
                &SearchOptions {
                    top_k: 2,
                    ..SearchOptions::default()
                },
            )
            .unwrap();
        let first_result_budget = estimate_search_result_tokens(&baseline[0]);

        let results = engine
            .search(
                &index,
                "shared",
                SearchMode::Text,
                &SearchOptions {
                    top_k: 2,
                    max_results: Some(2),
                    max_tokens: Some(first_result_budget),
                    ..SearchOptions::default()
                },
            )
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].chunk.id, baseline[0].chunk.id);
    }

    #[test]
    fn overfetches_to_replace_session_duplicates() {
        let chunks = vec![
            chunk(1, "src/a.rs", "rust", "fn shared() {}"),
            chunk(2, "src/b.rs", "rust", "fn shared() {}"),
            chunk(3, "src/c.rs", "rust", "fn shared() {}"),
        ];
        let index = SeekrIndex::build_from(&chunks, &vec![vec![1.0; 8]; 3], 8);
        let engine = SearchEngine::new(SearchConfig::default(), None);

        let results = engine
            .search(
                &index,
                "shared",
                SearchMode::Text,
                &SearchOptions {
                    top_k: 2,
                    excluded_chunk_ids: HashSet::from([1]),
                    ..SearchOptions::default()
                },
            )
            .unwrap();

        assert_eq!(
            results
                .iter()
                .map(|result| result.chunk.id)
                .collect::<Vec<_>>(),
            vec![2, 3]
        );
    }
}
