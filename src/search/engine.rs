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
use crate::search::text::{TextSearchOptions, search_text_regex};
use crate::search::{SearchMode, SearchResult};

#[derive(Debug, Clone)]
pub struct SearchOptions {
    pub top_k: usize,
    pub path_prefix: Option<PathBuf>,
    pub languages: Vec<String>,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            top_k: 20,
            path_prefix: None,
            languages: Vec::new(),
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
            index.chunk_count
        } else {
            options.top_k
        };
        let fused = self.search_fused(index, query, &mode, candidate_limit)?;
        let languages: HashSet<String> = options
            .languages
            .iter()
            .map(|language| language.to_lowercase())
            .collect();

        Ok(fused
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
                Some(SearchResult {
                    chunk: chunk.clone(),
                    score: result.fused_score,
                    source: mode.clone(),
                    matched_lines: result.matched_lines,
                })
            })
            .take(options.top_k)
            .collect())
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
                let text_results = search_text_regex(index, query, &self.text_options(top_k))?;
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
                    path_prefix: Some(PathBuf::from("src")),
                    languages: vec!["rust".to_string()],
                },
            )
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].chunk.id, 1);
    }
}
