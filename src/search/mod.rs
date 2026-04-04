//! Search engine module.
//!
//! Provides text regex search, semantic vector search, AST pattern search,
//! and RRF fusion ranking across multiple search backends.

pub mod ast_pattern;
pub mod fusion;
pub mod semantic;
pub mod text;

use crate::parser::CodeChunk;

/// Search mode selection.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SearchMode {
    /// Regex text search.
    Text,
    /// Semantic vector search.
    Semantic,
    /// AST pattern search (function signatures).
    Ast,
    /// Hybrid: combine text + semantic results via RRF fusion.
    Hybrid,
}

/// A search query.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SearchQuery {
    /// The query string.
    pub query: String,

    /// Search mode.
    pub mode: SearchMode,

    /// Maximum number of results to return.
    pub top_k: usize,

    /// Project path to search in.
    pub project_path: String,
}

/// A single search result.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SearchResult {
    /// The matched code chunk.
    pub chunk: CodeChunk,

    /// Relevance score (higher is better).
    pub score: f32,

    /// Which search mode produced this result.
    pub source: SearchMode,

    /// Matched line numbers (for text search).
    pub matched_lines: Vec<usize>,
}

/// Aggregated search response.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SearchResponse {
    /// Search results, sorted by relevance.
    pub results: Vec<SearchResult>,

    /// Total number of results before top-k truncation.
    pub total: usize,

    /// Search duration in milliseconds.
    pub duration_ms: u64,

    /// The query that was executed.
    pub query: SearchQuery,
}

impl std::fmt::Display for SearchMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SearchMode::Text => write!(f, "text"),
            SearchMode::Semantic => write!(f, "semantic"),
            SearchMode::Ast => write!(f, "ast"),
            SearchMode::Hybrid => write!(f, "hybrid"),
        }
    }
}

impl std::str::FromStr for SearchMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "text" => Ok(SearchMode::Text),
            "semantic" => Ok(SearchMode::Semantic),
            "ast" => Ok(SearchMode::Ast),
            "hybrid" => Ok(SearchMode::Hybrid),
            _ => Err(format!(
                "Unknown search mode: '{}'. Use: text, semantic, ast, hybrid",
                s
            )),
        }
    }
}
