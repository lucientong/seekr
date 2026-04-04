//! Text regex search.
//!
//! Full-text regex matching using the `regex` crate.
//! Supports case sensitivity and context line configuration.

use std::path::Path;

use regex::RegexBuilder;

use crate::error::SearchError;
use crate::index::store::SeekrIndex;
use crate::parser::CodeChunk;

/// Options for text search.
#[derive(Debug, Clone)]
pub struct TextSearchOptions {
    /// Case-sensitive matching.
    pub case_sensitive: bool,

    /// Number of context lines before/after a match.
    pub context_lines: usize,

    /// Maximum number of results.
    pub top_k: usize,
}

impl Default for TextSearchOptions {
    fn default() -> Self {
        Self {
            case_sensitive: false,
            context_lines: 2,
            top_k: 20,
        }
    }
}

/// A single text match within a code chunk.
#[derive(Debug, Clone)]
pub struct TextMatch {
    /// The chunk that matched.
    pub chunk_id: u64,

    /// Line numbers (0-indexed) that matched.
    pub matched_lines: Vec<usize>,

    /// Relevance score based on match count and density.
    pub score: f32,
}

/// Perform text regex search across the index.
///
/// Searches through all indexed code chunks using regex pattern matching.
/// Results are scored by the number and density of matches.
pub fn search_text_regex(
    index: &SeekrIndex,
    query: &str,
    options: &TextSearchOptions,
) -> Result<Vec<TextMatch>, SearchError> {
    let regex = RegexBuilder::new(query)
        .case_insensitive(!options.case_sensitive)
        .build()
        .map_err(|e| SearchError::InvalidRegex(e.to_string()))?;

    let mut matches: Vec<TextMatch> = Vec::new();

    for (chunk_id, chunk) in &index.chunks {
        let mut matched_lines = Vec::new();

        for (line_idx, line) in chunk.body.lines().enumerate() {
            if regex.is_match(line) {
                matched_lines.push(line_idx);
            }
        }

        if !matched_lines.is_empty() {
            let total_lines = chunk.body.lines().count().max(1) as f32;
            let match_count = matched_lines.len() as f32;

            // Score combines match count with density
            // More matches and higher density = higher score
            let density = match_count / total_lines;
            let score = match_count + density * 10.0;

            matches.push(TextMatch {
                chunk_id: *chunk_id,
                matched_lines,
                score,
            });
        }
    }

    // Sort by score descending
    matches.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Truncate to top-k
    matches.truncate(options.top_k);

    Ok(matches)
}

/// Perform text regex search directly on files (without index).
///
/// Scans the file system for regex matches. Useful for ad-hoc searches
/// before an index is built.
pub fn search_text_in_file(
    file_path: &Path,
    query: &str,
    case_sensitive: bool,
) -> Result<Vec<(usize, String)>, SearchError> {
    let regex = RegexBuilder::new(query)
        .case_insensitive(!case_sensitive)
        .build()
        .map_err(|e| SearchError::InvalidRegex(e.to_string()))?;

    let content = std::fs::read_to_string(file_path)
        .map_err(|e| SearchError::Index(crate::error::IndexError::Io(e)))?;

    let mut results = Vec::new();
    for (line_idx, line) in content.lines().enumerate() {
        if regex.is_match(line) {
            results.push((line_idx, line.to_string()));
        }
    }

    Ok(results)
}

/// Get context lines around matched lines.
///
/// Returns a list of (line_number, line_content, is_match) tuples.
pub fn get_match_context(
    chunk: &CodeChunk,
    matched_lines: &[usize],
    context_lines: usize,
) -> Vec<(usize, String, bool)> {
    let lines: Vec<&str> = chunk.body.lines().collect();
    let total = lines.len();
    let mut result: Vec<(usize, String, bool)> = Vec::new();
    let mut included: std::collections::HashSet<usize> = std::collections::HashSet::new();

    for &match_line in matched_lines {
        let start = match_line.saturating_sub(context_lines);
        let end = (match_line + context_lines + 1).min(total);

        for (line_idx, line) in lines.iter().enumerate().take(end).skip(start) {
            if included.insert(line_idx) {
                let is_match = matched_lines.contains(&line_idx);
                result.push((
                    line_idx + chunk.line_range.start, // absolute line number
                    line.to_string(),
                    is_match,
                ));
            }
        }
    }

    result.sort_by_key(|(line, _, _)| *line);
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::ChunkKind;
    use std::path::PathBuf;

    fn make_chunk(id: u64, body: &str) -> CodeChunk {
        CodeChunk {
            id,
            file_path: PathBuf::from("test.rs"),
            language: "rust".to_string(),
            kind: ChunkKind::Function,
            name: Some("test_fn".to_string()),
            signature: None,
            doc_comment: None,
            body: body.to_string(),
            byte_range: 0..body.len(),
            line_range: 0..body.lines().count(),
        }
    }

    #[test]
    fn test_text_search_regex() {
        let mut index = SeekrIndex::new(4);
        let chunk = make_chunk(1, "fn authenticate(user: &str) {\n    validate(user);\n}\n");
        let entry = crate::index::IndexEntry {
            chunk_id: 1,
            embedding: vec![0.1; 4],
            text_tokens: vec!["authenticate".to_string()],
        };
        index.add_entry(entry, chunk);

        let options = TextSearchOptions {
            case_sensitive: false,
            context_lines: 0,
            top_k: 10,
        };

        let results = search_text_regex(&index, "authenticate", &options).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].chunk_id, 1);
        assert!(!results[0].matched_lines.is_empty());
    }

    #[test]
    fn test_text_search_case_insensitive() {
        let mut index = SeekrIndex::new(4);
        let chunk = make_chunk(1, "fn Authenticate(user: &str) {}");
        let entry = crate::index::IndexEntry {
            chunk_id: 1,
            embedding: vec![0.1; 4],
            text_tokens: vec!["authenticate".to_string()],
        };
        index.add_entry(entry, chunk);

        let options = TextSearchOptions {
            case_sensitive: false,
            ..Default::default()
        };

        let results = search_text_regex(&index, "authenticate", &options).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_context_lines() {
        let chunk = make_chunk(1, "line 0\nline 1\nMATCH line 2\nline 3\nline 4\n");
        let context = get_match_context(&chunk, &[2], 1);

        assert!(context.len() >= 3); // at least match + 1 before + 1 after
        let line_nums: Vec<usize> = context.iter().map(|(l, _, _)| *l).collect();
        assert!(line_nums.contains(&1));
        assert!(line_nums.contains(&2));
        assert!(line_nums.contains(&3));
    }

    #[test]
    fn test_invalid_regex() {
        let index = SeekrIndex::new(4);
        let options = TextSearchOptions::default();

        let result = search_text_regex(&index, "[invalid", &options);
        assert!(result.is_err());
    }
}
