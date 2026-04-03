//! AST pattern search.
//!
//! Parses user-provided function signature patterns (e.g., "fn(string) -> number")
//! and matches them against indexed CodeChunk signatures.
//! Supports wildcards (`*`), optional keywords (`async`, `pub`), and fuzzy type matching.
//!
//! ## Pattern Syntax
//!
//! ```text
//! [async] [pub] fn_keyword([param_type, ...]) [-> return_type]
//! ```
//!
//! Examples:
//! - `fn(string) -> number`        — any function taking a string, returning a number
//! - `fn(*) -> Result`             — any function returning a Result
//! - `async fn(*)`                 — any async function
//! - `fn(*, *) -> bool`            — any function with 2 params returning bool
//! - `fn authenticate(*)`          — function named "authenticate" with any params
//! - `class User`                  — class named User
//! - `struct *Config`              — struct whose name ends with "Config"

use crate::error::SearchError;
use crate::index::store::SeekrIndex;
use crate::parser::{ChunkKind, CodeChunk};

/// A parsed AST search pattern.
#[derive(Debug, Clone)]
pub struct AstPattern {
    /// Optional qualifiers: "async", "pub", "static", etc.
    pub qualifiers: Vec<String>,

    /// The kind of construct to match.
    pub kind: PatternKind,

    /// Optional name pattern (supports `*` wildcard).
    pub name_pattern: Option<String>,

    /// Optional parameter type patterns (supports `*` wildcard).
    pub param_patterns: Option<Vec<String>>,

    /// Optional return type pattern (supports `*` wildcard).
    pub return_pattern: Option<String>,
}

/// The kind of pattern target.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PatternKind {
    /// Match functions or methods.
    Function,
    /// Match classes.
    Class,
    /// Match structs.
    Struct,
    /// Match enums.
    Enum,
    /// Match interfaces / traits.
    Interface,
    /// Match any kind.
    Any,
}

/// Result of an AST pattern match.
#[derive(Debug, Clone)]
pub struct AstMatch {
    /// The chunk ID that matched.
    pub chunk_id: u64,

    /// Match score (0.0 to 1.0).
    pub score: f32,
}

/// Parse a user-provided AST pattern string.
///
/// # Examples
/// ```
/// use seekr_code::search::ast_pattern::parse_pattern;
///
/// let pat = parse_pattern("fn(string) -> number").unwrap();
/// assert_eq!(pat.kind, seekr_code::search::ast_pattern::PatternKind::Function);
/// ```
pub fn parse_pattern(pattern: &str) -> Result<AstPattern, SearchError> {
    let pattern = pattern.trim();

    if pattern.is_empty() {
        return Err(SearchError::InvalidAstPattern(
            "Empty pattern".to_string(),
        ));
    }

    let tokens = tokenize_pattern(pattern);

    if tokens.is_empty() {
        return Err(SearchError::InvalidAstPattern(
            "Could not parse pattern".to_string(),
        ));
    }

    let mut idx = 0;
    let mut qualifiers = Vec::new();

    // Collect qualifiers (async, pub, static, etc.)
    while idx < tokens.len() {
        match tokens[idx].as_str() {
            "async" | "pub" | "static" | "export" | "private" | "protected" | "public"
            | "abstract" | "virtual" | "override" | "const" | "mut" | "unsafe" => {
                qualifiers.push(tokens[idx].clone());
                idx += 1;
            }
            _ => break,
        }
    }

    if idx >= tokens.len() {
        return Err(SearchError::InvalidAstPattern(
            "Pattern has only qualifiers, missing kind (fn, class, struct, etc.)".to_string(),
        ));
    }

    // Parse kind keyword
    let (kind, idx) = parse_kind(&tokens, idx)?;

    // Parse optional name
    let (name_pattern, idx) = parse_name(&tokens, idx);

    // Parse optional parameters (only for function-like kinds)
    let (param_patterns, idx) = if matches!(kind, PatternKind::Function | PatternKind::Any) {
        parse_params(&tokens, idx)?
    } else {
        (None, idx)
    };

    // Parse optional return type
    let return_pattern = parse_return_type(&tokens, idx);

    Ok(AstPattern {
        qualifiers,
        kind,
        name_pattern,
        param_patterns,
        return_pattern,
    })
}

/// Search chunks in the index using an AST pattern.
pub fn search_ast_pattern(
    index: &SeekrIndex,
    pattern: &str,
    top_k: usize,
) -> Result<Vec<AstMatch>, SearchError> {
    let parsed = parse_pattern(pattern)?;

    let mut matches: Vec<AstMatch> = Vec::new();

    for (_chunk_id, chunk) in &index.chunks {
        let score = match_chunk(&parsed, chunk);
        if score > 0.0 {
            matches.push(AstMatch {
                chunk_id: chunk.id,
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

    matches.truncate(top_k);

    Ok(matches)
}

/// Match a single chunk against a parsed AST pattern.
/// Returns a score from 0.0 (no match) to 1.0 (perfect match).
fn match_chunk(pattern: &AstPattern, chunk: &CodeChunk) -> f32 {
    let mut score = 0.0f32;
    let mut total_criteria = 0.0f32;

    // 1. Match kind (required, weight = 0.3)
    total_criteria += 0.3;
    if match_kind(&pattern.kind, &chunk.kind) {
        score += 0.3;
    } else {
        // Kind mismatch is a hard filter — return 0
        return 0.0;
    }

    // 2. Match qualifiers (weight = 0.1)
    if !pattern.qualifiers.is_empty() {
        total_criteria += 0.1;
        let sig_lower = chunk
            .signature
            .as_deref()
            .unwrap_or("")
            .to_lowercase();
        let body_start = chunk.body.lines().next().unwrap_or("").to_lowercase();
        let combined = format!("{} {}", sig_lower, body_start);

        let matched_quals = pattern
            .qualifiers
            .iter()
            .filter(|q| combined.contains(q.as_str()))
            .count();

        if pattern.qualifiers.len() > 0 {
            score += 0.1 * (matched_quals as f32 / pattern.qualifiers.len() as f32);
        }
    }

    // 3. Match name (weight = 0.3)
    if let Some(ref name_pat) = pattern.name_pattern {
        total_criteria += 0.3;
        if let Some(ref chunk_name) = chunk.name {
            if wildcard_match(name_pat, chunk_name) {
                score += 0.3;
            } else if chunk_name.to_lowercase().contains(&name_pat.to_lowercase().replace('*', ""))
            {
                // Partial name match gets partial credit
                score += 0.15;
            }
        }
    }

    // 4. Match parameter patterns (weight = 0.15)
    if let Some(ref param_pats) = pattern.param_patterns {
        total_criteria += 0.15;
        let sig = chunk.signature.as_deref().unwrap_or(&chunk.body);
        let chunk_params = extract_params_from_signature(sig);

        if param_pats.len() == 1 && param_pats[0] == "*" {
            // Wildcard: any params match
            score += 0.15;
        } else if param_pats.is_empty() && chunk_params.is_empty() {
            // Both empty: match
            score += 0.15;
        } else {
            let param_score = match_param_types(param_pats, &chunk_params);
            score += 0.15 * param_score;
        }
    }

    // 5. Match return type (weight = 0.15)
    if let Some(ref ret_pat) = pattern.return_pattern {
        total_criteria += 0.15;
        let sig = chunk.signature.as_deref().unwrap_or(&chunk.body);
        let chunk_ret = extract_return_type_from_signature(sig);

        if ret_pat == "*" {
            score += 0.15;
        } else if let Some(ref chunk_ret) = chunk_ret {
            if fuzzy_type_match(ret_pat, chunk_ret) {
                score += 0.15;
            } else if chunk_ret.to_lowercase().contains(&ret_pat.to_lowercase()) {
                score += 0.075; // partial credit
            }
        }
    }

    // Normalize: if no optional criteria were specified, boost the score
    if total_criteria > 0.0 {
        score / total_criteria
    } else {
        0.0
    }
}

// ============================================================
// Pattern Parsing Helpers
// ============================================================

/// Tokenize a pattern string into meaningful tokens.
fn tokenize_pattern(pattern: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut chars = pattern.chars().peekable();

    while let Some(ch) = chars.next() {
        match ch {
            '(' | ')' | ',' => {
                if !current.is_empty() {
                    tokens.push(std::mem::take(&mut current));
                }
                tokens.push(ch.to_string());
            }
            '-' if chars.peek() == Some(&'>') => {
                if !current.is_empty() {
                    tokens.push(std::mem::take(&mut current));
                }
                chars.next(); // consume '>'
                tokens.push("->".to_string());
            }
            ' ' | '\t' => {
                if !current.is_empty() {
                    tokens.push(std::mem::take(&mut current));
                }
            }
            _ => {
                current.push(ch);
            }
        }
    }

    if !current.is_empty() {
        tokens.push(current);
    }

    tokens
}

/// Parse the kind keyword from tokens.
fn parse_kind(tokens: &[String], idx: usize) -> Result<(PatternKind, usize), SearchError> {
    if idx >= tokens.len() {
        return Ok((PatternKind::Any, idx));
    }

    let kind_str = tokens[idx].to_lowercase();
    let kind = match kind_str.as_str() {
        "fn" | "func" | "function" | "def" | "method" => PatternKind::Function,
        "class" => PatternKind::Class,
        "struct" => PatternKind::Struct,
        "enum" => PatternKind::Enum,
        "interface" | "trait" | "protocol" => PatternKind::Interface,
        "*" => PatternKind::Any,
        _ => {
            // If it's not a recognized keyword, treat the first token as a kind "function"
            // (since most patterns are for functions) and this token as the name
            return Ok((PatternKind::Function, idx));
        }
    };

    Ok((kind, idx + 1))
}

/// Parse an optional name pattern.
fn parse_name(tokens: &[String], idx: usize) -> (Option<String>, usize) {
    if idx >= tokens.len() {
        return (None, idx);
    }

    // Next token is a name if it's not '(' or '->'
    if tokens[idx] != "(" && tokens[idx] != "->" && tokens[idx] != ")" && tokens[idx] != "," {
        (Some(tokens[idx].clone()), idx + 1)
    } else {
        (None, idx)
    }
}

/// Parse optional parameter type patterns from parentheses.
fn parse_params(
    tokens: &[String],
    idx: usize,
) -> Result<(Option<Vec<String>>, usize), SearchError> {
    if idx >= tokens.len() || tokens[idx] != "(" {
        return Ok((None, idx));
    }

    let mut params = Vec::new();
    let mut i = idx + 1; // skip '('

    while i < tokens.len() && tokens[i] != ")" {
        if tokens[i] == "," {
            i += 1;
            continue;
        }
        params.push(tokens[i].clone());
        i += 1;
    }

    if i < tokens.len() && tokens[i] == ")" {
        i += 1; // skip ')'
    }

    Ok((Some(params), i))
}

/// Parse optional return type from `->`.
fn parse_return_type(tokens: &[String], idx: usize) -> Option<String> {
    if idx + 1 < tokens.len() && tokens[idx] == "->" {
        // Collect all remaining tokens as the return type
        let ret_parts: Vec<&str> = tokens[idx + 1..].iter().map(|s| s.as_str()).collect();
        if ret_parts.is_empty() {
            None
        } else {
            Some(ret_parts.join(" "))
        }
    } else {
        None
    }
}

// ============================================================
// Matching Helpers
// ============================================================

/// Check if the pattern kind matches the chunk kind.
fn match_kind(pattern_kind: &PatternKind, chunk_kind: &ChunkKind) -> bool {
    match pattern_kind {
        PatternKind::Any => true,
        PatternKind::Function => matches!(chunk_kind, ChunkKind::Function | ChunkKind::Method),
        PatternKind::Class => matches!(chunk_kind, ChunkKind::Class),
        PatternKind::Struct => matches!(chunk_kind, ChunkKind::Struct),
        PatternKind::Enum => matches!(chunk_kind, ChunkKind::Enum),
        PatternKind::Interface => matches!(chunk_kind, ChunkKind::Interface),
    }
}

/// Wildcard string matching.
///
/// Supports `*` as a glob wildcard that matches any sequence of characters.
/// Case-insensitive.
fn wildcard_match(pattern: &str, text: &str) -> bool {
    let pattern = pattern.to_lowercase();
    let text = text.to_lowercase();

    if !pattern.contains('*') {
        return pattern == text;
    }

    let parts: Vec<&str> = pattern.split('*').collect();

    if parts.len() == 1 {
        return pattern == text;
    }

    let mut text_pos = 0;

    for (i, part) in parts.iter().enumerate() {
        if part.is_empty() {
            continue;
        }

        if i == 0 {
            // First part must be a prefix
            if !text[text_pos..].starts_with(part) {
                return false;
            }
            text_pos += part.len();
        } else if i == parts.len() - 1 {
            // Last part must be a suffix
            if !text[text_pos..].ends_with(part) {
                return false;
            }
        } else {
            // Middle parts must appear in order
            match text[text_pos..].find(part) {
                Some(pos) => text_pos += pos + part.len(),
                None => return false,
            }
        }
    }

    true
}

/// Extract parameter types from a function signature string.
///
/// Handles common signature formats:
/// - `fn foo(x: i32, y: String) -> bool`
/// - `def foo(x: int, y: str) -> bool`
/// - `function foo(x, y)`
fn extract_params_from_signature(sig: &str) -> Vec<String> {
    // Find content between first '(' and matching ')'
    let Some(open) = sig.find('(') else {
        return Vec::new();
    };

    let mut depth = 0;
    let mut close = None;

    for (i, ch) in sig[open..].char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    close = Some(open + i);
                    break;
                }
            }
            _ => {}
        }
    }

    let Some(close) = close else {
        return Vec::new();
    };

    let params_str = &sig[open + 1..close];
    if params_str.trim().is_empty() {
        return Vec::new();
    }

    // Split by commas (being careful of nested generics)
    split_params(params_str)
        .iter()
        .filter_map(|p| extract_type_from_param(p.trim()))
        .collect()
}

/// Split parameter list by commas, respecting nesting (generics, etc.).
fn split_params(params: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut depth = 0;

    for ch in params.chars() {
        match ch {
            '<' | '(' | '[' | '{' => {
                depth += 1;
                current.push(ch);
            }
            '>' | ')' | ']' | '}' => {
                depth -= 1;
                current.push(ch);
            }
            ',' if depth == 0 => {
                parts.push(std::mem::take(&mut current));
            }
            _ => current.push(ch),
        }
    }

    if !current.is_empty() {
        parts.push(current);
    }

    parts
}

/// Extract the type from a parameter declaration.
///
/// Handles:
/// - `x: i32` -> "i32"
/// - `x: &str` -> "str"
/// - `name: String` -> "String"
/// - `x int` -> "int" (Go style)
/// - `x` -> "x" (untyped, treated as name/type ambiguous)
fn extract_type_from_param(param: &str) -> Option<String> {
    let param = param.trim();
    if param.is_empty() {
        return None;
    }

    // Rust/Python style: `name: Type`
    if let Some(colon_pos) = param.find(':') {
        let type_part = param[colon_pos + 1..].trim();
        // Strip references (&, &mut)
        let type_part = type_part
            .trim_start_matches('&')
            .trim_start_matches("mut ")
            .trim();
        return Some(type_part.to_string());
    }

    // Go style: `name Type` (space separated, second token is type)
    let parts: Vec<&str> = param.split_whitespace().collect();
    if parts.len() >= 2 {
        return Some(parts.last().unwrap().to_string());
    }

    // Single token — could be type or name
    Some(param.to_string())
}

/// Extract return type from a function signature.
fn extract_return_type_from_signature(sig: &str) -> Option<String> {
    // Look for `->` (Rust/Python)
    if let Some(arrow_pos) = sig.find("->") {
        let ret = sig[arrow_pos + 2..].trim();
        // Strip trailing '{' or ':'
        let ret = ret.trim_end_matches(|c: char| c == '{' || c == ':' || c.is_whitespace());
        if !ret.is_empty() {
            return Some(ret.to_string());
        }
    }

    // Look for `: ReturnType` after closing paren (TypeScript/Go style)
    // Find the last ')' and check if there's `: Type` after it
    if let Some(close_paren) = sig.rfind(')') {
        let after = sig[close_paren + 1..].trim();
        if let Some(stripped) = after.strip_prefix(':') {
            let ret = stripped.trim().trim_end_matches(|c: char| c == '{' || c.is_whitespace());
            if !ret.is_empty() {
                return Some(ret.to_string());
            }
        }
    }

    None
}

/// Match parameter types between pattern and chunk.
/// Returns a score from 0.0 to 1.0.
fn match_param_types(pattern_params: &[String], chunk_params: &[String]) -> f32 {
    if pattern_params.is_empty() && chunk_params.is_empty() {
        return 1.0;
    }

    if pattern_params.is_empty() || chunk_params.is_empty() {
        return 0.0;
    }

    // Check parameter count match (with wildcards)
    let pattern_count = pattern_params.len();
    let chunk_count = chunk_params.len();

    // If pattern has a single `*`, match any number of params
    if pattern_count == 1 && pattern_params[0] == "*" {
        return 1.0;
    }

    // Count non-wildcard params in pattern
    let fixed_params: Vec<&String> = pattern_params.iter().filter(|p| p.as_str() != "*").collect();

    if fixed_params.len() > chunk_count {
        return 0.0; // More fixed params than chunk has
    }

    let mut matched = 0;
    let mut chunk_idx = 0;

    for pat_param in pattern_params {
        if pat_param == "*" {
            matched += 1;
            if chunk_idx < chunk_count {
                chunk_idx += 1;
            }
            continue;
        }

        // Try to match this pattern param against remaining chunk params
        while chunk_idx < chunk_count {
            if fuzzy_type_match(pat_param, &chunk_params[chunk_idx]) {
                matched += 1;
                chunk_idx += 1;
                break;
            }
            chunk_idx += 1;
        }
    }

    matched as f32 / pattern_params.len() as f32
}

/// Fuzzy type matching between pattern type and actual type.
///
/// Handles:
/// - Case-insensitive comparison
/// - Common type aliases: "string" matches "String", "&str", "str"
/// - "number" matches "i32", "f64", "int", "float", etc.
/// - "bool" matches "boolean", "bool"
/// - Wildcard `*` matches anything
/// - Prefix/suffix with `*`
fn fuzzy_type_match(pattern: &str, actual: &str) -> bool {
    let pattern = pattern.trim().to_lowercase();
    let actual = actual.trim().to_lowercase();

    if pattern == "*" {
        return true;
    }

    // Exact match
    if pattern == actual {
        return true;
    }

    // Wildcard matching
    if pattern.contains('*') {
        return wildcard_match(&pattern, &actual);
    }

    // Type group matching
    match pattern.as_str() {
        "string" | "str" => {
            matches!(
                actual.as_str(),
                "string" | "str" | "&str" | "std::string::string" | "text"
            )
        }
        "number" | "num" | "int" | "integer" => {
            matches!(
                actual.as_str(),
                "i8" | "i16"
                    | "i32"
                    | "i64"
                    | "i128"
                    | "isize"
                    | "u8"
                    | "u16"
                    | "u32"
                    | "u64"
                    | "u128"
                    | "usize"
                    | "f32"
                    | "f64"
                    | "int"
                    | "int8"
                    | "int16"
                    | "int32"
                    | "int64"
                    | "uint"
                    | "float"
                    | "float32"
                    | "float64"
                    | "number"
                    | "double"
            )
        }
        "bool" | "boolean" => {
            matches!(actual.as_str(), "bool" | "boolean")
        }
        "void" | "none" | "unit" | "()" => {
            matches!(actual.as_str(), "void" | "none" | "()" | "unit" | "null")
        }
        _ => {
            // Check if actual contains the pattern (partial match)
            actual.contains(&pattern) || pattern.contains(&actual)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::ChunkKind;
    use std::path::PathBuf;

    fn make_chunk(id: u64, kind: ChunkKind, name: &str, sig: &str, body: &str) -> CodeChunk {
        CodeChunk {
            id,
            file_path: PathBuf::from("test.rs"),
            language: "rust".to_string(),
            kind,
            name: Some(name.to_string()),
            signature: Some(sig.to_string()),
            doc_comment: None,
            body: body.to_string(),
            byte_range: 0..body.len(),
            line_range: 0..body.lines().count(),
        }
    }

    #[test]
    fn test_parse_simple_pattern() {
        let pat = parse_pattern("fn(string) -> number").unwrap();
        assert_eq!(pat.kind, PatternKind::Function);
        assert!(pat.name_pattern.is_none());
        assert_eq!(pat.param_patterns.as_ref().unwrap(), &["string"]);
        assert_eq!(pat.return_pattern.as_ref().unwrap(), "number");
    }

    #[test]
    fn test_parse_named_function_pattern() {
        let pat = parse_pattern("fn authenticate(*)").unwrap();
        assert_eq!(pat.kind, PatternKind::Function);
        assert_eq!(pat.name_pattern.as_ref().unwrap(), "authenticate");
        assert_eq!(pat.param_patterns.as_ref().unwrap(), &["*"]);
    }

    #[test]
    fn test_parse_async_pattern() {
        let pat = parse_pattern("async fn(*) -> Result").unwrap();
        assert_eq!(pat.kind, PatternKind::Function);
        assert!(pat.qualifiers.contains(&"async".to_string()));
        assert_eq!(pat.return_pattern.as_ref().unwrap(), "Result");
    }

    #[test]
    fn test_parse_class_pattern() {
        let pat = parse_pattern("class User").unwrap();
        assert_eq!(pat.kind, PatternKind::Class);
        assert_eq!(pat.name_pattern.as_ref().unwrap(), "User");
    }

    #[test]
    fn test_parse_struct_wildcard() {
        let pat = parse_pattern("struct *Config").unwrap();
        assert_eq!(pat.kind, PatternKind::Struct);
        assert_eq!(pat.name_pattern.as_ref().unwrap(), "*Config");
    }

    #[test]
    fn test_parse_multi_param() {
        let pat = parse_pattern("fn(string, number) -> bool").unwrap();
        assert_eq!(pat.kind, PatternKind::Function);
        let params = pat.param_patterns.as_ref().unwrap();
        assert_eq!(params.len(), 2);
        assert_eq!(params[0], "string");
        assert_eq!(params[1], "number");
        assert_eq!(pat.return_pattern.as_ref().unwrap(), "bool");
    }

    #[test]
    fn test_parse_empty_params() {
        let pat = parse_pattern("fn()").unwrap();
        assert_eq!(pat.kind, PatternKind::Function);
        assert!(pat.param_patterns.as_ref().unwrap().is_empty());
    }

    #[test]
    fn test_wildcard_match() {
        assert!(wildcard_match("*Config", "SeekrConfig"));
        assert!(wildcard_match("*Config", "AppConfig"));
        assert!(!wildcard_match("*Config", "ConfigManager"));
        assert!(wildcard_match("Auth*", "AuthService"));
        assert!(wildcard_match("*", "anything"));
        assert!(wildcard_match("exact", "exact"));
        assert!(!wildcard_match("exact", "notexact"));
    }

    #[test]
    fn test_fuzzy_type_match() {
        assert!(fuzzy_type_match("string", "String"));
        assert!(fuzzy_type_match("string", "&str"));
        assert!(fuzzy_type_match("number", "i32"));
        assert!(fuzzy_type_match("number", "f64"));
        assert!(fuzzy_type_match("bool", "boolean"));
        assert!(fuzzy_type_match("*", "anything"));
        assert!(fuzzy_type_match("Result*", "Result<String, Error>"));
    }

    #[test]
    fn test_extract_params_rust() {
        let params = extract_params_from_signature("fn authenticate(user: &str, password: String) -> bool");
        assert_eq!(params.len(), 2);
        assert_eq!(params[0], "str");
        assert_eq!(params[1], "String");
    }

    #[test]
    fn test_extract_return_type_rust() {
        let ret = extract_return_type_from_signature("fn foo(x: i32) -> Result<String, Error>");
        assert_eq!(ret, Some("Result<String, Error>".to_string()));
    }

    #[test]
    fn test_extract_return_type_arrow() {
        let ret = extract_return_type_from_signature("def foo(x: int) -> bool:");
        assert_eq!(ret, Some("bool".to_string()));
    }

    #[test]
    fn test_match_function_by_return_type() {
        let pat = parse_pattern("fn(*) -> Result").unwrap();

        let chunk = make_chunk(
            1,
            ChunkKind::Function,
            "authenticate",
            "fn authenticate(user: &str) -> Result<Token, Error>",
            "fn authenticate(user: &str) -> Result<Token, Error> { }",
        );

        let score = match_chunk(&pat, &chunk);
        assert!(score > 0.5, "Should match function returning Result, got {}", score);
    }

    #[test]
    fn test_match_function_by_name() {
        let pat = parse_pattern("fn authenticate(*)").unwrap();

        let chunk = make_chunk(
            1,
            ChunkKind::Function,
            "authenticate",
            "fn authenticate(user: &str, pass: &str) -> bool",
            "fn authenticate(user: &str, pass: &str) -> bool { }",
        );

        let score = match_chunk(&pat, &chunk);
        assert!(score > 0.5, "Should match by name, got {}", score);
    }

    #[test]
    fn test_no_match_wrong_kind() {
        let pat = parse_pattern("class Foo").unwrap();

        let chunk = make_chunk(
            1,
            ChunkKind::Function,
            "Foo",
            "fn Foo()",
            "fn Foo() {}",
        );

        let score = match_chunk(&pat, &chunk);
        assert_eq!(score, 0.0, "Should not match wrong kind");
    }

    #[test]
    fn test_search_ast_pattern_integration() {
        let mut index = SeekrIndex::new(4);

        // Add some chunks
        let chunks = vec![
            make_chunk(
                1,
                ChunkKind::Function,
                "authenticate",
                "fn authenticate(user: &str) -> Result<Token, AuthError>",
                "fn authenticate(user: &str) -> Result<Token, AuthError> { }",
            ),
            make_chunk(
                2,
                ChunkKind::Function,
                "calculate",
                "fn calculate(x: f64, y: f64) -> f64",
                "fn calculate(x: f64, y: f64) -> f64 { x + y }",
            ),
            make_chunk(
                3,
                ChunkKind::Struct,
                "AppConfig",
                "pub struct AppConfig",
                "pub struct AppConfig { pub port: u16 }",
            ),
        ];

        for chunk in &chunks {
            let entry = crate::index::IndexEntry {
                chunk_id: chunk.id,
                embedding: vec![0.1; 4],
                text_tokens: vec![],
            };
            index.add_entry(entry, chunk.clone());
        }

        // Search for functions returning Result
        let results = search_ast_pattern(&index, "fn(*) -> Result", 10).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].chunk_id, 1);

        // Search for structs with *Config name
        let results = search_ast_pattern(&index, "struct *Config", 10).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].chunk_id, 3);

        // Search for function named calculate
        let results = search_ast_pattern(&index, "fn calculate(*)", 10).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].chunk_id, 2);
    }

    #[test]
    fn test_empty_pattern_error() {
        let result = parse_pattern("");
        assert!(result.is_err());
    }
}
