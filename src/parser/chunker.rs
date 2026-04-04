//! Semantic code chunker.
//!
//! Traverses AST nodes to extract semantic code chunks (functions, classes,
//! methods, structs, etc.) from parsed source files.

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use tree_sitter::Node;

use crate::error::ParserError;
use crate::parser::treesitter::SupportedLanguage;
use crate::parser::{ChunkKind, CodeChunk, ParseResult};

/// Global chunk ID counter (monotonically increasing).
static CHUNK_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Generate a new unique chunk ID.
fn next_chunk_id() -> u64 {
    CHUNK_ID_COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// Minimum number of lines for a chunk to be considered meaningful.
const MIN_CHUNK_LINES: usize = 2;

/// Maximum number of lines for fallback line-based chunking.
const FALLBACK_CHUNK_SIZE: usize = 50;

/// Parse a source file and extract code chunks.
pub fn chunk_file(
    path: &Path,
    source: &str,
    lang: SupportedLanguage,
) -> Result<ParseResult, ParserError> {
    let tree = crate::parser::treesitter::parse_source(source, lang)?;
    let root = tree.root_node();

    let chunk_kinds = lang.chunk_node_kinds();

    let mut chunks = Vec::new();

    if chunk_kinds.is_empty() {
        // For non-code languages (JSON, TOML, etc.), create a single chunk
        // for the entire file if it's not too large
        if source.lines().count() <= FALLBACK_CHUNK_SIZE * 2 {
            chunks.push(CodeChunk {
                id: next_chunk_id(),
                file_path: path.to_path_buf(),
                language: lang.name().to_string(),
                kind: ChunkKind::Block,
                name: path.file_name().and_then(|f| f.to_str()).map(String::from),
                signature: None,
                doc_comment: None,
                body: source.to_string(),
                byte_range: 0..source.len(),
                line_range: 0..source.lines().count(),
            });
        }
    } else {
        // Walk the AST and extract chunks for matching node kinds
        extract_chunks_recursive(&root, source, path, lang, chunk_kinds, &mut chunks);

        // If no chunks were found via AST, fall back to line-based chunking
        if chunks.is_empty() {
            chunks = fallback_line_chunks(path, source, lang);
        }
    }

    Ok(ParseResult {
        chunks,
        language: lang.name().to_string(),
    })
}

/// Recursively walk the AST and extract chunks for matching node kinds.
fn extract_chunks_recursive(
    node: &Node,
    source: &str,
    file_path: &Path,
    lang: SupportedLanguage,
    chunk_kinds: &[&str],
    chunks: &mut Vec<CodeChunk>,
) {
    let kind = node.kind();

    if chunk_kinds.contains(&kind) {
        if let Some(chunk) = node_to_chunk(node, source, file_path, lang) {
            // Only add chunks that are meaningful (not too small)
            let line_count = chunk.line_range.end - chunk.line_range.start;
            if line_count >= MIN_CHUNK_LINES {
                chunks.push(chunk);
            }
        }
        // Don't recurse into matched nodes to avoid nested duplicates
        // (e.g., methods inside a class that's already extracted)
        // We DO want nested chunks for impl/class blocks though
        if should_recurse_into(kind) {
            let mut cursor = node.walk();
            if cursor.goto_first_child() {
                loop {
                    let child = cursor.node();
                    extract_chunks_recursive(&child, source, file_path, lang, chunk_kinds, chunks);
                    if !cursor.goto_next_sibling() {
                        break;
                    }
                }
            }
        }
    } else {
        // Continue searching in children
        let mut cursor = node.walk();
        if cursor.goto_first_child() {
            loop {
                let child = cursor.node();
                extract_chunks_recursive(&child, source, file_path, lang, chunk_kinds, chunks);
                if !cursor.goto_next_sibling() {
                    break;
                }
            }
        }
    }
}

/// Determine if we should recurse into a matched node to find nested chunks.
fn should_recurse_into(kind: &str) -> bool {
    matches!(
        kind,
        "impl_item"
            | "class_declaration"
            | "class_definition"
            | "class_specifier"
            | "interface_declaration"
            | "namespace_definition"
            | "module"
            | "mod_item"
            | "export_statement"
            | "decorated_definition"
    )
}

/// Convert a tree-sitter Node to a CodeChunk.
fn node_to_chunk(
    node: &Node,
    source: &str,
    file_path: &Path,
    lang: SupportedLanguage,
) -> Option<CodeChunk> {
    let start_byte = node.start_byte();
    let end_byte = node.end_byte();

    if end_byte <= start_byte || end_byte > source.len() {
        return None;
    }

    let body = source[start_byte..end_byte].to_string();
    let start_line = node.start_position().row;
    let end_line = node.end_position().row + 1; // exclusive

    let kind = classify_node_kind(node.kind(), lang);
    let name = extract_node_name(node, source);
    let signature = extract_signature(node, source, lang);
    let doc_comment = extract_doc_comment(node, source, start_line);

    Some(CodeChunk {
        id: next_chunk_id(),
        file_path: file_path.to_path_buf(),
        language: lang.name().to_string(),
        kind,
        name,
        signature,
        doc_comment,
        body,
        byte_range: start_byte..end_byte,
        line_range: start_line..end_line,
    })
}

/// Classify a tree-sitter node kind into a ChunkKind.
fn classify_node_kind(ts_kind: &str, _lang: SupportedLanguage) -> ChunkKind {
    match ts_kind {
        // Functions
        "function_item" | "function_definition" | "function_declaration" | "arrow_function" => {
            ChunkKind::Function
        }
        // Methods
        "method_definition"
        | "method_declaration"
        | "method"
        | "singleton_method"
        | "constructor_declaration" => ChunkKind::Method,
        // Classes
        "class_declaration" | "class_definition" | "class_specifier" => ChunkKind::Class,
        // Structs
        "struct_item" | "struct_specifier" => ChunkKind::Struct,
        // Enums
        "enum_item" | "enum_declaration" | "enum_specifier" => ChunkKind::Enum,
        // Interfaces / Traits
        "interface_declaration" | "trait_item" => ChunkKind::Interface,
        // Modules / Namespaces
        "mod_item" | "namespace_definition" | "module" => ChunkKind::Module,
        // Impl blocks (Rust) → treat as Module-level grouping
        "impl_item" => ChunkKind::Module,
        // Everything else
        _ => ChunkKind::Block,
    }
}

/// Extract the name of a node (e.g., function name, class name).
fn extract_node_name(node: &Node, source: &str) -> Option<String> {
    // Try common field names for the "name" of a construct
    for field_name in &["name", "declarator"] {
        if let Some(name_node) = node.child_by_field_name(field_name) {
            let name = &source[name_node.start_byte()..name_node.end_byte()];
            return Some(name.to_string());
        }
    }

    // For some languages, look at the first named child of specific type
    let mut cursor = node.walk();
    if cursor.goto_first_child() {
        loop {
            let child = cursor.node();
            if child.kind() == "identifier" || child.kind() == "type_identifier" {
                let name = &source[child.start_byte()..child.end_byte()];
                return Some(name.to_string());
            }
            if !cursor.goto_next_sibling() {
                break;
            }
        }
    }

    None
}

/// Extract a function/method signature (first line or up to the body).
fn extract_signature(node: &Node, source: &str, _lang: SupportedLanguage) -> Option<String> {
    let body = &source[node.start_byte()..node.end_byte()];

    // Find the first `{` or `:` (Python) to extract just the signature
    if let Some(pos) = body.find('{') {
        let sig = body[..pos].trim();
        if !sig.is_empty() {
            return Some(sig.to_string());
        }
    }

    // For Python-style (colon-based blocks)
    if let Some(pos) = body.find(':') {
        // Make sure it's a function/class colon, not a type annotation colon
        let before_colon = &body[..pos];
        if before_colon.contains("def ") || before_colon.contains("class ") {
            let sig = body[..=pos].trim();
            if !sig.is_empty() {
                return Some(sig.to_string());
            }
        }
    }

    // Fallback: first line
    let first_line = body.lines().next().map(|l| l.trim().to_string());
    first_line.filter(|l| !l.is_empty())
}

/// Extract documentation comment immediately before a node.
fn extract_doc_comment(node: &Node, source: &str, _node_start_line: usize) -> Option<String> {
    // Look at previous siblings for comment nodes
    let mut prev = node.prev_sibling();
    let mut comments = Vec::new();

    while let Some(sibling) = prev {
        let kind = sibling.kind();
        if kind == "line_comment" || kind == "comment" || kind == "block_comment" {
            let text = &source[sibling.start_byte()..sibling.end_byte()];
            comments.push(text.to_string());
            prev = sibling.prev_sibling();
        } else {
            break;
        }
    }

    if comments.is_empty() {
        return None;
    }

    // Reverse since we collected them backwards
    comments.reverse();
    let combined = comments.join("\n");

    // Clean up common comment prefixes
    let cleaned: String = combined
        .lines()
        .map(|line| {
            let trimmed = line.trim();
            if let Some(stripped) = trimmed.strip_prefix("///") {
                stripped.trim().to_string()
            } else if let Some(stripped) = trimmed.strip_prefix("//!") {
                stripped.trim().to_string()
            } else if let Some(stripped) = trimmed.strip_prefix("//") {
                stripped.trim().to_string()
            } else if let Some(stripped) = trimmed.strip_prefix('#') {
                stripped.trim().to_string()
            } else {
                trimmed.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("\n");

    if cleaned.trim().is_empty() {
        None
    } else {
        Some(cleaned)
    }
}

/// Fallback: split source into line-based chunks when AST chunking yields nothing.
fn fallback_line_chunks(file_path: &Path, source: &str, lang: SupportedLanguage) -> Vec<CodeChunk> {
    let lines: Vec<&str> = source.lines().collect();
    let total_lines = lines.len();

    if total_lines == 0 {
        return Vec::new();
    }

    let mut chunks = Vec::new();
    let mut offset = 0;

    for chunk_start in (0..total_lines).step_by(FALLBACK_CHUNK_SIZE) {
        let chunk_end = (chunk_start + FALLBACK_CHUNK_SIZE).min(total_lines);
        let chunk_lines = &lines[chunk_start..chunk_end];
        let body = chunk_lines.join("\n");
        let byte_start = offset;
        let byte_end = offset + body.len();
        offset = byte_end + 1; // +1 for the newline between chunks

        chunks.push(CodeChunk {
            id: next_chunk_id(),
            file_path: file_path.to_path_buf(),
            language: lang.name().to_string(),
            kind: ChunkKind::Block,
            name: Some(format!(
                "{}:L{}-L{}",
                file_path.file_name().unwrap_or_default().to_string_lossy(),
                chunk_start + 1,
                chunk_end
            )),
            signature: None,
            doc_comment: None,
            body,
            byte_range: byte_start..byte_end,
            line_range: chunk_start..chunk_end,
        });
    }

    chunks
}

/// Parse a file from disk: read it, detect language, and chunk it.
pub fn chunk_file_from_path(path: &Path) -> Result<Option<ParseResult>, ParserError> {
    let lang = match SupportedLanguage::from_path(path) {
        Some(l) => l,
        None => return Ok(None), // Unsupported language, skip
    };

    let source = std::fs::read_to_string(path).map_err(ParserError::Io)?;

    // Skip binary files
    if crate::scanner::filter::is_binary_content(source.as_bytes()) {
        return Ok(None);
    }

    let result = chunk_file(path, &source, lang)?;
    Ok(Some(result))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_rust_file() {
        let source = r#"
/// A greeting function.
fn greet(name: &str) -> String {
    format!("Hello, {}!", name)
}

/// A struct.
struct User {
    name: String,
    age: u32,
}

impl User {
    fn new(name: String, age: u32) -> Self {
        Self { name, age }
    }

    fn display(&self) -> String {
        format!("{} ({})", self.name, self.age)
    }
}
"#;
        let result = chunk_file(Path::new("test.rs"), source, SupportedLanguage::Rust).unwrap();

        assert_eq!(result.language, "rust");
        assert!(
            !result.chunks.is_empty(),
            "Should find at least some chunks"
        );

        // Should find the greet function
        let greet = result
            .chunks
            .iter()
            .find(|c| c.name.as_deref() == Some("greet"));
        assert!(greet.is_some(), "Should find greet function");

        let greet = greet.unwrap();
        assert_eq!(greet.kind, ChunkKind::Function);
        assert!(greet.doc_comment.is_some(), "Should extract doc comment");
        assert!(greet.signature.is_some(), "Should extract signature");
    }

    #[test]
    fn test_chunk_python_file() {
        let source = r#"
class Calculator:
    """A simple calculator."""

    def add(self, a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    def subtract(self, a: int, b: int) -> int:
        return a - b

def standalone_function(x: str) -> bool:
    return len(x) > 0
"#;
        let result = chunk_file(Path::new("calc.py"), source, SupportedLanguage::Python).unwrap();

        assert_eq!(result.language, "python");
        assert!(!result.chunks.is_empty());
    }

    #[test]
    fn test_chunk_javascript_file() {
        let source = r#"
function fetchData(url) {
    return fetch(url).then(r => r.json());
}

class EventEmitter {
    constructor() {
        this.listeners = {};
    }

    on(event, callback) {
        this.listeners[event] = callback;
    }
}
"#;
        let result =
            chunk_file(Path::new("app.js"), source, SupportedLanguage::JavaScript).unwrap();

        assert_eq!(result.language, "javascript");
        assert!(!result.chunks.is_empty());
    }

    #[test]
    fn test_fallback_chunking() {
        // Create a file with no recognizable AST patterns
        let source = "#!/bin/bash\necho 'hello'\necho 'world'\n".repeat(30);
        let result = chunk_file(Path::new("script.sh"), &source, SupportedLanguage::Bash).unwrap();

        // Bash has empty chunk_node_kinds, so should get a single block chunk
        assert!(!result.chunks.is_empty());
    }
}
