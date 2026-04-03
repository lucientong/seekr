//! Code parser module.
//!
//! Uses Tree-sitter for AST parsing and semantic chunking of source code.

pub mod chunker;
pub mod summary;
pub mod treesitter;

use std::ops::Range;
use std::path::PathBuf;

/// The kind of a code chunk.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ChunkKind {
    Function,
    Method,
    Class,
    Struct,
    Enum,
    Interface,
    Module,
    /// Fallback for chunks that don't match any specific kind.
    Block,
}

/// A semantic chunk of code extracted from a source file.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CodeChunk {
    /// Unique identifier for this chunk.
    pub id: u64,

    /// Path to the source file.
    pub file_path: PathBuf,

    /// Programming language.
    pub language: String,

    /// Kind of code construct.
    pub kind: ChunkKind,

    /// Name of the construct (e.g., function name).
    pub name: Option<String>,

    /// Full signature (e.g., `fn foo(x: i32) -> String`).
    pub signature: Option<String>,

    /// Documentation comment, if any.
    pub doc_comment: Option<String>,

    /// The full source text of this chunk.
    pub body: String,

    /// Byte range in the original file.
    pub byte_range: Range<usize>,

    /// Line range in the original file (0-indexed).
    pub line_range: Range<usize>,
}

/// Result of parsing a single file.
#[derive(Debug)]
pub struct ParseResult {
    /// Code chunks extracted from the file.
    pub chunks: Vec<CodeChunk>,

    /// The detected language.
    pub language: String,
}

impl std::fmt::Display for ChunkKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChunkKind::Function => write!(f, "function"),
            ChunkKind::Method => write!(f, "method"),
            ChunkKind::Class => write!(f, "class"),
            ChunkKind::Struct => write!(f, "struct"),
            ChunkKind::Enum => write!(f, "enum"),
            ChunkKind::Interface => write!(f, "interface"),
            ChunkKind::Module => write!(f, "module"),
            ChunkKind::Block => write!(f, "block"),
        }
    }
}
