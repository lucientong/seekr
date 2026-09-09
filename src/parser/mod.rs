//! Code parser module.
//!
//! Uses Tree-sitter for AST parsing and semantic chunking of source code.

pub mod callsites;
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

/// Surface-syntax call kinds extracted from Tree-sitter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CallKind {
    Function,
    Method,
    Constructor,
    Macro,
}

impl CallKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Function => "function",
            Self::Method => "method",
            Self::Constructor => "constructor",
            Self::Macro => "macro",
        }
    }
}

impl std::fmt::Display for CallKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// A name-level call-site mention (not a resolved reference edge).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CallSite {
    pub id: u64,
    pub file_path: PathBuf,
    pub language: String,
    /// Final identifier segment of the callee (`foo` in `a.b.foo`).
    pub callee_name: String,
    pub call_kind: CallKind,
    /// True when the callee was accessed via member / scoped path.
    pub is_member: bool,
    pub byte_range: Range<usize>,
    pub line_range: Range<usize>,
    pub enclosing_chunk_id: Option<u64>,
    pub body: String,
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

    /// Call-site mentions extracted from the same parse tree.
    pub call_sites: Vec<CallSite>,

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
