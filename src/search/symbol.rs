//! Lightweight name-based symbol navigation.
//!
//! This intentionally does not claim compiler- or LSP-grade resolution.

use std::path::PathBuf;

use crate::parser::{ChunkKind, CodeChunk};

#[derive(Debug, Clone, Default)]
pub struct SymbolOptions {
    pub path_prefix: Option<PathBuf>,
    pub languages: Vec<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct SymbolSummary {
    pub normalized_name: String,
    pub names: Vec<String>,
    pub kinds: Vec<String>,
    pub languages: Vec<String>,
    pub definition_count: usize,
}

/// Normalize a symbol for case-insensitive exact lookup.
pub fn normalize_symbol_name(name: &str) -> String {
    name.trim().to_lowercase()
}

pub(crate) fn is_indexable_symbol(chunk: &CodeChunk) -> bool {
    chunk
        .name
        .as_ref()
        .is_some_and(|name| !name.trim().is_empty())
        && matches!(
            chunk.kind,
            ChunkKind::Function
                | ChunkKind::Method
                | ChunkKind::Class
                | ChunkKind::Struct
                | ChunkKind::Enum
                | ChunkKind::Interface
                | ChunkKind::Module
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_case_and_surrounding_whitespace() {
        assert_eq!(normalize_symbol_name("  HTTPServer  "), "httpserver");
    }
}
