//! Tree-sitter integration.
//!
//! Uses individual tree-sitter language crates to parse source files into ASTs.
//! Supports language detection by file extension and AST traversal.

use std::path::Path;

use tree_sitter::{Language, Parser};

use crate::error::ParserError;

/// Supported languages and their Tree-sitter grammars.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SupportedLanguage {
    Rust,
    Python,
    JavaScript,
    TypeScript,
    Tsx,
    Go,
    Java,
    C,
    Cpp,
    Json,
    Toml,
    Yaml,
    Html,
    Css,
    Ruby,
    Bash,
}

impl SupportedLanguage {
    /// Detect the language from a file extension.
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "rs" => Some(Self::Rust),
            "py" | "pyi" | "pyx" => Some(Self::Python),
            "js" | "jsx" | "mjs" | "cjs" => Some(Self::JavaScript),
            "ts" | "mts" | "cts" => Some(Self::TypeScript),
            "tsx" => Some(Self::Tsx),
            "go" => Some(Self::Go),
            "java" => Some(Self::Java),
            "c" | "h" => Some(Self::C),
            "cc" | "cpp" | "cxx" | "hpp" | "hxx" => Some(Self::Cpp),
            "json" => Some(Self::Json),
            "toml" => Some(Self::Toml),
            "yaml" | "yml" => Some(Self::Yaml),
            "html" | "htm" => Some(Self::Html),
            "css" | "scss" => Some(Self::Css),
            "rb" => Some(Self::Ruby),
            "sh" | "bash" | "zsh" => Some(Self::Bash),
            _ => None,
        }
    }

    /// Detect the language from a file path.
    pub fn from_path(path: &Path) -> Option<Self> {
        // Check special filenames first
        if let Some(filename) = path.file_name().and_then(|f| f.to_str()) {
            match filename.to_lowercase().as_str() {
                "makefile" | "gnumakefile" => return Some(Self::Bash),
                "dockerfile" => return Some(Self::Bash),
                _ => {}
            }
        }

        // Check extension
        path.extension()
            .and_then(|ext| ext.to_str())
            .and_then(Self::from_extension)
    }

    /// Get the Tree-sitter language grammar.
    pub fn grammar(&self) -> Language {
        match self {
            Self::Rust => tree_sitter_rust::LANGUAGE.into(),
            Self::Python => tree_sitter_python::LANGUAGE.into(),
            Self::JavaScript => tree_sitter_javascript::LANGUAGE.into(),
            Self::TypeScript => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
            Self::Tsx => tree_sitter_typescript::LANGUAGE_TSX.into(),
            Self::Go => tree_sitter_go::LANGUAGE.into(),
            Self::Java => tree_sitter_java::LANGUAGE.into(),
            Self::C => tree_sitter_c::LANGUAGE.into(),
            Self::Cpp => tree_sitter_cpp::LANGUAGE.into(),
            Self::Json => tree_sitter_json::LANGUAGE.into(),
            Self::Toml => tree_sitter_toml_ng::LANGUAGE.into(),
            Self::Yaml => tree_sitter_yaml::LANGUAGE.into(),
            Self::Html => tree_sitter_html::LANGUAGE.into(),
            Self::Css => tree_sitter_css::LANGUAGE.into(),
            Self::Ruby => tree_sitter_ruby::LANGUAGE.into(),
            Self::Bash => tree_sitter_bash::LANGUAGE.into(),
        }
    }

    /// Get the language name as a string.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Rust => "rust",
            Self::Python => "python",
            Self::JavaScript => "javascript",
            Self::TypeScript => "typescript",
            Self::Tsx => "tsx",
            Self::Go => "go",
            Self::Java => "java",
            Self::C => "c",
            Self::Cpp => "cpp",
            Self::Json => "json",
            Self::Toml => "toml",
            Self::Yaml => "yaml",
            Self::Html => "html",
            Self::Css => "css",
            Self::Ruby => "ruby",
            Self::Bash => "bash",
        }
    }

    /// Get AST node kinds that represent interesting code constructs
    /// (functions, classes, methods, etc.) for this language.
    pub fn chunk_node_kinds(&self) -> &[&str] {
        match self {
            Self::Rust => &[
                "function_item",
                "impl_item",
                "struct_item",
                "enum_item",
                "trait_item",
                "mod_item",
                "const_item",
                "static_item",
                "type_item",
                "macro_definition",
            ],
            Self::Python => &[
                "function_definition",
                "class_definition",
                "decorated_definition",
            ],
            Self::JavaScript | Self::Tsx => &[
                "function_declaration",
                "class_declaration",
                "method_definition",
                "arrow_function",
                "export_statement",
            ],
            Self::TypeScript => &[
                "function_declaration",
                "class_declaration",
                "method_definition",
                "arrow_function",
                "interface_declaration",
                "type_alias_declaration",
                "export_statement",
            ],
            Self::Go => &[
                "function_declaration",
                "method_declaration",
                "type_declaration",
            ],
            Self::Java => &[
                "class_declaration",
                "method_declaration",
                "interface_declaration",
                "enum_declaration",
                "constructor_declaration",
            ],
            Self::C => &["function_definition", "struct_specifier", "enum_specifier"],
            Self::Cpp => &[
                "function_definition",
                "class_specifier",
                "struct_specifier",
                "enum_specifier",
                "namespace_definition",
            ],
            Self::Ruby => &["method", "class", "module", "singleton_method"],
            // For config/data languages, we don't chunk
            Self::Json | Self::Toml | Self::Yaml | Self::Html | Self::Css | Self::Bash => &[],
        }
    }
}

/// Create a parser configured for the given language.
pub fn create_parser(lang: SupportedLanguage) -> Result<Parser, ParserError> {
    let mut parser = Parser::new();
    parser
        .set_language(&lang.grammar())
        .map_err(|e| ParserError::UnsupportedLanguage(format!("{}: {}", lang.name(), e)))?;
    Ok(parser)
}

/// Parse source code with the given language and return the tree.
pub fn parse_source(
    source: &str,
    lang: SupportedLanguage,
) -> Result<tree_sitter::Tree, ParserError> {
    let mut parser = create_parser(lang)?;
    parser
        .parse(source, None)
        .ok_or_else(|| ParserError::ParseFailed {
            path: std::path::PathBuf::from("<string>"),
            reason: "Parser returned None".to_string(),
        })
}

impl std::fmt::Display for SupportedLanguage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_language_detection() {
        assert_eq!(
            SupportedLanguage::from_extension("rs"),
            Some(SupportedLanguage::Rust)
        );
        assert_eq!(
            SupportedLanguage::from_extension("py"),
            Some(SupportedLanguage::Python)
        );
        assert_eq!(
            SupportedLanguage::from_extension("tsx"),
            Some(SupportedLanguage::Tsx)
        );
        assert_eq!(SupportedLanguage::from_extension("unknown"), None);
    }

    #[test]
    fn test_parse_rust() {
        let source = "fn hello() -> String { \"world\".to_string() }";
        let tree = parse_source(source, SupportedLanguage::Rust).unwrap();
        let root = tree.root_node();
        assert_eq!(root.kind(), "source_file");
        assert!(root.child_count() > 0);
    }

    #[test]
    fn test_parse_python() {
        let source = "def greet(name: str) -> str:\n    return f\"Hello, {name}\"";
        let tree = parse_source(source, SupportedLanguage::Python).unwrap();
        let root = tree.root_node();
        assert_eq!(root.kind(), "module");
    }

    #[test]
    fn test_parse_javascript() {
        let source = "function add(a, b) { return a + b; }";
        let tree = parse_source(source, SupportedLanguage::JavaScript).unwrap();
        let root = tree.root_node();
        assert_eq!(root.kind(), "program");
    }

    #[test]
    fn test_all_grammars_load() {
        let languages = [
            SupportedLanguage::Rust,
            SupportedLanguage::Python,
            SupportedLanguage::JavaScript,
            SupportedLanguage::TypeScript,
            SupportedLanguage::Tsx,
            SupportedLanguage::Go,
            SupportedLanguage::Java,
            SupportedLanguage::C,
            SupportedLanguage::Cpp,
            SupportedLanguage::Json,
            SupportedLanguage::Toml,
            SupportedLanguage::Yaml,
            SupportedLanguage::Html,
            SupportedLanguage::Css,
            SupportedLanguage::Ruby,
            SupportedLanguage::Bash,
        ];

        for lang in languages {
            let parser = create_parser(lang);
            assert!(
                parser.is_ok(),
                "Failed to create parser for {}",
                lang.name()
            );
        }
    }
}
