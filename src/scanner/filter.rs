//! File filter module.
//!
//! Provides binary file detection, file size limits,
//! and file type whitelist/blacklist filtering.

use std::path::Path;

/// Maximum number of bytes to check for binary detection.
const BINARY_CHECK_SIZE: usize = 8192;

/// Percentage of null bytes threshold for binary detection.
const BINARY_NULL_THRESHOLD: f64 = 0.01;

/// Known source code file extensions.
const SOURCE_EXTENSIONS: &[&str] = &[
    // Rust
    "rs",
    // Python
    "py",
    "pyi",
    "pyx",
    // JavaScript / TypeScript
    "js",
    "jsx",
    "mjs",
    "cjs",
    "ts",
    "tsx",
    "mts",
    "cts",
    // Go
    "go",
    // Java / Kotlin
    "java",
    "kt",
    "kts",
    // C / C++
    "c",
    "h",
    "cc",
    "cpp",
    "cxx",
    "hpp",
    "hxx",
    // C#
    "cs",
    // Ruby
    "rb",
    // PHP
    "php",
    // Swift
    "swift",
    // Scala
    "scala",
    // Shell
    "sh",
    "bash",
    "zsh",
    "fish",
    // Web
    "html",
    "htm",
    "css",
    "scss",
    "sass",
    "less",
    // Data / Config
    "json",
    "yaml",
    "yml",
    "toml",
    "xml",
    "ini",
    "cfg",
    // Markdown / Docs
    "md",
    "rst",
    "txt",
    // SQL
    "sql",
    // Lua
    "lua",
    // Dart
    "dart",
    // Elixir / Erlang
    "ex",
    "exs",
    "erl",
    // Haskell
    "hs",
    // OCaml
    "ml",
    "mli",
    // Zig
    "zig",
    // Protobuf
    "proto",
    // Dockerfile
    "dockerfile",
    // Makefile
    "makefile",
];

/// Check if a file is likely a binary file by examining its content.
///
/// Reads up to the first 8KB and checks for null bytes.
/// Returns `true` if the file appears to be binary.
pub fn is_binary_file(path: &Path) -> bool {
    match std::fs::read(path) {
        Ok(content) => is_binary_content(&content),
        Err(_) => true, // If we can't read it, treat as binary (skip it)
    }
}

/// Check if content appears to be binary.
///
/// Uses null-byte heuristic on the first 8KB.
pub fn is_binary_content(content: &[u8]) -> bool {
    let check_len = content.len().min(BINARY_CHECK_SIZE);
    if check_len == 0 {
        return false; // Empty files are not binary
    }

    let slice = &content[..check_len];
    let null_count = slice.iter().filter(|&&b| b == 0).count();
    let null_ratio = null_count as f64 / check_len as f64;

    null_ratio > BINARY_NULL_THRESHOLD
}

/// Check if a file has a recognized source code extension.
pub fn is_source_file(path: &Path) -> bool {
    // Check for special filenames (no extension)
    if let Some(filename) = path.file_name().and_then(|f| f.to_str()) {
        let lower = filename.to_lowercase();
        if matches!(
            lower.as_str(),
            "makefile" | "dockerfile" | "rakefile" | "gemfile" | "cmakelists.txt"
        ) {
            return true;
        }
    }

    // Check extension
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| {
            let lower = ext.to_lowercase();
            SOURCE_EXTENSIONS.contains(&lower.as_str())
        })
        .unwrap_or(false)
}

/// Check if a file should be included for indexing.
///
/// A file passes if:
/// 1. It has a recognized source code extension
/// 2. It is not too large (checked by size, not reading content)
/// 3. It is not binary (checked lazily, only if other checks pass)
pub fn should_index_file(path: &Path, size: u64, max_file_size: u64) -> bool {
    // Size check first (cheapest)
    if size > max_file_size {
        tracing::debug!(path = %path.display(), size, max = max_file_size, "Skipping oversized file");
        return false;
    }

    // Extension check (cheap)
    if !is_source_file(path) {
        return false;
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_detection() {
        // Text content
        assert!(!is_binary_content(b"Hello, world!\nThis is text."));

        // Binary content (lots of null bytes)
        let mut binary = vec![0u8; 1000];
        binary[0] = b'E';
        binary[1] = b'L';
        binary[2] = b'F';
        assert!(is_binary_content(&binary));

        // Empty content
        assert!(!is_binary_content(b""));
    }

    #[test]
    fn test_source_file_detection() {
        assert!(is_source_file(Path::new("main.rs")));
        assert!(is_source_file(Path::new("app.py")));
        assert!(is_source_file(Path::new("index.ts")));
        assert!(is_source_file(Path::new("main.go")));
        assert!(is_source_file(Path::new("config.toml")));
        assert!(is_source_file(Path::new("Makefile")));

        // Non-source files
        assert!(!is_source_file(Path::new("image.png")));
        assert!(!is_source_file(Path::new("data.bin")));
        assert!(!is_source_file(Path::new("archive.tar.gz")));
    }

    #[test]
    fn test_should_index_file() {
        let max_size = 10 * 1024 * 1024; // 10 MB

        assert!(should_index_file(Path::new("main.rs"), 1000, max_size));
        assert!(!should_index_file(
            Path::new("main.rs"),
            max_size + 1,
            max_size
        ));
        assert!(!should_index_file(Path::new("image.png"), 1000, max_size));
    }
}
