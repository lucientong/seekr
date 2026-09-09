//! Configuration management for Seekr.
//!
//! Loads configuration from `~/.seekr/config.toml` with sensible defaults.
//! CLI arguments can override config file values.

use crate::error::ConfigError;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

/// Default server port for HTTP API.
const DEFAULT_PORT: u16 = 7720;

/// Default batch size for embedding computation.
const DEFAULT_BATCH_SIZE: usize = 32;

/// Default RRF fusion parameter k.
const DEFAULT_RRF_K: u32 = 60;

/// Default maximum file size for indexing (10 MB).
const DEFAULT_MAX_FILE_SIZE: u64 = 10 * 1024 * 1024;

/// Default number of context lines for text search results.
const DEFAULT_CONTEXT_LINES: usize = 2;

/// Default top-k results for semantic search.
const DEFAULT_TOP_K: usize = 20;

/// Main configuration structure for Seekr.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SeekrConfig {
    /// Directory for storing index files.
    /// Default: `~/.seekr/indexes/`
    pub index_dir: PathBuf,

    /// Directory for storing downloaded ONNX models.
    /// Default: `~/.seekr/models/`
    pub model_dir: PathBuf,

    /// Embedding model name.
    /// Default: "all-MiniLM-L6-v2"
    pub embed_model: String,

    /// File glob patterns to exclude from scanning.
    pub exclude_patterns: Vec<String>,

    /// Maximum file size (in bytes) to index.
    pub max_file_size: u64,

    /// Server configuration.
    pub server: ServerConfig,

    /// Search configuration.
    pub search: SearchConfig,

    /// Embedding configuration.
    pub embedding: EmbeddingConfig,
}

/// Server-specific configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ServerConfig {
    /// Host address to bind. Default: "127.0.0.1"
    pub host: String,

    /// Port number. Default: 7720
    pub port: u16,
}

/// Search-specific configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SearchConfig {
    /// Number of context lines for text search results.
    pub context_lines: usize,

    /// Top-k results for semantic search.
    pub top_k: usize,

    /// RRF fusion parameter k.
    pub rrf_k: u32,

    /// Minimum score threshold for semantic search results (0.0 - 1.0).
    pub score_threshold: f32,
}

/// Embedding-specific configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct EmbeddingConfig {
    /// Batch size for embedding computation.
    pub batch_size: usize,
}

/// Project-local indexing configuration loaded from `.seekr.toml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct WorkspaceConfig {
    /// Workspace-relative directories to scan.
    pub roots: Vec<PathBuf>,
    /// Optional whitelist globs, relative to the workspace root.
    pub include: Vec<String>,
    /// Additional blacklist globs, relative to the workspace root.
    pub exclude: Vec<String>,
    /// Optional language whitelist.
    pub languages: Vec<String>,
    /// Optional project-specific maximum file size.
    pub max_file_size: Option<u64>,
}

impl Default for WorkspaceConfig {
    fn default() -> Self {
        Self {
            roots: vec![PathBuf::from(".")],
            include: Vec::new(),
            exclude: Vec::new(),
            languages: Vec::new(),
            max_file_size: None,
        }
    }
}

/// Validated scan settings formed by merging global and project config.
#[derive(Debug, Clone)]
pub struct ScanConfig {
    pub workspace_root: PathBuf,
    pub roots: Vec<PathBuf>,
    pub include_patterns: Vec<String>,
    pub exclude_patterns: Vec<String>,
    pub languages: HashSet<String>,
    pub max_file_size: u64,
}

impl Default for SeekrConfig {
    fn default() -> Self {
        let seekr_dir = default_seekr_dir();
        Self {
            index_dir: seekr_dir.join("indexes"),
            model_dir: seekr_dir.join("models"),
            embed_model: "all-MiniLM-L6-v2".to_string(),
            exclude_patterns: vec![
                "*.min.js".to_string(),
                "*.min.css".to_string(),
                "*.lock".to_string(),
                "package-lock.json".to_string(),
                "yarn.lock".to_string(),
            ],
            max_file_size: DEFAULT_MAX_FILE_SIZE,
            server: ServerConfig::default(),
            search: SearchConfig::default(),
            embedding: EmbeddingConfig::default(),
        }
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: DEFAULT_PORT,
        }
    }
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            context_lines: DEFAULT_CONTEXT_LINES,
            top_k: DEFAULT_TOP_K,
            rrf_k: DEFAULT_RRF_K,
            score_threshold: 0.0,
        }
    }
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            batch_size: DEFAULT_BATCH_SIZE,
        }
    }
}

impl SeekrConfig {
    /// Load configuration from the default config file path.
    ///
    /// If the config file does not exist, returns default configuration
    /// and creates the default config file.
    pub fn load() -> std::result::Result<Self, ConfigError> {
        let config_path = default_config_path();
        Self::load_from(&config_path)
    }

    /// Load configuration from a specific file path.
    pub fn load_from(path: &Path) -> std::result::Result<Self, ConfigError> {
        if !path.exists() {
            let config = Self::default();
            // Attempt to create default config file, but don't fail if we can't
            if let Err(e) = config.save_to(path) {
                tracing::warn!(
                    "Could not write default config to {}: {}",
                    path.display(),
                    e
                );
            }
            return Ok(config);
        }

        let content = std::fs::read_to_string(path)?;
        let config: SeekrConfig =
            toml::from_str(&content).map_err(|e| ConfigError::ParseError(e.to_string()))?;
        Ok(config)
    }

    /// Save configuration to a specific file path.
    pub fn save_to(&self, path: &Path) -> std::result::Result<(), ConfigError> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let content =
            toml::to_string_pretty(self).map_err(|e| ConfigError::ParseError(e.to_string()))?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Get the index directory for a specific project path.
    ///
    /// Each project gets its own isolated index directory based on
    /// a blake3 hash of the canonical project path.
    pub fn project_index_dir(&self, project_path: &Path) -> PathBuf {
        let canonical = project_path
            .canonicalize()
            .unwrap_or_else(|_| project_path.to_path_buf());
        let hash = blake3::hash(canonical.to_string_lossy().as_bytes());
        // Use first 16 hex chars for readability
        let hex = hash.to_hex();
        let short_hash = &hex.as_str()[..16];
        self.index_dir.join(short_hash)
    }

    /// Load and validate project-local scan settings.
    pub fn scan_config(&self, workspace_root: &Path) -> Result<ScanConfig, ConfigError> {
        let workspace_root = workspace_root.canonicalize()?;
        let config_path = workspace_root.join(".seekr.toml");
        let workspace = if config_path.exists() {
            let content = std::fs::read_to_string(&config_path)?;
            toml::from_str::<WorkspaceConfig>(&content)
                .map_err(|error| ConfigError::ParseError(error.to_string()))?
        } else {
            WorkspaceConfig::default()
        };
        let configured_roots = if workspace.roots.is_empty() {
            vec![PathBuf::from(".")]
        } else {
            workspace.roots
        };
        let mut roots = Vec::new();
        for configured_root in configured_roots {
            let candidate = if configured_root.is_absolute() {
                configured_root
            } else {
                workspace_root.join(configured_root)
            };
            let root = candidate
                .canonicalize()
                .map_err(|error| ConfigError::InvalidValue {
                    key: "roots".to_string(),
                    reason: format!("cannot resolve '{}': {error}", candidate.display()),
                })?;
            if !root.starts_with(&workspace_root) {
                return Err(ConfigError::InvalidValue {
                    key: "roots".to_string(),
                    reason: format!(
                        "'{}' is outside workspace '{}'",
                        root.display(),
                        workspace_root.display()
                    ),
                });
            }
            if !root.is_dir() {
                return Err(ConfigError::InvalidValue {
                    key: "roots".to_string(),
                    reason: format!("'{}' is not a directory", root.display()),
                });
            }
            if !roots.contains(&root) {
                roots.push(root);
            }
        }
        roots.sort();

        let mut languages = HashSet::new();
        for language in &workspace.languages {
            if language.trim().is_empty() {
                continue;
            }
            let supported = crate::parser::treesitter::SupportedLanguage::from_name(language)
                .ok_or_else(|| ConfigError::InvalidValue {
                    key: "languages".to_string(),
                    reason: format!("unsupported language '{language}'"),
                })?;
            languages.insert(supported.name().to_string());
        }
        let mut exclude_patterns = self.exclude_patterns.clone();
        exclude_patterns.extend(workspace.exclude);

        Ok(ScanConfig {
            workspace_root,
            roots,
            include_patterns: workspace.include,
            exclude_patterns,
            languages,
            max_file_size: workspace.max_file_size.unwrap_or(self.max_file_size),
        })
    }
}

/// Get the default Seekr data directory (`~/.seekr/`).
fn default_seekr_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".seekr")
}

/// Get the default config file path (`~/.seekr/config.toml`).
pub fn default_config_path() -> PathBuf {
    default_seekr_dir().join("config.toml")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = SeekrConfig::default();
        assert_eq!(config.server.port, 7720);
        assert_eq!(config.embed_model, "all-MiniLM-L6-v2");
        assert_eq!(config.embedding.batch_size, 32);
        assert_eq!(config.search.rrf_k, 60);
    }

    #[test]
    fn test_project_index_dir_isolation() {
        let config = SeekrConfig::default();
        let dir_a = config.project_index_dir(Path::new("/home/user/project-a"));
        let dir_b = config.project_index_dir(Path::new("/home/user/project-b"));
        assert_ne!(
            dir_a, dir_b,
            "Different projects should have different index dirs"
        );
    }

    #[test]
    fn test_load_nonexistent_returns_default() {
        let config = SeekrConfig::load_from(Path::new("/nonexistent/path/config.toml")).unwrap();
        assert_eq!(config.server.port, DEFAULT_PORT);
    }

    #[test]
    fn loads_and_resolves_workspace_scan_config() {
        let workspace = tempfile::tempdir().unwrap();
        std::fs::create_dir(workspace.path().join("core")).unwrap();
        std::fs::write(
            workspace.path().join(".seekr.toml"),
            r#"
roots = ["core", "core"]
include = ["core/**/*.rs"]
exclude = ["**/*.generated.rs"]
languages = ["rs", "python"]
max_file_size = 1024
"#,
        )
        .unwrap();

        let scan = SeekrConfig::default()
            .scan_config(workspace.path())
            .unwrap();
        assert_eq!(
            scan.roots,
            vec![workspace.path().join("core").canonicalize().unwrap()]
        );
        assert_eq!(scan.include_patterns, vec!["core/**/*.rs"]);
        assert!(
            scan.exclude_patterns
                .contains(&"**/*.generated.rs".to_string())
        );
        assert_eq!(
            scan.languages,
            HashSet::from(["rust".to_string(), "python".to_string()])
        );
        assert_eq!(scan.max_file_size, 1024);
    }

    #[test]
    fn rejects_workspace_roots_outside_project() {
        let parent = tempfile::tempdir().unwrap();
        let workspace = parent.path().join("workspace");
        let outside = parent.path().join("outside");
        std::fs::create_dir(&workspace).unwrap();
        std::fs::create_dir(&outside).unwrap();
        std::fs::write(workspace.join(".seekr.toml"), "roots = [\"../outside\"]").unwrap();

        assert!(matches!(
            SeekrConfig::default().scan_config(&workspace),
            Err(ConfigError::InvalidValue { key, .. }) if key == "roots"
        ));
    }
}
