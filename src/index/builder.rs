//! Unified full and incremental indexing application service.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::config::SeekrConfig;
use crate::embedder::batch::BatchEmbedder;
use crate::embedder::traits::Embedder;
use crate::error::{ParserError, ScannerError, SeekrError};
use crate::index::IndexEntry;
use crate::index::incremental::IncrementalState;
use crate::index::store::{SeekrIndex, tokenize_for_index_pub};
use crate::parser::CodeChunk;
use crate::parser::chunker::chunk_file;
use crate::parser::summary::generate_summary;
use crate::parser::treesitter::SupportedLanguage;
use crate::scanner::filter::should_index_file;
use crate::scanner::walker::walk_workspace_root;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuildStatus {
    Built,
    UpToDate,
    Empty,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuildStage {
    Scanning,
    Parsing,
    Embedding,
    Saving,
}

#[derive(Debug, Clone, Copy)]
pub struct BuildProgress {
    pub stage: BuildStage,
    pub completed: usize,
    pub total: usize,
}

#[derive(Debug)]
pub struct BuildReport {
    pub status: BuildStatus,
    pub project_path: PathBuf,
    pub index_dir: PathBuf,
    pub index: SeekrIndex,
    pub files_found: usize,
    pub files_skipped: usize,
    pub files_parsed: usize,
    pub changed_files: usize,
    pub unchanged_files: usize,
    pub deleted_files: usize,
    pub duration: Duration,
}

/// Owns the only scan → incremental → parse → embed → merge/save pipeline.
pub struct IndexBuilder {
    config: SeekrConfig,
    embedder: Arc<dyn Embedder>,
}

impl IndexBuilder {
    pub fn new(config: SeekrConfig, embedder: Arc<dyn Embedder>) -> Self {
        Self { config, embedder }
    }

    pub fn build(&self, project_path: &Path, force: bool) -> Result<BuildReport, SeekrError> {
        self.build_with_progress(project_path, force, |_| {})
    }

    pub fn build_with_progress<F>(
        &self,
        project_path: &Path,
        force: bool,
        mut progress: F,
    ) -> Result<BuildReport, SeekrError>
    where
        F: FnMut(BuildProgress),
    {
        let started = Instant::now();
        let project_path = project_path
            .canonicalize()
            .unwrap_or_else(|_| project_path.to_path_buf());
        let index_dir = self.config.project_index_dir(&project_path);
        let state_path = index_dir.join("incremental_state.json");

        progress(BuildProgress {
            stage: BuildStage::Scanning,
            completed: 0,
            total: 0,
        });
        let scan_config = self.config.scan_config(&project_path)?;
        let mut entries = BTreeMap::new();
        let mut files_skipped = 0;
        let mut scan_errors = 0;
        for root in &scan_config.roots {
            let scan_result = walk_workspace_root(root, &scan_config)?;
            files_skipped += scan_result.skipped;
            scan_errors += scan_result.errors;
            for entry in scan_result.entries {
                match entry.path.canonicalize() {
                    Ok(path) => {
                        entries.entry(path).or_insert(entry);
                    }
                    Err(_) => scan_errors += 1,
                }
            }
        }
        if scan_errors > 0 {
            return Err(ScannerError::Incomplete {
                errors: scan_errors,
            }
            .into());
        }
        let discovered_files = entries.len();
        let file_paths: Vec<PathBuf> = entries
            .into_iter()
            .filter(|(path, entry)| {
                should_index_file(path, entry.size, scan_config.max_file_size)
                    && SupportedLanguage::from_path(path).is_some_and(|language| {
                        scan_config.languages.is_empty()
                            || scan_config.languages.contains(language.name())
                    })
            })
            .map(|(path, _)| path)
            .collect();
        files_skipped += discovered_files.saturating_sub(file_paths.len());
        let files_found = file_paths.len();

        let has_index =
            index_dir.join("index.bin").exists() || index_dir.join("index.json").exists();
        // Probe every existing index before deciding whether incremental state
        // is usable. This guarantees legacy formats always require --force.
        let loaded_index = if !force && has_index {
            Some(SeekrIndex::load(&index_dir)?)
        } else {
            None
        };
        let loaded_index = match loaded_index {
            Some(index) if !index.mentions_valid() => {
                tracing::warn!(
                    path = %index_dir.display(),
                    "Mentions sidecar missing or invalid; rebuilding index"
                );
                None
            }
            other => other,
        };
        let incremental = loaded_index.is_some() && state_path.exists();
        let index = loaded_index;
        let mut state = if incremental {
            IncrementalState::load(&state_path)?
        } else {
            IncrementalState::default()
        };

        // Change detection reads immutable snapshots. Parsing and state
        // updates below use those same bytes, avoiding stale TOCTOU commits.
        let changes = state.detect_changes(&file_paths)?;
        let files_to_process = &changes.changed;

        if incremental && files_to_process.is_empty() && changes.deleted.is_empty() {
            return Ok(BuildReport {
                status: BuildStatus::UpToDate,
                project_path,
                index_dir,
                index: index.expect("incremental mode always loads an index"),
                files_found,
                files_skipped,
                files_parsed: 0,
                changed_files: 0,
                unchanged_files: changes.unchanged.len(),
                deleted_files: 0,
                duration: started.elapsed(),
            });
        }

        struct ParsedFile {
            path: PathBuf,
            content_hash: String,
            mtime: std::time::SystemTime,
            chunks: Vec<CodeChunk>,
            call_sites: Vec<crate::parser::CallSite>,
        }

        // Complete all fallible parse work before mutating the old index.
        let mut parsed_files = Vec::with_capacity(files_to_process.len());
        let mut files_parsed = 0;
        for (position, snapshot) in files_to_process.iter().enumerate() {
            let language = SupportedLanguage::from_path(&snapshot.path).ok_or_else(|| {
                ParserError::UnsupportedLanguage(snapshot.path.display().to_string())
            })?;
            let (chunks, call_sites) =
                if crate::scanner::filter::is_binary_content(&snapshot.content) {
                    (Vec::new(), Vec::new())
                } else {
                    let source = std::str::from_utf8(&snapshot.content).map_err(|error| {
                        ParserError::ParseFailed {
                            path: snapshot.path.clone(),
                            reason: format!("source is not valid UTF-8: {error}"),
                        }
                    })?;
                    let result = chunk_file(&snapshot.path, source, language)?;
                    files_parsed += 1;
                    (result.chunks, result.call_sites)
                };
            parsed_files.push(ParsedFile {
                path: snapshot.path.clone(),
                content_hash: snapshot.content_hash.clone(),
                mtime: snapshot.mtime,
                chunks,
                call_sites,
            });
            progress(BuildProgress {
                stage: BuildStage::Parsing,
                completed: position + 1,
                total: files_to_process.len(),
            });
        }

        let chunks: Vec<&CodeChunk> = parsed_files
            .iter()
            .flat_map(|file| file.chunks.iter())
            .collect();
        let summaries: Vec<String> = chunks.iter().map(|chunk| generate_summary(chunk)).collect();
        let embeddings = if summaries.is_empty() {
            Vec::new()
        } else {
            BatchEmbedder::new(Arc::clone(&self.embedder), self.config.embedding.batch_size)
                .embed_all_with_progress(&summaries, |completed, total| {
                    progress(BuildProgress {
                        stage: BuildStage::Embedding,
                        completed,
                        total,
                    });
                })?
        };
        let embedding_dim = index
            .as_ref()
            .map(|index| index.embedding_dim())
            .unwrap_or_else(|| self.embedder.dimension());

        // Parsing and embedding succeeded for every changed file. It is now
        // safe to replace old per-file data and advance incremental state.
        let mut index = index.unwrap_or_else(|| SeekrIndex::new(embedding_dim));
        let removed_ids = state.apply_deletions(&changes.deleted);
        index.remove_chunks(&removed_ids);
        for file_path in &changes.deleted {
            index.remove_file_call_sites(file_path);
        }
        for file in &parsed_files {
            index.remove_chunks(&state.chunk_ids_for_file(&file.path));
            index.remove_file_call_sites(&file.path);
        }
        for (chunk, embedding) in chunks.iter().zip(&embeddings) {
            index.try_add_entry(index_entry(chunk, embedding), (*chunk).clone())?;
        }
        index.rebuild_hnsw();
        index.set_format_version(crate::INDEX_VERSION);

        for file in parsed_files {
            let chunk_ids = file.chunks.iter().map(|chunk| chunk.id).collect();
            index.replace_file_call_sites(&file.path, file.call_sites);
            state.update_snapshot(file.path, file.mtime, file.content_hash, chunk_ids);
        }

        progress(BuildProgress {
            stage: BuildStage::Saving,
            completed: 0,
            total: 1,
        });
        index.save(&index_dir)?;
        state.save(&state_path)?;
        progress(BuildProgress {
            stage: BuildStage::Saving,
            completed: 1,
            total: 1,
        });

        let status = if index.chunk_count() == 0 {
            BuildStatus::Empty
        } else {
            BuildStatus::Built
        };
        Ok(BuildReport {
            status,
            project_path,
            index_dir,
            index,
            files_found,
            files_skipped,
            files_parsed,
            changed_files: files_to_process.len(),
            unchanged_files: changes.unchanged.len(),
            deleted_files: changes.deleted.len(),
            duration: started.elapsed(),
        })
    }
}

fn index_entry(chunk: &CodeChunk, embedding: &[f32]) -> IndexEntry {
    IndexEntry {
        chunk_id: chunk.id,
        embedding: embedding.to_vec(),
        text_tokens: tokenize_for_index_pub(&chunk.body),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedder::batch::DummyEmbedder;

    #[test]
    fn incremental_build_reports_up_to_date_and_tracks_deletions() {
        let project = tempfile::tempdir().unwrap();
        let index_root = tempfile::tempdir().unwrap();
        let source = project.path().join("lib.rs");
        std::fs::write(&source, "pub fn answer() -> u32 {\n    42\n}\n").unwrap();

        let mut config = SeekrConfig {
            index_dir: index_root.path().to_path_buf(),
            ..SeekrConfig::default()
        };
        config.embedding.batch_size = 2;
        let builder = IndexBuilder::new(config, Arc::new(DummyEmbedder::new(8)));

        let first = builder.build(project.path(), false).unwrap();
        assert_eq!(first.status, BuildStatus::Built);
        assert!(first.index.chunk_count() > 0);

        let second = builder.build(project.path(), false).unwrap();
        assert_eq!(second.status, BuildStatus::UpToDate);

        std::fs::remove_file(source).unwrap();
        let third = builder.build(project.path(), false).unwrap();
        assert_eq!(third.status, BuildStatus::Empty);
        assert_eq!(third.deleted_files, 1);
        assert_eq!(third.index.chunk_count(), 0);
    }

    #[test]
    fn parse_failure_preserves_previous_index_and_retries() {
        let project = tempfile::tempdir().unwrap();
        let index_root = tempfile::tempdir().unwrap();
        let source = project.path().join("lib.rs");
        std::fs::write(&source, "pub fn durable() -> u32 {\n    42\n}\n").unwrap();
        let config = SeekrConfig {
            index_dir: index_root.path().to_path_buf(),
            ..SeekrConfig::default()
        };
        let builder = IndexBuilder::new(config.clone(), Arc::new(DummyEmbedder::new(8)));
        let first = builder.build(project.path(), false).unwrap();
        assert!(!first.index.symbol_definitions("durable").is_empty());

        std::fs::write(&source, [0xff, 0xfe, 0xfd]).unwrap();
        assert!(builder.build(project.path(), false).is_err());

        let preserved = SeekrIndex::load(&config.project_index_dir(project.path())).unwrap();
        assert_eq!(preserved.symbol_definitions("durable").len(), 1);

        // The failed bytes were not committed to incremental state, so a
        // subsequent valid write is processed rather than reported up-to-date.
        std::fs::write(&source, "pub fn recovered() -> u32 {\n    7\n}\n").unwrap();
        let recovered = builder.build(project.path(), false).unwrap();
        assert_eq!(recovered.index.symbol_definitions("recovered").len(), 1);
        assert!(recovered.index.symbol_definitions("durable").is_empty());
    }

    #[test]
    fn missing_mentions_sidecar_triggers_self_healing_rebuild() {
        let project = tempfile::tempdir().unwrap();
        let index_root = tempfile::tempdir().unwrap();
        std::fs::write(
            project.path().join("lib.rs"),
            "pub fn helper() {\n    println!(\"ok\");\n}\npub fn caller() {\n    helper();\n}\n",
        )
        .unwrap();
        let config = SeekrConfig {
            index_dir: index_root.path().to_path_buf(),
            ..SeekrConfig::default()
        };
        let builder = IndexBuilder::new(config.clone(), Arc::new(DummyEmbedder::new(8)));
        let first = builder.build(project.path(), false).unwrap();
        assert!(!first.index.callers("helper").is_empty());

        let index_dir = config.project_index_dir(project.path());
        std::fs::remove_file(index_dir.join("mentions.bin")).unwrap();
        let healed = builder.build(project.path(), false).unwrap();
        assert_eq!(healed.status, BuildStatus::Built);
        assert!(!healed.index.callers("helper").is_empty());
        assert!(index_dir.join("mentions.bin").exists());
    }

    #[test]
    fn builds_deduplicated_multi_root_workspace_with_language_filter() {
        let project = tempfile::tempdir().unwrap();
        let index_root = tempfile::tempdir().unwrap();
        let root_a = project.path().join("root-a");
        let nested = root_a.join("nested");
        let root_b = project.path().join("root-b");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::create_dir(&root_b).unwrap();
        std::fs::write(root_a.join("a.rs"), "fn alpha() {}").unwrap();
        std::fs::write(nested.join("nested.rs"), "fn nested() {}").unwrap();
        std::fs::write(root_b.join("b.py"), "def beta():\n    pass\n").unwrap();
        std::fs::write(root_b.join("ignored.js"), "function ignored() {}").unwrap();
        std::fs::write(
            project.path().join(".seekr.toml"),
            r#"
roots = ["root-a", "root-a/nested", "root-b"]
languages = ["rust", "python"]
"#,
        )
        .unwrap();
        let config = SeekrConfig {
            index_dir: index_root.path().to_path_buf(),
            ..SeekrConfig::default()
        };
        let builder = IndexBuilder::new(config, Arc::new(DummyEmbedder::new(8)));

        let report = builder.build(project.path(), false).unwrap();
        let indexed_paths: std::collections::HashSet<_> = report
            .index
            .iter_chunks()
            .map(|(_, chunk)| chunk.file_path.clone())
            .collect();

        assert_eq!(report.files_found, 3);
        assert_eq!(indexed_paths.len(), 3);
        assert!(indexed_paths.iter().all(|path| {
            matches!(
                SupportedLanguage::from_path(path),
                Some(SupportedLanguage::Rust | SupportedLanguage::Python)
            )
        }));
    }
}
