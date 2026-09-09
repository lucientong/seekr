//! Unified full and incremental indexing application service.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::config::SeekrConfig;
use crate::embedder::batch::BatchEmbedder;
use crate::embedder::traits::Embedder;
use crate::error::SeekrError;
use crate::index::IndexEntry;
use crate::index::incremental::IncrementalState;
use crate::index::store::{SeekrIndex, tokenize_for_index_pub};
use crate::parser::CodeChunk;
use crate::parser::chunker::chunk_file_from_path;
use crate::parser::summary::generate_summary;
use crate::scanner::filter::should_index_file;
use crate::scanner::walker::walk_directory;

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
        let scan_result = walk_directory(&project_path, &self.config)?;
        let file_paths: Vec<PathBuf> = scan_result
            .entries
            .iter()
            .filter(|entry| should_index_file(&entry.path, entry.size, self.config.max_file_size))
            .map(|entry| entry.path.clone())
            .collect();
        let files_found = file_paths.len();

        let has_index =
            index_dir.join("index.bin").exists() || index_dir.join("index.json").exists();
        let incremental = !force && has_index && state_path.exists();
        let (mut index, mut state) = if incremental {
            (
                Some(SeekrIndex::load(&index_dir)?),
                IncrementalState::load(&state_path)?,
            )
        } else {
            (None, IncrementalState::default())
        };

        let changes = state.detect_changes(&file_paths);
        let files_to_process = if incremental {
            changes.changed.clone()
        } else {
            file_paths.clone()
        };

        if incremental && files_to_process.is_empty() && changes.deleted.is_empty() {
            return Ok(BuildReport {
                status: BuildStatus::UpToDate,
                project_path,
                index_dir,
                index: index.expect("incremental mode always loads an index"),
                files_found,
                files_skipped: scan_result.skipped,
                files_parsed: 0,
                changed_files: 0,
                unchanged_files: changes.unchanged.len(),
                deleted_files: 0,
                duration: started.elapsed(),
            });
        }

        if let Some(index) = index.as_mut() {
            let removed_ids = state.apply_deletions(&changes.deleted);
            index.remove_chunks(&removed_ids);
            for file_path in &files_to_process {
                index.remove_chunks(&state.chunk_ids_for_file(file_path));
            }
        }

        let mut chunks = Vec::new();
        let mut files_parsed = 0;
        for (position, file_path) in files_to_process.iter().enumerate() {
            match chunk_file_from_path(file_path) {
                Ok(Some(parse_result)) => {
                    chunks.extend(parse_result.chunks);
                    files_parsed += 1;
                }
                Ok(None) => {}
                Err(error) => {
                    tracing::debug!(
                        path = %file_path.display(),
                        %error,
                        "Failed to parse file"
                    );
                }
            }
            progress(BuildProgress {
                stage: BuildStage::Parsing,
                completed: position + 1,
                total: files_to_process.len(),
            });
        }

        let summaries: Vec<String> = chunks.iter().map(generate_summary).collect();
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
            .map(|index| index.embedding_dim)
            .unwrap_or_else(|| self.embedder.dimension());

        let mut index = if let Some(mut index) = index {
            for (chunk, embedding) in chunks.iter().zip(&embeddings) {
                index.try_add_entry(index_entry(chunk, embedding), chunk.clone())?;
            }
            index.rebuild_hnsw();
            index
        } else {
            SeekrIndex::try_build_from(&chunks, &embeddings, embedding_dim)?
        };
        index.version = crate::INDEX_VERSION;

        for file_path in &files_to_process {
            let chunk_ids = chunks
                .iter()
                .filter(|chunk| chunk.file_path == *file_path)
                .map(|chunk| chunk.id)
                .collect();
            let content = std::fs::read(file_path)?;
            state.update_file(file_path.clone(), &content, chunk_ids);
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

        let status = if index.chunk_count == 0 {
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
            files_skipped: scan_result.skipped,
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
        assert!(first.index.chunk_count > 0);

        let second = builder.build(project.path(), false).unwrap();
        assert_eq!(second.status, BuildStatus::UpToDate);

        std::fs::remove_file(source).unwrap();
        let third = builder.build(project.path(), false).unwrap();
        assert_eq!(third.status, BuildStatus::Empty);
        assert_eq!(third.deleted_files, 1);
        assert_eq!(third.index.chunk_count, 0);
    }
}
