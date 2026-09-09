//! Shared long-lived project engines for server entry points.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;

use crate::config::SeekrConfig;
use crate::embedder::onnx::OnnxEmbedder;
use crate::embedder::traits::Embedder;
use crate::error::{IndexError, SeekrError, ServerError};
use crate::index::builder::{BuildStatus, IndexBuilder};
use crate::index::store::SeekrIndex;
use crate::search::engine::{SearchEngine, SearchOptions};
use crate::search::{SearchMode, SearchResult};

pub struct ProjectEngine {
    project_path: PathBuf,
    config: SeekrConfig,
    embedder: Arc<dyn Embedder>,
    index: Arc<RwLock<SeekrIndex>>,
    build_lock: Mutex<()>,
    has_persisted_index: AtomicBool,
}

#[derive(Debug)]
pub struct ProjectBuildReport {
    pub status: BuildStatus,
    pub project_path: PathBuf,
    pub index_dir: PathBuf,
    pub chunk_count: usize,
    pub embedding_dim: usize,
    pub files_found: usize,
    pub files_skipped: usize,
    pub files_parsed: usize,
    pub changed_files: usize,
    pub unchanged_files: usize,
    pub deleted_files: usize,
    pub duration: Duration,
}

impl ProjectEngine {
    fn load(project_path: PathBuf, config: SeekrConfig) -> Result<Self, SeekrError> {
        let embedder: Arc<dyn Embedder> = Arc::new(OnnxEmbedder::new(&config.model_dir)?);
        Self::load_with_embedder(project_path, config, embedder)
    }

    fn load_with_embedder(
        project_path: PathBuf,
        config: SeekrConfig,
        embedder: Arc<dyn Embedder>,
    ) -> Result<Self, SeekrError> {
        let index_dir = config.project_index_dir(&project_path);
        let (index, has_persisted_index) = match SeekrIndex::load(&index_dir) {
            Ok(index) => (index, true),
            Err(IndexError::NotFound(_)) => (SeekrIndex::new(embedder.dimension()), false),
            Err(error) => return Err(error.into()),
        };
        Ok(Self {
            project_path,
            config,
            embedder,
            index: Arc::new(RwLock::new(index)),
            build_lock: Mutex::new(()),
            has_persisted_index: AtomicBool::new(has_persisted_index),
        })
    }

    pub fn project_path(&self) -> &Path {
        &self.project_path
    }

    pub fn shared_index(&self) -> Arc<RwLock<SeekrIndex>> {
        Arc::clone(&self.index)
    }

    pub fn search(
        &self,
        query: &str,
        mode: SearchMode,
        options: &SearchOptions,
    ) -> Result<Vec<SearchResult>, SeekrError> {
        if !self.has_persisted_index.load(Ordering::Acquire) {
            return Err(
                IndexError::NotFound(self.config.project_index_dir(&self.project_path)).into(),
            );
        }
        let index = self
            .index
            .read()
            .map_err(|error| ServerError::Internal(format!("Index lock poisoned: {error}")))?;
        SearchEngine::new(self.config.search.clone(), Some(Arc::clone(&self.embedder)))
            .search(&index, query, mode, options)
            .map_err(Into::into)
    }

    pub fn build(&self, force: bool) -> Result<ProjectBuildReport, SeekrError> {
        let _build_guard = self
            .build_lock
            .lock()
            .map_err(|error| ServerError::Internal(format!("Build lock poisoned: {error}")))?;
        let report = IndexBuilder::new(self.config.clone(), Arc::clone(&self.embedder))
            .build(&self.project_path, force)?;
        let result = ProjectBuildReport {
            status: report.status,
            project_path: report.project_path,
            index_dir: report.index_dir,
            chunk_count: report.index.chunk_count,
            embedding_dim: report.index.embedding_dim,
            files_found: report.files_found,
            files_skipped: report.files_skipped,
            files_parsed: report.files_parsed,
            changed_files: report.changed_files,
            unchanged_files: report.unchanged_files,
            deleted_files: report.deleted_files,
            duration: report.duration,
        };
        let mut index = self
            .index
            .write()
            .map_err(|error| ServerError::Internal(format!("Index lock poisoned: {error}")))?;
        *index = report.index;
        self.has_persisted_index.store(true, Ordering::Release);
        Ok(result)
    }

    pub async fn search_async(
        self: Arc<Self>,
        query: String,
        mode: SearchMode,
        options: SearchOptions,
    ) -> Result<Vec<SearchResult>, ServerError> {
        tokio::task::spawn_blocking(move || self.search(&query, mode, &options))
            .await
            .map_err(|error| ServerError::Internal(format!("Search task failed: {error}")))?
            .map_err(|error| ServerError::Internal(error.to_string()))
    }

    pub async fn build_async(
        self: Arc<Self>,
        force: bool,
    ) -> Result<ProjectBuildReport, ServerError> {
        tokio::task::spawn_blocking(move || self.build(force))
            .await
            .map_err(|error| ServerError::Internal(format!("Index task failed: {error}")))?
            .map_err(|error| ServerError::Internal(error.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedder::batch::DummyEmbedder;

    #[test]
    fn build_replaces_shared_index_for_subsequent_searches() {
        let project = tempfile::tempdir().unwrap();
        let index_root = tempfile::tempdir().unwrap();
        let source = project.path().join("lib.rs");
        std::fs::write(&source, "pub fn before() {\n    println!(\"before\");\n}\n").unwrap();
        let config = SeekrConfig {
            index_dir: index_root.path().to_path_buf(),
            ..SeekrConfig::default()
        };
        let engine = ProjectEngine::load_with_embedder(
            project.path().to_path_buf(),
            config,
            Arc::new(DummyEmbedder::new(8)),
        )
        .unwrap();

        engine.build(false).unwrap();
        let before = engine
            .search("before", SearchMode::Text, &SearchOptions::default())
            .unwrap();
        assert!(!before.is_empty());

        std::fs::write(&source, "pub fn after() {\n    println!(\"after\");\n}\n").unwrap();
        engine.build(false).unwrap();
        let after = engine
            .search("after", SearchMode::Text, &SearchOptions::default())
            .unwrap();
        assert!(!after.is_empty());
    }
}

#[derive(Clone)]
pub struct EngineRegistry {
    config: SeekrConfig,
    engines: Arc<RwLock<HashMap<PathBuf, Arc<ProjectEngine>>>>,
}

impl EngineRegistry {
    pub fn new(config: SeekrConfig) -> Self {
        Self {
            config,
            engines: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub fn config(&self) -> &SeekrConfig {
        &self.config
    }

    pub fn get_or_create(&self, project_path: &Path) -> Result<Arc<ProjectEngine>, SeekrError> {
        let project_path = project_path
            .canonicalize()
            .unwrap_or_else(|_| project_path.to_path_buf());
        if let Some(engine) = self
            .engines
            .read()
            .map_err(|error| ServerError::Internal(format!("Registry lock poisoned: {error}")))?
            .get(&project_path)
            .cloned()
        {
            return Ok(engine);
        }

        let candidate = Arc::new(ProjectEngine::load(
            project_path.clone(),
            self.config.clone(),
        )?);
        let mut engines = self
            .engines
            .write()
            .map_err(|error| ServerError::Internal(format!("Registry lock poisoned: {error}")))?;
        Ok(Arc::clone(engines.entry(project_path).or_insert(candidate)))
    }

    pub async fn get_or_create_async(
        &self,
        project_path: PathBuf,
    ) -> Result<Arc<ProjectEngine>, ServerError> {
        let registry = self.clone();
        tokio::task::spawn_blocking(move || registry.get_or_create(&project_path))
            .await
            .map_err(|error| ServerError::Internal(format!("Engine load task failed: {error}")))?
            .map_err(|error| ServerError::Internal(error.to_string()))
    }
}
