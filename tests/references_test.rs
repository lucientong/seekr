//! Name-level references / callers quality and integration checks.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use seekr_code::config::SeekrConfig;
use seekr_code::embedder::traits::Embedder;
use seekr_code::error::EmbedderError;
use seekr_code::index::builder::IndexBuilder;
use seekr_code::index::store::SeekrIndex;

struct DummyEmbedder {
    dim: usize,
}

impl DummyEmbedder {
    fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl Embedder for DummyEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedderError> {
        let mut embedding = vec![0.0; self.dim];
        for (index, byte) in text.bytes().enumerate() {
            embedding[index % self.dim] += byte as f32;
        }
        let norm = embedding.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 0.0 {
            for value in &mut embedding {
                *value /= norm;
            }
        }
        Ok(embedding)
    }

    fn dimension(&self) -> usize {
        self.dim
    }
}

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn build_project(project: &Path, index_root: &Path) -> SeekrConfig {
    let config = SeekrConfig {
        index_dir: index_root.to_path_buf(),
        ..SeekrConfig::default()
    };
    let embedder: Arc<dyn Embedder> = Arc::new(DummyEmbedder::new(8));
    IndexBuilder::new(config.clone(), embedder)
        .build(project, true)
        .unwrap();
    config
}

#[test]
fn references_and_callers_for_rust_authenticate() {
    let dir = tempfile::tempdir().unwrap();
    let project = dir.path().join("project");
    std::fs::create_dir_all(project.join("src")).unwrap();
    for name in ["sample.rs", "sample.py", "sample.ts"] {
        std::fs::copy(fixtures_dir().join(name), project.join("src").join(name)).unwrap();
    }
    let config = build_project(&project, &dir.path().join("indexes"));
    let index_dir = config.project_index_dir(&project);
    let index = SeekrIndex::load(&index_dir).unwrap();

    let callers = index.callers("find_user_by_name");
    assert!(!callers.is_empty(), "expected callers of find_user_by_name");
    assert!(
        callers
            .iter()
            .any(|hit| hit.mention.callee_name == "find_user_by_name")
    );
    assert!(
        callers
            .iter()
            .all(|hit| hit.disclaimer.contains("not compiler/LSP"))
    );
    assert!(callers.iter().all(|hit| !hit.reasons.is_empty()));
}

#[test]
fn ambiguous_same_name_reduces_confidence() {
    let dir = tempfile::tempdir().unwrap();
    let project = dir.path().join("project");
    std::fs::create_dir_all(&project).unwrap();
    std::fs::write(
        project.join("a.rs"),
        "fn helper() {\n    let x = 1;\n}\nfn one() {\n    helper();\n}\n",
    )
    .unwrap();
    std::fs::write(
        project.join("b.rs"),
        "fn helper() {\n    let y = 2;\n}\nfn two() {\n    helper();\n}\n",
    )
    .unwrap();

    let config = build_project(&project, &dir.path().join("indexes"));
    let index = SeekrIndex::load(&config.project_index_dir(&project)).unwrap();
    let hits = index.references("helper");
    assert!(hits.len() >= 2);
    assert!(
        hits.iter()
            .any(|hit| hit.reasons.iter().any(|r| r == "ambiguous_definitions"))
    );
}

#[test]
fn incremental_delete_clears_call_sites() {
    let dir = tempfile::tempdir().unwrap();
    let project = dir.path().join("project");
    std::fs::create_dir_all(&project).unwrap();
    let source = project.join("main.rs");
    std::fs::write(
        &source,
        "fn helper() {\n    let x = 1;\n}\nfn main() {\n    helper();\n}\n",
    )
    .unwrap();

    let config = SeekrConfig {
        index_dir: dir.path().join("indexes"),
        ..SeekrConfig::default()
    };
    let embedder: Arc<dyn Embedder> = Arc::new(DummyEmbedder::new(8));
    IndexBuilder::new(config.clone(), Arc::clone(&embedder))
        .build(&project, true)
        .unwrap();

    let index_dir = config.project_index_dir(&project);
    let index = SeekrIndex::load(&index_dir).unwrap();
    assert!(!index.callers("helper").is_empty());

    std::fs::remove_file(&source).unwrap();
    IndexBuilder::new(config, embedder)
        .build(&project, false)
        .unwrap();
    let index = SeekrIndex::load(&index_dir).unwrap();
    assert!(index.callers("helper").is_empty());
}

#[test]
fn call_site_ids_stable_across_rebuild() {
    let dir = tempfile::tempdir().unwrap();
    let project = dir.path().join("project");
    std::fs::create_dir_all(&project).unwrap();
    std::fs::write(
        project.join("main.rs"),
        "fn main() {\n    foo();\n}\nfn foo() {\n    let x = 1;\n}\n",
    )
    .unwrap();

    let config = SeekrConfig {
        index_dir: dir.path().join("indexes"),
        ..SeekrConfig::default()
    };
    let embedder: Arc<dyn Embedder> = Arc::new(DummyEmbedder::new(8));
    IndexBuilder::new(config.clone(), Arc::clone(&embedder))
        .build(&project, true)
        .unwrap();
    let index_dir = config.project_index_dir(&project);
    let first: Vec<u64> = SeekrIndex::load(&index_dir)
        .unwrap()
        .callers("foo")
        .into_iter()
        .map(|hit| hit.mention.id)
        .collect();
    IndexBuilder::new(config, embedder)
        .build(&project, true)
        .unwrap();
    let second: Vec<u64> = SeekrIndex::load(&index_dir)
        .unwrap()
        .callers("foo")
        .into_iter()
        .map(|hit| hit.mention.id)
        .collect();
    assert_eq!(first, second);
    assert!(!first.is_empty());
}
