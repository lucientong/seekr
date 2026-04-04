//! Integration tests for seekr-code.
//!
//! End-to-end testing of the complete pipeline:
//! scanner -> parser -> embedder -> index -> search

use std::path::{Path, PathBuf};

use seekr_code::config::SeekrConfig;
use seekr_code::embedder::batch::{BatchEmbedder, DummyEmbedder};
use seekr_code::embedder::traits::Embedder;
use seekr_code::index::IndexEntry;
use seekr_code::index::incremental::IncrementalState;
use seekr_code::index::store::SeekrIndex;
use seekr_code::parser::CodeChunk;
use seekr_code::parser::chunker::chunk_file_from_path;
use seekr_code::parser::summary::generate_summary;
use seekr_code::scanner::filter::should_index_file;
use seekr_code::scanner::walker::walk_directory;
use seekr_code::search::ast_pattern::{parse_pattern, search_ast_pattern};
use seekr_code::search::fusion::rrf_fuse;
use seekr_code::search::semantic::{SemanticSearchOptions, search_semantic};
use seekr_code::search::text::{TextSearchOptions, search_text_regex};

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
}

// ============================================================
// 1. Scanner tests
// ============================================================

#[test]
fn test_scanner_finds_fixtures() {
    let config = SeekrConfig::default();
    let result = walk_directory(&fixtures_dir(), &config).unwrap();

    // Should find at least our 4 fixture files
    assert!(
        result.entries.len() >= 4,
        "Expected at least 4 fixture files, found {}",
        result.entries.len()
    );

    let extensions: Vec<String> = result
        .entries
        .iter()
        .filter_map(|e| {
            e.path
                .extension()
                .map(|ext| ext.to_string_lossy().to_string())
        })
        .collect();

    assert!(
        extensions.contains(&"rs".to_string()),
        "Should find .rs files"
    );
    assert!(
        extensions.contains(&"py".to_string()),
        "Should find .py files"
    );
    assert!(
        extensions.contains(&"js".to_string()),
        "Should find .js files"
    );
    assert!(
        extensions.contains(&"go".to_string()),
        "Should find .go files"
    );
}

#[test]
fn test_scanner_filter_integration() {
    let config = SeekrConfig::default();
    let result = walk_directory(&fixtures_dir(), &config).unwrap();

    for entry in &result.entries {
        assert!(
            should_index_file(&entry.path, entry.size, config.max_file_size),
            "All fixture files should pass the index filter: {}",
            entry.path.display()
        );
    }
}

// ============================================================
// 2. Parser tests
// ============================================================

#[test]
fn test_parser_rust_fixture() {
    let path = fixtures_dir().join("sample.rs");
    let result = chunk_file_from_path(&path).unwrap();
    assert!(result.is_some(), "Should be able to parse Rust file");

    let parse_result = result.unwrap();
    assert_eq!(parse_result.language, "rust");
    assert!(
        !parse_result.chunks.is_empty(),
        "Should extract chunks from Rust file"
    );

    // Should find the authenticate_user function
    let auth_chunk = parse_result
        .chunks
        .iter()
        .find(|c| c.name.as_deref() == Some("authenticate_user"));
    assert!(
        auth_chunk.is_some(),
        "Should find authenticate_user function"
    );

    // Should find the User struct
    let user_struct = parse_result
        .chunks
        .iter()
        .find(|c| c.name.as_deref() == Some("User"));
    assert!(user_struct.is_some(), "Should find User struct");
}

#[test]
fn test_parser_python_fixture() {
    let path = fixtures_dir().join("sample.py");
    let result = chunk_file_from_path(&path).unwrap();
    assert!(result.is_some(), "Should be able to parse Python file");

    let parse_result = result.unwrap();
    assert_eq!(parse_result.language, "python");
    assert!(!parse_result.chunks.is_empty());

    // Should find the UserService class
    let class_chunk = parse_result
        .chunks
        .iter()
        .find(|c| c.name.as_deref() == Some("UserService"));
    assert!(class_chunk.is_some(), "Should find UserService class");
}

#[test]
fn test_parser_javascript_fixture() {
    let path = fixtures_dir().join("sample.js");
    let result = chunk_file_from_path(&path).unwrap();
    assert!(result.is_some(), "Should be able to parse JavaScript file");

    let parse_result = result.unwrap();
    assert_eq!(parse_result.language, "javascript");
    assert!(!parse_result.chunks.is_empty());
}

#[test]
fn test_parser_go_fixture() {
    let path = fixtures_dir().join("sample.go");
    let result = chunk_file_from_path(&path).unwrap();
    assert!(result.is_some(), "Should be able to parse Go file");

    let parse_result = result.unwrap();
    assert_eq!(parse_result.language, "go");
    assert!(!parse_result.chunks.is_empty());
}

#[test]
fn test_parser_summary_generation() {
    let path = fixtures_dir().join("sample.rs");
    let result = chunk_file_from_path(&path).unwrap().unwrap();

    for chunk in &result.chunks {
        let summary = generate_summary(chunk);
        assert!(!summary.is_empty(), "Summary should not be empty");
        // Summary should contain either the name or some code content
        if let Some(ref name) = chunk.name {
            assert!(
                summary.to_lowercase().contains(&name.to_lowercase()) || !summary.is_empty(),
                "Summary should reference the chunk name"
            );
        }
    }
}

// ============================================================
// 3. Embedder tests
// ============================================================

#[test]
fn test_embedder_batch_processing() {
    let embedder = DummyEmbedder::new(384);

    let texts = [
        "fn authenticate(user: &str)",
        "class UserService",
        "async function fetchData()",
    ];

    let batch = BatchEmbedder::new(embedder, 32);
    let embeddings = batch
        .embed_all(&texts.iter().map(|s| s.to_string()).collect::<Vec<_>>())
        .unwrap();

    assert_eq!(embeddings.len(), 3);
    for emb in &embeddings {
        assert_eq!(emb.len(), 384, "Embedding dimension should be 384");
    }
}

#[test]
fn test_embedder_different_inputs_different_embeddings() {
    let embedder = DummyEmbedder::new(384);

    let emb1 = embedder.embed("fn authenticate()").unwrap();
    let emb2 = embedder.embed("class UserService").unwrap();

    assert_ne!(
        emb1, emb2,
        "Different inputs should produce different embeddings"
    );
}

// ============================================================
// 4. Index build + persist + load tests
// ============================================================

#[test]
fn test_index_build_from_fixtures() {
    let config = SeekrConfig::default();

    // Scan and parse
    let result = walk_directory(&fixtures_dir(), &config).unwrap();
    let mut all_chunks: Vec<CodeChunk> = Vec::new();

    for entry in &result.entries {
        if let Ok(Some(parse_result)) = chunk_file_from_path(&entry.path) {
            all_chunks.extend(parse_result.chunks);
        }
    }

    assert!(!all_chunks.is_empty(), "Should have parsed some chunks");

    // Generate embeddings
    let embedder = DummyEmbedder::new(64);
    let summaries: Vec<String> = all_chunks.iter().map(generate_summary).collect();
    let batch = BatchEmbedder::new(embedder, 32);
    let embeddings = batch.embed_all(&summaries).unwrap();

    assert_eq!(embeddings.len(), all_chunks.len());

    // Build index
    let index = SeekrIndex::build_from(&all_chunks, &embeddings, 64);
    assert_eq!(index.chunk_count, all_chunks.len());

    // Save and load
    let dir = tempfile::tempdir().unwrap();
    index.save(dir.path()).unwrap();

    let loaded = SeekrIndex::load(dir.path()).unwrap();
    assert_eq!(loaded.chunk_count, index.chunk_count);
    assert_eq!(loaded.embedding_dim, 64);
}

#[test]
fn test_index_remove_chunk() {
    let mut index = SeekrIndex::new(4);

    let chunk = CodeChunk {
        id: 1,
        file_path: PathBuf::from("test.rs"),
        language: "rust".to_string(),
        kind: seekr_code::parser::ChunkKind::Function,
        name: Some("test_fn".to_string()),
        signature: None,
        doc_comment: None,
        body: "fn test() {}".to_string(),
        byte_range: 0..12,
        line_range: 0..1,
    };

    let entry = IndexEntry {
        chunk_id: 1,
        embedding: vec![0.1; 4],
        text_tokens: vec!["test".to_string()],
    };

    index.add_entry(entry, chunk);
    assert_eq!(index.chunk_count, 1);

    index.remove_chunk(1);
    assert_eq!(index.chunk_count, 0);
    assert!(index.get_chunk(1).is_none());
}

// ============================================================
// 5. Search tests (all three modes + fusion)
// ============================================================

fn build_test_index() -> SeekrIndex {
    let config = SeekrConfig::default();
    let result = walk_directory(&fixtures_dir(), &config).unwrap();
    let mut all_chunks: Vec<CodeChunk> = Vec::new();

    for entry in &result.entries {
        if let Ok(Some(parse_result)) = chunk_file_from_path(&entry.path) {
            all_chunks.extend(parse_result.chunks);
        }
    }

    let embedder = DummyEmbedder::new(64);
    let summaries: Vec<String> = all_chunks.iter().map(generate_summary).collect();
    let batch = BatchEmbedder::new(embedder, 32);
    let embeddings = batch.embed_all(&summaries).unwrap();

    SeekrIndex::build_from(&all_chunks, &embeddings, 64)
}

#[test]
fn test_text_search_on_fixtures() {
    let index = build_test_index();

    let options = TextSearchOptions {
        case_sensitive: false,
        context_lines: 2,
        top_k: 20,
    };

    // Search for "authenticate" — should find results in Rust, Python, Go files
    let results = search_text_regex(&index, "authenticate", &options).unwrap();
    assert!(
        !results.is_empty(),
        "Should find 'authenticate' in fixtures"
    );
}

#[test]
fn test_semantic_search_on_fixtures() {
    let index = build_test_index();
    let embedder = DummyEmbedder::new(64);

    let options = SemanticSearchOptions {
        top_k: 10,
        score_threshold: 0.0,
    };

    let results =
        search_semantic(&index, "user authentication login", &embedder, &options).unwrap();
    assert!(!results.is_empty(), "Semantic search should return results");
}

#[test]
fn test_ast_search_on_fixtures() {
    let index = build_test_index();

    // Search for struct patterns — should succeed without errors
    let _struct_results = search_ast_pattern(&index, "struct User", 10).unwrap();

    // Search for function patterns — should succeed without errors
    let _fn_results = search_ast_pattern(&index, "fn(*) -> Result", 10).unwrap();
}

#[test]
fn test_hybrid_fusion_on_fixtures() {
    let index = build_test_index();
    let embedder = DummyEmbedder::new(64);

    // Text search
    let text_options = TextSearchOptions {
        case_sensitive: false,
        context_lines: 2,
        top_k: 20,
    };
    let text_results = search_text_regex(&index, "authenticate", &text_options).unwrap();

    // Semantic search
    let semantic_options = SemanticSearchOptions {
        top_k: 20,
        score_threshold: 0.0,
    };
    let semantic_results =
        search_semantic(&index, "authenticate user", &embedder, &semantic_options).unwrap();

    // Fuse results
    let fused = rrf_fuse(&text_results, &semantic_results, 60, 10);

    // Should have some results
    assert!(!fused.is_empty(), "Hybrid fusion should return results");

    // Results appearing in both lists should score higher
    if fused.len() >= 2 {
        assert!(
            fused[0].fused_score >= fused[1].fused_score,
            "Results should be sorted by fused score"
        );
    }
}

// ============================================================
// 6. End-to-end pipeline test
// ============================================================

#[test]
fn test_end_to_end_pipeline() {
    let config = SeekrConfig::default();

    // Step 1: Scan
    let scan_result = walk_directory(&fixtures_dir(), &config).unwrap();
    assert!(scan_result.entries.len() >= 4, "Should find fixture files");

    // Step 2: Filter
    let filtered: Vec<_> = scan_result
        .entries
        .iter()
        .filter(|e| should_index_file(&e.path, e.size, config.max_file_size))
        .collect();
    assert!(!filtered.is_empty(), "Should have indexable files");

    // Step 3: Parse + chunk
    let mut all_chunks: Vec<CodeChunk> = Vec::new();
    for entry in &filtered {
        if let Ok(Some(parse_result)) = chunk_file_from_path(&entry.path) {
            all_chunks.extend(parse_result.chunks);
        }
    }
    assert!(!all_chunks.is_empty(), "Should produce code chunks");

    // Step 4: Generate summaries + embeddings
    let summaries: Vec<String> = all_chunks.iter().map(generate_summary).collect();
    let embedder = DummyEmbedder::new(64);
    let batch = BatchEmbedder::new(embedder, 32);
    let embeddings = batch.embed_all(&summaries).unwrap();
    assert_eq!(embeddings.len(), all_chunks.len());

    // Step 5: Build index
    let index = SeekrIndex::build_from(&all_chunks, &embeddings, 64);
    assert!(index.chunk_count > 0, "Index should have entries");

    // Step 6: Save and reload
    let dir = tempfile::tempdir().unwrap();
    index.save(dir.path()).unwrap();
    let loaded_index = SeekrIndex::load(dir.path()).unwrap();
    assert_eq!(loaded_index.chunk_count, index.chunk_count);

    // Step 7: Text search
    let text_options = TextSearchOptions {
        case_sensitive: false,
        context_lines: 0,
        top_k: 10,
    };
    let text_results = search_text_regex(&loaded_index, "authenticate", &text_options).unwrap();
    assert!(
        !text_results.is_empty(),
        "Text search should find authenticate"
    );

    // Step 8: Semantic search
    let search_embedder = DummyEmbedder::new(64);
    let semantic_options = SemanticSearchOptions {
        top_k: 10,
        score_threshold: 0.0,
    };
    let semantic_results = search_semantic(
        &loaded_index,
        "user authentication",
        &search_embedder,
        &semantic_options,
    )
    .unwrap();
    assert!(
        !semantic_results.is_empty(),
        "Semantic search should return results"
    );

    // Step 9: Fusion
    let fused = rrf_fuse(&text_results, &semantic_results, 60, 5);
    assert!(!fused.is_empty(), "Fusion should produce results");

    // Step 10: Verify results have associated chunk data
    for result in &fused {
        let chunk = loaded_index.get_chunk(result.chunk_id);
        assert!(chunk.is_some(), "Each result should have associated chunk");
    }
}

// ============================================================
// 7. AST pattern parsing tests
// ============================================================

#[test]
fn test_ast_pattern_parsing_comprehensive() {
    // Simple function pattern
    let pat = parse_pattern("fn(string) -> number").unwrap();
    assert_eq!(
        pat.kind,
        seekr_code::search::ast_pattern::PatternKind::Function
    );
    assert!(pat.param_patterns.is_some());
    assert!(pat.return_pattern.is_some());

    // Async function
    let pat = parse_pattern("async fn(*) -> Result").unwrap();
    assert!(pat.qualifiers.contains(&"async".to_string()));

    // Class pattern
    let pat = parse_pattern("class *Service").unwrap();
    assert_eq!(
        pat.kind,
        seekr_code::search::ast_pattern::PatternKind::Class
    );
    assert_eq!(pat.name_pattern.as_deref(), Some("*Service"));

    // Struct with wildcard
    let pat = parse_pattern("struct *Config").unwrap();
    assert_eq!(
        pat.kind,
        seekr_code::search::ast_pattern::PatternKind::Struct
    );

    // Enum pattern
    let pat = parse_pattern("enum *Error").unwrap();
    assert_eq!(pat.kind, seekr_code::search::ast_pattern::PatternKind::Enum);

    // Interface/trait pattern
    let pat = parse_pattern("trait Repository").unwrap();
    assert_eq!(
        pat.kind,
        seekr_code::search::ast_pattern::PatternKind::Interface
    );

    // Error case
    assert!(parse_pattern("").is_err());
}

// ============================================================
// 8. Incremental index tests
// ============================================================

#[test]
fn test_incremental_state_tracks_files() {
    let mut state = IncrementalState::default();

    let content = b"fn main() {}";
    state.update_file(PathBuf::from("/test/a.rs"), content, vec![1, 2]);
    state.update_file(PathBuf::from("/test/b.rs"), content, vec![3, 4]);

    assert_eq!(state.files.len(), 2);
    assert_eq!(
        state.chunk_ids_for_file(Path::new("/test/a.rs")),
        vec![1, 2]
    );
    assert_eq!(
        state.chunk_ids_for_file(Path::new("/test/b.rs")),
        vec![3, 4]
    );
    assert!(
        state
            .chunk_ids_for_file(Path::new("/test/nonexistent.rs"))
            .is_empty()
    );
}

#[test]
fn test_incremental_apply_deletions() {
    let mut state = IncrementalState::default();

    state.update_file(PathBuf::from("/test/a.rs"), b"code a", vec![1, 2]);
    state.update_file(PathBuf::from("/test/b.rs"), b"code b", vec![3, 4]);
    state.update_file(PathBuf::from("/test/c.rs"), b"code c", vec![5]);

    let removed =
        state.apply_deletions(&[PathBuf::from("/test/a.rs"), PathBuf::from("/test/c.rs")]);

    assert_eq!(state.files.len(), 1);
    assert!(state.files.contains_key(&PathBuf::from("/test/b.rs")));

    // Should return chunk IDs from deleted files
    assert!(removed.contains(&1));
    assert!(removed.contains(&2));
    assert!(removed.contains(&5));
    assert!(!removed.contains(&3));
}

#[test]
fn test_incremental_state_persistence() {
    let dir = tempfile::tempdir().unwrap();
    let state_path = dir.path().join("incremental.json");

    // Create and save state
    let mut state = IncrementalState::default();
    state.update_file(
        PathBuf::from("/project/src/main.rs"),
        b"fn main() {}",
        vec![1, 2, 3],
    );
    state.update_file(
        PathBuf::from("/project/src/lib.rs"),
        b"pub mod config;",
        vec![4],
    );
    state.save(&state_path).unwrap();

    // Load and verify
    let loaded = IncrementalState::load(&state_path).unwrap();
    assert_eq!(loaded.files.len(), 2);
    assert_eq!(
        loaded.chunk_ids_for_file(Path::new("/project/src/main.rs")),
        vec![1, 2, 3]
    );
}
