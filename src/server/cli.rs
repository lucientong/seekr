//! CLI subcommand implementations.
//!
//! Handles search result formatting (colored terminal + JSON output),
//! index building orchestration, and status display.

use std::path::Path;
use std::time::Instant;

use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};

use crate::config::SeekrConfig;
use crate::embedder::batch::{BatchEmbedder, DummyEmbedder};
use crate::embedder::traits::Embedder;
use crate::error::SeekrError;
use crate::index::incremental::IncrementalState;
use crate::index::store::SeekrIndex;
use crate::parser::CodeChunk;
use crate::parser::chunker::chunk_file_from_path;
use crate::parser::summary::generate_summary;
use crate::scanner::filter::should_index_file;
use crate::scanner::walker::walk_directory;
use crate::search::ast_pattern::search_ast_pattern;
use crate::search::fusion::{
    fuse_ast_only, fuse_semantic_only, fuse_text_only, rrf_fuse, rrf_fuse_three,
};
use crate::search::semantic::{SemanticSearchOptions, search_semantic};
use crate::search::text::{TextSearchOptions, search_text_regex};
use crate::search::{SearchMode, SearchQuery, SearchResponse, SearchResult};

/// Execute the `seekr-code index` command.
///
/// Scans the project directory, parses source files into chunks,
/// generates embeddings, and builds + persists the search index.
/// Supports incremental indexing: only re-processes changed files
/// unless `--force` is specified.
pub fn cmd_index(
    project_path: &str,
    force: bool,
    config: &SeekrConfig,
    json_output: bool,
) -> Result<(), SeekrError> {
    let project_path = Path::new(project_path)
        .canonicalize()
        .unwrap_or_else(|_| Path::new(project_path).to_path_buf());

    let start = Instant::now();
    let index_dir = config.project_index_dir(&project_path);
    let state_path = index_dir.join("incremental_state.json");

    // Step 1: Scan files
    if !json_output {
        eprintln!("{} Scanning project...", "→".blue());
    }

    let scan_result = walk_directory(&project_path, config)?;
    let entries: Vec<_> = scan_result
        .entries
        .iter()
        .filter(|e| should_index_file(&e.path, e.size, config.max_file_size))
        .collect();

    if !json_output {
        eprintln!(
            "  {} {} files found ({} skipped)",
            "✓".green(),
            entries.len(),
            scan_result.skipped,
        );
    }

    // Step 2: Incremental change detection
    let all_file_paths: Vec<_> = entries.iter().map(|e| e.path.clone()).collect();
    let mut incr_state = if force {
        if !json_output {
            eprintln!("  {} Force mode: full rebuild", "ℹ".blue());
        }
        IncrementalState::default()
    } else {
        IncrementalState::load(&state_path).unwrap_or_default()
    };

    let changes = incr_state.detect_changes(&all_file_paths);
    let files_to_process = if force {
        all_file_paths.clone()
    } else {
        changes.changed.clone()
    };

    // Handle deletions from existing index
    let mut existing_index = if !force {
        SeekrIndex::load(&index_dir).ok()
    } else {
        None
    };

    if !changes.deleted.is_empty() {
        if let Some(ref mut idx) = existing_index {
            let removed_ids = incr_state.apply_deletions(&changes.deleted);
            idx.remove_chunks(&removed_ids);
            if !json_output {
                eprintln!(
                    "  {} Removed {} chunks from {} deleted files",
                    "✓".green(),
                    removed_ids.len(),
                    changes.deleted.len(),
                );
            }
        }
    }

    if !force && files_to_process.is_empty() && changes.deleted.is_empty() {
        if !json_output {
            eprintln!(
                "{} Index is up to date ({} files unchanged).",
                "✓".green(),
                changes.unchanged.len(),
            );
        }
        if json_output {
            let status = serde_json::json!({
                "status": "up_to_date",
                "project": project_path.display().to_string(),
                "unchanged_files": changes.unchanged.len(),
            });
            println!(
                "{}",
                serde_json::to_string_pretty(&status).unwrap_or_default()
            );
        }
        return Ok(());
    }

    if !json_output && !force {
        eprintln!(
            "  {} {} changed, {} unchanged, {} deleted",
            "ℹ".blue(),
            files_to_process.len(),
            changes.unchanged.len(),
            changes.deleted.len(),
        );
    }

    // Step 3: Parse & chunk changed files
    if !json_output {
        eprintln!("{} Parsing source files...", "→".blue());
    }

    let pb = if !json_output {
        let pb = ProgressBar::new(files_to_process.len() as u64);
        pb.set_style(
            ProgressStyle::with_template("  {bar:40.cyan/blue} {pos}/{len} {msg}")
                .unwrap()
                .progress_chars("██░"),
        );
        Some(pb)
    } else {
        None
    };

    let mut new_chunks: Vec<CodeChunk> = Vec::new();
    let mut parsed_files = 0;

    // If incremental, remove old chunks for changed files from existing index
    if let Some(ref mut idx) = existing_index {
        for file_path in &files_to_process {
            let old_chunk_ids = incr_state.chunk_ids_for_file(file_path);
            if !old_chunk_ids.is_empty() {
                idx.remove_chunks(&old_chunk_ids);
            }
        }
    }

    for file_path in &files_to_process {
        match chunk_file_from_path(file_path) {
            Ok(Some(parse_result)) => {
                new_chunks.extend(parse_result.chunks);
                parsed_files += 1;
            }
            Ok(None) => {}
            Err(e) => {
                tracing::debug!(path = %file_path.display(), error = %e, "Failed to parse file");
            }
        }

        if let Some(ref pb) = pb {
            pb.inc(1);
        }
    }

    if let Some(pb) = pb {
        pb.finish_and_clear();
    }

    if !json_output {
        eprintln!(
            "  {} {} new chunks from {} files",
            "✓".green(),
            new_chunks.len(),
            parsed_files,
        );
    }

    if new_chunks.is_empty() && existing_index.is_none() {
        if !json_output {
            eprintln!("{} No code chunks found. Nothing to index.", "⚠".yellow());
        }
        return Ok(());
    }

    // Step 4: Generate summaries for embedding
    let summaries: Vec<String> = new_chunks.iter().map(generate_summary).collect();

    // Step 5: Generate embeddings
    if !json_output && !new_chunks.is_empty() {
        eprintln!("{} Generating embeddings...", "→".blue());
    }

    let embeddings = if new_chunks.is_empty() {
        Vec::new()
    } else {
        match create_embedder(config) {
            Ok(embedder) => {
                let batch = BatchEmbedder::new(embedder, config.embedding.batch_size);
                let pb_embed = if !json_output {
                    let pb = ProgressBar::new(summaries.len() as u64);
                    pb.set_style(
                        ProgressStyle::with_template(
                            "  {bar:40.green/blue} {pos}/{len} embeddings",
                        )
                        .unwrap()
                        .progress_chars("██░"),
                    );
                    Some(pb)
                } else {
                    None
                };

                let result = batch.embed_all_with_progress(&summaries, |completed, _total| {
                    if let Some(ref pb) = pb_embed {
                        pb.set_position(completed as u64);
                    }
                })?;

                if let Some(pb) = pb_embed {
                    pb.finish_and_clear();
                }

                result
            }
            Err(e) => {
                tracing::warn!("ONNX embedder unavailable ({}), using dummy embedder", e);
                if !json_output {
                    eprintln!(
                        "  {} ONNX model unavailable, using placeholder embeddings",
                        "⚠".yellow()
                    );
                }
                let dummy = DummyEmbedder::new(384);
                let batch = BatchEmbedder::new(dummy, config.embedding.batch_size);
                batch.embed_all(&summaries)?
            }
        }
    };

    let embedding_dim = embeddings
        .first()
        .map(|e: &Vec<f32>| e.len())
        .or_else(|| existing_index.as_ref().map(|idx| idx.embedding_dim))
        .unwrap_or(384);

    if !json_output && !new_chunks.is_empty() {
        eprintln!(
            "  {} {} embeddings generated (dim={})",
            "✓".green(),
            embeddings.len(),
            embedding_dim,
        );
    }

    // Step 6: Build or merge index
    if !json_output {
        eprintln!("{} Building index...", "→".blue());
    }

    let index = if let Some(mut idx) = existing_index {
        // Merge new chunks into existing index
        for (chunk, embedding) in new_chunks.iter().zip(embeddings.iter()) {
            let text_tokens = crate::index::store::tokenize_for_index_pub(&chunk.body);
            let entry = crate::index::IndexEntry {
                chunk_id: chunk.id,
                embedding: embedding.clone(),
                text_tokens,
            };
            idx.add_entry(entry, chunk.clone());
        }
        idx
    } else {
        SeekrIndex::build_from(&new_chunks, &embeddings, embedding_dim)
    };

    // Save index
    index.save(&index_dir)?;

    // Step 7: Update incremental state
    for file_path in &files_to_process {
        let chunk_ids: Vec<u64> = new_chunks
            .iter()
            .filter(|c| c.file_path == *file_path)
            .map(|c| c.id)
            .collect();
        if let Ok(content) = std::fs::read(file_path) {
            incr_state.update_file(file_path.clone(), &content, chunk_ids);
        }
    }
    let _ = incr_state.save(&state_path);

    let elapsed = start.elapsed();

    if json_output {
        let status = serde_json::json!({
            "status": "ok",
            "project": project_path.display().to_string(),
            "chunks": index.chunk_count,
            "files_parsed": parsed_files,
            "embedding_dim": embedding_dim,
            "incremental": !force,
            "changed_files": files_to_process.len(),
            "deleted_files": changes.deleted.len(),
            "index_dir": index_dir.display().to_string(),
            "duration_ms": elapsed.as_millis(),
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&status).unwrap_or_default()
        );
    } else {
        eprintln!(
            "  {} Index built: {} chunks in {:.1}s{}",
            "✓".green(),
            index.chunk_count,
            elapsed.as_secs_f64(),
            if !force { " (incremental)" } else { "" },
        );
        eprintln!("  {} Saved to {}", "✓".green(), index_dir.display(),);
    }

    Ok(())
}

/// Execute the `seekr-code search` command.
pub fn cmd_search(
    query: &str,
    mode: &str,
    top_k: usize,
    project_path: &str,
    config: &SeekrConfig,
    json_output: bool,
) -> Result<(), SeekrError> {
    let project_path = Path::new(project_path)
        .canonicalize()
        .unwrap_or_else(|_| Path::new(project_path).to_path_buf());

    let start = Instant::now();

    // Parse search mode
    let search_mode: SearchMode = mode
        .parse()
        .map_err(|e: String| SeekrError::Search(crate::error::SearchError::InvalidRegex(e)))?;

    // Load index
    let index_dir = config.project_index_dir(&project_path);
    let index = SeekrIndex::load(&index_dir).inspect_err(|_e| {
        tracing::error!(
            "Failed to load index from {}. Run `seekr-code index` first.",
            index_dir.display()
        );
    })?;

    // Execute search based on mode
    let fused_results = match &search_mode {
        SearchMode::Text => {
            let options = TextSearchOptions {
                case_sensitive: false,
                context_lines: config.search.context_lines,
                top_k,
            };
            let text_results = search_text_regex(&index, query, &options)?;
            fuse_text_only(&text_results, top_k)
        }
        SearchMode::Semantic => {
            let embedder = create_embedder_for_search(config)?;
            let options = SemanticSearchOptions {
                top_k,
                score_threshold: config.search.score_threshold,
            };
            let semantic_results = search_semantic(&index, query, embedder.as_ref(), &options)?;
            fuse_semantic_only(&semantic_results, top_k)
        }
        SearchMode::Hybrid => {
            // Run text, semantic, and AST search, then fuse with 3-way RRF
            let text_options = TextSearchOptions {
                case_sensitive: false,
                context_lines: config.search.context_lines,
                top_k,
            };
            let text_results = search_text_regex(&index, query, &text_options)?;

            let embedder = create_embedder_for_search(config)?;
            let semantic_options = SemanticSearchOptions {
                top_k,
                score_threshold: config.search.score_threshold,
            };
            let semantic_results =
                search_semantic(&index, query, embedder.as_ref(), &semantic_options)?;

            // Try AST pattern search — it's fine if the query doesn't parse as an AST pattern
            let ast_results = search_ast_pattern(&index, query, top_k).unwrap_or_default();

            if ast_results.is_empty() {
                // 2-way fusion when no AST matches
                rrf_fuse(&text_results, &semantic_results, config.search.rrf_k, top_k)
            } else {
                // 3-way fusion with AST
                rrf_fuse_three(
                    &text_results,
                    &semantic_results,
                    &ast_results,
                    config.search.rrf_k,
                    top_k,
                )
            }
        }
        SearchMode::Ast => {
            let ast_results = search_ast_pattern(&index, query, top_k)?;
            if ast_results.is_empty() && !json_output {
                eprintln!(
                    "{} No AST pattern matches found for '{}'",
                    "⚠".yellow(),
                    query,
                );
                eprintln!(
                    "  {} Pattern syntax: fn(string) -> number, async fn(*) -> Result, struct *Config",
                    "ℹ".blue(),
                );
            }
            fuse_ast_only(&ast_results, top_k)
        }
    };

    let elapsed = start.elapsed();

    // Build response — propagate matched_lines from fusion results
    let results: Vec<SearchResult> = fused_results
        .iter()
        .filter_map(|fused| {
            index.get_chunk(fused.chunk_id).map(|chunk| SearchResult {
                chunk: chunk.clone(),
                score: fused.fused_score,
                source: search_mode.clone(),
                matched_lines: fused.matched_lines.clone(),
            })
        })
        .collect();

    let total = results.len();

    if json_output {
        let response = SearchResponse {
            results,
            total,
            duration_ms: elapsed.as_millis() as u64,
            query: SearchQuery {
                query: query.to_string(),
                mode: search_mode,
                top_k,
                project_path: project_path.display().to_string(),
            },
        };
        println!(
            "{}",
            serde_json::to_string_pretty(&response).unwrap_or_default()
        );
    } else {
        print_results_colored(&results, &elapsed);
    }

    Ok(())
}

/// Execute the `seekr-code status` command.
pub fn cmd_status(
    project_path: &str,
    config: &SeekrConfig,
    json_output: bool,
) -> Result<(), SeekrError> {
    let project_path = Path::new(project_path)
        .canonicalize()
        .unwrap_or_else(|_| Path::new(project_path).to_path_buf());

    let index_dir = config.project_index_dir(&project_path);

    // Check for v2 bincode index first, fall back to v1 JSON index
    let exists = index_dir.join("index.bin").exists() || index_dir.join("index.json").exists();

    if json_output {
        let status = if exists {
            match SeekrIndex::load(&index_dir) {
                Ok(index) => serde_json::json!({
                    "indexed": true,
                    "project": project_path.display().to_string(),
                    "index_dir": index_dir.display().to_string(),
                    "chunks": index.chunk_count,
                    "embedding_dim": index.embedding_dim,
                    "version": index.version,
                }),
                Err(e) => serde_json::json!({
                    "indexed": true,
                    "project": project_path.display().to_string(),
                    "index_dir": index_dir.display().to_string(),
                    "error": e.to_string(),
                }),
            }
        } else {
            serde_json::json!({
                "indexed": false,
                "project": project_path.display().to_string(),
                "index_dir": index_dir.display().to_string(),
                "message": "No index found. Run `seekr-code index` to build one.",
            })
        };
        println!(
            "{}",
            serde_json::to_string_pretty(&status).unwrap_or_default()
        );
    } else if exists {
        match SeekrIndex::load(&index_dir) {
            Ok(index) => {
                eprintln!("📊 Index status for {}", project_path.display());
                eprintln!("  {} Project: {}", "•".blue(), project_path.display());
                eprintln!("  {} Index dir: {}", "•".blue(), index_dir.display());
                eprintln!(
                    "  {} Chunks: {}",
                    "•".blue(),
                    index.chunk_count.to_string().green()
                );
                eprintln!("  {} Embedding dim: {}", "•".blue(), index.embedding_dim,);
                eprintln!("  {} Version: {}", "•".blue(), index.version);
            }
            Err(e) => {
                eprintln!("{} Index found but could not load: {}", "⚠".yellow(), e);
            }
        }
    } else {
        eprintln!(
            "{} No index found for {}",
            "⚠".yellow(),
            project_path.display()
        );
        eprintln!(
            "  Run `seekr-code index {}` to build one.",
            project_path.display()
        );
    }

    Ok(())
}

/// Print search results with colored terminal output.
fn print_results_colored(results: &[SearchResult], elapsed: &std::time::Duration) {
    if results.is_empty() {
        eprintln!("{} No results found.", "⚠".yellow());
        return;
    }

    eprintln!(
        "\n🔍 {} results in {:.1}ms\n",
        results.len(),
        elapsed.as_secs_f64() * 1000.0,
    );

    for (i, result) in results.iter().enumerate() {
        let file_path = result.chunk.file_path.display();
        let kind = &result.chunk.kind;
        let name = result.chunk.name.as_deref().unwrap_or("<unnamed>");
        let score = result.score;

        // Header line
        println!(
            "{} {} {} {} (score: {:.4})",
            format!("[{}]", i + 1).dimmed(),
            file_path.to_string().cyan(),
            format!("{}", kind).dimmed(),
            name.yellow().bold(),
            score,
        );

        // Show line range
        let line_start = result.chunk.line_range.start + 1; // 1-indexed
        let line_end = result.chunk.line_range.end;
        println!("    {} L{}-L{}", "│".dimmed(), line_start, line_end,);

        // Show signature or first few lines of body
        if let Some(ref sig) = result.chunk.signature {
            println!("    {} {}", "│".dimmed(), sig.green());
        } else {
            // Show first 3 lines
            for (j, line) in result.chunk.body.lines().take(3).enumerate() {
                let trimmed = line.trim();
                if !trimmed.is_empty() {
                    println!("    {} {}", "│".dimmed(), trimmed);
                }
                if j == 2 && result.chunk.body.lines().count() > 3 {
                    println!("    {} {}", "│".dimmed(), "...".dimmed());
                }
            }
        }

        println!();
    }
}

/// Create an embedder for indexing (OnnxEmbedder or fall back to DummyEmbedder).
fn create_embedder(config: &SeekrConfig) -> Result<Box<dyn Embedder>, SeekrError> {
    match crate::embedder::onnx::OnnxEmbedder::new(&config.model_dir) {
        Ok(embedder) => Ok(Box::new(embedder)),
        Err(e) => Err(SeekrError::Embedder(
            crate::error::EmbedderError::OnnxError(format!(
                "Failed to create ONNX embedder: {}",
                e
            )),
        )),
    }
}

/// Create an embedder for search queries.
/// Falls back to DummyEmbedder if OnnxEmbedder is unavailable.
fn create_embedder_for_search(config: &SeekrConfig) -> Result<Box<dyn Embedder>, SeekrError> {
    match crate::embedder::onnx::OnnxEmbedder::new(&config.model_dir) {
        Ok(embedder) => Ok(Box::new(embedder)),
        Err(_e) => {
            tracing::warn!("ONNX embedder unavailable for search, using dummy embedder");
            Ok(Box::new(DummyEmbedder::new(384)))
        }
    }
}
