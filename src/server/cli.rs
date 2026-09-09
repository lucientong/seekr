//! CLI subcommand implementations.
//!
//! Handles search result formatting (colored terminal + JSON output),
//! index building orchestration, and status display.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use colored::Colorize;

use crate::config::SeekrConfig;
use crate::embedder::traits::Embedder;
use crate::error::SeekrError;
use crate::index::builder::{BuildStatus, IndexBuilder};
use crate::index::store::SeekrIndex;
use crate::search::engine::{SearchEngine, SearchOptions};
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
    if !json_output {
        eprintln!("{} Indexing {}...", "→".blue(), project_path.display());
    }
    let embedder = Arc::from(create_embedder(config)?);
    let report = IndexBuilder::new(config.clone(), embedder).build(&project_path, force)?;

    if json_output {
        let status = serde_json::json!({
            "status": build_status_name(report.status),
            "project": report.project_path.display().to_string(),
            "chunks": report.index.chunk_count,
            "files_found": report.files_found,
            "files_skipped": report.files_skipped,
            "files_parsed": report.files_parsed,
            "embedding_dim": report.index.embedding_dim,
            "incremental": !force,
            "changed_files": report.changed_files,
            "unchanged_files": report.unchanged_files,
            "deleted_files": report.deleted_files,
            "index_dir": report.index_dir.display().to_string(),
            "duration_ms": report.duration.as_millis(),
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&status).unwrap_or_default()
        );
    } else {
        match report.status {
            BuildStatus::UpToDate => eprintln!(
                "{} Index is up to date ({} files unchanged).",
                "✓".green(),
                report.unchanged_files,
            ),
            BuildStatus::Empty => eprintln!("{} No code chunks found.", "⚠".yellow()),
            BuildStatus::Built => eprintln!(
                "  {} Index built: {} chunks in {:.1}s{}",
                "✓".green(),
                report.index.chunk_count,
                report.duration.as_secs_f64(),
                if !force { " (incremental)" } else { "" },
            ),
        }
        eprintln!("  {} Saved to {}", "✓".green(), report.index_dir.display());
    }

    Ok(())
}

fn build_status_name(status: BuildStatus) -> &'static str {
    match status {
        BuildStatus::Built => "ok",
        BuildStatus::UpToDate => "up_to_date",
        BuildStatus::Empty => "empty",
    }
}

/// Execute the `seekr-code search` command.
pub fn cmd_search(
    query: &str,
    mode: &str,
    project_path: &str,
    mut options: SearchOptions,
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

    let embedder = if matches!(search_mode, SearchMode::Semantic | SearchMode::Hybrid) {
        Some(Arc::from(create_embedder_for_search(config)?))
    } else {
        None
    };
    let engine = SearchEngine::new(config.search.clone(), embedder);
    options.path_prefix = options.path_prefix.map(|path| {
        if path.is_absolute() {
            path
        } else {
            project_path.join(path)
        }
    });
    let top_k = options.top_k;
    let max_results = options.max_results;
    let max_tokens = options.max_tokens;
    let results = engine.search(&index, query, search_mode.clone(), &options)?;

    if search_mode == SearchMode::Ast && results.is_empty() && !json_output {
        eprintln!(
            "{} No AST pattern matches found for '{}'",
            "⚠".yellow(),
            query
        );
        eprintln!(
            "  {} Pattern syntax: fn(string) -> number, async fn(*) -> Result, struct *Config",
            "ℹ".blue(),
        );
    }

    let elapsed = start.elapsed();
    let total = results.len();
    let estimated_tokens = crate::search::token_budget::estimate_search_results_tokens(&results);

    if json_output {
        let response = SearchResponse {
            results,
            total,
            estimated_tokens,
            duration_ms: elapsed.as_millis() as u64,
            query: SearchQuery {
                query: query.to_string(),
                mode: search_mode,
                top_k,
                max_results,
                max_tokens,
                session_id: None,
                project_path: project_path.display().to_string(),
            },
        };
        println!(
            "{}",
            serde_json::to_string_pretty(&response).unwrap_or_default()
        );
    } else {
        print_results_colored(&results, &elapsed, config.search.context_lines);
    }

    Ok(())
}

pub fn cmd_clean(
    project_path: &str,
    config: &SeekrConfig,
    json_output: bool,
) -> Result<(), SeekrError> {
    let project_path = Path::new(project_path)
        .canonicalize()
        .unwrap_or_else(|_| Path::new(project_path).to_path_buf());
    let index_dir = config.project_index_dir(&project_path);
    let removed = index_dir.exists();
    if removed {
        std::fs::remove_dir_all(&index_dir)?;
    }

    if json_output {
        println!(
            "{}",
            serde_json::json!({
                "status": "ok",
                "project": project_path.display().to_string(),
                "index_dir": index_dir.display().to_string(),
                "removed": removed,
            })
        );
    } else if removed {
        eprintln!("{} Removed index at {}", "✓".green(), index_dir.display());
    } else {
        eprintln!(
            "{} No index found for {}",
            "ℹ".blue(),
            project_path.display()
        );
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
fn print_results_colored(
    results: &[SearchResult],
    elapsed: &std::time::Duration,
    context_lines: usize,
) {
    if results.is_empty() {
        eprintln!("{} No results found.", "⚠".yellow());
        return;
    }

    eprintln!(
        "\n🔍 {} results · ~{} estimated tokens · {:.1}ms\n",
        results.len(),
        crate::search::token_budget::estimate_search_results_tokens(results),
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

        if !result.matched_lines.is_empty() {
            for (line, content, is_match) in crate::search::text::get_match_context(
                &result.chunk,
                &result.matched_lines,
                context_lines,
            ) {
                let marker = if is_match { ">" } else { "│" };
                let rendered = if is_match {
                    content.red().bold()
                } else {
                    content.normal()
                };
                println!("  {} {:>5} {}", marker, line + 1, rendered);
            }
        } else if let Some(ref sig) = result.chunk.signature {
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

/// Create the production ONNX embedder.
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
fn create_embedder_for_search(config: &SeekrConfig) -> Result<Box<dyn Embedder>, SeekrError> {
    match crate::embedder::onnx::OnnxEmbedder::new(&config.model_dir) {
        Ok(embedder) => Ok(Box::new(embedder)),
        Err(e) => Err(SeekrError::Embedder(e)),
    }
}
