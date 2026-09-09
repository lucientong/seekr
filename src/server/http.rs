//! HTTP API server.
//!
//! REST API built with axum, bound to 127.0.0.1 (configurable port):
//! - `POST /search` — Search code with various modes
//! - `POST /index`  — Trigger index build for a project
//! - `GET  /status` — Query index status for a project

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use tokio::net::TcpListener;

use crate::config::SeekrConfig;
use crate::embedder::batch::BatchEmbedder;
use crate::embedder::traits::Embedder;
use crate::index::store::SeekrIndex;
use crate::parser::CodeChunk;
use crate::parser::chunker::chunk_file_from_path;
use crate::parser::summary::generate_summary;
use crate::scanner::filter::should_index_file;
use crate::scanner::walker::walk_directory;
use crate::search::engine::{SearchEngine, SearchOptions};
use crate::search::{SearchMode, SearchQuery, SearchResponse};

/// Shared application state for HTTP handlers.
pub struct AppState {
    pub config: SeekrConfig,
}

// ============================================================
// Request / Response types
// ============================================================

/// Request body for `POST /search`.
#[derive(Debug, Deserialize)]
pub struct SearchRequest {
    /// Search query string.
    pub query: String,

    /// Search mode: "text", "semantic", "ast", or "hybrid".
    #[serde(default = "default_mode")]
    pub mode: String,

    /// Maximum number of results.
    #[serde(default = "default_top_k")]
    pub top_k: usize,

    /// Project path to search in.
    #[serde(default = "default_path")]
    pub project_path: String,

    /// Optional path prefix relative to the project root.
    #[serde(default)]
    pub path_prefix: Option<String>,

    /// Optional language names to include.
    #[serde(default)]
    pub languages: Vec<String>,
}

fn default_mode() -> String {
    "hybrid".to_string()
}
fn default_top_k() -> usize {
    20
}
fn default_path() -> String {
    ".".to_string()
}

/// Request body for `POST /index`.
#[derive(Debug, Deserialize)]
pub struct IndexRequest {
    /// Project path to index.
    #[serde(default = "default_path")]
    pub path: String,

    /// Force full re-index.
    #[serde(default)]
    pub force: bool,
}

/// Response for `POST /index`.
#[derive(Debug, Serialize)]
pub struct IndexResponse {
    pub status: String,
    pub project: String,
    pub chunks: usize,
    pub files_parsed: usize,
    pub embedding_dim: usize,
    pub duration_ms: u128,
}

/// Request params for `GET /status`.
#[derive(Debug, Deserialize)]
pub struct StatusQuery {
    /// Project path to check (default: ".").
    #[serde(default = "default_path")]
    pub path: String,
}

/// Response for `GET /status`.
#[derive(Debug, Serialize)]
pub struct StatusResponse {
    pub indexed: bool,
    pub project: String,
    pub index_dir: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunks: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding_dim: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

/// API error response.
#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: String,
    pub details: Option<String>,
}

// ============================================================
// Server startup
// ============================================================

/// Start the HTTP API server.
pub async fn start_http_server(
    host: &str,
    port: u16,
    config: SeekrConfig,
) -> Result<(), crate::error::ServerError> {
    let state = Arc::new(AppState { config });

    let app = Router::new()
        .route("/search", post(handle_search))
        .route("/index", post(handle_index))
        .route("/status", get(handle_status))
        .route("/health", get(handle_health))
        .with_state(state);

    let addr = format!("{}:{}", host, port);
    tracing::info!(address = %addr, "Starting HTTP server");

    let listener =
        TcpListener::bind(&addr)
            .await
            .map_err(|e| crate::error::ServerError::BindFailed {
                address: addr.clone(),
                source: e,
            })?;

    tracing::info!(address = %addr, "HTTP server listening");

    axum::serve(listener, app)
        .await
        .map_err(|e| crate::error::ServerError::Internal(format!("Server error: {}", e)))?;

    Ok(())
}

// ============================================================
// Handlers
// ============================================================

/// `GET /health` — Simple health check.
async fn handle_health() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "ok",
        "version": crate::VERSION,
    }))
}

/// `POST /search` — Execute a code search.
async fn handle_search(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, (StatusCode, Json<ErrorResponse>)> {
    let config = &state.config;
    let start = Instant::now();

    // Parse search mode
    let search_mode: SearchMode = req.mode.parse().map_err(|e: String| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Invalid search mode".to_string(),
                details: Some(e),
            }),
        )
    })?;

    // Resolve project path
    let project_path = Path::new(&req.project_path)
        .canonicalize()
        .unwrap_or_else(|_| Path::new(&req.project_path).to_path_buf());

    // Load index
    let index_dir = config.project_index_dir(&project_path);
    let index = SeekrIndex::load(&index_dir).map_err(|e| {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: "Index not found".to_string(),
                details: Some(format!(
                    "No index at {}. Run `seekr-code index` first. Error: {}",
                    index_dir.display(),
                    e,
                )),
            }),
        )
    })?;

    let embedder = if matches!(search_mode, SearchMode::Semantic | SearchMode::Hybrid) {
        Some(Arc::from(create_embedder(config).map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: "Embedder unavailable".to_string(),
                    details: Some(e),
                }),
            )
        })?))
    } else {
        None
    };
    let path_prefix = req.path_prefix.as_ref().map(|prefix| {
        let path = Path::new(prefix);
        if path.is_absolute() {
            path.to_path_buf()
        } else {
            project_path.join(path)
        }
    });
    let engine = SearchEngine::new(config.search.clone(), embedder);
    let top_k = req.top_k;
    let results = engine
        .search(
            &index,
            &req.query,
            search_mode.clone(),
            &SearchOptions {
                top_k,
                path_prefix,
                languages: req.languages,
            },
        )
        .map_err(|e| {
            let status = if matches!(
                e,
                crate::error::SearchError::Embedder(_)
                    | crate::error::SearchError::EmbedderUnavailable
            ) {
                StatusCode::INTERNAL_SERVER_ERROR
            } else {
                StatusCode::BAD_REQUEST
            };
            (
                status,
                Json(ErrorResponse {
                    error: "Search failed".to_string(),
                    details: Some(e.to_string()),
                }),
            )
        })?;

    let elapsed = start.elapsed();
    let total = results.len();

    let response = SearchResponse {
        results,
        total,
        duration_ms: elapsed.as_millis() as u64,
        query: SearchQuery {
            query: req.query,
            mode: search_mode,
            top_k,
            project_path: project_path.display().to_string(),
        },
    };

    Ok(Json(response))
}

/// `POST /index` — Trigger index build for a project.
async fn handle_index(
    State(state): State<Arc<AppState>>,
    Json(req): Json<IndexRequest>,
) -> Result<Json<IndexResponse>, (StatusCode, Json<ErrorResponse>)> {
    let config = &state.config;
    let start = Instant::now();

    let project_path = Path::new(&req.path)
        .canonicalize()
        .unwrap_or_else(|_| Path::new(&req.path).to_path_buf());

    // Step 1: Scan files
    let scan_result = walk_directory(&project_path, config).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "Scan failed".to_string(),
                details: Some(e.to_string()),
            }),
        )
    })?;

    let entries: Vec<_> = scan_result
        .entries
        .iter()
        .filter(|e| should_index_file(&e.path, e.size, config.max_file_size))
        .collect();

    // Step 2: Parse & chunk
    let mut all_chunks: Vec<CodeChunk> = Vec::new();
    let mut parsed_files = 0;

    for entry in &entries {
        match chunk_file_from_path(&entry.path) {
            Ok(Some(parse_result)) => {
                all_chunks.extend(parse_result.chunks);
                parsed_files += 1;
            }
            Ok(None) => {}
            Err(e) => {
                tracing::debug!(path = %entry.path.display(), error = %e, "Failed to parse file");
            }
        }
    }

    if all_chunks.is_empty() {
        return Ok(Json(IndexResponse {
            status: "empty".to_string(),
            project: project_path.display().to_string(),
            chunks: 0,
            files_parsed: 0,
            embedding_dim: 0,
            duration_ms: start.elapsed().as_millis(),
        }));
    }

    // Step 3: Generate summaries
    let summaries: Vec<String> = all_chunks.iter().map(generate_summary).collect();

    // Step 4: Generate embeddings
    let embedder = create_embedder(config).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "Embedder unavailable".to_string(),
                details: Some(e),
            }),
        )
    })?;
    let batch = BatchEmbedder::new(embedder, config.embedding.batch_size);
    let embeddings = batch.embed_all(&summaries).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "Embedding failed".to_string(),
                details: Some(e.to_string()),
            }),
        )
    })?;

    let embedding_dim = embeddings
        .first()
        .map(|e: &Vec<f32>| e.len())
        .unwrap_or(384);

    // Step 5: Build and save index
    let index =
        SeekrIndex::try_build_from(&all_chunks, &embeddings, embedding_dim).map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: "Index build failed".to_string(),
                    details: Some(e.to_string()),
                }),
            )
        })?;
    let index_dir = config.project_index_dir(&project_path);
    index.save(&index_dir).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "Index save failed".to_string(),
                details: Some(e.to_string()),
            }),
        )
    })?;

    let elapsed = start.elapsed();

    Ok(Json(IndexResponse {
        status: "ok".to_string(),
        project: project_path.display().to_string(),
        chunks: all_chunks.len(),
        files_parsed: parsed_files,
        embedding_dim,
        duration_ms: elapsed.as_millis(),
    }))
}

/// `GET /status` — Query index status.
async fn handle_status(
    State(state): State<Arc<AppState>>,
    axum::extract::Query(query): axum::extract::Query<StatusQuery>,
) -> Json<StatusResponse> {
    let config = &state.config;

    let project_path = Path::new(&query.path)
        .canonicalize()
        .unwrap_or_else(|_| Path::new(&query.path).to_path_buf());

    let index_dir = config.project_index_dir(&project_path);
    // Check for v2 bincode index first, fall back to v1 JSON index
    let index_exists =
        index_dir.join("index.bin").exists() || index_dir.join("index.json").exists();

    if !index_exists {
        return Json(StatusResponse {
            indexed: false,
            project: project_path.display().to_string(),
            index_dir: index_dir.display().to_string(),
            chunks: None,
            embedding_dim: None,
            version: None,
            error: None,
            message: Some("No index found. Run `seekr-code index` first.".to_string()),
        });
    }

    match SeekrIndex::load(&index_dir) {
        Ok(index) => Json(StatusResponse {
            indexed: true,
            project: project_path.display().to_string(),
            index_dir: index_dir.display().to_string(),
            chunks: Some(index.chunk_count),
            embedding_dim: Some(index.embedding_dim),
            version: Some(index.version),
            error: None,
            message: None,
        }),
        Err(e) => Json(StatusResponse {
            indexed: true,
            project: project_path.display().to_string(),
            index_dir: index_dir.display().to_string(),
            chunks: None,
            embedding_dim: None,
            version: None,
            error: Some(e.to_string()),
            message: None,
        }),
    }
}

/// Create the production ONNX embedder.
fn create_embedder(config: &SeekrConfig) -> Result<Box<dyn Embedder>, String> {
    crate::embedder::onnx::OnnxEmbedder::new(&config.model_dir)
        .map(|embedder| Box::new(embedder) as Box<dyn Embedder>)
        .map_err(|e| e.to_string())
}
