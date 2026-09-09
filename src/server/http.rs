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
use crate::index::store::SeekrIndex;
use crate::search::engine::SearchOptions;
use crate::search::{SearchMode, SearchQuery, SearchResponse};
use crate::server::state::EngineRegistry;

/// Shared application state for HTTP handlers.
pub struct AppState {
    pub registry: EngineRegistry,
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
    start_http_server_with_registry(host, port, EngineRegistry::new(config)).await
}

pub async fn start_http_server_with_registry(
    host: &str,
    port: u16,
    registry: EngineRegistry,
) -> Result<(), crate::error::ServerError> {
    let state = Arc::new(AppState { registry });

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

    let engine = state
        .registry
        .get_or_create_async(project_path.clone())
        .await
        .map_err(|error| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: "Project engine unavailable".to_string(),
                    details: Some(error.to_string()),
                }),
            )
        })?;
    let path_prefix = req.path_prefix.as_ref().map(|prefix| {
        let path = Path::new(prefix);
        if path.is_absolute() {
            path.to_path_buf()
        } else {
            project_path.join(path)
        }
    });
    let top_k = req.top_k;
    let results = engine
        .search_async(
            req.query.clone(),
            search_mode.clone(),
            SearchOptions {
                top_k,
                path_prefix,
                languages: req.languages,
            },
        )
        .await
        .map_err(|e| {
            let details = e.to_string();
            let status = if details.contains("Index not found") {
                StatusCode::NOT_FOUND
            } else {
                StatusCode::INTERNAL_SERVER_ERROR
            };
            (
                status,
                Json(ErrorResponse {
                    error: "Search failed".to_string(),
                    details: Some(details),
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
    let project_path = Path::new(&req.path)
        .canonicalize()
        .unwrap_or_else(|_| Path::new(&req.path).to_path_buf());
    let engine = state
        .registry
        .get_or_create_async(project_path)
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: "Project engine unavailable".to_string(),
                    details: Some(e.to_string()),
                }),
            )
        })?;
    let report = engine.build_async(req.force).await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "Indexing failed".to_string(),
                details: Some(e.to_string()),
            }),
        )
    })?;

    Ok(Json(IndexResponse {
        status: match report.status {
            crate::index::builder::BuildStatus::Built => "ok",
            crate::index::builder::BuildStatus::UpToDate => "up_to_date",
            crate::index::builder::BuildStatus::Empty => "empty",
        }
        .to_string(),
        project: report.project_path.display().to_string(),
        chunks: report.chunk_count,
        files_parsed: report.files_parsed,
        embedding_dim: report.embedding_dim,
        duration_ms: report.duration.as_millis(),
    }))
}

/// `GET /status` — Query index status.
async fn handle_status(
    State(state): State<Arc<AppState>>,
    axum::extract::Query(query): axum::extract::Query<StatusQuery>,
) -> Json<StatusResponse> {
    let config = state.registry.config();

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
