//! HTTP API server.
//!
//! REST API built with axum, bound to 127.0.0.1 (configurable port):
//! - `POST /search` — Search code with various modes
//! - `POST /index`  — Trigger index build for a project
//! - `POST /references` — Name-level definition↔mention joins
//! - `POST /callers` — Name-level caller mentions
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
use crate::search::references::{CallerHit, ReferenceHit, ReferenceOptions};
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

    /// Optional final result count limit, capped by `top_k`.
    #[serde(default)]
    pub max_results: Option<usize>,

    /// Optional conservative estimated token budget.
    #[serde(default)]
    pub max_tokens: Option<usize>,

    /// Optional session identifier for suppressing previously returned chunks.
    #[serde(default)]
    pub session_id: Option<String>,

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

/// Request body for `POST /references` and `POST /callers`.
#[derive(Debug, Deserialize)]
pub struct ReferenceRequest {
    pub name: String,
    #[serde(default = "default_path")]
    pub project_path: String,
    #[serde(default)]
    pub path_prefix: Option<String>,
    #[serde(default)]
    pub languages: Vec<String>,
    #[serde(default)]
    pub limit: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct ReferencesResponse {
    pub name: String,
    pub disclaimer: &'static str,
    pub hits: Vec<ReferenceHit>,
}

#[derive(Debug, Serialize)]
pub struct CallersResponse {
    pub name: String,
    pub disclaimer: &'static str,
    pub hits: Vec<CallerHit>,
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
        .route("/references", post(handle_references))
        .route("/callers", post(handle_callers))
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
    let session_id = req.session_id.filter(|session_id| !session_id.is_empty());
    if session_id
        .as_ref()
        .is_some_and(|session_id| session_id.len() > 128)
    {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Invalid session ID".to_string(),
                details: Some("session_id must not exceed 128 bytes".to_string()),
            }),
        ));
    }
    let excluded_chunk_ids = session_id
        .as_deref()
        .map(|session_id| {
            state
                .registry
                .session_dedup()
                .seen_chunk_ids(&project_path, session_id)
        })
        .unwrap_or_default();

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
    let max_results = req.max_results;
    let max_tokens = req.max_tokens;
    let results = engine
        .search_async(
            req.query.clone(),
            search_mode.clone(),
            SearchOptions {
                top_k,
                max_results,
                max_tokens,
                path_prefix,
                languages: req.languages,
                excluded_chunk_ids,
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
    if let Some(session_id) = session_id.as_deref() {
        state.registry.session_dedup().record_returned(
            &project_path,
            session_id,
            results.iter().map(|result| result.chunk.id),
        );
    }

    let elapsed = start.elapsed();
    let total = results.len();
    let estimated_tokens = crate::search::token_budget::estimate_search_results_tokens(&results);

    let response = SearchResponse {
        results,
        total,
        estimated_tokens,
        duration_ms: elapsed.as_millis() as u64,
        query: SearchQuery {
            query: req.query,
            mode: search_mode,
            top_k,
            max_results,
            max_tokens,
            session_id,
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

async fn handle_references(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ReferenceRequest>,
) -> Result<Json<ReferencesResponse>, (StatusCode, Json<ErrorResponse>)> {
    let name = req.name.trim();
    if name.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Missing name".to_string(),
                details: None,
            }),
        ));
    }
    let (project_path, options) = reference_request_parts(&req);
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
    let name = name.to_string();
    let hits = tokio::task::spawn_blocking(move || engine.references(&name, &options))
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: "References lookup failed".to_string(),
                    details: Some(e.to_string()),
                }),
            )
        })?
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: "References lookup failed".to_string(),
                    details: Some(e.to_string()),
                }),
            )
        })?;
    Ok(Json(ReferencesResponse {
        name: req.name,
        disclaimer: crate::search::references::REFERENCES_DISCLAIMER,
        hits,
    }))
}

async fn handle_callers(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ReferenceRequest>,
) -> Result<Json<CallersResponse>, (StatusCode, Json<ErrorResponse>)> {
    let name = req.name.trim();
    if name.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Missing name".to_string(),
                details: None,
            }),
        ));
    }
    let (project_path, options) = reference_request_parts(&req);
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
    let name = name.to_string();
    let hits = tokio::task::spawn_blocking(move || engine.callers(&name, &options))
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: "Callers lookup failed".to_string(),
                    details: Some(e.to_string()),
                }),
            )
        })?
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: "Callers lookup failed".to_string(),
                    details: Some(e.to_string()),
                }),
            )
        })?;
    Ok(Json(CallersResponse {
        name: req.name,
        disclaimer: crate::search::references::REFERENCES_DISCLAIMER,
        hits,
    }))
}

fn reference_request_parts(req: &ReferenceRequest) -> (std::path::PathBuf, ReferenceOptions) {
    let project_path = Path::new(&req.project_path)
        .canonicalize()
        .unwrap_or_else(|_| Path::new(&req.project_path).to_path_buf());
    let path_prefix = req.path_prefix.as_ref().map(|prefix| {
        let prefix = Path::new(prefix);
        if prefix.is_absolute() {
            prefix.to_path_buf()
        } else {
            project_path.join(prefix)
        }
    });
    (
        project_path,
        ReferenceOptions {
            path_prefix,
            languages: req.languages.clone(),
            limit: req.limit,
        },
    )
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
            chunks: Some(index.chunk_count()),
            embedding_dim: Some(index.embedding_dim()),
            version: Some(index.format_version()),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_request_remains_compatible_without_agent_limits() {
        let request: SearchRequest =
            serde_json::from_value(serde_json::json!({ "query": "authentication" })).unwrap();

        assert_eq!(request.top_k, 20);
        assert_eq!(request.max_results, None);
        assert_eq!(request.max_tokens, None);
        assert_eq!(request.session_id, None);
    }
}
