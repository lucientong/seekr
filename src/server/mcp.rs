//! MCP Server protocol implementation.
//!
//! Implements Model Context Protocol (MCP) over stdio transport.
//! Registers three tools:
//! - `seekr_search`: Search code
//! - `seekr_index`: Trigger index build
//! - `seekr_status`: View index status
//!
//! The MCP protocol uses JSON-RPC 2.0 over stdin/stdout.

use std::io::{BufRead, Write};
use std::path::Path;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::config::SeekrConfig;
use crate::embedder::batch::{BatchEmbedder, DummyEmbedder};
use crate::embedder::traits::Embedder;
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
use crate::search::{SearchMode, SearchResult};

// ============================================================
// JSON-RPC 2.0 types
// ============================================================

/// JSON-RPC request.
#[derive(Debug, Deserialize)]
struct JsonRpcRequest {
    jsonrpc: String,
    id: Option<Value>,
    method: String,
    #[serde(default)]
    params: Option<Value>,
}

/// JSON-RPC response.
#[derive(Debug, Serialize)]
struct JsonRpcResponse {
    jsonrpc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<JsonRpcError>,
}

/// JSON-RPC error.
#[derive(Debug, Serialize)]
struct JsonRpcError {
    code: i32,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<Value>,
}

impl JsonRpcResponse {
    fn success(id: Option<Value>, result: Value) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: Some(result),
            error: None,
        }
    }

    fn error(id: Option<Value>, code: i32, message: String) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: None,
            error: Some(JsonRpcError {
                code,
                message,
                data: None,
            }),
        }
    }
}

// ============================================================
// MCP Protocol constants
// ============================================================

const MCP_PROTOCOL_VERSION: &str = "2024-11-05";
const SEEKR_MCP_NAME: &str = "seekr-code";
const SEEKR_MCP_VERSION: &str = env!("CARGO_PKG_VERSION");

// JSON-RPC error codes
const ERROR_PARSE: i32 = -32700;
const ERROR_INVALID_REQUEST: i32 = -32600;
const ERROR_METHOD_NOT_FOUND: i32 = -32601;
const ERROR_INTERNAL: i32 = -32603;

// ============================================================
// MCP Server
// ============================================================

/// Run the MCP Server over stdio.
///
/// Reads JSON-RPC requests from stdin (one per line) and writes
/// responses to stdout. This blocks until stdin is closed.
pub fn run_mcp_stdio(config: &SeekrConfig) -> Result<(), crate::error::ServerError> {
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut stdout = stdout.lock();

    tracing::info!("MCP Server starting on stdio");

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                tracing::error!("Failed to read stdin: {}", e);
                break;
            }
        };

        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let request: JsonRpcRequest = match serde_json::from_str(line) {
            Ok(req) => req,
            Err(e) => {
                let resp = JsonRpcResponse::error(None, ERROR_PARSE, format!("Parse error: {}", e));
                write_response(&mut stdout, &resp);
                continue;
            }
        };

        if request.jsonrpc != "2.0" {
            let resp = JsonRpcResponse::error(
                request.id,
                ERROR_INVALID_REQUEST,
                "Invalid JSON-RPC version, expected 2.0".to_string(),
            );
            write_response(&mut stdout, &resp);
            continue;
        }

        let response = handle_request(&request, config);
        write_response(&mut stdout, &response);
    }

    tracing::info!("MCP Server shutting down");
    Ok(())
}

/// Write a JSON-RPC response to stdout (one line).
fn write_response(writer: &mut impl Write, response: &JsonRpcResponse) {
    if let Ok(json) = serde_json::to_string(response) {
        let _ = writeln!(writer, "{}", json);
        let _ = writer.flush();
    }
}

/// Route an incoming MCP request to the appropriate handler.
fn handle_request(request: &JsonRpcRequest, config: &SeekrConfig) -> JsonRpcResponse {
    match request.method.as_str() {
        // MCP lifecycle
        "initialize" => handle_initialize(request),
        "initialized" => {
            // Notification — no response needed, but we return a result anyway
            // since some clients expect it
            JsonRpcResponse::success(request.id.clone(), Value::Null)
        }
        "ping" => JsonRpcResponse::success(request.id.clone(), serde_json::json!({})),

        // MCP discovery
        "tools/list" => handle_tools_list(request),

        // MCP tool invocation
        "tools/call" => handle_tools_call(request, config),

        // Unknown method
        _ => JsonRpcResponse::error(
            request.id.clone(),
            ERROR_METHOD_NOT_FOUND,
            format!("Method not found: {}", request.method),
        ),
    }
}

// ============================================================
// MCP Lifecycle handlers
// ============================================================

fn handle_initialize(request: &JsonRpcRequest) -> JsonRpcResponse {
    JsonRpcResponse::success(
        request.id.clone(),
        serde_json::json!({
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {
                "tools": {}
            },
            "serverInfo": {
                "name": SEEKR_MCP_NAME,
                "version": SEEKR_MCP_VERSION,
            }
        }),
    )
}

// ============================================================
// MCP Tools discovery
// ============================================================

fn handle_tools_list(request: &JsonRpcRequest) -> JsonRpcResponse {
    let tools = serde_json::json!({
        "tools": [
            {
                "name": "seekr_search",
                "description": "Search code in a project using text regex, semantic vector, AST pattern, or hybrid mode. Returns ranked code chunks matching the query.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query. For text mode: regex pattern. For semantic mode: natural language description. For AST mode: function signature pattern (e.g., 'fn(string) -> number'). For hybrid mode: any query."
                        },
                        "mode": {
                            "type": "string",
                            "description": "Search mode: 'text', 'semantic', 'ast', or 'hybrid' (default).",
                            "enum": ["text", "semantic", "ast", "hybrid"],
                            "default": "hybrid"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Maximum number of results to return (default: 20).",
                            "default": 20
                        },
                        "project_path": {
                            "type": "string",
                            "description": "Absolute or relative path to the project directory to search in.",
                            "default": "."
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "seekr_index",
                "description": "Build or rebuild the code search index for a project. Scans source files, parses them into semantic chunks, generates embeddings, and builds a searchable index.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the project directory to index.",
                            "default": "."
                        },
                        "force": {
                            "type": "boolean",
                            "description": "Force full re-index, ignoring incremental state.",
                            "default": false
                        }
                    }
                }
            },
            {
                "name": "seekr_status",
                "description": "Get the index status for a project. Returns information about whether the project is indexed, how many chunks exist, and the index version.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the project directory to check.",
                            "default": "."
                        }
                    }
                }
            }
        ]
    });

    JsonRpcResponse::success(request.id.clone(), tools)
}

// ============================================================
// MCP Tools invocation
// ============================================================

fn handle_tools_call(request: &JsonRpcRequest, config: &SeekrConfig) -> JsonRpcResponse {
    let params = match &request.params {
        Some(p) => p,
        None => {
            return JsonRpcResponse::error(
                request.id.clone(),
                ERROR_INVALID_REQUEST,
                "Missing params".to_string(),
            );
        }
    };

    let tool_name = params.get("name").and_then(|v| v.as_str()).unwrap_or("");
    let arguments = params
        .get("arguments")
        .cloned()
        .unwrap_or(Value::Object(Default::default()));

    match tool_name {
        "seekr_search" => handle_tool_search(request.id.clone(), &arguments, config),
        "seekr_index" => handle_tool_index(request.id.clone(), &arguments, config),
        "seekr_status" => handle_tool_status(request.id.clone(), &arguments, config),
        _ => JsonRpcResponse::error(
            request.id.clone(),
            ERROR_METHOD_NOT_FOUND,
            format!("Unknown tool: {}", tool_name),
        ),
    }
}

/// Handle `seekr_search` tool call.
fn handle_tool_search(
    id: Option<Value>,
    arguments: &Value,
    config: &SeekrConfig,
) -> JsonRpcResponse {
    let query = arguments
        .get("query")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let mode_str = arguments
        .get("mode")
        .and_then(|v| v.as_str())
        .unwrap_or("hybrid");
    let top_k = arguments
        .get("top_k")
        .and_then(|v| v.as_u64())
        .unwrap_or(20) as usize;
    let project_path_str = arguments
        .get("project_path")
        .and_then(|v| v.as_str())
        .unwrap_or(".");

    if query.is_empty() {
        return JsonRpcResponse::error(id, ERROR_INVALID_REQUEST, "Missing query".to_string());
    }

    let search_mode: SearchMode = match mode_str.parse() {
        Ok(m) => m,
        Err(e) => return JsonRpcResponse::error(id, ERROR_INVALID_REQUEST, e),
    };

    let project_path = Path::new(project_path_str)
        .canonicalize()
        .unwrap_or_else(|_| Path::new(project_path_str).to_path_buf());

    let index_dir = config.project_index_dir(&project_path);
    let index = match SeekrIndex::load(&index_dir) {
        Ok(idx) => idx,
        Err(e) => {
            return JsonRpcResponse::error(
                id,
                ERROR_INTERNAL,
                format!("Failed to load index: {}. Run `seekr-code index` first.", e),
            );
        }
    };

    let start = Instant::now();

    let fused_results = match execute_search(&search_mode, query, &index, config, top_k) {
        Ok(results) => results,
        Err(e) => return JsonRpcResponse::error(id, ERROR_INTERNAL, e),
    };

    let elapsed = start.elapsed();

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

    // Format results as MCP content
    let content = format_results_for_mcp(&results, elapsed.as_millis() as u64);

    JsonRpcResponse::success(
        id,
        serde_json::json!({
            "content": [{
                "type": "text",
                "text": content,
            }]
        }),
    )
}

/// Handle `seekr_index` tool call.
fn handle_tool_index(
    id: Option<Value>,
    arguments: &Value,
    config: &SeekrConfig,
) -> JsonRpcResponse {
    let path_str = arguments
        .get("path")
        .and_then(|v| v.as_str())
        .unwrap_or(".");

    let project_path = Path::new(path_str)
        .canonicalize()
        .unwrap_or_else(|_| Path::new(path_str).to_path_buf());

    let start = Instant::now();

    // Scan
    let scan_result = match walk_directory(&project_path, config) {
        Ok(r) => r,
        Err(e) => {
            return JsonRpcResponse::error(id, ERROR_INTERNAL, format!("Scan failed: {}", e));
        }
    };

    let entries: Vec<_> = scan_result
        .entries
        .iter()
        .filter(|e| should_index_file(&e.path, e.size, config.max_file_size))
        .collect();

    // Parse
    let mut all_chunks: Vec<CodeChunk> = Vec::new();
    let mut parsed_files = 0;

    for entry in &entries {
        if let Ok(Some(parse_result)) = chunk_file_from_path(&entry.path) {
            all_chunks.extend(parse_result.chunks);
            parsed_files += 1;
        }
    }

    if all_chunks.is_empty() {
        return JsonRpcResponse::success(
            id,
            serde_json::json!({
                "content": [{
                    "type": "text",
                    "text": "No code chunks found in the project. Nothing to index.",
                }]
            }),
        );
    }

    // Embed
    let summaries: Vec<String> = all_chunks.iter().map(generate_summary).collect();

    let embeddings = match create_embedder(config) {
        Ok(embedder) => {
            let batch = BatchEmbedder::new(embedder, config.embedding.batch_size);
            match batch.embed_all(&summaries) {
                Ok(e) => e,
                Err(e) => {
                    return JsonRpcResponse::error(
                        id,
                        ERROR_INTERNAL,
                        format!("Embedding failed: {}", e),
                    );
                }
            }
        }
        Err(e) => {
            return JsonRpcResponse::error(
                id,
                ERROR_INTERNAL,
                format!("Embedder creation failed: {}", e),
            );
        }
    };

    let embedding_dim = embeddings
        .first()
        .map(|e: &Vec<f32>| e.len())
        .unwrap_or(384);

    // Build and save
    let index = SeekrIndex::build_from(&all_chunks, &embeddings, embedding_dim);
    let index_dir = config.project_index_dir(&project_path);

    if let Err(e) = index.save(&index_dir) {
        return JsonRpcResponse::error(id, ERROR_INTERNAL, format!("Index save failed: {}", e));
    }

    let elapsed = start.elapsed();

    let message = format!(
        "Index built successfully!\n\
         • Project: {}\n\
         • Files parsed: {}\n\
         • Code chunks: {}\n\
         • Embedding dim: {}\n\
         • Duration: {:.1}s",
        project_path.display(),
        parsed_files,
        all_chunks.len(),
        embedding_dim,
        elapsed.as_secs_f64(),
    );

    JsonRpcResponse::success(
        id,
        serde_json::json!({
            "content": [{
                "type": "text",
                "text": message,
            }]
        }),
    )
}

/// Handle `seekr_status` tool call.
fn handle_tool_status(
    id: Option<Value>,
    arguments: &Value,
    config: &SeekrConfig,
) -> JsonRpcResponse {
    let path_str = arguments
        .get("path")
        .and_then(|v| v.as_str())
        .unwrap_or(".");

    let project_path = Path::new(path_str)
        .canonicalize()
        .unwrap_or_else(|_| Path::new(path_str).to_path_buf());

    let index_dir = config.project_index_dir(&project_path);
    // Check for v2 bincode index first, fall back to v1 JSON index
    let index_exists =
        index_dir.join("index.bin").exists() || index_dir.join("index.json").exists();

    let message = if !index_exists {
        format!(
            "No index found for {}.\n\
             Run `seekr-code index {}` to build one.",
            project_path.display(),
            project_path.display(),
        )
    } else {
        match SeekrIndex::load(&index_dir) {
            Ok(index) => format!(
                "Index status for {}:\n\
                 • Indexed: yes\n\
                 • Chunks: {}\n\
                 • Embedding dim: {}\n\
                 • Version: {}\n\
                 • Index dir: {}",
                project_path.display(),
                index.chunk_count,
                index.embedding_dim,
                index.version,
                index_dir.display(),
            ),
            Err(e) => format!(
                "Index found but could not load: {}\n\
                 Try rebuilding with `seekr-code index {}`.",
                e,
                project_path.display(),
            ),
        }
    };

    JsonRpcResponse::success(
        id,
        serde_json::json!({
            "content": [{
                "type": "text",
                "text": message,
            }]
        }),
    )
}

// ============================================================
// Shared helpers
// ============================================================

use crate::search::fusion::FusedResult;

/// Execute search across different modes — shared by MCP and HTTP.
fn execute_search(
    mode: &SearchMode,
    query: &str,
    index: &SeekrIndex,
    config: &SeekrConfig,
    top_k: usize,
) -> Result<Vec<FusedResult>, String> {
    match mode {
        SearchMode::Text => {
            let options = TextSearchOptions {
                case_sensitive: false,
                context_lines: config.search.context_lines,
                top_k,
            };
            let results = search_text_regex(index, query, &options).map_err(|e| e.to_string())?;
            Ok(fuse_text_only(&results, top_k))
        }
        SearchMode::Semantic => {
            let embedder = create_embedder(config)?;
            let options = SemanticSearchOptions {
                top_k,
                score_threshold: config.search.score_threshold,
            };
            let results = search_semantic(index, query, embedder.as_ref(), &options)
                .map_err(|e| e.to_string())?;
            Ok(fuse_semantic_only(&results, top_k))
        }
        SearchMode::Hybrid => {
            let text_options = TextSearchOptions {
                case_sensitive: false,
                context_lines: config.search.context_lines,
                top_k,
            };
            let text_results =
                search_text_regex(index, query, &text_options).map_err(|e| e.to_string())?;

            let embedder = create_embedder(config)?;
            let semantic_options = SemanticSearchOptions {
                top_k,
                score_threshold: config.search.score_threshold,
            };
            let semantic_results =
                search_semantic(index, query, embedder.as_ref(), &semantic_options)
                    .map_err(|e| e.to_string())?;

            // Try AST pattern search — silently degrade if query isn't a valid AST pattern
            let ast_results = search_ast_pattern(index, query, top_k).unwrap_or_default();

            if ast_results.is_empty() {
                // 2-way fusion when no AST matches
                Ok(rrf_fuse(
                    &text_results,
                    &semantic_results,
                    config.search.rrf_k,
                    top_k,
                ))
            } else {
                // 3-way fusion with AST
                Ok(rrf_fuse_three(
                    &text_results,
                    &semantic_results,
                    &ast_results,
                    config.search.rrf_k,
                    top_k,
                ))
            }
        }
        SearchMode::Ast => {
            let results = search_ast_pattern(index, query, top_k).map_err(|e| e.to_string())?;
            Ok(fuse_ast_only(&results, top_k))
        }
    }
}

/// Create an embedder. Falls back to DummyEmbedder if ONNX is unavailable.
fn create_embedder(config: &SeekrConfig) -> Result<Box<dyn Embedder>, String> {
    match crate::embedder::onnx::OnnxEmbedder::new(&config.model_dir) {
        Ok(embedder) => Ok(Box::new(embedder)),
        Err(_) => {
            tracing::warn!("ONNX embedder unavailable, using dummy embedder");
            Ok(Box::new(DummyEmbedder::new(384)))
        }
    }
}

/// Format search results into a readable text block for MCP tool output.
fn format_results_for_mcp(results: &[SearchResult], duration_ms: u64) -> String {
    if results.is_empty() {
        return "No results found.".to_string();
    }

    let mut output = format!("Found {} results in {}ms:\n\n", results.len(), duration_ms);

    for (i, result) in results.iter().enumerate() {
        let name = result.chunk.name.as_deref().unwrap_or("<unnamed>");
        let file_path = result.chunk.file_path.display();
        let line_start = result.chunk.line_range.start + 1;
        let line_end = result.chunk.line_range.end;

        output.push_str(&format!(
            "---\n[{}] {} ({}) in {} L{}-L{} (score: {:.4})\n",
            i + 1,
            name,
            result.chunk.kind,
            file_path,
            line_start,
            line_end,
            result.score,
        ));

        // Show signature or first few lines
        if let Some(ref sig) = result.chunk.signature {
            output.push_str(&format!("  Signature: {}\n", sig));
        }

        // Show first 5 lines of body
        let body_preview: String = result
            .chunk
            .body
            .lines()
            .take(5)
            .collect::<Vec<&str>>()
            .join("\n");
        output.push_str(&format!("```\n{}\n```\n\n", body_preview));
    }

    output
}
