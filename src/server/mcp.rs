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
use crate::index::builder::BuildStatus;
use crate::index::store::SeekrIndex;
use crate::search::engine::SearchOptions;
use crate::search::references::ReferenceOptions;
use crate::search::symbol::SymbolOptions;
use crate::search::{SearchMode, SearchResult};
use crate::server::state::EngineRegistry;

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
const MCP_CONNECTION_SESSION_ID: &str = "mcp-stdio-connection";

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
    let registry = EngineRegistry::new(config.clone());

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
            if request.id.is_none() {
                continue;
            }
            let resp = JsonRpcResponse::error(
                request.id,
                ERROR_INVALID_REQUEST,
                "Invalid JSON-RPC version, expected 2.0".to_string(),
            );
            write_response(&mut stdout, &resp);
            continue;
        }

        if let Some(response) = handle_request(&request, &registry, MCP_CONNECTION_SESSION_ID) {
            write_response(&mut stdout, &response);
        }
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
fn handle_request(
    request: &JsonRpcRequest,
    registry: &EngineRegistry,
    session_id: &str,
) -> Option<JsonRpcResponse> {
    if request.id.is_none() {
        match request.method.as_str() {
            "notifications/initialized" | "initialized" => {
                tracing::debug!("MCP client initialized");
            }
            _ => tracing::debug!(method = %request.method, "Ignoring JSON-RPC notification"),
        }
        return None;
    }

    Some(match request.method.as_str() {
        // MCP lifecycle
        "initialize" => handle_initialize(request),
        "ping" => JsonRpcResponse::success(request.id.clone(), serde_json::json!({})),

        // MCP discovery
        "tools/list" => handle_tools_list(request),

        // MCP tool invocation
        "tools/call" => handle_tools_call(request, registry, session_id),

        // Unknown method
        _ => JsonRpcResponse::error(
            request.id.clone(),
            ERROR_METHOD_NOT_FOUND,
            format!("Method not found: {}", request.method),
        ),
    })
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
                            "description": "Maximum number of candidates to retrieve (default: 20).",
                            "default": 20
                        },
                        "max_results": {
                            "type": "integer",
                            "minimum": 0,
                            "description": "Optional final result count limit. Cannot exceed top_k."
                        },
                        "max_tokens": {
                            "type": "integer",
                            "minimum": 0,
                            "description": "Optional conservative estimated token budget for returned results."
                        },
                        "project_path": {
                            "type": "string",
                            "description": "Absolute or relative path to the project directory to search in.",
                            "default": "."
                        },
                        "path_prefix": {
                            "type": "string",
                            "description": "Optional path prefix relative to the project root."
                        },
                        "languages": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Optional language names to include."
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "seekr_symbol_definition",
                "description": "Find all indexed definitions sharing a normalized symbol name. This is lightweight name-based lookup, not compiler- or LSP-grade resolution.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Exact symbol name to find (case-insensitive)."
                        },
                        "project_path": {
                            "type": "string",
                            "description": "Project directory containing the index.",
                            "default": "."
                        },
                        "path_prefix": {
                            "type": "string",
                            "description": "Optional path prefix relative to the project root."
                        },
                        "languages": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Optional language names to include."
                        }
                    },
                    "required": ["name"]
                }
            },
            {
                "name": "seekr_symbols",
                "description": "List the lightweight name-based symbol catalog. Results are index-derived and are not compiler- or LSP-grade symbols.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "prefix": {
                            "type": "string",
                            "description": "Optional case-insensitive symbol name prefix."
                        },
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "default": 100,
                            "description": "Maximum number of normalized symbols to return."
                        },
                        "project_path": {
                            "type": "string",
                            "description": "Project directory containing the index.",
                            "default": "."
                        }
                    }
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
                "name": "seekr_references",
                "description": "Find name-level references (definition ↔ call-site mentions) for a symbol. Tree-sitter name inference only — not compiler/LSP resolution. Returns confidence scores and reasons.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Callee / definition name to resolve (case-insensitive)."
                        },
                        "project_path": {
                            "type": "string",
                            "description": "Project directory containing the index.",
                            "default": "."
                        },
                        "path_prefix": {
                            "type": "string",
                            "description": "Optional path prefix relative to the project root."
                        },
                        "languages": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Optional language names to include."
                        },
                        "limit": {
                            "type": "integer",
                            "minimum": 0,
                            "description": "Optional maximum number of reference hits."
                        }
                    },
                    "required": ["name"]
                }
            },
            {
                "name": "seekr_callers",
                "description": "Find name-level callers (call-site mentions) of a symbol. Tree-sitter name inference only — not compiler/LSP resolution. Returns confidence scores and reasons.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Callee name to find callers for (case-insensitive)."
                        },
                        "project_path": {
                            "type": "string",
                            "description": "Project directory containing the index.",
                            "default": "."
                        },
                        "path_prefix": {
                            "type": "string",
                            "description": "Optional path prefix relative to the project root."
                        },
                        "languages": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Optional language names to include."
                        },
                        "limit": {
                            "type": "integer",
                            "minimum": 0,
                            "description": "Optional maximum number of caller hits."
                        }
                    },
                    "required": ["name"]
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

fn handle_tools_call(
    request: &JsonRpcRequest,
    registry: &EngineRegistry,
    session_id: &str,
) -> JsonRpcResponse {
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
        "seekr_search" => handle_tool_search(request.id.clone(), &arguments, registry, session_id),
        "seekr_symbol_definition" => {
            handle_tool_symbol_definition(request.id.clone(), &arguments, registry)
        }
        "seekr_symbols" => handle_tool_symbols(request.id.clone(), &arguments, registry),
        "seekr_references" => handle_tool_references(request.id.clone(), &arguments, registry),
        "seekr_callers" => handle_tool_callers(request.id.clone(), &arguments, registry),
        "seekr_index" => handle_tool_index(request.id.clone(), &arguments, registry),
        "seekr_status" => handle_tool_status(request.id.clone(), &arguments, registry.config()),
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
    registry: &EngineRegistry,
    session_id: &str,
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
    let max_results = arguments
        .get("max_results")
        .and_then(|v| v.as_u64())
        .map(|value| value as usize);
    let max_tokens = arguments
        .get("max_tokens")
        .and_then(|v| v.as_u64())
        .map(|value| value as usize);
    let project_path_str = arguments
        .get("project_path")
        .and_then(|v| v.as_str())
        .unwrap_or(".");
    let path_prefix = arguments
        .get("path_prefix")
        .and_then(|v| v.as_str())
        .map(Path::new);
    let languages = arguments
        .get("languages")
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_str)
                .map(String::from)
                .collect()
        })
        .unwrap_or_default();

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

    let engine = match registry.get_or_create(&project_path) {
        Ok(engine) => engine,
        Err(e) => {
            return JsonRpcResponse::error(
                id,
                ERROR_INTERNAL,
                format!("Failed to load project engine: {e}"),
            );
        }
    };

    let start = Instant::now();

    let path_prefix = path_prefix.map(|prefix| {
        if prefix.is_absolute() {
            prefix.to_path_buf()
        } else {
            project_path.join(prefix)
        }
    });
    let excluded_chunk_ids = registry
        .session_dedup()
        .seen_chunk_ids(&project_path, session_id);
    let results = match engine.search(
        query,
        search_mode,
        &SearchOptions {
            top_k,
            max_results,
            max_tokens,
            path_prefix,
            languages,
            excluded_chunk_ids,
        },
    ) {
        Ok(results) => results,
        Err(e) => return JsonRpcResponse::error(id, ERROR_INTERNAL, e.to_string()),
    };
    registry.session_dedup().record_returned(
        &project_path,
        session_id,
        results.iter().map(|result| result.chunk.id),
    );

    let elapsed = start.elapsed();

    // Format results as MCP content
    let content = format_results_for_mcp(
        &results,
        elapsed.as_millis() as u64,
        registry.config().search.context_lines,
    );

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

fn handle_tool_symbol_definition(
    id: Option<Value>,
    arguments: &Value,
    registry: &EngineRegistry,
) -> JsonRpcResponse {
    let name = arguments
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim();
    if name.is_empty() {
        return JsonRpcResponse::error(
            id,
            ERROR_INVALID_REQUEST,
            "Missing symbol name".to_string(),
        );
    }
    let project_path = resolve_project_path(arguments, "project_path");
    let engine = match registry.get_or_create(&project_path) {
        Ok(engine) => engine,
        Err(error) => return JsonRpcResponse::error(id, ERROR_INTERNAL, error.to_string()),
    };
    let path_prefix = arguments
        .get("path_prefix")
        .and_then(Value::as_str)
        .map(|prefix| {
            let prefix = Path::new(prefix);
            if prefix.is_absolute() {
                prefix.to_path_buf()
            } else {
                project_path.join(prefix)
            }
        });
    let languages = arguments
        .get("languages")
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_str)
                .map(String::from)
                .collect()
        })
        .unwrap_or_default();
    let definitions = match engine.symbol_definitions(
        name,
        &SymbolOptions {
            path_prefix,
            languages,
        },
    ) {
        Ok(definitions) => definitions,
        Err(error) => return JsonRpcResponse::error(id, ERROR_INTERNAL, error.to_string()),
    };

    let mut output = format!(
        "Lightweight name-based lookup (not compiler/LSP resolution). Found {} definition candidate(s) for '{}':\n\n",
        definitions.len(),
        name
    );
    for (position, chunk) in definitions.iter().enumerate() {
        output.push_str(&format!(
            "[{}] {} {} in {} L{}-L{}\n",
            position + 1,
            chunk.kind,
            chunk.name.as_deref().unwrap_or("<unnamed>"),
            chunk.file_path.display(),
            chunk.line_range.start + 1,
            chunk.line_range.end
        ));
        if let Some(signature) = &chunk.signature {
            output.push_str(&format!("    {signature}\n"));
        }
    }
    JsonRpcResponse::success(
        id,
        serde_json::json!({ "content": [{ "type": "text", "text": output }] }),
    )
}

fn reference_options_from_args(arguments: &Value, project_path: &Path) -> ReferenceOptions {
    let path_prefix = arguments
        .get("path_prefix")
        .and_then(Value::as_str)
        .map(|prefix| {
            let prefix = Path::new(prefix);
            if prefix.is_absolute() {
                prefix.to_path_buf()
            } else {
                project_path.join(prefix)
            }
        });
    let languages = arguments
        .get("languages")
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_str)
                .map(String::from)
                .collect()
        })
        .unwrap_or_default();
    let limit = arguments
        .get("limit")
        .and_then(Value::as_u64)
        .map(|value| value as usize);
    ReferenceOptions {
        path_prefix,
        languages,
        limit,
    }
}

fn handle_tool_references(
    id: Option<Value>,
    arguments: &Value,
    registry: &EngineRegistry,
) -> JsonRpcResponse {
    let name = arguments
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim();
    if name.is_empty() {
        return JsonRpcResponse::error(
            id,
            ERROR_INVALID_REQUEST,
            "Missing symbol name".to_string(),
        );
    }
    let project_path = resolve_project_path(arguments, "project_path");
    let engine = match registry.get_or_create(&project_path) {
        Ok(engine) => engine,
        Err(error) => return JsonRpcResponse::error(id, ERROR_INTERNAL, error.to_string()),
    };
    let options = reference_options_from_args(arguments, &project_path);
    let hits = match engine.references(name, &options) {
        Ok(hits) => hits,
        Err(error) => return JsonRpcResponse::error(id, ERROR_INTERNAL, error.to_string()),
    };

    let mut output = format!(
        "Tree-sitter name-level inference only — not compiler/LSP resolution.\n\
         Found {} reference hit(s) for '{}':\n\n",
        hits.len(),
        name
    );
    for (position, hit) in hits.iter().enumerate() {
        output.push_str(&format!(
            "[{}] confidence={:.2} reasons={:?}\n    def: {} {} in {} L{}-L{}\n    mention: {} {:?} in {} L{}-L{}\n    snippet: {}\n\n",
            position + 1,
            hit.confidence,
            hit.reasons,
            hit.definition.kind,
            hit.definition.name.as_deref().unwrap_or("<unnamed>"),
            hit.definition.file_path.display(),
            hit.definition.line_range.start + 1,
            hit.definition.line_range.end,
            hit.mention.call_kind,
            hit.mention.callee_name,
            hit.mention.file_path.display(),
            hit.mention.line_range.start + 1,
            hit.mention.line_range.end,
            hit.mention.body.lines().next().unwrap_or(""),
        ));
    }
    JsonRpcResponse::success(
        id,
        serde_json::json!({ "content": [{ "type": "text", "text": output }] }),
    )
}

fn handle_tool_callers(
    id: Option<Value>,
    arguments: &Value,
    registry: &EngineRegistry,
) -> JsonRpcResponse {
    let name = arguments
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim();
    if name.is_empty() {
        return JsonRpcResponse::error(
            id,
            ERROR_INVALID_REQUEST,
            "Missing symbol name".to_string(),
        );
    }
    let project_path = resolve_project_path(arguments, "project_path");
    let engine = match registry.get_or_create(&project_path) {
        Ok(engine) => engine,
        Err(error) => return JsonRpcResponse::error(id, ERROR_INTERNAL, error.to_string()),
    };
    let options = reference_options_from_args(arguments, &project_path);
    let hits = match engine.callers(name, &options) {
        Ok(hits) => hits,
        Err(error) => return JsonRpcResponse::error(id, ERROR_INTERNAL, error.to_string()),
    };

    let mut output = format!(
        "Tree-sitter name-level inference only — not compiler/LSP resolution.\n\
         Found {} caller hit(s) for '{}':\n\n",
        hits.len(),
        name
    );
    for (position, hit) in hits.iter().enumerate() {
        let caller = hit
            .caller_chunk
            .as_ref()
            .map(|chunk| {
                format!(
                    "{} {}",
                    chunk.kind,
                    chunk.name.as_deref().unwrap_or("<unnamed>")
                )
            })
            .unwrap_or_else(|| "<unknown enclosing chunk>".to_string());
        output.push_str(&format!(
            "[{}] confidence={:.2} reasons={:?}\n    caller: {}\n    mention: {} {:?} in {} L{}-L{}\n    snippet: {}\n\n",
            position + 1,
            hit.confidence,
            hit.reasons,
            caller,
            hit.mention.call_kind,
            hit.mention.callee_name,
            hit.mention.file_path.display(),
            hit.mention.line_range.start + 1,
            hit.mention.line_range.end,
            hit.mention.body.lines().next().unwrap_or(""),
        ));
    }
    JsonRpcResponse::success(
        id,
        serde_json::json!({ "content": [{ "type": "text", "text": output }] }),
    )
}

fn handle_tool_symbols(
    id: Option<Value>,
    arguments: &Value,
    registry: &EngineRegistry,
) -> JsonRpcResponse {
    let prefix = arguments.get("prefix").and_then(Value::as_str);
    let limit = arguments
        .get("limit")
        .and_then(Value::as_u64)
        .unwrap_or(100)
        .clamp(1, 1_000) as usize;
    let project_path = resolve_project_path(arguments, "project_path");
    let engine = match registry.get_or_create(&project_path) {
        Ok(engine) => engine,
        Err(error) => return JsonRpcResponse::error(id, ERROR_INTERNAL, error.to_string()),
    };
    let symbols = match engine.symbols(prefix, limit) {
        Ok(symbols) => symbols,
        Err(error) => return JsonRpcResponse::error(id, ERROR_INTERNAL, error.to_string()),
    };
    let mut output = format!(
        "Lightweight indexed symbol catalog (not compiler/LSP symbols). Found {} symbol(s):\n\n",
        symbols.len()
    );
    for symbol in symbols {
        output.push_str(&format!(
            "- {} [{}] — {} definition(s), languages: {}\n",
            symbol.names.join(" / "),
            symbol.kinds.join(", "),
            symbol.definition_count,
            symbol.languages.join(", ")
        ));
    }
    JsonRpcResponse::success(
        id,
        serde_json::json!({ "content": [{ "type": "text", "text": output }] }),
    )
}

fn resolve_project_path(arguments: &Value, field: &str) -> std::path::PathBuf {
    let path = arguments.get(field).and_then(Value::as_str).unwrap_or(".");
    Path::new(path)
        .canonicalize()
        .unwrap_or_else(|_| Path::new(path).to_path_buf())
}

/// Handle `seekr_index` tool call.
fn handle_tool_index(
    id: Option<Value>,
    arguments: &Value,
    registry: &EngineRegistry,
) -> JsonRpcResponse {
    let path_str = arguments
        .get("path")
        .and_then(|v| v.as_str())
        .unwrap_or(".");
    let force = arguments
        .get("force")
        .and_then(Value::as_bool)
        .unwrap_or(false);

    let project_path = Path::new(path_str)
        .canonicalize()
        .unwrap_or_else(|_| Path::new(path_str).to_path_buf());

    let engine = match registry.get_or_create(&project_path) {
        Ok(engine) => engine,
        Err(e) => {
            return JsonRpcResponse::error(
                id,
                ERROR_INTERNAL,
                format!("Project engine creation failed: {}", e),
            );
        }
    };
    let report = match engine.build(force) {
        Ok(report) => report,
        Err(e) => {
            return JsonRpcResponse::error(id, ERROR_INTERNAL, format!("Indexing failed: {}", e));
        }
    };

    let message = format!(
        "Index {}.\n\
         • Project: {}\n\
         • Files parsed: {}\n\
         • Code chunks: {}\n\
         • Embedding dim: {}\n\
         • Duration: {:.1}s",
        match report.status {
            BuildStatus::Built => "built successfully",
            BuildStatus::UpToDate => "is already up to date",
            BuildStatus::Empty => "contains no code chunks",
        },
        report.project_path.display(),
        report.files_parsed,
        report.chunk_count,
        report.embedding_dim,
        report.duration.as_secs_f64(),
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
                index.chunk_count(),
                index.embedding_dim(),
                index.format_version(),
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

/// Format search results into a readable text block for MCP tool output.
fn format_results_for_mcp(
    results: &[SearchResult],
    duration_ms: u64,
    context_lines: usize,
) -> String {
    if results.is_empty() {
        return "No results found.".to_string();
    }

    let estimated_tokens = crate::search::token_budget::estimate_search_results_tokens(results);
    let mut output = format!(
        "Found {} results (~{} estimated tokens) in {}ms:\n\n",
        results.len(),
        estimated_tokens,
        duration_ms
    );

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

        let body_preview = if result.matched_lines.is_empty() {
            result
                .chunk
                .body
                .lines()
                .take(5)
                .collect::<Vec<&str>>()
                .join("\n")
        } else {
            crate::search::text::get_match_context(
                &result.chunk,
                &result.matched_lines,
                context_lines,
            )
            .into_iter()
            .map(|(line, content, is_match)| {
                format!(
                    "{} {:>5} {}",
                    if is_match { ">" } else { " " },
                    line + 1,
                    content
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
        };
        output.push_str(&format!("```\n{}\n```\n\n", body_preview));
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_tool_schema_exposes_agent_limits() {
        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(Value::from(1)),
            method: "tools/list".to_string(),
            params: None,
        };
        let response = handle_tools_list(&request);
        let search_schema = &response.result.unwrap()["tools"][0]["inputSchema"]["properties"];

        assert_eq!(search_schema["max_results"]["type"], "integer");
        assert_eq!(search_schema["max_tokens"]["type"], "integer");
    }

    #[test]
    fn tool_schema_exposes_lightweight_symbol_navigation() {
        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(Value::from(1)),
            method: "tools/list".to_string(),
            params: None,
        };
        let response = handle_tools_list(&request);
        let result = response.result.unwrap();
        let names: Vec<&str> = result["tools"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|tool| tool["name"].as_str())
            .collect();

        assert!(names.contains(&"seekr_symbol_definition"));
        assert!(names.contains(&"seekr_symbols"));
        assert!(names.contains(&"seekr_references"));
        assert!(names.contains(&"seekr_callers"));
    }

    #[test]
    fn notifications_do_not_produce_responses() {
        let registry = EngineRegistry::new(SeekrConfig::default());
        for method in [
            "notifications/initialized",
            "initialized",
            "unknown/notification",
        ] {
            let request = JsonRpcRequest {
                jsonrpc: "2.0".to_string(),
                id: None,
                method: method.to_string(),
                params: None,
            };
            assert!(handle_request(&request, &registry, MCP_CONNECTION_SESSION_ID).is_none());
        }
    }
}
