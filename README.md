# seekr-code

[![CI](https://github.com/lucientong/seekr/actions/workflows/ci.yml/badge.svg)](https://github.com/lucientong/seekr/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/seekr-code.svg)](https://crates.io/crates/seekr-code)
[![Downloads](https://img.shields.io/crates/d/seekr-code.svg)](https://crates.io/crates/seekr-code)
[![License](https://img.shields.io/crates/l/seekr-code.svg)](LICENSE)

A semantic code search engine, smarter than grep.

Supports **text regex** + **semantic vector** + **AST pattern** search — 100% local, no data leaves your machine.

[中文文档](README_CN.md)

## Features

- 🔍 **Text Search** — High-performance regex matching across code
- 🧠 **Semantic Search** — Local ONNX-based embedding with HuggingFace WordPiece tokenizer + HNSW ANN index, find code by meaning
- 🌳 **AST Pattern Search** — Match function signatures, structs, classes via Tree-sitter (e.g., `fn(*) -> Result`)
- 🔗 **References / Callers** — Name-level Tree-sitter call-site mentions with confidence scores (not compiler/LSP resolution)
- ⚡ **Hybrid Mode** — Combine text + semantic + AST via 3-way Reciprocal Rank Fusion (RRF)
- 📡 **MCP Server** — Model Context Protocol support for AI editor integration
- 🌐 **HTTP API** — REST API for integration with other tools
- 🔄 **Incremental Indexing** — Only re-process changed files
- 👁️ **Watch Daemon** — Real-time file monitoring with automatic incremental re-indexing
- 🗂️ **15 Languages** — Rust, Python, JavaScript, TypeScript, Go, Java, C, C++, Ruby, Bash, HTML, CSS, JSON, TOML, YAML

> **Breaking change (v2.0):** On-disk index format is now **v4** (`SEEKRIDX` framed + flat vector store + `mentions.bin` sidecar). Older indexes are **not** auto-migrated — run `seekr-code index --force`.
## Installation

### From crates.io

```bash
cargo install seekr-code
```

### From source

```bash
git clone https://github.com/lucientong/seekr.git
cd seekr
cargo install --path .
```

After installation, the `seekr-code` binary will be available in your `$PATH`.

### Requirements

- Rust 1.85.0 or later
- A C/C++ compiler (for building tree-sitter grammars)

## Quick Start

### 1. Build an index

```bash
# Index the current project
seekr-code index

# Index a specific project path
seekr-code index /path/to/project

# Force a full rebuild (ignore incremental state)
seekr-code index --force
```

### 2. Search code

```bash
# Hybrid search (default — combines text + semantic + AST)
seekr-code search "authenticate user"

# Text regex search
seekr-code search "fn.*authenticate" --mode text

# Semantic search (search by meaning)
seekr-code search "user login validation" --mode semantic

# Agent-oriented search with final result and estimated token budgets
seekr-code search "authentication flow" --max-results 8 --max-tokens 4000

# AST pattern search
seekr-code search "fn(*) -> Result" --mode ast
seekr-code search "struct *Config" --mode ast
seekr-code search "async fn(*)" --mode ast
```

### 3. Check index status

```bash
seekr-code status
```

### 4. JSON output

All commands support `--json` for machine-readable output:

```bash
seekr-code search "authenticate" --json
seekr-code index --json
seekr-code status --json
```

## Server Mode

### HTTP API

```bash
# Start the HTTP API server (default: 127.0.0.1:7720)
seekr-code serve

# Custom host and port
seekr-code serve --host 0.0.0.0 --port 8080

# Start with watch daemon — auto re-index on file changes
seekr-code serve --watch /path/to/project
```

The HTTP server only accepts project paths under the directory where it was
started. Binding to a non-loopback address exposes source-code search and
indexing APIs to the network; use a firewall or authenticated reverse proxy.

**Endpoints:**

| Method | Path          | Description                                      |
|--------|---------------|--------------------------------------------------|
| POST   | /search       | Search code                                      |
| POST   | /index        | Trigger index build                              |
| POST   | /references   | Name-level definition ↔ mention joins            |
| POST   | /callers      | Name-level caller mentions                       |
| GET    | /status       | Query index status                               |
| GET    | /health       | Health check                                     |

**Example:**

```bash
curl -X POST http://127.0.0.1:7720/search \
  -H "Content-Type: application/json" \
  -d '{"query": "authenticate user", "mode": "hybrid", "top_k": 10, "session_id": "agent-task-42", "max_tokens": 4000}'

curl -X POST http://127.0.0.1:7720/callers \
  -H "Content-Type: application/json" \
  -d '{"name": "authenticate_user", "project_path": ".", "limit": 20}'
```

Repeated searches with the same `session_id` suppress chunks already returned in that project.
`/references` and `/callers` are Tree-sitter name-level inference only — not compiler/LSP resolution.
### MCP Server (AI Editor Integration)

```bash
# Start as MCP server over stdio
seekr-code serve --mcp
```

**MCP Tools:**

- `seekr_search` — Search code (text, semantic, AST, hybrid modes)
- `seekr_symbol_definition` — Find all name-matched definition candidates
- `seekr_symbols` — Browse the lightweight indexed symbol catalog
- `seekr_references` — Join definitions to call-site mentions by normalized name
- `seekr_callers` — List call-site mentions of a symbol
- `seekr_index` — Build/rebuild the search index
- `seekr_status` — Get index status

`seekr_search` automatically suppresses repeated chunks for the lifetime of each MCP connection.
Symbol / references / callers tools are lightweight index lookups, not compiler- or LSP-grade resolution.
**Example MCP configuration** (e.g., for Claude Desktop, CodeBuddy, etc.):

```json
{
  "mcpServers": {
    "seekr-code": {
      "command": "seekr-code",
      "args": ["serve", "--mcp"]
    }
  }
}
```

## AST Pattern Syntax

```text
[async] [pub] fn [name]([param_types, ...]) [-> return_type]
class ClassName
struct StructName
enum EnumName
trait TraitName
```

**Examples:**

| Pattern                   | Matches                                    |
|---------------------------|--------------------------------------------|
| `fn(string) -> number`    | Functions taking a string, returning number |
| `fn(*) -> Result`         | Any function returning Result               |
| `async fn(*)`             | Any async function                          |
| `fn authenticate(*)`      | Functions named "authenticate"              |
| `struct *Config`           | Structs ending with "Config"               |
| `class *Service`           | Classes ending with "Service"              |
| `enum *Error`              | Enums ending with "Error"                  |

## Configuration

Configuration file: `~/.seekr/config.toml`

```toml
# Index storage directory (must be an absolute path; `~` is not expanded)
index_dir = "/absolute/path/to/.seekr/indexes"

# ONNX model directory
model_dir = "/absolute/path/to/.seekr/models"

# Embedding model name
embed_model = "all-MiniLM-L6-v2"

# Maximum file size to index (bytes)
max_file_size = 10485760

[server]
host = "127.0.0.1"
port = 7720

[search]
context_lines = 2
top_k = 20
rrf_k = 60

[embedding]
batch_size = 32
```

Project-specific indexing can be configured in `<workspace>/.seekr.toml`:

```toml
# All roots must remain inside the workspace. Nested roots are de-duplicated.
roots = ["packages/core", "packages/web"]

# Globs are relative to the workspace root.
include = ["packages/**/*.rs", "packages/**/*.ts", "packages/**/*.tsx"]
exclude = ["**/*.generated.rs", "**/fixtures/**"]

# Optional index-time language whitelist.
languages = ["rust", "typescript", "tsx"]

# Optional override of the global byte limit.
max_file_size = 5242880
```

Invalid TOML, unsupported languages, missing roots, and roots outside the workspace fail explicitly. All entry points, including watch mode, use the same workspace configuration.

## How It Works

1. **Scanner** — Walks the project directory, respects `.gitignore`, filters by file type/size
2. **Parser** — Uses Tree-sitter to parse source files into semantic code chunks and call-site mentions
3. **Embedder** — Generates vector embeddings using ONNX Runtime + all-MiniLM-L6-v2 with HuggingFace WordPiece tokenizer
4. **Index** — Flat contiguous vector store + inverted text index + HNSW sidecar + mentions sidecar (`SEEKRIDX` v4)
5. **Search** — Text regex, semantic HNSW ANN (with brute-force KNN fallback), AST pattern matching, fused via 3-way RRF
6. **References** — Name-level definition ↔ mention joins with confidence reasons
7. **Watch** — Optional file system monitoring with debounced incremental re-indexing

## Breaking Changes (2.0)

- Index format bumped to **v4**. Legacy v3 (and earlier) indexes are rejected with an explicit rebuild prompt.
- Rebuild with: `seekr-code index --force`
- There is **no** automatic migration.
- Optional design notes: [ast-grep gate](docs/ast-grep-gate.md) (not integrated), [quality gate / no default reranker](docs/quality-gate.md).

## Benchmarks

Run the benchmark suite with:

```bash
cargo bench --bench search_bench
```

Benchmarks cover:
- Index construction (100 / 500 / 1000 chunks)
- Vector search latency (500 / 1000 / 5000 chunks)
- Text search latency (inverted index)
- Cosine similarity computation (384d)
- Index save/load throughput (bincode)

## Environment Variables

| Variable    | Description                                       |
|-------------|---------------------------------------------------|
| `SEEKR_LOG` | Log level filter (e.g., `seekr_code=debug`)       |
| `RUST_LOG`  | Fallback log level if `SEEKR_LOG` is not set      |

## License

[Apache License 2.0](LICENSE)

## Author

[lucientong](https://github.com/lucientong)
