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
- ⚡ **Hybrid Mode** — Combine all three via 3-way Reciprocal Rank Fusion (RRF) for best results
- 📡 **MCP Server** — Model Context Protocol support for AI editor integration
- 🌐 **HTTP API** — REST API for integration with other tools
- 🔄 **Incremental Indexing** — Only re-process changed files
- 👁️ **Watch Daemon** — Real-time file monitoring with automatic incremental re-indexing
- 🗂️ **15 Languages** — Rust, Python, JavaScript, TypeScript, Go, Java, C, C++, Ruby, Bash, HTML, CSS, JSON, TOML, YAML

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

**Endpoints:**

| Method | Path      | Description          |
|--------|-----------|----------------------|
| POST   | /search   | Search code          |
| POST   | /index    | Trigger index build  |
| GET    | /status   | Query index status   |
| GET    | /health   | Health check         |

**Example:**

```bash
curl -X POST http://127.0.0.1:7720/search \
  -H "Content-Type: application/json" \
  -d '{"query": "authenticate user", "mode": "hybrid", "top_k": 10}'
```

### MCP Server (AI Editor Integration)

```bash
# Start as MCP server over stdio
seekr-code serve --mcp
```

**MCP Tools:**

- `seekr_search` — Search code (text, semantic, AST, hybrid modes)
- `seekr_index` — Build/rebuild the search index
- `seekr_status` — Get index status

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
# Index storage directory
index_dir = "~/.seekr/indexes"

# ONNX model directory
model_dir = "~/.seekr/models"

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

## How It Works

1. **Scanner** — Walks the project directory, respects `.gitignore`, filters by file type/size
2. **Parser** — Uses Tree-sitter to parse source files into semantic code chunks (functions, classes, structs, etc.)
3. **Embedder** — Generates vector embeddings using ONNX Runtime + all-MiniLM-L6-v2 with HuggingFace WordPiece tokenizer
4. **Index** — Builds inverted text index + HNSW vector index, persisted to disk via bincode binary format
5. **Search** — Text regex, semantic HNSW ANN (with brute-force KNN fallback), AST pattern matching, fused via 3-way RRF
6. **Watch** — Optional file system monitoring with debounced incremental re-indexing

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
