# Seekr Architecture

> **seekr-code** — A semantic code search engine, smarter than grep.

This document describes the internal architecture, module design, data flow, and key design decisions of seekr-code v1.0.0.

## Table of Contents

- [Overview](#overview)
- [High-Level Architecture](#high-level-architecture)
- [Module Map](#module-map)
  - [Scanner (`scanner`)](#scanner-scanner)
  - [Parser (`parser`)](#parser-parser)
  - [Embedder (`embedder`)](#embedder-embedder)
  - [Index (`index`)](#index-index)
  - [Search (`search`)](#search-search)
  - [Server (`server`)](#server-server)
  - [Config & Error (`config`, `error`)](#config--error-config-error)
- [Data Flow](#data-flow)
  - [Indexing Pipeline](#indexing-pipeline)
  - [Search Pipeline](#search-pipeline)
  - [Watch Daemon Pipeline](#watch-daemon-pipeline)
- [Core Data Structures](#core-data-structures)
- [Design Principles](#design-principles)
- [Technology Stack](#technology-stack)

---

## Overview

Seekr is a **100% local** code search engine that combines three complementary search strategies:

1. **Text regex search** — High-performance regular expression matching (like ripgrep)
2. **Semantic vector search** — ONNX-based local embedding model + HNSW approximate nearest neighbor search (with brute-force KNN fallback)
3. **AST pattern search** — Tree-sitter powered function signature pattern matching

These three modes can be used independently or combined via **Reciprocal Rank Fusion (RRF)** for hybrid search that understands both text patterns and code semantics.

No data ever leaves your machine. The entire pipeline — scanning, parsing, embedding, indexing, and search — runs locally.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interfaces                          │
│  ┌───────────┐    ┌──────────────┐    ┌───────────────────┐     │
│  │    CLI     │    │  HTTP REST   │    │  MCP (JSON-RPC)   │     │
│  │ (clap)    │    │  (axum)      │    │  (stdio)          │     │
│  └─────┬─────┘    └──────┬───────┘    └────────┬──────────┘     │
│        │                 │                      │               │
│        └────────────┬────┴──────────────────────┘               │
│                     │                                           │
│  ┌──────────────────▼──────────────────────────────────────┐    │
│  │                  Search Engine                           │    │
│  │  ┌──────────┐  ┌───────────┐  ┌──────────────────────┐  │    │
│  │  │  Text    │  │ Semantic  │  │    AST Pattern       │  │    │
│  │  │  Regex   │  │  Vector   │  │    Matching          │  │    │
│  │  └────┬─────┘  └─────┬─────┘  └──────────┬───────────┘  │    │
│  │       └──────────┬───┴────────────────────┘              │    │
│  │                  │                                       │    │
│  │          ┌───────▼───────┐                               │    │
│  │          │  RRF Fusion   │                               │    │
│  │          │  (3-way)      │                               │    │
│  │          └───────────────┘                               │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │                  Index Engine                            │    │
│  │  ┌──────────┐  ┌──────────┐  ┌────────────────────────┐ │    │
│  │  │ Vector   │  │ Inverted │  │   Chunk Metadata       │ │    │
│  │  │ Store    │  │ Index    │  │   Store                │ │    │
│  │  └──────────┘  └──────────┘  └────────────────────────┘ │    │
│  │                                                          │    │
│  │  ┌──────────┐  ┌──────────────┐  ┌──────────────────┐   │    │
│  │  │ bincode  │  │ Incremental  │  │   Mmap Store     │   │    │
│  │  │ persist  │  │ State        │  │   (zero-copy)    │   │    │
│  │  └──────────┘  └──────────────┘  └──────────────────┘   │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │                  Processing Pipeline                       │   │
│  │  ┌──────────┐   ┌──────────┐    ┌──────────────────┐      │   │
│  │  │ Scanner  │──▶│ Parser   │──▶ │   Embedder       │      │   │
│  │  │ (ignore) │   │(tree-    │    │   (ONNX          │      │   │
│  │  │          │   │ sitter)  │    │    all-MiniLM)   │      │   │
│  │  └──────────┘   └──────────┘    └──────────────────┘      │   │
│  └───────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌────────────┐   ┌──────────────────────────────────────────┐   │
│  │   Config   │   │   Watch Daemon (notify + tokio)          │   │
│  └────────────┘   └──────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

## Module Map

The project is organized into 7 core modules plus 2 infrastructure modules:

```
src/
├── lib.rs              # Library root: exports all modules, INDEX_VERSION, VERSION
├── main.rs             # Binary entry point: CLI argument parsing (clap)
├── config.rs           # Configuration management (TOML-based)
├── error.rs            # Unified error types (thiserror)
├── scanner/            # File discovery and watching
│   ├── mod.rs          # Module re-exports
│   ├── walker.rs       # Parallel directory traversal (ignore crate)
│   ├── filter.rs       # File filtering (extension, size, binary detection)
│   └── watcher.rs      # File system event monitoring (notify crate)
├── parser/             # Code parsing and chunking
│   ├── mod.rs          # Core types (CodeChunk, ChunkKind, ParseResult)
│   ├── treesitter.rs   # Language support (16 languages) and grammar management
│   ├── chunker.rs      # AST-aware semantic chunking
│   └── summary.rs      # Structured code summary generation
├── embedder/           # Embedding model management
│   ├── mod.rs          # Module re-exports
│   ├── traits.rs       # Embedder trait (pluggable backend interface)
│   ├── onnx.rs         # ONNX Runtime backend (all-MiniLM-L6-v2)
│   └── batch.rs        # BatchEmbedder wrapper + DummyEmbedder
├── index/              # Index storage and management
│   ├── mod.rs          # Core types (IndexEntry, SearchHit)
│   ├── store.rs        # SeekrIndex: vector + inverted + metadata storage
│   ├── mmap_store.rs   # Memory-mapped index (zero-copy reads)
│   └── incremental.rs  # Incremental indexing (mtime + blake3 change detection)
├── search/             # Search algorithms
│   ├── mod.rs          # Module re-exports
│   ├── text.rs         # Regex-based text search with context lines
│   ├── semantic.rs     # Vector similarity search (cosine KNN)
│   ├── ast_pattern.rs  # AST pattern parser and matcher
│   └── fusion.rs       # RRF fusion (2-way and 3-way)
└── server/             # Interface layer
    ├── mod.rs          # Module re-exports
    ├── cli.rs          # CLI command handlers (search, index, status)
    ├── http.rs         # HTTP REST API (axum)
    ├── mcp.rs          # MCP Server (JSON-RPC 2.0 over stdio)
    └── daemon.rs       # Watch daemon for real-time incremental indexing
```

### Scanner (`scanner`)

The scanner module discovers files eligible for indexing within a project directory.

| File | Responsibility |
|------|---------------|
| `walker.rs` | Parallel directory traversal using the `ignore` crate (`build_parallel()`). Respects `.gitignore` rules and custom exclude patterns from config. Collects results via `Mutex<Vec<PathBuf>>`. |
| `filter.rs` | File eligibility checking: size limit (default 10MB), extension whitelist (60+ extensions), binary content detection (null byte ratio > 1% in first 8KB). |
| `watcher.rs` | File system event monitoring via `notify` crate. Provides both sync (`FileWatcher` with `mpsc::channel`) and async (`start_async_watcher` with `tokio::sync::mpsc`) watchers. Includes `dedup_events()` for HashMap-based event deduplication. |

**Key types:**
- `FileEvent` — enum: `Changed(PathBuf)`, `Deleted(PathBuf)`
- `FileWatcher` — sync file watcher wrapper

### Parser (`parser`)

The parser module transforms source files into semantic code chunks using Tree-sitter AST parsing.

| File | Responsibility |
|------|---------------|
| `mod.rs` | Core data types: `CodeChunk`, `ChunkKind` (Function, Method, Class, Struct, Enum, Interface, Trait, Impl, Module, Other), `ParseResult`. |
| `treesitter.rs` | `SupportedLanguage` enum with 16 variants (Rust, Python, JavaScript, TypeScript, Tsx, Go, Java, C, Cpp, Json, Toml, Yaml, Html, Css, Ruby, Bash). Provides `from_extension()`, `grammar()`, `chunk_node_kinds()` per language. |
| `chunker.rs` | AST-aware chunking via `chunk_file()` / `chunk_file_from_path()`. Recursively traverses AST to extract semantic units. Falls back to 50-line blocks (`fallback_line_chunks()`) when AST yields no chunks. Uses global `AtomicU64` counter for unique chunk IDs. Minimum chunk size: 2 lines. |
| `summary.rs` | Generates structured summaries for embedding: combines kind+name, language, signature, doc_comment (truncated to 500 chars), and body snippet (first 5 meaningful lines). |

**Key types:**
- `CodeChunk` — id, file_path, language, kind, name, signature, doc_comment, body, byte_range, line_range
- `ChunkKind` — Function, Method, Class, Struct, Enum, Interface, Trait, Impl, Module, Other
- `ParseResult` — file_path, language, chunks vec

### Embedder (`embedder`)

The embedder module provides pluggable text-to-vector embedding backends.

| File | Responsibility |
|------|---------------|
| `traits.rs` | `Embedder` trait with `embed()`, `embed_batch()`, `dimension()`. Requires `Send + Sync`. Blanket implementation for `Box<dyn Embedder>`. |
| `onnx.rs` | `OnnxEmbedder` — ONNX Runtime backend for all-MiniLM-L6-v2 (quantized, 384-dim). Auto-downloads model and tokenizer from HuggingFace. Uses HuggingFace `tokenizers` crate with WordPiece tokenization. Truncation at 256 tokens. Mean Pooling + L2 normalization. ONNX optimization level 3, 4 intra-op threads. Atomic file downloads (tmp + rename). |
| `batch.rs` | `BatchEmbedder<E>` — wraps any `Embedder` with configurable batch size and progress callback. `DummyEmbedder` — deterministic pseudo-random embeddings for testing. |

**Key design:** The `Embedder` trait is the extension point for future model backends (e.g., GGML, custom models). The current implementation uses a `Mutex<Session>` for thread-safe ONNX inference.

### Index (`index`)

The index module manages persistent storage for vectors, inverted index, and chunk metadata.

| File | Responsibility |
|------|---------------|
| `mod.rs` | Core types: `IndexEntry` (chunk_id + embedding + text_tokens), `SearchHit` (chunk_id + score). |
| `store.rs` | `SeekrIndex` — the main index structure. Contains `vectors: HashMap<u64, Vec<f32>>`, `inverted_index: HashMap<String, Vec<(u64, u32)>>`, `chunks: HashMap<u64, CodeChunk>`. Provides `add_entry()`, `remove_chunk()`, `search_vector()` (brute-force cosine KNN), `search_text()` (inverted index with TF scoring). Serialization: `save()` writes bincode to `index.bin`, `load()` tries bincode first then falls back to JSON for v1 migration. |
| `mmap_store.rs` | `MmapIndex` — memory-mapped read-only index with zero-copy via `memmap2`. Single `mmap: Mmap` field, `as_bytes()` returns `&self.mmap` directly. |
| `incremental.rs` | `IncrementalState` — tracks file states (`mtime` + `blake3` content hash + chunk IDs). `detect_changes()` uses mtime as fast-path, blake3 as confirmation. `apply_deletions()` returns chunk IDs to remove. |

**Key design:** Index v2 uses bincode for serialization (replacing JSON in v1), providing significantly faster save/load times. Backward compatibility is maintained by falling back to `index.json` when `index.bin` is not found.

### Search (`search`)

The search module implements three independent search strategies and a fusion algorithm.

| File | Responsibility |
|------|---------------|
| `text.rs` | `search_text_regex()` — regex matching across all chunks. Scoring: `match_count + density * 10`. `get_match_context()` — extracts context lines around matches. |
| `semantic.rs` | `search_semantic()` — embeds query string, then calls `index.search_vector()` for HNSW approximate nearest neighbor search (with brute-force fallback). |
| `ast_pattern.rs` | `AstPattern` — parsed pattern with qualifiers (async, pub, static), kind, name_pattern, param_patterns, return_pattern. `parse_pattern()` — custom pattern syntax parser supporting wildcards `*`. `match_chunk()` — multi-dimensional weighted scoring (kind 0.3, name 0.3, params 0.15, return 0.15, qualifiers 0.1). `fuzzy_type_match()` — type group matching (string/number/bool/void groups). |
| `fusion.rs` | `FusedResult` — combined score from all sources. `rrf_fuse()` — 2-way RRF (text + semantic). `rrf_fuse_three()` — 3-way RRF (text + semantic + AST). Formula: `score = sum(1/(k + rank + 1))`, k=60 by default. Also provides `fuse_text_only()`, `fuse_semantic_only()`, `fuse_ast_only()` single-source wrappers. |

**Key design:** AST pattern search can fail gracefully — if AST parsing fails or returns no results, hybrid mode silently degrades to 2-way RRF (text + semantic) instead of erroring out.

### Server (`server`)

The server module provides three interface layers sharing the same search/index logic.

| File | Responsibility |
|------|---------------|
| `cli.rs` | CLI command handlers: `cmd_index()` (full pipeline with incremental support, progress bars), `cmd_search()` (4 modes with colored output), `cmd_status()` (index status report). JSON output support via `--json` flag. |
| `http.rs` | axum-based HTTP REST API. Routes: `POST /search`, `POST /index`, `GET /status`, `GET /health`. Uses `Arc<AppState>` with shared config and embedder. |
| `mcp.rs` | MCP (Model Context Protocol) Server over stdio. Implements JSON-RPC 2.0. Tools: `seekr_search`, `seekr_index`, `seekr_status`. `execute_search()` — shared search logic for all modes including 3-way RRF. |
| `daemon.rs` | Watch daemon for real-time incremental indexing. `run_watch_daemon()` — async main loop with 500ms debounce, event dedup, error recovery. Uses `Arc<RwLock<SeekrIndex>>` for concurrent access with HTTP server. |

**Key design:** All three server interfaces (CLI, HTTP, MCP) share the same underlying search/index logic, ensuring consistent behavior across interfaces.

### Config & Error (`config`, `error`)

| File | Responsibility |
|------|---------------|
| `config.rs` | `SeekrConfig` — TOML-based configuration with `index_dir`, `model_dir`, `embed_model`, `exclude_patterns`, `max_file_size`, plus sub-configs for server, search, and embedding. `project_index_dir()` — project isolation via blake3 hash (first 16 hex chars of canonical path hash). Default: `~/.seekr/config.toml`. |
| `error.rs` | `SeekrError` — top-level error enum aggregating all module errors via `#[from]`. Module-specific errors: `ScannerError`, `ParserError`, `EmbedderError`, `IndexError`, `SearchError`, `ServerError`, `ConfigError`. All defined with `thiserror`. |

## Data Flow

### Indexing Pipeline

```
User: `seekr-code index /path/to/project`

  ┌──────────────────────────────────────────────────────────┐
  │ 1. SCAN: Discover files                                  │
  │    walker.rs → parallel traverse with .gitignore support │
  │    filter.rs → check extension, size, binary content     │
  │    Result: Vec<PathBuf>                                  │
  └────────────────────────┬─────────────────────────────────┘
                           │
  ┌────────────────────────▼─────────────────────────────────┐
  │ 2. INCREMENTAL CHECK                                     │
  │    incremental.rs → compare mtime + blake3 hash          │
  │    Result: changed/new/deleted file lists                │
  └────────────────────────┬─────────────────────────────────┘
                           │
  ┌────────────────────────▼─────────────────────────────────┐
  │ 3. PARSE: Extract semantic chunks                        │
  │    treesitter.rs → detect language, load grammar         │
  │    chunker.rs → AST traversal → CodeChunk extraction     │
  │    summary.rs → generate structured summary per chunk    │
  │    Fallback: 50-line blocks for unknown/failed parses    │
  │    Result: Vec<CodeChunk>                                │
  └────────────────────────┬─────────────────────────────────┘
                           │
  ┌────────────────────────▼─────────────────────────────────┐
  │ 4. EMBED: Generate vector representations                │
  │    onnx.rs → tokenize (WordPiece) → ONNX inference       │
  │    Mean Pooling → L2 normalization → 384-dim vector      │
  │    batch.rs → batched processing (batch_size=32)         │
  │    Input: structured summary text (NOT raw code)         │
  │    Result: Vec<Vec<f32>>                                 │
  └────────────────────────┬─────────────────────────────────┘
                           │
  ┌────────────────────────▼─────────────────────────────────┐
  │ 5. INDEX: Store in persistent index                      │
  │    store.rs → add to vectors + inverted_index + chunks   │
  │    tokenize_for_index() → text tokens for inverted index │
  │    save() → bincode serialize to index.bin               │
  │    Result: index.bin + incremental_state.json            │
  └──────────────────────────────────────────────────────────┘
```

### Search Pipeline

```
User: `seekr-code search "handle user authentication" --mode hybrid`

  ┌──────────────────────────────────────────────────────────┐
  │ 1. LOAD: Load index from disk                           │
  │    store.rs → load index.bin (bincode)                  │
  │    Fallback: load index.json (v1 JSON)                  │
  └────────────────────────┬─────────────────────────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
  ┌──────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐
  │ Text Search │  │  Semantic   │  │ AST Pattern │
  │ text.rs     │  │ semantic.rs │  │ ast_pattern │
  │ regex match │  │ embed query │  │ parse query │
  │ + TF score  │  │ cosine KNN  │  │ match chunks│
  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
         │                │                 │
         └─────────┬──────┴─────────────────┘
                   │
          ┌────────▼────────┐
          │  RRF Fusion     │
          │  fusion.rs      │
          │  3-way combine  │
          │  k=60           │
          └────────┬────────┘
                   │
          ┌────────▼────────┐
          │  Ranked Results  │
          │  FusedResult[]   │
          └─────────────────┘
```

### Watch Daemon Pipeline

```
seekr-code serve --watch /path/to/project

  ┌──────────────┐     ┌──────────────┐
  │ HTTP Server  │     │ Watch Daemon │
  │ (axum)       │     │ (notify)     │
  └──────┬───────┘     └──────┬───────┘
         │                    │
         │  Arc<RwLock<       │  File system events
         │  SeekrIndex>>      │  (500ms debounce)
         │                    │
         └────────┬───────────┘
                  │
         Shared mutable index
         (read for search, write for updates)

  Daemon loop:
  1. Receive file events from notify
  2. Debounce 500ms, deduplicate by path
  3. For each changed file: parse → embed → update index
  4. For each deleted file: remove chunks from index
  5. Save updated index to disk
```

## Core Data Structures

### CodeChunk

The fundamental unit of indexed code:

```rust
pub struct CodeChunk {
    pub id: u64,              // Unique chunk ID (AtomicU64 counter)
    pub file_path: PathBuf,   // Source file path
    pub language: String,     // Programming language name
    pub kind: ChunkKind,      // Function, Class, Struct, etc.
    pub name: String,         // Symbol name
    pub signature: Option<String>,    // Function/method signature
    pub doc_comment: Option<String>,  // Documentation comment
    pub body: String,         // Full source code body
    pub byte_range: (usize, usize),   // Byte offset range
    pub line_range: (usize, usize),   // Line number range
}
```

### SeekrIndex

The main index structure with three sub-indexes:

```rust
pub struct SeekrIndex {
    pub version: u32,         // INDEX_VERSION (currently 2)
    pub vectors: HashMap<u64, Vec<f32>>,           // Vector index
    pub inverted_index: HashMap<String, Vec<(u64, u32)>>,  // Text index
    pub chunks: HashMap<u64, CodeChunk>,           // Metadata store
    pub embedding_dim: usize, // 384 for all-MiniLM-L6-v2
    pub chunk_count: usize,   // Total chunks
    // hnsw: Option<HnswMap> — in-memory HNSW graph (not serialized)
}
```

### FusedResult

Combined search result from multiple sources:

```rust
pub struct FusedResult {
    pub chunk_id: u64,
    pub fused_score: f32,
    pub text_score: Option<f32>,
    pub semantic_score: Option<f32>,
    pub ast_score: Option<f32>,
    pub matched_lines: Vec<usize>,
}
```

## Design Principles

1. **100% Local Execution** — No network calls during search. Model and data stay on disk. Network is only used for initial model download.

2. **Pluggable Backends** — The `Embedder` trait allows swapping embedding models without changing the rest of the pipeline. `Box<dyn Embedder>` enables runtime polymorphism.

3. **Graceful Degradation** — AST pattern search failures silently degrade to 2-way fusion. Missing index falls back to helpful error messages. V1 JSON index auto-migrates to v2 bincode.

4. **Project Isolation** — Each project gets its own index directory via blake3 hash of the canonical path. No cross-project data contamination.

5. **Incremental by Default** — Indexing uses mtime fast-path + blake3 content hash confirmation to avoid re-processing unchanged files. The `--force` flag overrides this.

6. **Triple Interface Model** — CLI, HTTP REST, and MCP Server all share the same search/index logic, ensuring behavioral consistency across all access methods.

7. **Zero-Copy Where Possible** — Memory-mapped index uses `memmap2` for zero-copy reads. Bincode serialization eliminates JSON parsing overhead.

8. **Structured Summaries for Embedding** — Instead of feeding raw code to the embedding model, structured summaries (kind+name+language+signature+doc+snippet) produce better semantic representations for natural language queries.

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Language | Rust (Edition 2024, MSRV 1.85.0) | Performance + safety |
| CLI | clap 4.5 | Argument parsing |
| Async Runtime | tokio 1.50 | Async I/O, channels, spawning |
| HTTP Server | axum 0.8 | REST API |
| AST Parsing | tree-sitter 0.25 + 16 grammars | Language-aware code chunking |
| Embedding | ort 2.0 (ONNX Runtime) | Local neural inference |
| Tokenizer | tokenizers 0.21 (HuggingFace) | WordPiece tokenization |
| File Traversal | ignore 0.4 | .gitignore-aware parallel walk |
| File Watching | notify 8.0 | File system event monitoring |
| Serialization | bincode 1.3 + serde 1.0 | Fast binary index persistence |
| Memory Mapping | memmap2 0.9 | Zero-copy index reads |
| Hashing | blake3 1.6 | Content hashing + project isolation |
| Error Handling | thiserror 2.0 + anyhow 1.0 | Structured errors |
| Regex | regex 1.11 | Text search matching |
| HTTP Client | reqwest 0.12 | Model download |
| Progress | indicatif 0.17 | CLI progress bars |
| Terminal Color | colored 3.0 | Colored CLI output |
| Config | toml 0.8 | Configuration file format |
| Benchmarks | criterion 0.5 | Performance benchmarks |
