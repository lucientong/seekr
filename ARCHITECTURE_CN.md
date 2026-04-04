# Seekr 架构文档

> **seekr-code** — 比 grep 更智能的语义代码搜索引擎。

本文档详细介绍 seekr-code v1.0.0 的内部架构、模块设计、数据流和关键设计决策。

---

## 概述

Seekr 是一个 **100% 本地运行** 的代码搜索引擎，组合了三种互补的搜索策略：

1. **文本正则搜索** — 高性能正则表达式匹配（类似 ripgrep）
2. **语义向量搜索** — 基于 ONNX 的本地 embedding 模型 + HNSW 近似最近邻搜索（暴力 KNN 兜底）
3. **AST 模式搜索** — 基于 Tree-sitter 的函数签名模式匹配

三种模式可独立使用或通过 **倒数排名融合（RRF）** 组合为混合搜索。所有数据永远不离开你的机器。

## 高层架构

```
┌──────────────────────────────────────────────────────────┐
│                      用户接口层                           │
│  ┌─────────┐  ┌────────────┐  ┌─────────────────────┐   │
│  │  CLI    │  │ HTTP REST  │  │ MCP (JSON-RPC)      │   │
│  │ (clap)  │  │ (axum)     │  │ (stdio)             │   │
│  └────┬────┘  └─────┬──────┘  └──────────┬──────────┘   │
│       └─────────┬───┴────────────────────┘               │
│                 ▼                                        │
│  ┌───────────────────────────────────────────────────┐   │
│  │ 搜索引擎: 文本正则 + 语义向量 + AST 模式 → RRF   │   │
│  └───────────────────────────────────────────────────┘   │
│  ┌───────────────────────────────────────────────────┐   │
│  │ 索引引擎: 向量存储 + 倒排索引 + Chunk 元数据     │   │
│  │           bincode 持久化 / mmap 零拷贝 / 增量索引 │   │
│  └───────────────────────────────────────────────────┘   │
│  ┌───────────────────────────────────────────────────┐   │
│  │ 处理管线: 扫描器(ignore) → 解析器(tree-sitter)   │   │
│  │           → 嵌入引擎(ONNX all-MiniLM)            │   │
│  └───────────────────────────────────────────────────┘   │
│  ┌────────────┐ ┌────────────────────────────────────┐   │
│  │ 配置(TOML) │ │ Watch 守护进程 (notify + tokio)    │   │
│  └────────────┘ └────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────┘
```

## 模块详解

项目由 7 个核心模块 + 2 个基础设施模块组成：

```
src/
├── lib.rs              # 库入口：导出所有模块、INDEX_VERSION、VERSION
├── main.rs             # 二进制入口：CLI 参数解析 (clap)
├── config.rs           # 配置管理（基于 TOML）
├── error.rs            # 统一错误类型 (thiserror)
├── scanner/            # 文件发现与监听
│   ├── walker.rs       # 并行目录遍历（ignore crate）
│   ├── filter.rs       # 文件过滤（扩展名、大小、二进制检测）
│   └── watcher.rs      # 文件系统事件监听（notify crate）
├── parser/             # 代码解析与分块
│   ├── treesitter.rs   # 16 种语言支持和语法管理
│   ├── chunker.rs      # AST 感知的语义分块
│   └── summary.rs      # 结构化代码摘要生成
├── embedder/           # 嵌入模型管理
│   ├── traits.rs       # Embedder trait（可插拔后端接口）
│   ├── onnx.rs         # ONNX Runtime 后端（all-MiniLM-L6-v2）
│   └── batch.rs        # BatchEmbedder + DummyEmbedder
├── index/              # 索引存储与管理
│   ├── store.rs        # SeekrIndex：向量 + 倒排 + 元数据
│   ├── mmap_store.rs   # 内存映射索引（零拷贝）
│   └── incremental.rs  # 增量索引（mtime + blake3）
├── search/             # 搜索算法
│   ├── text.rs         # 正则文本搜索
│   ├── semantic.rs     # 向量相似度搜索
│   ├── ast_pattern.rs  # AST 模式匹配
│   └── fusion.rs       # RRF 融合（二路/三路）
└── server/             # 接口层
    ├── cli.rs          # CLI 命令处理器
    ├── http.rs         # HTTP REST API (axum)
    ├── mcp.rs          # MCP 服务器 (JSON-RPC 2.0)
    └── daemon.rs       # Watch 守护进程
```

### Scanner（扫描器）

- **walker.rs** — 使用 `ignore` crate 并行遍历，遵守 `.gitignore`，通过 `Mutex<Vec<PathBuf>>` 收集结果
- **filter.rs** — 大小限制（10MB）、60+ 种扩展名白名单、二进制检测（前 8KB 空字节比 > 1%）
- **watcher.rs** — 同步/异步文件监听，`dedup_events()` 事件去重

### Parser（解析器）

- **treesitter.rs** — 16 种语言（Rust, Python, JS, TS, Tsx, Go, Java, C, C++, JSON, TOML, YAML, HTML, CSS, Ruby, Bash）
- **chunker.rs** — AST 递归提取语义块，回退 50 行分块，`AtomicU64` 全局 ID，最小 2 行
- **summary.rs** — 生成结构化摘要用于嵌入：kind+name + 语言 + 签名 + 文档注释(≤500字) + 代码片段(5行)

核心类型：`CodeChunk`（id, file_path, language, kind, name, signature, doc_comment, body, byte_range, line_range）

### Embedder（嵌入引擎）

- **traits.rs** — `Embedder` trait（`embed()`, `embed_batch()`, `dimension()`），要求 `Send + Sync`
- **onnx.rs** — all-MiniLM-L6-v2 量化版，384 维，WordPiece 分词，截断 256 tokens，Mean Pooling + L2 归一化
- **batch.rs** — 可配置批大小 + 进度回调，`DummyEmbedder` 用于测试

### Index（索引引擎）

- **store.rs** — `SeekrIndex`：向量 HashMap + 倒排索引 + chunk 元数据。bincode 序列化(`index.bin`)，兼容 v1 JSON 回退
- **mmap_store.rs** — `memmap2` 零拷贝读取
- **incremental.rs** — mtime 快速路径 + blake3 内容哈希确认

### Search（搜索引擎）

- **text.rs** — 正则匹配，`match_count + density * 10` 评分
- **semantic.rs** — 嵌入查询 → HNSW 近似最近邻搜索（暴力 KNN 兜底）
- **ast_pattern.rs** — 自定义模式语法，通配符 `*`，加权评分（kind 0.3, name 0.3, params 0.15, return 0.15, qualifiers 0.1）
- **fusion.rs** — RRF 公式 `score = sum(1/(k + rank + 1))`，k=60，AST 失败时静默降级为二路融合

### Server（服务层）

- **cli.rs** — `cmd_index()`/`cmd_search()`/`cmd_status()`，支持 `--json` 输出
- **http.rs** — axum REST API：`POST /search`, `POST /index`, `GET /status`, `GET /health`
- **mcp.rs** — JSON-RPC 2.0 over stdio，工具：`seekr_search`, `seekr_index`, `seekr_status`
- **daemon.rs** — 500ms 防抖，事件去重，`Arc<RwLock<SeekrIndex>>` 并发访问

## 数据流

### 索引管线

```
扫描(walker+filter) → 增量检查(mtime+blake3) → AST解析(tree-sitter)
→ 摘要生成(summary) → 向量嵌入(ONNX) → 存储索引(bincode → index.bin)
```

### 搜索管线

```
加载索引(index.bin) → 并行执行三路搜索 → RRF 融合 → 排序输出
    ├── 文本正则 (regex + TF)
    ├── 语义向量 (embed query → cosine KNN)
    └── AST 模式 (parse pattern → match chunks, 可降级)
```

### Watch 守护进程

HTTP 服务器与 Watch 守护进程共享 `Arc<RwLock<SeekrIndex>>`。守护进程循环：接收事件 → 500ms 防抖 → 去重 → 对变更文件解析嵌入更新索引 → 对删除文件移除 chunks → 保存。

## 核心数据结构

| 结构体 | 描述 |
|--------|------|
| `CodeChunk` | 索引代码的基本单位（id, file_path, language, kind, name, signature, doc_comment, body, ranges） |
| `SeekrIndex` | 主索引（version, vectors HashMap, inverted_index HashMap, chunks HashMap, embedding_dim, chunk_count） |
| `FusedResult` | 多源融合结果（chunk_id, fused_score, text/semantic/ast_score, matched_lines） |
| `AstPattern` | AST 搜索模式（qualifiers, kind, name_pattern, param_patterns, return_pattern） |
| `IncrementalState` | 增量状态（files HashMap: path → mtime + blake3 hash + chunk_ids） |

## 设计原则

1. **100% 本地执行** — 搜索无网络调用，仅初次下载模型需联网
2. **可插拔后端** — `Embedder` trait 支持未来替换嵌入模型
3. **优雅降级** — AST 失败降级为二路融合；v1 索引自动迁移到 v2
4. **项目隔离** — blake3(canonical_path) 前 16 hex 字符作为索引目录名
5. **默认增量** — mtime 快速路径 + blake3 确认，避免重处理未变文件
6. **三重接口** — CLI / HTTP / MCP 共享搜索索引逻辑，行为一致
7. **零拷贝** — memmap2 零拷贝读取，bincode 替代 JSON 消除解析开销
8. **HNSW 不持久化** — HNSW 图 `#[serde(skip)]`，加载时从向量数据重建，避免磁盘格式依赖 `instant-distance` 内部结构
8. **结构化摘要** — 输入嵌入模型的是 kind+name+language+signature+doc+snippet 而非原始代码

## 技术栈

| 组件 | 技术 | 用途 |
|------|------|------|
| 语言 | Rust (Edition 2024, MSRV 1.85.0) | 性能 + 安全 |
| CLI | clap 4.5 | 参数解析 |
| 异步运行时 | tokio 1.50 | 异步 I/O |
| HTTP | axum 0.8 | REST API |
| AST 解析 | tree-sitter 0.25 + 16 语法 | 语言感知分块 |
| 嵌入 | ort 2.0 (ONNX Runtime) | 本地神经推理 |
| 分词 | tokenizers 0.21 (HuggingFace) | WordPiece |
| 文件遍历 | ignore 0.4 | .gitignore 感知 |
| 文件监听 | notify 8.0 | 文件系统事件 |
| 序列化 | bincode 1.3 + serde 1.0 | 快速二进制持久化 |
| 内存映射 | memmap2 0.9 | 零拷贝读取 |
| 哈希 | blake3 1.6 | 内容哈希 + 项目隔离 |
| 错误处理 | thiserror 2.0 + anyhow 1.0 | 结构化错误 |
| 正则 | regex 1.11 | 文本搜索 |
| HTTP 客户端 | reqwest 0.12 | 模型下载 |
| 进度条 | indicatif 0.17 | CLI 进度显示 |
| 终端颜色 | colored 3.0 | 彩色输出 |
| 配置 | toml 0.8 | 配置文件 |
| 近邻搜索 | instant-distance 0.6 | HNSW 近似最近邻 |
| 基准测试 | criterion 0.5 | 性能基准 |
