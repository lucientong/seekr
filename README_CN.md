# seekr-code

语义化代码搜索引擎，比 grep 更智能。

支持**文本正则** + **语义向量** + **AST 模式**搜索 — 100% 本地运行，数据不会离开你的机器。

[English](README.md)

## 功能特性

- 🔍 **文本搜索** — 高性能正则表达式匹配
- 🧠 **语义搜索** — 基于 ONNX 的本地 Embedding + HuggingFace WordPiece 分词器 + HNSW 近似最近邻索引，按语义查找代码
- 🌳 **AST 模式搜索** — 通过 Tree-sitter 匹配函数签名、结构体、类（如 `fn(*) -> Result`）
- ⚡ **混合模式** — 三路倒数排名融合（3-way RRF）组合全部三种搜索，获得最佳结果
- 📡 **MCP 服务器** — 支持 Model Context Protocol，接入 AI 编辑器
- 🌐 **HTTP API** — REST API，方便与其他工具集成
- 🔄 **增量索引** — 仅重新处理变更文件
- 👁️ **Watch Daemon** — 实时文件监控，自动增量重建索引
- 🗂️ **多语言支持** — Rust、Python、JavaScript、TypeScript、Go、Java、C、C++、Ruby、Bash、HTML、CSS、JSON、TOML、YAML

## 安装

### 从 crates.io 安装

```bash
cargo install seekr-code
```

### 从源码安装

```bash
git clone https://github.com/lucientong/seekr.git
cd seekr
cargo install --path .
```

安装后，`seekr-code` 命令即可在终端中使用。

### 系统要求

- Rust 1.85.0 及以上
- C/C++ 编译器（用于编译 tree-sitter 语法库）

## 快速开始

### 1. 构建索引

```bash
# 索引当前项目
seekr-code index

# 索引指定路径
seekr-code index /path/to/project

# 强制完全重建索引
seekr-code index --force
```

### 2. 搜索代码

```bash
# 混合搜索（默认，组合文本 + 语义 + AST）
seekr-code search "authenticate user"

# 文本正则搜索
seekr-code search "fn.*authenticate" --mode text

# 语义搜索（按含义查找）
seekr-code search "用户登录验证" --mode semantic

# AST 模式搜索
seekr-code search "fn(*) -> Result" --mode ast
seekr-code search "struct *Config" --mode ast
seekr-code search "async fn(*)" --mode ast
```

### 3. 查看索引状态

```bash
seekr-code status
```

### 4. JSON 输出

所有命令支持 `--json` 参数，输出机器可读的 JSON 格式：

```bash
seekr-code search "authenticate" --json
seekr-code index --json
seekr-code status --json
```

## 服务器模式

### HTTP API

```bash
# 启动 HTTP API 服务器（默认：127.0.0.1:7720）
seekr-code serve

# 自定义地址和端口
seekr-code serve --host 0.0.0.0 --port 8080

# 启动并开启文件监听 — 文件变更时自动重建索引
seekr-code serve --watch /path/to/project
```

**接口列表：**

| 方法   | 路径      | 说明           |
|--------|-----------|----------------|
| POST   | /search   | 搜索代码       |
| POST   | /index    | 触发索引构建   |
| GET    | /status   | 查询索引状态   |
| GET    | /health   | 健康检查       |

**示例：**

```bash
curl -X POST http://127.0.0.1:7720/search \
  -H "Content-Type: application/json" \
  -d '{"query": "authenticate user", "mode": "hybrid", "top_k": 10}'
```

### MCP 服务器（AI 编辑器集成）

```bash
# 以 MCP 服务器模式启动（通过 stdio）
seekr-code serve --mcp
```

**MCP 工具：**

- `seekr_search` — 搜索代码，支持文本、语义、AST 和混合模式
- `seekr_index` — 构建/重建搜索索引
- `seekr_status` — 获取索引状态

**MCP 配置示例**（Claude Desktop、CodeBuddy 等）：

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

## AST 模式语法

```text
[async] [pub] fn [名称]([参数类型, ...]) [-> 返回类型]
class 类名
struct 结构体名
enum 枚举名
trait Trait名
```

**示例：**

| 模式                     | 匹配                              |
|--------------------------|------------------------------------|
| `fn(string) -> number`   | 接受 string 参数、返回 number 的函数 |
| `fn(*) -> Result`        | 任何返回 Result 的函数              |
| `async fn(*)`            | 任何 async 函数                    |
| `fn authenticate(*)`     | 名为 "authenticate" 的函数         |
| `struct *Config`         | 名称以 "Config" 结尾的结构体       |
| `class *Service`         | 名称以 "Service" 结尾的类          |
| `enum *Error`            | 名称以 "Error" 结尾的枚举          |

## 配置

配置文件路径：`~/.seekr/config.toml`

```toml
# 索引存储目录
index_dir = "~/.seekr/indexes"

# ONNX 模型目录
model_dir = "~/.seekr/models"

# Embedding 模型名称
embed_model = "all-MiniLM-L6-v2"

# 索引文件大小上限（字节）
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

## 工作原理

1. **扫描器（Scanner）** — 遍历项目目录，遵循 `.gitignore`，按文件类型/大小过滤
2. **解析器（Parser）** — 使用 Tree-sitter 将源文件解析为语义代码块（函数、类、结构体等）
3. **嵌入器（Embedder）** — 使用 ONNX Runtime + all-MiniLM-L6-v2 + HuggingFace WordPiece 分词器生成向量嵌入
4. **索引（Index）** — 构建倒排文本索引 + HNSW 向量索引，通过 bincode 二进制格式持久化到磁盘
5. **搜索（Search）** — 文本正则、语义 HNSW 近似最近邻搜索（暴力 KNN 兜底）、AST 模式匹配，通过三路 RRF 融合
6. **监听（Watch）** — 可选的文件系统监控，支持防抖增量重建索引

## 性能基准

运行基准测试：

```bash
cargo bench --bench search_bench
```

基准测试覆盖：
- 索引构建（100 / 500 / 1000 个代码块）
- 向量搜索延迟（500 / 1000 / 5000 个代码块）
- 文本搜索延迟（倒排索引）
- 余弦相似度计算（384 维）
- 索引保存/加载吞吐量（bincode）

## 环境变量

| 变量         | 说明                                        |
|-------------|---------------------------------------------|
| `SEEKR_LOG` | 日志级别过滤器（如 `seekr_code=debug`）     |
| `RUST_LOG`  | 备用日志级别（当 `SEEKR_LOG` 未设置时生效） |

## 许可证

[Apache License 2.0](LICENSE)

## 作者

[lucientong](https://github.com/lucientong)
