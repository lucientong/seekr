# ast-grep 技术门禁报告（Seekr v2.0）

日期：2026-09-10  
目标版本：`ast-grep-core` / `ast-grep-language` **0.44.1**（MSRV 1.85）  
结论：**未通过门禁 → 不引入依赖，保留现有跨语言 AST DSL**

## 1. 现有 DSL 基线

| 指标 | 结果 |
|------|------|
| Rust F1（score≥0.8） | ≥ 0.90（`tests/ast_golden_test.rs`） |
| TypeScript F1 | ≥ 0.90 |
| Python F1 | ≥ 0.90 |
| 整体 F1 | ≥ 0.95 |
| 结构误报（错误 kind） | 0 |
| Release 二进制体积 | **38,312,016** bytes（≈ 36.55 MiB） |
| Release 增量编译（touch `src/lib.rs`） | ≈ **61.4 s** |

Golden fixtures：`tests/fixtures/sample.rs` / `sample.ts` / `sample.py`。

## 2. Prototype 验证

尝试以 optional feature 引入：

```toml
ast-grep-core = { version = "=0.44.1", optional = true }
ast-grep-language = { version = "=0.44.1", default-features = false,
  features = ["tree-sitter-rust", "tree-sitter-python", "tree-sitter-typescript"],
  optional = true }
```

### 阻断性失败：`tree-sitter` `links` 冲突

- Seekr 使用 `tree-sitter ^0.25`
- ast-grep 0.44.1 需要 `tree-sitter ^0.26.3`
- 两者均声明 `links = "tree-sitter"`，**Cargo 拒绝在同一依赖图中共存**
- 因此无法在不升级全库 tree-sitter 生态的前提下完成体积/编译门禁实测

独立探针（`/tmp/astgrep-probe`，仅 3 种 grammar）可编译；release 探针体积约 **5.0 MiB**，且自带第二套 grammar（Rust/Python/TypeScript），预示即便解决 `links` 冲突，合并进 Seekr 后增量也极易超过 **≤2 MiB 且 ≤5%** 的体积门禁。

### 能力抽检（独立探针）

在独立 crate 中，`$VAR` / `$$$ARGS` / async / Unicode 标识符 / 泛型等 pattern 可用；但该结果**不能**抵消与 Seekr 主依赖的链接冲突。

## 3. 门禁判定

| 门禁 | 要求 | 结果 |
|------|------|------|
| 整体 F1 | ≥ 0.95 | DSL 基线通过；ast-grep 未能接入主 crate |
| 每语言 F1 | ≥ 0.90 | DSL 基线通过 |
| 结构误报 | 0 | DSL 基线通过 |
| Release 体积增幅 | ≤5% 且 ≤2 MiB | **无法测量 / 预期不达标**（links 冲突 + 重复 grammar） |
| 干净编译增幅 | ≤15% | **无法测量**（无法解析依赖） |

**最终决定：撤回 prototype，不合并 ast-grep。** 现有签名 DSL 继续作为 Hybrid AST 路径；raw ast-grep pattern 入口不提供。

## 4. 若未来重试

需同时满足：

1. 将 Seekr 的 tree-sitter + 全部 grammar 升级到与目标 ast-grep 兼容的单一 `tree-sitter` 版本；或
2. ast-grep 提供可复用宿主已加载 `TSLanguage`、且不强制第二套 `links` 的集成方式；并且
3. 体积/编译门禁仍须重新实测通过。
