# 检索质量门禁报告（Seekr v2.0）

日期：2026-09-10  
Corpus：`tests/retrieval_quality_test.rs` **v1**  
结论：**首轮 Hybrid 已达标 → 不引入默认 reranker**

## 1. 离线分项（始终运行）

| 指标 | 结果 |
|------|------|
| 多语言 BM25 Recall@5 | ≥ 0.95 |
| 多语言 BM25 MRR | ≥ 0.95 |
| Dummy Hybrid smoke | 有结果即可（不设语义门槛） |

## 2. 真实模型 Hybrid 门禁

触发条件：`SEEKR_QUALITY_GATE=1`（CI `quality` job / 本地显式开启）。

| 指标 | 首轮实测 | Floor（baseline − 0.05） |
|------|----------|---------------------------|
| Recall@5 | **1.000** | 0.950 |
| MRR | **1.000** | 0.950 |

模型：本地 all-MiniLM-L6-v2 量化 ONNX（`SEEKR_MODEL_DIR` 或 `~/.seekr/models`）。  
CI：`.github/workflows/ci.yml` 的 `quality` job 缓存模型目录。

## 3. Reranker 决策

计划约定：仅当 Hybrid 未达门槛 **且** cross-encoder 实验能补上时，才引入默认关闭的 `Reranker` trait + 第二 ONNX 模型。

首轮 Hybrid 已达 Recall@5/MRR 门槛，因此：

- **不增加** reranker trait / 候选池 / 第二模型
- **不增加** 默认模型体积与查询延迟
- 明确结论：**不需要 reranker**

若未来 corpus 扩大导致 Hybrid 跌破 floor，再重新评估可选 reranker。
