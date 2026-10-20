# 哪些是跑出来的，哪些只是读出来的

诚实分三档：**实测** / **读码** / **读文档（Q&A/README，未独立验证）**。

## 实测（本机，Windows，可复现）

`labs/recipe-lab`（Rust，`cargo run --bin 03_check`）：

| 断言 | 结果 |
| --- | --- |
| 渲染 5 token，以 `<｜Assistant｜></think>` 结尾 | PASS |
| 注入 `<｜User｜>`：recipe 1 个控制符 / 朴素 2 个 | PASS（13 vs 10 id） |
| 普通 token 拼的 `</think>` 留在 reasoning；真 special token 切到 content | PASS |
| DSML 解析成 tool_calls delta，参数拼回 `{"city": "Beijing"}`，finish=tool_calls | PASS |

`labs/eplb-sim/sim.py`（纯 torch，无需 GPU）：

| 项 | 结果 |
| --- | --- |
| 合成负载下 naive vs 重排的 peak load | 268.8 → 191.0（layer0） |
| 热点专家副本数 | 3（冷门 1） |

**这些是我自己的输入，不是 DeepSeek 的 benchmark**，只能证明策略形状对，不能引用成性能数字。

## 读码确认（行号见 `03-anchors.md`）

- `Prompt` 的 special_ranges 在构造时记录，`PromptTokenizer` 用清空 added vocabulary 的副本编普通文本；
- `ChunkKind` 三态决定 `</think>` / DSML 是内容还是标记；
- 流式解析是 KMP + Stage 状态机，UTF-8 不完整时缓冲；
- `resolve_max_tokens` 的三档默认与「先检查 requested、再夹到剩余上下文」的顺序；
- `OutputConstraint` 只声明，不执行；
- Scheduler 在 C++，`_schedule_prefill` / `_schedule_decode` 分开；
- `DeferredRelease` 在下一步 `pre_sampler_sync` 释放；
- EPLB 两级贪心、向上取整、冷启动提前 1/4 周期；
- `KVSTORE_GRAIN=128`、ring = 134 / 396、`MAX_ROLLBACK_TOKENS = nextn+2`；
- `_init_yoco` 的两条 assert；
- `FlopsEstimator` 排除 draft 计算。

## 读文档（Q&A PDF / README，未独立验证）

这些数字来自仓库自带的 `doc/DeepSeek V4.1 Flash 推理部署 Q&A.pdf`，我**没有**对应的集群去验证：

- 单集群 155 万亿 tokens/天；每卡每小时 38 美元（按谷价）；
- P:D ≈ 1:1，高效最小单元 64P+64D；P 用 EP2~8/DP2~8，D 用 EP64~128；
- EPLB 生产参数与 `moe_lb`：prefill 1.074/1.045，decode 1.090；
- 3FS ~100 PiB SSD，TTL 3 天（agent）/ 7 天（chatbot），3 天 TTL 损失 0.5–1% 命中率；
- SWA 每序列 1,415,040 B（20 层 × 134 × 528 B）；
- DSpark 动态验证：5.00 → 3.2–4.4 个 draft，接受长度 3.29 vs 3.20，+4~7% 吞吐；
- 负载均衡后能达到完全均衡离线 benchmark 的 80%。

## 跑不了的（环境限制）

| 尝试 | 结果 |
| --- | --- |
| `pip install deepseek-recipe`（Windows，Py 3.14 / 3.12） | 失败：sdist 现场编译 `opencv v0.93.7` 的 build script 报错（缺 OpenCV 4.11+ / libclang） |
| 用 Rust crate 代替 | 成功，`labs/recipe-lab` 就是这么跑的 |
| `python examples/offline_inference.py` | 未尝试：需要 Linux + SM100/昇腾 950 + V4.1-Flash 权重 |
| 导入 `deepinfer`（Windows） | 失败在 `os.uname()`（`deepinfer/__init__.py` 只支持 Linux） |
| 单独跑 `eplb/policy.py` | 成功（用 `importlib` 绕过包 `__init__`），见 `labs/eplb-sim` |

## 一个容易误读的点

DeepInfer 的 benchmark 用**全接受**计账：`avg_SPS × requests × (1+nextn) / cp`。真实接受率低于 1 时，实际吞吐低于这个数。README 的 Pareto 图是同一套口径，横向比较没问题，别当成「每步一定出 6 个 token」。
