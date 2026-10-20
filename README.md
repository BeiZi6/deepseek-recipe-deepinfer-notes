# DeepSeek 两个新仓库：学习笔记 + 可跑实验

2026-10-20 抓取的两个仓库（commit 见下），一个是**协议层**，一个是**推理引擎层**。

| 仓库 | 一句话 | 规模 | 你能在本机跑吗 |
| --- | --- | --- | --- |
| [deepseek-recipe](https://github.com/deepseek-ai/deepseek-recipe) | API 协议 ↔ 对话 ↔ prompt ↔ token id ↔ 协议响应 的双向编解码 | ~20.8k 行 Rust + 1.3k 行 Python | **能**，`labs/recipe-lab` 已验证 |
| [DeepInfer](https://github.com/deepseek-ai/DeepInfer) | 生产推理引擎（离线版），V4.1-Flash 专用 | ~47.7k 行 Python + 5.1k 行 C++（不含 vendor） | **不能**，需 Linux + SM100/昇腾 950 |

抓取版本：`deepseek-recipe@935817c`（2026-10-20）、`DeepInfer@8b16149`（2026-10-20）。

## 先看这两张图

**Recipe = 一个纯函数式的翻译层，没有模型、没有 HTTP、没有工具执行：**

```
Chat Completions ┐
Responses        ├─► Conversation ─► Prompt(text + special_ranges) ─► token ids
Messages         ┘        ▲                                              │
                          │                                              ▼
     协议响应 ◄── StreamProcessor ◄── 后端 chunk（Text / Tokens / Finish）
```

**DeepInfer = 围绕「一次 forward 一步」组织的流水线，CPU 和 GPU 各自不闲着：**

```
CPU:  schedule(N+1) ──► launch forward(N+1) ──► post-process(N-1) ──► launch sample(N+1)
GPU:        forward(N) ─────────────► sample(N)
```

## 学习顺序（约 3–4 小时）

1. `notes/01-recipe.md` — 协议层怎么做到「注入不了 special token」，以及流式解析状态机。
2. `labs/recipe-lab/` — 跑 `cargo run --bin 03_check`，四条断言把上面的说法钉死。
3. `notes/02-deepinfer.md` — EP/DP/CP、EPLB、SWA+YoCo、DSpark、零开销调度、结构化输出。
4. `labs/eplb-sim/sim.py` — 用 20 行跑通 EPLB 的贪心重排，看 peak load 从 268 降到 191。
5. `notes/03-anchors.md` — 想深挖时按 file:line 直接跳。
6. `notes/04-verified.md` — 哪些结论是**跑出来的**，哪些只是**读出来的**。
7. `tweet.md` — 推文。

## 目录

```
notes/            四份笔记
labs/recipe-lab/  Rust 实验（渲染 / 流式 / 断言检查）
labs/eplb-sim/    EPLB 策略的最小复现
tweet.md          推文
.research/        两个仓库的浅克隆（gitignore，可删）
```

## 两条最容易踩的坑

- **`pip install deepseek-recipe` 在 Windows 上装不上**：sdist 要现场编译 OpenCV 绑定，实测在 `opencv v0.93.7` 的 build script 处失败。用 Rust crate，或上 Linux。
- **DeepInfer 不是 vLLM 的替代品**：只有离线 `LLM.generate()`，没有 OpenAI server，KV cache 存储 / PD 分离 / 负载均衡都**不在**这个 release 里。它的价值是「看 DeepSeek 自己怎么调这套硬件」。
