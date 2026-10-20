# 阅读地图（file:line 直接跳）

行号对应 `deepseek-recipe@935817c` / `DeepInfer@8b16149`。克隆在 `.research/` 下。

## deepseek-recipe

### 数据结构

| 位置 | 看什么 |
| --- | --- |
| `deepseek-recipe-core/src/prompt.rs:26` | `Prompt` = text + special_ranges |
| `:54` / `:74` / `:111` | `push_special` / `split_off` / `special_ranges` |
| `deepseek-recipe-core/src/special_tokens.rs` | 16 个控制符的拼写（`<｜User｜>`、`<think>`、`｜DSML｜`…） |
| `deepseek-recipe-core/src/conversation.rs` | 协议无关的 `Conversation`（thinking_mode / tools / response_format） |

### 渲染

| 位置 | 看什么 |
| --- | --- |
| `deepseek-recipe-encoding/src/v4/dsv41.rs:250` | `render_conversation`：插 system、挂工具说明、挂 schema |
| `:21` | `Reasoning Effort: 50/75/100` 的映射 |
| `:70` / `:91` | 工具说明 / DSML 工具调用块的渲染 |
| `:134` | `render_message`：thinking marker、reasoning 回放、prefix 续写 |
| `:321` | `normalize_messages`：合并连续 user、按 call 顺序排 tool result |
| `deepseek-recipe-encoding/src/v4/mod.rs:24` | assistant 正文里允许出现的 grounding 控制符白名单 |
| `:197` | `push_assistant_content`：Aho-Corasick 找 grounding token |
| `deepseek-recipe-encoding/src/v4/dsv4.rs:33` | V4 差异：system 消息不带 system token |
| `deepseek-recipe/src/protocol/*/request/convert.rs` | 图片：V4 换成 `[Unsupported Image]`，V4.1 走 image placeholder |

### 编码

| 位置 | 看什么 |
| --- | --- |
| `deepseek-recipe-encoding/src/tokenizer.rs:48` | `PromptTokenizer::new`：克隆 + 清空 added vocabulary |
| `:56` | `with_added_vocabulary(AddedVocabulary::new())` ← 安全边界的实现 |
| `:90` | `encode_ids`：切段 → 批量编码 → 拼 special id |

### 流式解析

| 位置 | 看什么 |
| --- | --- |
| `deepseek-recipe/src/stream/mod.rs:66` | `OutputChunk`（Start/Content/Reasoning/ToolCall*/Finish） |
| `:102` | `ChunkGenerator` trait：一个 OutputChunk → N 个协议 chunk |
| `deepseek-recipe/src/stream/processor.rs:19` | `StreamProcessor` |
| `:53` | `with_tokenizer`：挂上才能收 `Tokens` chunk |
| `:75` | `into_protocol_stream`：主循环、stop sequence、usage 累计 |
| `deepseek-recipe/src/stream/output_parser.rs:36` | `ParsingOptions`（tool_calls / json / thinking stage / stop seq） |
| `:83` | `ChunkKind`：Text vs PlainTokens vs SpecialTokens ← 防伪造的核心 |
| `:328` | `add_tool_calls_open_branch`：DSML 开标签的分支 |
| `:751` | 手写 KMP failure table |
| `deepseek-recipe/src/stream/decoder.rs:14` | `U+FFFD` 作为「字符没收全」的哨兵 |
| `:34` | `StreamTokenizer` trait 的三条实现者义务 |

### 请求/响应

| 位置 | 看什么 |
| --- | --- |
| `deepseek-recipe/src/request/mod.rs:27` | `InferenceOptions`（temperature/top_p/logprobs/seed/constraint） |
| `:63` | `OutputConstraint`：json / regex / tool_call |
| `:158/162/166` | 8192 / 65536 / 131072 三档默认 |
| `deepseek-recipe/src/protocol/*/request/convert.rs` | 三种协议的字段映射（每个 500–970 行，挑差异看） |

## DeepInfer

### 调度与执行

| 位置 | 看什么 |
| --- | --- |
| `csrc/engine/scheduler.hpp:79` | `Scheduler` 构造：所有上限、`nextn`、`prompt_limit` |
| `:351` | `_schedule_prefill`：chunked prefill、block+state 联合分配、图片边界 |
| `:559` | `_schedule_decode`：抢占、clone state、MTP slot |
| `deepinfer/engine/llm_engine.py:860` | `step()`：schedule → forward → sample → post-process |
| `:67` | `DeferredRelease`：延后一步释放，和 GPU 重叠 |
| `:830` | `pre_sampler_sync`：真正释放点 |
| `deepinfer/engine/engine_type.py` | NORMAL / PROMPT / COMPLETION |
| `deepinfer/worker/model_runner.py:308` | `execute_main_model` |

### EPLB

| 位置 | 看什么 |
| --- | --- |
| `deepinfer/model_executor/eplb/__init__.py:21` | `resolve_num_extra_experts`：向上取整到 ep_size 的倍数 |
| `:42` | `ExpertLoadBalancer`：统计窗口、重排周期、冷启动提前 1/4 周期 |
| `:146` | `_update_rearrangement` |
| `deepinfer/model_executor/eplb/policy.py:73` | `rebalance_experts_global`：先复制热点，再摆位置 |
| `:12` | `_batch_greedy_indices`：按历史峰值做贪心 |
| `deepinfer/model_executor/eplb/shuffle.py` | NVLink symmetric memory 搬权重 |

### KV / SWA / YoCo

| 位置 | 看什么 |
| --- | --- |
| `deepinfer/worker/cache_engine.py:32` | `KVSTORE_GRAIN = 128` |
| `:88` | `get_inference_swa_block_size` = window + 回退 |
| `:101` | `get_kvstore_swa_block_size` = 推理 ring + extra + keep |
| `:113` | `_is_dropped_swa_layer`：按 `n_prefill_layer` 切 |
| `deepinfer/model_executor/models/v4_modules/yoco.py` | YoCo 开关与 consumer 判定 |
| `deepinfer/model_executor/models/deepseek.py:100` | `_init_yoco` 的两条 assert |
| `deepinfer/model_executor/models/v4_modules/attn.py` | MLA + sparse index + SWA 的运行时 |

### 投机 / 结构化输出 / 一致性

| 位置 | 看什么 |
| --- | --- |
| `deepinfer/model_executor/models/mtp/dflash.py:46` | `ConfidenceHead`：预测接受率 |
| `:82` | `DFlash`：draft 层、`max_MTP_value` |
| `csrc/engine/mtp_varlen.hpp` | 变长验证的 C++ 接口 |
| `deepinfer/structure_output/matcher.py:20` | `MAX_ROLLBACK_TOKENS = nextn + 2` |
| `:58` | `XGrammarMatcher`：trigger / bound / rollback |
| `deepinfer/structure_output/compiler.py` | 编译（异步） |
| `deepinfer/model_executor/layers/sampler.py` | mask 怎么进采样 |
| `csrc/engine/flops_estimator.hpp:28` | FP8 等效 FLOPs，draft 不算进 MFU |

### 环境变量速查

`deepinfer/environ.py` 里全部有默认值，最常改的：

```
DEEPINFER_USE_NEXTN_DECODE=5          # draft 数，0 = 关投机
DEEPINFER_ENABLE_MTP_VARLEN=1         # DSpark 变长验证
DEEPINFER_ZERO_OVERHEAD_SCHEDULING=1  # 异步调度（默认开）
DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE=compact
DEEPINFER_YOCO_PREFILL_KEEP_TOKENS=140
DEEPINFER_BATCH_INVARIANT=1           # 仅 NVIDIA
DEEPINFER_EXPERT_COUNT_WINDOW_SIZE=1000
DEEPINFER_EXPERT_REARRANGEMENT_STEP_INTERVAL=10000
```
