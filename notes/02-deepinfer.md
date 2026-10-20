# DeepInfer：一个「只为一种模型、一种硬件」写的推理引擎

## 先划清边界

README 自己写得很清楚，这个 release **只有离线推理**。要变成完整服务，还缺：

- KV cache 存储系统（3FS 那套）
- VE / Prefill / Decode 三段分离
- 跨 DP 副本的负载均衡

代码里能看到的骨架是齐的：`EngineType.NORMAL / PROMPT / COMPLETION`（`deepinfer/engine/engine_type.py`）就是 PE/CE 分离的接口，`csrc/engine/scheduler.hpp` 里也按 engine type 分支。缺的是「谁来路由、谁来存 KV」。

支持的组合只有一个：**DeepSeek-V4.1-Flash + SM100（或昇腾 950）+ Linux + CUDA 13 / CANN 9.2 + Python 3.12**。没有通用模型加载路径，`v4_modules/` 里全是 V4 专有结构（MLA、Indexer、Engram、HC、YoCo）。

## 并行：为什么生产不用 TP

Q&A 的原话：Prefill `EP2~8/DP2~8`，Decode `EP64~128/DP64~128`（Wide-EP），VE 单卡，**暂未用 TP、CP**。

代码里 CP 是支持的（`--cp`、`prepare_cp_mirror`、`cp_broadcast_multi_tensor`），但生产不开。原因在 Q&A 里：Attention 部分用 DP（`#DP=#EP`），因为 MLA 的 KV 已经被压得很小，按 head 切（TP）带来的通信不划算；而 MoE 专家必须切（EP），否则单卡放不下。

Wide-EP 的收益是**访存**：每步只读本卡那几个专家，EP 越大每卡专家越少。代价是一个实例的所有卡在同一个通信域，**一张卡挂全实例挂**——所以 Q&A 里 failover 的做法是「按健康卡数重新定 EP 大小」，而不是原地恢复。

## EPLB：两级贪心，先复制热点，再摆位置

`model_executor/eplb/` 干两件事：

1. `resolve_num_extra_experts()`（`__init__.py:21`）把冗余专家数向上取整，使 `num_routed + num_extra` 能被 `ep_size` 整除；
2. `rebalance_experts_global()`（`policy.py:73`）每 M 步重排一次。

重排分两阶段：

**阶段一：给热点专家加副本。** 对每个 expert 算「每副本负载」，每轮复制当前最热的那个（`policy.py` 里的 `score.max` / `scatter`），共复制 `num_redundant` 次。注意它优化的是「历史 N 步里最坏那步的峰值」，不是平均值——`values.sum(0).argmin()` 前面先做了 `max`。

**阶段二：把物理副本摆到卡上。** `_batch_greedy_indices()`（`policy.py:12`）按负载降序，逐个 expert 试每个 rank，选「放进去之后，历史各步的 rank 最大负载之和最小」的那个。

我用合成负载跑了一遍（`labs/eplb-sim/sim.py`，8 专家 / 12 物理槽 / 6 rank × 2 槽）：

```
naive 布局   peak per layer: [268.8, 255.4, 274.5]
重排后       peak per layer: [191.0, 180.6, 191.5]
副本数（layer0）: [3, 2, 2, 1, 1, 1, 1, 1]
```

这是**我自己造的负载**，不是他们的 benchmark；但它验证了策略的形状：热点专家拿到 3 个副本，冷门 1 个。

Q&A 给的生产参数：Prefill 每卡 4 个冗余起步、窗口 100、每 1000 步重排，`moe_lb` 1.074（EP4DP4）/1.045（EP2DP2）；Decode 每卡 1 个冗余、窗口 1000、每 10000 步重排，`moe_lb` 1.090。`moe_lb` = 最慢 rank / 均值。

权重 shuffle 走 NVLink symmetric memory（`expert_model_parallel_shuffle_inplace`），因为一次重排要把专家权重在卡间搬家，走 PCIe 太慢。

## SWA + YoCo：把「重算」压成常数

这是整个仓库最值得读的部分，因为它同时动了**缓存格式**、**调度**和**模型结构**。

背景：V4.1 是 CED 架构，encoder 层产出 Block KV，decoder 层（后 20 层）用 SWA（滑动窗口 128）。如果 decoder 的 SWA KV 丢了，精确重建要重跑 2560 个 token。

三个关键常数：

| 常数 | 值 | 出处 |
| --- | --- | --- |
| 落盘粒度 | 128 token | `cache_engine.py:32 KVSTORE_GRAIN` |
| ring 长度（推理） | window + 最大回退 = 128 + 6 = 134 | `get_inference_swa_block_size`（:88） |
| ring 长度（落盘） | 134 + 128(extra) + 134(keep) = 396 | `get_kvstore_swa_block_size`（:101） |

回退 6 从哪来：`DEEPINFER_MAX_BACKTRACKING_TOKENS = max(env, nextn + 1)`，nextn 默认 5 → 6。

**YoCo 截断**（`models/v4_modules/yoco.py`、`deepseek.py` 的 `_init_yoco`）：prefill 时 decoder 层只重算 prompt 末尾那一小段（Q&A 说 128 个 token；README 的默认配置保留 140，覆盖窗口 128 + 回退 6），产出**只给 decode 用、不进前缀缓存**。所以总 prefill 计算量近乎减半。

注意 `DEEPINFER_YOCO_PREFILL_KEEP_TOKENS` 的代码默认值是 0（`environ.py`），必须显式设置；README 给的 140 是 V4.1 的推荐值。

代价是两条硬约束（代码里是 assert）：

- DFlash 的 target 层必须**全在** consumer 半区或**全不在**；
- Engram 层不能出现在 consumer 半区。

命中规则也因此变复杂：最大命中长度 `L-134`；先查 Block KV 得到 `L1`（128 的倍数），再看 `[L1-128, L1]` 里有没有 SWA KV，没有就退一块重算。

SWA KV 不进分页 block，而是放在独立的 state cache，每个序列一个定长 slot，按「是否持久化」切成 `[stored region | dropped region]`。Q&A 给了精确账：20 层 × 134 token × 528 B = **1,415,040 B/序列**。

## DSpark：用置信度决定「验证几个 draft」

普通投机解码固定验证 k 个 draft，接受率低的 token 白算。DSpark 多一个 `ConfidenceHead`（`models/mtp/dflash.py:46`），预测每个 draft token 的接受概率，然后按「预期有效吞吐」排序，只验证前缀子集。

Q&A 的数字：draft 一次出 5 个，动态长度把验证数从 5.00 砍到 3.2–4.4（约 -30%），平均接受长度基本不变（3.29 vs 3.20），在 EP64/DP64、每 rank ~60 并发时折合 **+4~7% 吞吐**。

和异步调度的配合很微妙：`schedule(N+1)` 跑的时候第 N 步的 confidence 还没回传，所以 CPU 只定**总预算**，逐请求长度在 GPU 上用 tilelang kernel 重算（累乘 + 排序 + remap scatter），结果写回 pinned buffer。

`MAX_ROLLBACK_TOKENS = nextn + 2`（`structure_output/matcher.py:20`）——投机解码最多回退 nextn 个 token，结构化输出的自动机必须能回滚这么深，否则 mask 会错。

## 零开销调度：把 CPU 那段空转填掉

朴素 decode 一步：`schedule(N) → forward(N) → sample(N) → post-process(N)`。`post-process` 要等 GPU 把 sample 结果同步回来，这期间 GPU 空转。

DeepInfer 的做法（`engine/llm_engine.py:860 step` + Q&A）：

```
GPU:  forward(N) ──────────► sample(N)
CPU:       schedule(N+1) → launch_forward(N+1) → post-process(N-1) → launch_sample(N+1)
```

难点是资源生命周期：第 N 步的 `SequenceMetadata`、`RequestOutput` 还被调用方读着，不能马上回收。`DeferredRelease`（`llm_engine.py:67`）把它们攒起来，在**下一步的 `pre_sampler_sync`** 释放——那时 GPU 正在算，CPU 释放正好重叠。

`DEEPINFER_ZERO_OVERHEAD_SCHEDULING` 默认对 NORMAL/COMPLETION 开启（`engine_type.py` 的 `default_zero_overhead_scheduling`）。

## 结构化输出：编译离线，运行时只查表

`structure_output/` 用 xgrammar：

1. 请求进来时把 regex / JSON schema / EBNF 编译成语法自动机（`compiler.py`），编译在独立进程/线程池，**不占 decode 关键路径**；
2. 每步 decode 用 `XGrammarMatcher`（`matcher.py:58`）推进自动机，产出 vocab bitmask；
3. sampler 把非法 token 的 logit 置 -inf。

三个工程细节：

- **trigger tokens**：可以等模型先自由生成一段，出现指定 token 序列后才开始约束；
- **bound token**：触发后先强制走完 trigger 序列再进 grammar；
- **rollback**：投机解码被拒时要回退自动机状态，深度 = nextn+2。

`SOMatcherBuilder.pump()` 在 `step()` 里被调用——编译是异步的，但**取结果**在关键路径上，所以用超时（`DEEPINFER_XGRAMMAR_TIMEOUT`）兜底。

## 那些「一致性」开关

| 开关 | 作用 | 限制 |
| --- | --- | --- |
| `DEEPINFER_BATCH_INVARIANT=1` | 不同 batch 组合下结果逐位一致 | 仅 NVIDIA |
| 再加 `DEEPINFER_USE_NEXTN_DECODE=0` + 固定 `seq_seed` | 完全确定性采样 | 仅 NVIDIA |

Batch invariant 的代价是放弃一些融合 kernel（`attn.py` 里 `use_fused_norm_rope_attn_rope_cast` 会因它关闭）。这就是「可复现」和「快」的真实兑换率。

## benchmark 数字怎么读

`examples/README.md` 写得很老实：

- decode throughput = `avg_SPS × decode_num_requests × (1 + nextn) / cp`，按**全接受**计账；
- profiler 会让 step time 膨胀最多 50%（bsz=1），所以 trace 只能看 kernel 组成，不能拿来算 SPS；
- profiler 退出后还有 7–11% 的残留减速，所以测量必须排在 profiler 之前。

换句话说：**README 首页的 Pareto 图是可信的，但你自己复现时别把 profiler 开着算吞吐。**

## 如果你的机器跑不了

完全正常。这个仓库的正确用法是「读」，不是「跑」：

1. 先读 `csrc/engine/scheduler.hpp` 的 `_schedule_prefill` / `_schedule_decode`（:351 / :559），理解 waiting/running/prompted 三队列和 block+state 联合分配；
2. 再读 `eplb/policy.py`（能单独跑，纯 torch）；
3. 再读 `cache_engine.py` 的三个常数和 `yoco.py` 的边界 assert；
4. 最后读 `llm_engine.py:860 step`，把 CPU/GPU 两条时间线对上。
