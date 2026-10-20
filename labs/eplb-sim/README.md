# EPLB 最小复现

只 import `DeepInfer/deepinfer/model_executor/eplb/policy.py`，不 import `deepinfer` 包（那个包在非 Linux 上会挂在 `os.uname()`）。

```bash
python labs/eplb-sim/sim.py
```

输出：

```
naive      peak per layer: [268.8, 255.4, 274.5]
rebalanced peak per layer: [191.0, 180.6, 191.5]
copies per expert (layer0): [3, 2, 2, 1, 1, 1, 1, 1]
placement (layer0): [1, 5, 1, 6, 3, 7, 0, 2, 0, 2, 0, 4]
PASS hot expert replicated, peak lowered
```

负载是合成的（Zipf 形状 + 8 步抖动），**不是** DeepSeek 的 benchmark。它只验证两件事：

1. 热点专家会拿到更多副本（3 个 vs 1 个）；
2. 贪心摆位把「历史各步的最慢 rank」压下来了。

需要 `torch`（CPU 版即可）和 `.research/DeepInfer`。
