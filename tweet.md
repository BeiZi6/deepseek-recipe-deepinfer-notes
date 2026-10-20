# 推文

## 主推（单条，约 600 字）

DeepSeek 今天开源的两个仓库，其实是一句话的两半：

**deepseek-recipe** 管「协议怎么变成 token」，**DeepInfer** 管「token 怎么变成钱」。

Recipe 最漂亮的地方不是支持三种 API，是它把 special token 的**字节区间**在构造 prompt 时就记下来了。用户输入里的 `<｜User｜>` 走的是「删掉 added vocabulary」的 tokenizer，只能被编成 5 个普通 token。我实测同一段注入：朴素编码 10 个 id、含 2 个控制符；recipe 13 个 id、只有模板那 1 个。

注意方向：**被注入的版本更短**。按 token 数审计根本发现不了。

输出侧同理：模型用普通 token 拼出 `</think>` 只是正文，只有真的 special token（128822）才切段。所以它敢用 KMP 做增量解析、敢在 chunk 边界上缓冲 UTF-8。

DeepInfer 是另一套思路：只服务一个模型、一种卡，把能省的全省。

· Decode 用 EP64~128 的 Wide-EP，不为别的，就为每卡少读几个专家
· EPLB 两级贪心：先给热点专家加副本，再按历史峰值摆位置。我拿合成负载跑了下，peak load 268→191
· SWA KV 不进分页块，单独 state cache，134 token 的 ring；落盘再加两段余量到 396
· YoCo 让 decoder 只重算最后 128 个 token（保留 140 覆盖回退），prefill 计算量近乎减半，代价是两条 assert
· 零开销调度：GPU 算 forward(N) 时 CPU 跑 schedule(N+1)，资源延后一步释放

两个仓库的分工特别清楚：Recipe 说「这次要 JSON」，DeepInfer 用 xgrammar 把非法 token 的 logit 打成 -inf。一个声明约束，一个执行约束。

顺手记两个坑：
1. `pip install deepseek-recipe` 在 Windows 装不上，要现场编 OpenCV；用 Rust crate 就行
2. DeepInfer 只有离线 `generate()`，KV 存储 / PD 分离 / 负载均衡都不在这个 release 里

学习仓库（含可跑的 Rust 实验和 EPLB 模拟）在评论区。

## 短钩子（如果只想发一条）

DeepSeek 这两个仓库最好对照着读：Recipe 解决「用户能不能伪造控制符」，DeepInfer 解决「一张卡能不能少读几个专家」。

前者靠**构造时记录字节区间**，后者靠**两级贪心 + 延后一步释放**。

一个防注入，一个抢吞吐，都是把「事后补救」换成「事前设计」。
