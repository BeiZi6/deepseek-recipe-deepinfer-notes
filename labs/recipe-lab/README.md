# recipe-lab

三个可运行的 Rust 程序，只依赖 crates.io 上的 `deepseek-recipe 0.2`，不需要模型、GPU、API key。

```bash
cargo run --bin 03_check     # 四条断言，验证笔记里的全部说法
cargo run --bin 01_render    # 打印三种请求渲染出的 prompt
cargo run --bin 02_stream    # 把一段假的 DSML 输出变成 OpenAI SSE chunk
```

`03_check` 期望输出：

```
PASS render: 5 tokens, ends with <|Assistant|></think>
PASS injection: recipe 1 vs naive 2 control tokens (13 vs 10 ids)
PASS token boundary: spelling stays reasoning, special id switches to content
PASS stream: DSML -> tool_calls deltas, finish_reason=tool_calls
```

## tokenizer

`tokenizer.json` 没有入库（6.6 MB）。二选一：

```bash
# A. 从浅克隆拿
git clone --depth 1 https://github.com/deepseek-ai/deepseek-recipe ../../.research/deepseek-recipe

# B. 显式指定任意 V4.1 tokenizer
export DEEPSEEK_TOKENIZER=/path/to/tokenizer.json
```

## 为什么不用 Python 包

`pip install deepseek-recipe` 要现场用 maturin 编译 OpenCV 绑定，Windows 上实测在 `opencv v0.93.7` 的 build script 失败。Rust crate 是同一份代码，且 `cargo` 有预编译依赖。
