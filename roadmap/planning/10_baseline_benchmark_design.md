# Baseline Benchmark Harness 设计

## 目标

设计一个用于 NanoCache-V 的 baseline benchmark harness。

本阶段只设计，不实现。

该 benchmark 的目标不是证明量化有效，而是为后续 M1 profiler、M2 backend、M3 quantized KV cache 提供可信对照。

## 非目标

本阶段不做：

- INT8/INT4 benchmark。
- Triton/CUDA kernel benchmark。
- 多模态 benchmark 实现。
- 与 vLLM/SGLang 的正式对比。
- 未经 profiler 支撑的性能结论。

## Benchmark 层级

### Level 0: Sanity Benchmark

目的：

- 确认模型能跑。
- 确认 tokenizer、forward、scheduler、KV cache 基本可用。

模型：

- Qwen3-4B-Instruct-2507。

输入：

- 1 条短 prompt。
- 1 组 token id prompt。

输出：

- 是否成功生成。
- 是否出现 CUDA/Triton/FlashAttention 错误。

不用于性能结论。

### Level 1: Baseline Throughput Benchmark

目的：

- 建立 FP16/BF16 baseline。
- 分离 prefill 和 decode 指标。

模型：

- Qwen3-4B-Instruct-2507。

指标：

- prefill tokens/s。
- decode tokens/s。
- total wall time。
- total generated tokens。
- GPU peak memory。

### Level 2: Long Context Matrix

目的：

- 观察 context length 增长时 KV cache 显存和 decode latency 的变化。

建议矩阵：

```text
context_len: 1K, 4K, 8K, 16K, 32K
batch_size: 1, 4, 8, 16
output_len: 128, 512
```

主模型：

- Qwen3-4B-Instruct-2507。

注意：

- 32K 需要根据显存实际情况调整。
- 如果 OOM，必须记录 OOM 点，不允许删掉失败结果。

### Level 3: Long Decode / Reasoning Benchmark

目的：

- 模拟 thinking/reasoning 长输出。
- 观察 TPOT 和 KV cache 增长。

模型：

- Qwen3-4B-Thinking-2507，可选。

建议设置：

```text
context_len: 1K, 4K, 8K
output_len: 1K, 2K, 4K
batch_size: 1, 4
```

### Level 4: Visual Prefix KV Benchmark

目的：

- 评估多模态输入带来的 visual token 数量。
- 评估 visual prefix KV cache 显存占用。
- 评估 visual prefix reuse 和 quantization 的收益。

目标模型：

- Qwen3.5-2B。

建议指标：

- image count。
- image resolution。
- visual token count。
- visual prefix KV bytes。
- reusable visual prefix hit ratio。
- visual prefix quantization error。
- decode latency with reused visual prefix。

该层级属于多模态阶段，不在 M0 实现。

## Benchmark 参数

命令行参数建议：

```text
--model
--num-seqs
--input-len
--output-len
--input-len-min
--input-len-max
--output-len-min
--output-len-max
--warmup-iters
--repeat-iters
--seed
--enforce-eager
--max-model-len
--max-num-seqs
--max-num-batched-tokens
--enable-chunked-prefill
--max-chunk-size
--output-json
```

后续 KV cache 参数：

```text
--kv-cache-dtype
--kv-quant-scheme
--kv-group-size
--kv-recent-window
```

M0 不实现后续参数，只预留设计。

## Timing 规则

必须使用：

```python
torch.cuda.synchronize()
start = time.perf_counter()
...
torch.cuda.synchronize()
elapsed = time.perf_counter() - start
```

对于 kernel/microbenchmark，后续使用 CUDA events。

端到端 benchmark 可以先使用 wall time + synchronize，但必须明确包含 CPU scheduler overhead。

## Warmup 规则

至少两类 warmup：

1. Engine warmup：
   - 初始化模型。
   - 跑一次短 prompt。
   - 触发 Triton/FlashAttention/CUDA Graph 相关初始化。

2. Workload warmup：
   - 用与正式 benchmark 类似的 shape 跑 `warmup_iters` 次。
   - warmup 结果不计入统计。

## Repeat 规则

正式 benchmark 必须重复：

```text
repeat_iters >= 5 for expensive model benchmark
repeat_iters >= 30 for microbenchmark
```

输出：

- median。
- p90。
- min。
- max。
- mean 可选，但不能只看 mean。

## Memory 统计

每轮 benchmark 前：

```python
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
```

每轮 benchmark 后记录：

```text
torch.cuda.memory_allocated()
torch.cuda.memory_reserved()
torch.cuda.max_memory_allocated()
torch.cuda.mem_get_info()
```

输出字段：

```text
allocated_mb
reserved_mb
peak_allocated_mb
free_mb
total_mb
```

## 输出格式

优先输出 JSONL，每个 case 一行。

示例：

```json
{
  "case": "baseline_decode",
  "model": "Qwen3-4B-Instruct-2507",
  "dtype": "bf16",
  "kv_cache_dtype": "fp",
  "num_seqs": 8,
  "input_len": 4096,
  "output_len": 512,
  "warmup_iters": 3,
  "repeat_iters": 5,
  "latency_s": {
    "median": 12.3,
    "p90": 12.9,
    "min": 12.1,
    "max": 13.4
  },
  "throughput": {
    "prefill_tok_s": 0.0,
    "decode_tok_s": 0.0,
    "total_tok_s": 0.0
  },
  "memory_mb": {
    "allocated": 0.0,
    "reserved": 0.0,
    "peak_allocated": 0.0,
    "free": 0.0,
    "total": 0.0
  },
  "notes": []
}
```

## Prefill / Decode 分离

当前 `LLMEngine.generate()` 只在 tqdm 中维护 prefill/decode throughput，不返回结构化 step metrics。

M0 benchmark 设计要求：

- 不直接相信总 wall time 能代表 decode。
- M1 profiler 需要在 `LLMEngine.step()` 记录每一步是 prefill 还是 decode。
- benchmark harness 后续应读取 profiler summary 来分离 prefill/decode。

因此 M0 阶段如果先改 `bench.py`，只能得到 baseline total throughput，不能作为最终性能证据。

## Benchmark 验收标准

一个 benchmark case 合格必须满足：

- 打印 model/config。
- 打印 input/output shape。
- 有 warmup。
- 有 repeat。
- timing 前后 synchronize。
- 有 memory stats。
- 输出 median/p90/min/max。
- 明确是否包含 CPU scheduler overhead。
- 失败/OOM 也记录。

## 当前阶段结论

M0-T4 只完成设计，不实现。

下一步进入 M0-T5：

- 确认本地模型路径。
- 设计 sanity run 命令。
- 确认依赖检查命令。
