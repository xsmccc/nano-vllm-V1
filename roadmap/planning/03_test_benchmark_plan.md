# Test 与 Benchmark 计划

## Correctness Tests

### Quant/Dequant Roundtrip

输入：

```text
x: [num_tokens, num_kv_heads, head_dim]
```

检查：

- max absolute error。
- mean absolute error。
- relative L2 error。
- cosine similarity。
- 不允许 NaN 或 Inf。

### Store/Load Correctness

输入：

```text
k, v: [num_tokens, num_kv_heads, head_dim]
slot_mapping: [num_tokens]
cache: [num_blocks, block_size, num_kv_heads, head_dim]
```

检查：

- 每个 token 都写到预期 slot。
- `slot_mapping == -1` 的 token 被跳过。
- random non-contiguous slot order 正确。

### Attention Correctness

对比：

- FP baseline attention output。
- quantized cache dequantized attention output。

检查：

- output error。
- logits error。
- optional top-k agreement。

### Lifecycle Correctness

场景：

- allocate 和 deallocate。
- prefix cache hit。
- copy-on-write。
- swap out 和 swap in。
- hybrid recent/old blocks。

检查：

- metadata 与 physical block ID 保持对齐。
- shared writable blocks 不会被错误量化。
- deallocation 后不会残留 stale scale/zero metadata。

### No-Silent-Fallback Test

如果用户请求 `kv_cache_dtype=int4`，但某个必要 kernel/path 不支持：

- 必须 raise 清晰错误。
- 不允许静默使用 FP16/BF16。

## Benchmark 要求

每个 benchmark 必须包含：

- warmup iterations。
- repeated measured iterations。
- timing 前后使用 `torch.cuda.synchronize()`。
- GPU memory stats。
- median、p90、min、max。
- 输出 input shape 和 cache layout。
- 输出 model/config 信息。

## Benchmark 指标

需要报告：

- prefill tokens/s。
- decode tokens/s。
- TTFT。
- TPOT。
- KV cache bytes/token。
- peak GPU memory。
- reserved GPU memory。
- allocated GPU memory。
- quantization time。
- dequantization time。
- attention time。
- memory saving ratio。

## Benchmark 输出格式

优先使用结构化输出：

```text
case: decode_int8_kv
model: Qwen...
dtype: bf16
kv_mode: int8
shape:
  batch: ...
  context_len: ...
  block_size: ...
  kv_heads: ...
  head_dim: ...
latency_ms:
  median: ...
  p90: ...
  min: ...
  max: ...
memory_mb:
  allocated: ...
  reserved: ...
  peak: ...
notes:
  ...
```

## 性能结论规则

没有 benchmark 或 profiler 输出支持时，不允许写：

- faster。
- slower。
- saves memory。
- bottleneck。
- optimized。

只能写成 hypothesis 或 expected behavior。
