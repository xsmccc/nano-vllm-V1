# 当前 store_kvcache Triton Kernel 解析

## 为什么重要

`store_kvcache_kernel` 是当前 nano-vLLM 中 KV cache 写入的关键路径。

后续 FP8/INT8/INT4 KV cache 的 store path 很可能从这里演化：

- FP baseline store。
- quantized store。
- INT4 pack。
- metadata 写入。

因此需要先理解当前 kernel 的 shape、layout 和 memory access。

## 当前代码位置

文件：

```text
nanovllm/layers/attention.py
```

入口：

```python
store_kvcache(key, value, k_cache, v_cache, slot_mapping)
```

Triton kernel：

```python
store_kvcache_kernel[(N,)](...)
```

## 输入 Shape

当前 Python 入口中：

```text
key:
  shape = [N, num_heads, head_dim]

value:
  shape = [N, num_heads, head_dim]

k_cache/v_cache:
  logical shape = [num_blocks, block_size, num_heads, head_dim]
  kernel 中按 [num_blocks * block_size, num_heads * head_dim] 理解

slot_mapping:
  shape = [N]
```

其中：

```text
D = num_heads * head_dim
```

## Program 映射

当前 kernel：

```python
idx = tl.program_id(0)
```

含义：

- grid = `(N,)`
- 每个 Triton program 负责一个 token。
- `idx` 表示第几个 input token。

## 写入逻辑

1. 读取当前 token 的目标 slot：

```python
slot = tl.load(slot_mapping_ptr + idx)
```

2. 如果 `slot == -1`，跳过。

3. 读取当前 token 的 K/V：

```text
key offset = idx * key_stride + arange(0, D)
value offset = idx * value_stride + arange(0, D)
```

4. 写入 cache：

```text
cache offset = slot * D + arange(0, D)
```

## Memory Layout

要求：

```python
key.stride(-1) == 1
value.stride(-1) == 1
key.stride(1) == head_dim
value.stride(1) == head_dim
k_cache.stride(1) == D
v_cache.stride(1) == D
```

这说明 kernel 假设每个 token 的 `[num_heads, head_dim]` 是连续的一段。

## 当前性能直觉

当前 kernel 做的是纯 memory copy：

```text
read key/value from temporary qkv output
write key/value to KV cache
```

主要瓶颈大概率是 memory bandwidth，而不是 compute。

但这是 hypothesis，后续必须通过 benchmark/profiler 验证。

## 后续量化改造方向

### INT8 Store

可能变化：

- 读取 FP K/V。
- 计算 scale/zero。
- 写 INT8 data。
- 写 scale/zero metadata。

### INT4 Store

额外变化：

- 两个 4-bit value pack 到一个 uint8。
- group-wise scale/zero。
- head_dim 必须考虑 group size 对齐。

### KIVI-Style

Key 和 Value 的 metadata layout 不同：

- Key 偏 per-channel。
- Value 偏 per-token/group-wise。

## 常见坑

1. 没有处理 `slot == -1`。
2. 假设 input tensor contiguous，但实际 stride 不满足。
3. INT4 pack 后 logical shape 和 physical storage shape 混淆。
4. scale/zero metadata 写错 block/slot。
5. 量化 store path 不支持某 dtype，却静默 fallback 到 FP store。

## 可以做的小实验

1. 用随机 key/value 和随机 slot_mapping 测 store correctness。
2. 写 PyTorch reference store，与 Triton store 对比。
3. 改变 N、num_heads、head_dim，测 store latency。
4. 后续加入 INT8 quant store，与 FP store 比较 memory traffic。
