# Prefill / Decode 与 Paged KV Cache

## 为什么重要

KV cache 量化主要影响 decode 阶段。

Prefill 阶段一次处理 prompt token，计算量大；decode 阶段每步通常只处理每条 sequence 的一个新 token，但需要读取越来越长的历史 KV cache。长上下文下，decode 阶段对 KV cache memory bandwidth 和显存容量非常敏感。

## 核心概念

### Prefill

Prefill 输入完整 prompt 或 prompt chunk。

在当前 nano-vLLM 中：

```text
prepare_prefill(seqs)
  -> input_ids: [total_new_prefill_tokens]
  -> positions: [total_new_prefill_tokens]
  -> slot_mapping: [total_new_prefill_tokens]
  -> cu_seqlens_q / cu_seqlens_k
  -> block_tables optional
```

Prefill 会把 prompt 的 K/V 写入 KV cache。

### Decode

Decode 每步每条 sequence 只处理一个 token。

在当前 nano-vLLM 中：

```text
prepare_decode(seqs)
  -> input_ids: [batch_size]
  -> positions: [batch_size]
  -> slot_mapping: [batch_size]
  -> context_lens: [batch_size]
  -> block_tables: [batch_size, max_num_blocks]
```

Decode 的 attention 会读取该 sequence 的历史 KV cache。

### Paged KV Cache

KV cache 被切成固定大小 block。

当前 layout：

```text
kv_cache:
  [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]

per-layer k_cache/v_cache:
  [num_blocks, block_size, num_kv_heads, head_dim]
```

sequence 通过 `block_table` 间接引用 physical blocks：

```text
logical block index -> physical block id
```

global slot：

```text
slot = physical_block_id * block_size + offset_inside_block
```

## 在 NanoCache-V 中对应哪段代码

- `nanovllm/engine/sequence.py`
  - `block_table`
  - `num_blocks`
  - `last_block_num_tokens`
- `nanovllm/engine/block_manager.py`
  - block allocate/deallocate。
  - prefix cache。
  - CoW。
  - swap。
- `nanovllm/engine/model_runner.py`
  - `prepare_prefill`
  - `prepare_decode`
  - `allocate_kv_cache`
- `nanovllm/layers/attention.py`
  - `store_kvcache`
  - FlashAttention prefill/decode 调用。

## 常见坑

1. 把 logical block index 和 physical block id 混淆。
2. 量化 metadata 绑定到 sequence，而不是 physical block。
3. 对 partial block 做不可逆量化，导致后续 decode 写入困难。
4. 忽视 prefix cache 下多个 sequence 共享同一 physical block。
5. CoW 时只复制 data，不复制 metadata。
6. swap 时忘记 metadata。

## 可以做的小实验

1. 构造两个共享 prefix 的 prompt，观察 `seq.block_table` 是否复用 block。
2. 构造长输出，观察 decode 过程中何时追加新 block。
3. 降低可用 KV block 数，触发 swap out/in。

## 与当前任务关系

M0 阶段已经梳理了 KV cache lifecycle。

M1 profiler 要把 prefill/decode 分开记录。M2 backend 抽象必须保持 paged cache 语义不变。
