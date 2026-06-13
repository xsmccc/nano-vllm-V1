# 当前仓库结构说明

## 仓库状态审计

审计时间：M0-T1。

当前工作目录有两层 git repo：

```text
/home/xsmccc/nano-vllm
  .git
  nano-vllm-V1/
    .git
```

外层 repo：

```text
path: /home/xsmccc/nano-vllm
branch: main
remote: https://github.com/GeeeekExplorer/nano-vllm
```

外层 `git status` 显示原始 nano-vLLM 文件被删除，并出现未跟踪目录 `nano-vllm-V1/`。这说明当前工作副本已经从外层原始仓库迁移/复制出一个内层项目目录。

内层 repo：

```text
path: /home/xsmccc/nano-vllm/nano-vllm-V1
branch: main
remote: https://github.com/xsmccc/nano-vllm-V1.git
latest commit: 12941cb feat: add Chunked Prefill, KV Cache Swap, Copy-on-Write, comprehensive Chinese comments
```

内层 repo 当前 tracked runtime 文件包括：

```text
bench.py
example.py
nanovllm/
pyproject.toml
README.md
```

当前未跟踪文件只包括本项目新增文档目录：

```text
roadmap/
learnning/
```

M0 结论：

- 后续所有 NanoCache-V 开发都以 `/home/xsmccc/nano-vllm/nano-vllm-V1` 作为项目根目录。
- 功能 commit 应该在内层 repo 完成。
- 外层 repo 的删除状态暂时不处理，避免把目录迁移和功能开发混在一起。
- 当前阶段只新增文档，不修改 runtime 代码。
- 第一条建议 commit 应为纯文档提交，例如 `docs: add NanoCache-V roadmap and M0 planning`。

风险：

- 如果误在外层 repo 提交，会把大量删除文件和内层目录迁移混成一个提交。
- 如果后续要清理仓库结构，需要单独做 repo migration commit，不能和 KV cache 功能混合。

## KV Cache 分配路径

文件：`nanovllm/engine/model_runner.py`

当前 KV cache 在 `ModelRunner.allocate_kv_cache()` 中分配。

逻辑 layout：

```text
kv_cache:
  shape = [2, num_layers, num_blocks, block_size, num_kv_heads_per_rank, head_dim]
  dim 0 = 0 表示 K
  dim 0 = 1 表示 V
```

每层 attention module 拿到：

```text
k_cache = kv_cache[0, layer_id]
v_cache = kv_cache[1, layer_id]

shape = [num_blocks, block_size, num_kv_heads_per_rank, head_dim]
```

## KV Cache 写入路径

文件：`nanovllm/layers/attention.py`

当前 K/V tensor：

```text
k: [num_tokens, num_kv_heads_per_rank, head_dim]
v: [num_tokens, num_kv_heads_per_rank, head_dim]
```

`store_kvcache()` 使用 `slot_mapping` 把每个 token 写入 paged cache。

全局 slot 计算方式：

```text
slot = block_id * block_size + token_offset_inside_block
```

## KV Cache 读取路径

当前读取路径交给 FlashAttention：

- Prefill: `flash_attn_varlen_func`
- Decode: `flash_attn_with_kvcache`

关键限制：

INT8/INT4 quantized cache 不能直接传给当前 FlashAttention path，除非后续实现兼容的 attention kernel。因此第一版 quantized backend 很可能需要先显式 dequant 到 FP scratch buffer，再调用 FlashAttention。

这意味着：

- 第一版量化路径的主要收益是显存节省。
- 第一版量化路径不应轻易宣称 decode 更快。
- 性能结论必须由 profiler 和 benchmark 支撑。

## Block Management

文件：`nanovllm/engine/block_manager.py`

职责：

- GPU block allocate/deallocate。
- Prefix cache hash lookup。
- Copy-on-write。
- GPU/CPU swap。

Quantized KV cache 必须和这些生命周期事件保持一致，否则容易出现 block metadata 错位、scale/zero 失效、shared block 被错误写入等问题。

## KV Cache 生命周期

### 1. Allocate

触发位置：

```text
Scheduler.schedule()
  -> BlockManager.allocate(seq)
```

发生阶段：

```text
Prefill
```

行为：

- Scheduler 从 waiting 队列取新 sequence。
- BlockManager 根据 `seq.num_blocks` 分配 physical block。
- 如果 prefix cache 命中，复用已有 block，并增加 `ref_count`。
- 如果 prefix cache miss，分配新的 free block。
- 分配结果写入 `seq.block_table`。

重要字段：

```text
seq.block_table: logical block index -> physical block id
block.ref_count: shared prefix cache ownership
block.hash: full block token hash
block.token_ids: full block token content
```

量化影响：

- quant metadata 必须跟 physical block id 绑定，而不是跟 sequence logical index 绑定。
- prefix cache hit 时，同一个 physical block 可能被多个 sequence 共享。
- shared block 被量化前必须确认不会再被写入。

### 2. Global Slot Mapping

触发位置：

```text
ModelRunner.prepare_prefill()
ModelRunner.prepare_decode()
```

映射关系：

```text
global_slot = physical_block_id * block_size + offset_inside_block
```

用途：

- `slot_mapping` 告诉 `store_kvcache` 当前 token 的 K/V 应该写到哪个 cache slot。

量化影响：

- quantized store kernel 必须使用同样的 slot mapping。
- scale/zero metadata 的写入位置必须和 data slot 一致。

### 3. Store K/V

触发位置：

```text
Attention.forward()
  -> store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
```

当前输入：

```text
k: [num_tokens, num_kv_heads_per_rank, head_dim]
v: [num_tokens, num_kv_heads_per_rank, head_dim]
k_cache/v_cache: [num_blocks, block_size, num_kv_heads_per_rank, head_dim]
slot_mapping: [num_tokens]
```

当前实现：

- Triton kernel 每个 program 处理一个 token。
- 根据 `slot_mapping[idx]` 找到 cache slot。
- 将当前 token 的 K/V 写入 FP cache。
- `slot == -1` 表示 padding，跳过。

量化影响：

- FP baseline backend 可以保持当前写入语义。
- INT8/INT4 backend 需要在 store 阶段计算 quantized data 和 metadata。
- K/V 的 quantization granularity 不同，metadata shape 也不同。

### 4. Read K/V

触发位置：

```text
Attention.forward()
```

Prefill read path：

```text
flash_attn_varlen_func(q, k, v, ...)
```

Decode read path：

```text
flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache, ...)
```

当前限制：

- FlashAttention 当前 path 期望 FP-compatible K/V cache。
- INT8/INT4 cache 不能直接作为 drop-in replacement。

量化影响：

- 第一版 quantized backend 应该使用 dequant-to-scratch reference path。
- 不允许请求 INT8/INT4 时静默使用 FP cache。
- 后续优化目标是 fused dequant load 或 quantized attention microkernel。

### 5. Copy-on-Write

触发位置：

```text
Scheduler.schedule()
  -> BlockManager.copy_on_write(seq)
LLMEngine.step()
  -> ModelRunner.copy_kv_blocks(cow_pairs)
```

发生条件：

- decode 阶段即将写入最后一个 block。
- 该 block 的 `ref_count > 1`。

行为：

- BlockManager 分配新 physical block。
- 更新当前 sequence 的 `block_table[-1]`。
- 降低旧 block 的 `ref_count`。
- ModelRunner 在 GPU 上复制旧 block 的 K/V 到新 block。

量化影响：

- 如果旧 block 是 quantized，复制时必须复制 quantized data 和 metadata。
- 如果旧 block 是 FP，复制时必须复制 FP data。
- CoW 后新旧 block 的 quantization state 不能混淆。

### 6. Swap Out / Swap In

触发位置：

```text
Scheduler.preempt()
  -> BlockManager.swap_out(seq)
LLMEngine.step()
  -> ModelRunner.swap_blocks(..., "out")

Scheduler.schedule()
  -> BlockManager.swap_in(seq)
LLMEngine.step()
  -> ModelRunner.swap_blocks(..., "in")
```

行为：

- GPU block 不足时，将 sequence 的 KV cache 从 GPU copy 到 CPU pinned memory。
- GPU 有空间后，再从 CPU pinned memory copy 回 GPU。
- `seq.block_table` 在 swapped 状态下暂时记录 CPU block id。

量化影响：

- 如果 GPU cache 支持 quantized format，CPU cache 也需要保存同样 format 或有明确转换规则。
- metadata 也必须随 block swap。
- swap map 中的 id 语义会随 direction 改变，backend 需要非常小心。

### 7. Deallocate

触发位置：

```text
Scheduler.postprocess()
  -> BlockManager.deallocate(seq)
Scheduler.preempt()
  -> BlockManager.deallocate(seq)  # CPU swap 空间不足时
```

行为：

- sequence 结束或被彻底抢占时释放 block。
- block `ref_count` 减一。
- `ref_count == 0` 时 physical block 回到 free list。
- `seq.block_table` 清空。

量化影响：

- quantized metadata 必须在 block 释放时标记无效或等待下次覆盖。
- 不能让 stale scale/zero metadata 被新 sequence 误用。

### KV Cache Lifecycle Summary

```text
waiting seq
  -> allocate physical blocks
  -> build block_table
  -> prepare slot_mapping
  -> store K/V
  -> read K/V in attention
  -> decode append
      -> optional CoW
      -> optional new block allocation
  -> optional swap out/in
  -> deallocate when finished
```

对 NanoCache-V 的核心约束：

- data、metadata、block state 必须都以 physical block 为核心管理。
- partial block 不能随意量化。
- shared block 不能在 ownership 不清楚时量化。
- swap 和 CoW 不能只搬 data，不搬 metadata。

## Scheduler

文件：`nanovllm/engine/scheduler.py`

职责：

- prefill scheduling。
- chunked prefill。
- decode scheduling。
- CoW trigger。
- swap trigger。

Hybrid recent/old policy 应该围绕 block lifecycle 谨慎加入，不应该直接在 attention 层里用 token index 临时判断。

## 当前推理主链路

### 用户入口

```text
LLM.generate(prompts, sampling_params)
```

位置：

- `nanovllm/llm.py`
- `nanovllm/engine/llm_engine.py`

职责：

- 接收字符串或 token id prompt。
- 创建 `Sequence`。
- 加入 scheduler waiting 队列。
- 循环调用 `step()`，直到所有请求完成。

### 单步执行

```text
LLMEngine.step()
```

主要流程：

```text
Scheduler.schedule()
  -> optional copy_kv_blocks()
  -> optional swap_blocks()
  -> ModelRunner.run(seqs, is_prefill)
  -> Scheduler.postprocess(seqs, token_ids)
```

这里是后续 profiler 的重要 hook 点，因为它能看到：

- 本轮是 prefill 还是 decode。
- 本轮有多少 seq。
- 是否发生 CoW。
- 是否发生 swap in/out。
- 本轮执行耗时。

### Scheduler

```text
Scheduler.schedule()
```

职责：

- 从 waiting/running/swapped 队列中选择本轮要跑的 sequences。
- prefill 阶段分配 KV blocks。
- decode 阶段检查是否需要 append 新 block。
- 触发 copy-on-write。
- 触发 swap out / swap in。

输出：

```text
seqs: list[Sequence]
is_prefill: bool
cow_pairs: list[(src_block_id, dst_block_id)]
swap_in_map: list[(cpu_block_id, gpu_block_id)]
swap_out_map: list[(gpu_block_id, cpu_block_id)]
```

### ModelRunner

```text
ModelRunner.run(seqs, is_prefill)
```

职责：

- 准备 input ids、positions、slot mapping、block tables。
- 调用模型 forward。
- 调用 sampler 产生 token ids。
- 在 chunked prefill 场景下维护 `num_computed_tokens`。

prefill 准备：

```text
prepare_prefill(seqs)
  -> input_ids: [total_prefill_tokens]
  -> positions: [total_prefill_tokens]
  -> context:
       cu_seqlens_q
       cu_seqlens_k
       max_seqlen_q
       max_seqlen_k
       slot_mapping
       block_tables optional
```

decode 准备：

```text
prepare_decode(seqs)
  -> input_ids: [batch_size]
  -> positions: [batch_size]
  -> context:
       slot_mapping
       context_lens
       block_tables
```

### Prefill 输入输出细节

输入：

```text
seqs: list[Sequence]
is_prefill: True
```

每条 sequence 的关键字段：

```text
seq.token_ids
seq.num_cached_tokens
seq.num_computed_tokens
seq.block_table
seq.num_blocks
```

`prepare_prefill` 输出：

```text
input_ids:
  shape = [total_new_prefill_tokens]
  dtype = int64

positions:
  shape = [total_new_prefill_tokens]
  dtype = int64

slot_mapping:
  shape = [total_new_prefill_tokens]
  dtype = int32

cu_seqlens_q:
  shape = [num_seqs + 1]
  dtype = int32

cu_seqlens_k:
  shape = [num_seqs + 1]
  dtype = int32

block_tables:
  shape = [num_seqs, max_num_blocks_per_seq] or None
  dtype = int32
```

Prefill 阶段需要写入当前新 token 的 K/V cache。若 prefix cache 命中，则 Q 只覆盖未缓存 token，但 K/V 的可见长度包含 cached prefix。

### Decode 输入输出细节

输入：

```text
seqs: list[Sequence]
is_prefill: False
```

每条 sequence decode 时只取一个 token：

```text
input_ids:
  shape = [batch_size]
  dtype = int64

positions:
  shape = [batch_size]
  dtype = int64

slot_mapping:
  shape = [batch_size]
  dtype = int32

context_lens:
  shape = [batch_size]
  dtype = int32

block_tables:
  shape = [batch_size, max_num_blocks_per_seq]
  dtype = int32
```

Decode 阶段每条 sequence 追加一个 token，对应的 K/V 写入当前 sequence 最后一个 physical block 的最后一个 slot。

### Model Forward

```text
Qwen3ForCausalLM.forward(input_ids, positions)
  -> Qwen3Model.forward
  -> each Qwen3DecoderLayer
  -> Qwen3Attention.forward
```

每层 attention：

```text
hidden_states
  -> qkv_proj
  -> q, k, v reshape
  -> optional q_norm/k_norm
  -> rotary_emb
  -> Attention.forward(q, k, v)
```

### Attention

```text
Attention.forward(q, k, v)
```

当前流程：

1. 如果 KV cache 已分配，先调用 `store_kvcache(k, v, k_cache, v_cache, slot_mapping)`。
2. 如果是 prefill，调用 `flash_attn_varlen_func`。
3. 如果是 decode，调用 `flash_attn_with_kvcache`。

后续 KV quantization 主要会影响：

- step 1 的 store path。
- step 3 的 cache read path。
- quantized metadata 与 block lifecycle 的一致性。

## M1 Profiler 可能 Hook 点

后续 profiler 不应该一开始侵入每个 kernel。M1 建议先从 Python 层可控 hook 开始。

### LLMEngine.step

可观测内容：

- 本轮是否 prefill。
- scheduled seq 数量。
- 本轮 wall time。
- prefill tokens 或 decode batch size。
- 是否发生 CoW。
- 是否发生 swap in/out。

意义：

- 这是端到端 step 级别 profiler 的入口。
- 可以统计 TTFT/TPOT 的上层信息。

### Scheduler.schedule

可观测内容：

- waiting/running/swapped 队列长度。
- 本轮调度出的 seq 数。
- prefill/decode 分支。
- cow_pairs 数量。
- swap_in_map / swap_out_map 数量。

意义：

- 用于解释性能波动是否来自调度、swap 或 CoW。

### ModelRunner.prepare_prefill / prepare_decode

可观测内容：

- input token 数。
- batch size。
- max sequence length。
- block table shape。
- slot mapping shape。

意义：

- 用于把模型耗时和实际输入规模对应起来。

### ModelRunner.run_model

可观测内容：

- model forward 时间。
- CUDA Graph replay 或 eager 路径。
- input shape。

意义：

- 区分 Python 调度开销和模型 forward 开销。

### Attention.forward

可观测内容：

- 每层 K/V store 是否发生。
- prefill 或 decode attention path。
- q/k/v shape。
- k_cache/v_cache shape。

意义：

- 后续 KV cache quantization 的关键路径。
- 第一版 profiler 可以只统计少量层或汇总，避免过高 overhead。

### ModelRunner.allocate_kv_cache

可观测内容：

- KV cache layout。
- dtype。
- num blocks。
- block bytes。
- total KV bytes。
- GPU memory before/after allocation。

意义：

- 这是 KV cache memory profiler 的第一入口。

## 当前 Benchmark 缺口

文件：`bench.py`

当前脚本只报告总 throughput。

缺失：

- 被测 workload 的正式 warmup。
- timing 前后的 CUDA synchronize。
- repeated experiments。
- GPU memory statistics。
- prefill/decode 分离指标。
- percentile latency。
- profiler output。

后续第一步应先把 benchmark 规范化，再进行性能优化。
