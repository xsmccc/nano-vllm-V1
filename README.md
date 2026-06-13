# Prism-Infer

基于 [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) 构建的多模态推理引擎，聚焦 Qwen3-VL 适配与 KV Cache 优化。

## 新增特性

### 1. Chunked Prefill

**问题**：长 prompt（如 4K tokens）的 Prefill 会独占 GPU 数百毫秒，阻塞所有 Decode 序列的生成，导致 TTFT 尾延迟飙升。

**方案**：将长 prompt 切分为 `max_chunk_size`（默认 512）大小的 chunk，分多轮处理。每轮 Scheduler 可以把未完成 Prefill 的序列和 Decode 序列混合调度。

**涉及文件**：
- `engine/scheduler.py` — `enable_chunked_prefill` 分支，chunk 大小计算
- `engine/sequence.py` — `num_computed_tokens`、`is_prefill_finished`、`remaining_prefill_tokens`
- `engine/model_runner.py` — `run()` 中的 chunk 截断 + 恢复逻辑

### 2. KV Cache Swap (GPU ↔ CPU)

**问题**：原版在 GPU block 不足时直接 deallocate（丢弃 KV Cache），被驱逐的序列需要从头重新 Prefill，浪费计算。

**方案**：引入 CPU 端 pinned memory KV Cache 作为 swap 空间。block 不足时先 swap_out 到 CPU 保留 KV，GPU 有空间后 swap_in 换回继续生成。两级驱逐策略：优先 swap → CPU 也满才 deallocate。

**涉及文件**：
- `engine/block_manager.py` — `swap_out()`、`swap_in()`、`can_swap_out()`、`can_swap_in()`、CPU block 管理
- `engine/scheduler.py` — `swapped` 队列、`preempt()` 两级驱逐
- `engine/model_runner.py` — `swap_blocks()`、CPU KV Cache 分配（`pin_memory=True`）

### 3. Copy-on-Write (CoW)

**问题**：Prefix Caching 让多条序列共享同一个物理 KV block（`ref_count > 1`），但 Decode 时写入新 KV 会污染其他序列的数据。

**方案**：Decode 前检查最后一个 block 的 `ref_count`，若 > 1 则分配新 block 并在 GPU 上复制 KV 数据，更新当前序列的 block_table。

**涉及文件**：
- `engine/block_manager.py` — `copy_on_write()`
- `engine/scheduler.py` — Decode 分支的 CoW 检查
- `engine/model_runner.py` — `copy_kv_blocks()`

## 代码注释

为全部 19 个源文件添加了详尽的中文注释（新增约 1000 行），包括：

- 每个类/方法的功能说明和 C++ 类比
- PagedAttention 核心数据流的逐行解释
- CUDA Graph、Tensor Parallel、KV Cache 分配的实现细节
- Triton kernel（store_kvcache）和 FlashAttention 调用路径的解析

## 架构概览

```
用户请求 → LLMEngine.add_request() → Scheduler.waiting 队列
                                          │
         ┌────────────────────────────────┘
         ▼
    Scheduler.schedule()
    ├─ Prefill: allocate blocks → BlockManager 分配/Prefix Cache 复用
    ├─ Decode: can_append → CoW 检查 → may_append
    ├─ Swap Out: GPU 不足 → KV Cache → CPU pinned memory
    └─ Swap In: GPU 有空间 → CPU KV Cache → GPU
         │
         ▼
    ModelRunner.run()
    ├─ prepare_prefill/decode: block_table → slot_mapping (GPU tensor)
    ├─ Attention: Triton 写 KV Cache + FlashAttention 读 KV Cache
    └─ Sampler: logits → token_id
         │
         ▼
    Scheduler.postprocess() → append_token → 检查 EOS/max_tokens
```

## Quick Start

```bash
pip install -e .

python example.py
```

```python
from nanovllm import LLM, SamplingParams

llm = LLM("/path/to/Qwen3-0.6B", enforce_eager=True)
# 启用 Chunked Prefill:
# llm = LLM("/path/to/model", enable_chunked_prefill=True, max_chunk_size=512)

sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
outputs = llm.generate(["Hello, Nano-vLLM."], sampling_params)
print(outputs[0]["text"])
```

## 项目结构

```
nanovllm/
├── engine/
│   ├── sequence.py        # Sequence 数据结构 + block_table (页表)
│   ├── block_manager.py   # 物理 block 分配/释放/Prefix Cache/CoW/Swap
│   ├── scheduler.py       # 三队列调度 (waiting/running/swapped) + Chunked Prefill
│   ├── model_runner.py    # KV Cache 分配 + CUDA Graph + Swap 数据搬运 + 前向推理
│   └── llm_engine.py      # 推理主循环: schedule → run → postprocess
├── layers/
│   ├── attention.py       # Triton store_kvcache + FlashAttention (Prefill/Decode)
│   ├── linear.py          # 线性层 + torch.compile
│   ├── rotary_embedding.py # RoPE 位置编码
│   └── ...                # activation, layernorm, sampler, embed_head
├── models/
│   └── qwen3.py           # Qwen3 模型实现
├── utils/
│   ├── context.py         # 全局上下文 (is_prefill, slot_mapping, block_tables)
│   └── loader.py          # 权重加载 (safetensors + TP 切分)
├── config.py              # 配置类
├── sampling_params.py     # 采样参数
└── llm.py                 # 对外入口
```

## 与原版差异

| | 原版 nano-vllm | 本版本 |
|---|---|---|
| Chunked Prefill | ✗ | ✓ (`max_chunk_size` 可配置) |
| Swap (GPU↔CPU) | ✗ | ✓ (pinned memory, 两级驱逐) |
| Copy-on-Write | ✗ | ✓ (Prefix Cache 共享 block 写前复制) |
| 代码注释 | 少量英文 | 全面中文注释 (~1000 行) |
| 模型支持 | Qwen3 | Qwen3 (同) |

## Acknowledgements

- 原版 [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) by GeeeekExplorer
- [vLLM](https://github.com/vllm-project/vllm) — PagedAttention 论文实现
- [FlashAttention](https://github.com/Dao-AILab/flash-attention) — 高效 Attention kernel

## License

[MIT](LICENSE)
