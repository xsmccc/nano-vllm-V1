# NanoCache-V 项目总纲

## 项目目标

NanoCache-V 是一个基于 nano-vLLM-V1 的 KV Cache 量化压缩与性能分析系统。

目标场景是长上下文与多模态推理。这个项目应该体现真实的 LLM/VLM inference infra 能力：有 correctness test、有 profiler、有 benchmark、有显存分析、有 kernel 优化路径，而不是只写一个能跑的 demo。

## 核心范围

1. 在 nano-vLLM-V1 中实现 KV Cache profiler。
2. 支持 FP16/BF16 baseline KV cache。
3. 支持 FP8 KV cache。
4. 支持 INT8 KV cache。
5. 支持 INT4 group-wise KV cache。
6. 实现 KIVI-style KV quantization：
   - Key 偏 per-channel quantization。
   - Value 偏 per-token 或 group-wise quantization。
7. 实现 hybrid KV policy：
   - recent tokens 保持 FP16/BF16。
   - old tokens 使用 INT8/INT4。
8. 实现 visual prefix KV cache profiling、reuse 和 quantization。
9. 实现 MLA-style compressed KV cache microbenchmark。
10. 对关键路径实现 Triton 优先的 microkernel，后续再探索 CUDA/TileLang/CUTLASS。

## 当前仓库事实

当前 KV cache 主路径集中在：

- `nanovllm/engine/model_runner.py`
  - 分配全局 KV tensor。
  - 当前 layout: `[2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]`。
- `nanovllm/layers/attention.py`
  - Triton `store_kvcache` 负责写入当前 K/V。
  - FlashAttention 负责 prefill/decode 阶段读取 paged KV cache。
- `nanovllm/engine/block_manager.py`
  - 负责 block 分配、prefix cache、copy-on-write、swap。
- `nanovllm/engine/scheduler.py`
  - 负责 prefill/decode 调度，并触发 CoW/swap。
- `bench.py`
  - 当前只是简单吞吐测试，还不满足 NanoCache-V 的 benchmark 要求。

## 不可妥协的工程规则

1. 请求量化 KV 模式时，不允许静默 fallback 到 FP16/BF16。
2. 每个实现必须说明：
   - input shape。
   - output shape。
   - memory layout。
   - quantization error 来源。
   - 预期性能瓶颈。
3. 每个功能都必须先有 correctness test，再谈性能。
4. 每个 benchmark 必须包含：
   - warmup。
   - `torch.cuda.synchronize()`。
   - repeated runs。
   - GPU memory statistics。
   - median 和 tail latency。
5. 代码改动必须小步提交：
   - 一次 commit 只改一个模块或一个行为。
   - 不混合 refactor、feature、formatting。
6. 没有测量的数据只能称为 hypothesis，不能称为 conclusion。

## AI 协作原则

详细规则见 `roadmap/planning/06_ai_collaboration_rules.md`。该文件保持英文，方便后续直接作为 AI agent 的开发约束使用。

核心原则：

1. AI 必须先读代码，再提实现。
2. AI 不允许编造 benchmark 结果。
3. AI 不允许在没有测量的情况下声称 faster/slower。
4. AI 不允许隐藏 unsupported path。
5. AI 不允许做无关重构。
6. AI 必须明确说明 blocker。
7. AI 写 test/benchmark 时必须同时写清楚验收标准。
8. AI 写 kernel 时必须解释 layout、stride、mask 和 memory traffic。
9. AI 对不确定的 API 或论文细节必须先验证。
10. 优先写简单正确的 reference implementation，再写 optimized kernel。

## 高层里程碑

### M0: 仓库梳理与 Baseline

目标：让当前 nano-vLLM-V1 的状态可理解、可复现。

交付物：

- 仓库结构说明。
- baseline inference sanity run。
- 当前 KV cache layout 文档。
- 符合规范的 baseline benchmark 脚本。

### M1: KV Cache Profiler

目标：让 KV cache 的显存和运行时行为可见。

交付物：

- profiler module。
- per-step prefill/decode timing。
- KV block occupancy 统计。
- prefix cache、CoW、swap event counters。
- GPU memory summary。

### M2: KV Cache Backend 抽象

目标：把裸 tensor 管理替换成 backend interface，同时保持 FP16/BF16 行为不变。

交付物：

- FP baseline backend。
- backend interface。
- tests 证明 baseline 行为未改变。

### M3: Quantized KV Reference Path

目标：实现 FP8/INT8/INT4 存储，以及 dequant-to-scratch correctness path。

交付物：

- quant/dequant utilities。
- metadata layout。
- roundtrip correctness tests。
- attention output comparison tests。
- no-silent-fallback tests。

### M4: KIVI-Style Policy

目标：实现 K/V 非对称量化策略。

交付物：

- Key per-channel quantization。
- Value per-token 或 group-wise quantization。
- error analysis。
- decode benchmark。

### M5: Hybrid Recent/Old KV Policy

目标：recent blocks 保持 FP，old blocks 量化。

交付物：

- block-level quantization state。
- recent-window policy。
- prefix cache 和 CoW 兼容测试。
- memory saving benchmark。

### M6: 多模态 Visual Prefix KV

目标：将 KV cache profiler、reuse 和 quantization 扩展到 Qwen3.5-2B visual prefix 场景。

交付物：

- Qwen3.5-2B config / processor 分析。
- visual token 数量统计。
- visual prefix KV cache memory profiler。
- visual prefix KV reuse 设计。
- visual prefix KV quantization reference path。

### M7: Compressed KV Microbenchmark

目标：独立于完整模型，研究 compressed KV cache 设计。

交付物：

- synthetic compressed KV benchmark。
- decompression overhead measurement。
- memory bandwidth analysis。
- compressed KV 设计说明。

### M8: Kernel Optimization

目标：用实测过的 kernel 替换慢 reference operation。

建议顺序：

- Triton quant store。
- Triton dequant load。
- INT4 pack/unpack。
- fused dequant-to-scratch。
- CUDA core version。
- TileLang/CUTLASS 只在 bottleneck 被测出来之后再探索。

## 硬件计划

主路径：

- NVIDIA RTX 5090 32GB 多卡租赁。
- 先走 NVIDIA，因为当前仓库依赖 PyTorch CUDA、Triton、FlashAttention 和 CUDA Graph。

次路径：

- 国产 64GB GPU 可以作为后续兼容性研究。
- 在框架和 kernel 兼容性明确之前，不放到主线关键路径。

## 目标模型计划

第一阶段 KV cache 底座模型：

```text
Qwen3-4B-Instruct-2507
```

第二阶段多模态目标模型：

```text
Qwen3.5-2B
```

可选同架构压力 workload：

```text
Qwen3-4B-Thinking-2507
```

Qwen3-4B-Instruct-2507 用于先完成 attention KV cache profiler、量化、压缩、benchmark 和 kernel 路线。Qwen3.5-2B 用于后续 visual prefix KV cache profiling、reuse 和 quantization。

不要硬编码模型结构。适配前必须检查 HuggingFace config：

- `num_hidden_layers`
- `num_attention_heads`
- `num_key_value_heads`
- `head_dim`
- `hidden_size`
- `torch_dtype`
- `max_position_embeddings`
- `rope_scaling`

多模态阶段还必须检查：

- `vision_config`
- `text_config`
- `image_token_id`
- `video_token_id`
- processor 输出格式
- visual token 数量
- multimodal special tokens

详细路线见 `roadmap/planning/07_model_route.md`。

## 算子后端计划

算子后端分三层：

- PyTorch reference：用于 correctness、metadata layout 和 error analysis。
- Triton optimized：用于 quantized store、dequant-to-scratch、INT4 pack/unpack，是主优化后端。
- CUDA optional：用于关键 kernel 的对照实现和底层能力展示，不阻塞主线。

CUTLASS 和 TileLang 作为后续调研，不作为当前阶段必须交付。

## 文档地图

- `roadmap/All_guide.md`: 项目总纲。
- `roadmap/planning/00_current_repo.md`: 当前代码结构与 KV 路径。
- `roadmap/planning/01_milestones.md`: 里程碑拆解。
- `roadmap/planning/02_commit_plan.md`: 小步提交计划。
- `roadmap/planning/03_test_benchmark_plan.md`: test 和 benchmark 标准。
- `roadmap/planning/04_risk_register.md`: 风险表。
- `roadmap/planning/05_interview_story.md`: 秋招项目叙事。
- `roadmap/planning/06_ai_collaboration_rules.md`: AI 协作规则，保持英文。
- `roadmap/planning/07_model_route.md`: 模型适配路线。
- `roadmap/planning/08_phase_m0_execution.md`: 当前 M0 阶段执行计划。
- `roadmap/planning/09_acceptance_workflow.md`: 任务验收流程。
- `roadmap/planning/10_baseline_benchmark_design.md`: baseline benchmark harness 设计。
- `roadmap/planning/11_sanity_run_plan.md`: baseline sanity run 计划与当前环境 blocker。
- `learnning/`: 技术学习笔记。
