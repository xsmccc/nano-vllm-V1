# M0 当前阶段执行计划

## 阶段定位

M0 不是实现 KV quantization 的阶段。

M0 的目标是把当前 nano-vLLM-V1 变成一个可审计、可复现、可继续演进的 baseline 工程。

如果 M0 做不好，后续 M1/M2/M3 的 profiler、backend 抽象和量化实现都会缺少可信基线。

## M0 目标

完成以下事情：

1. 确认仓库状态和提交边界。
2. 梳理当前推理主链路。
3. 梳理当前 KV cache 生命周期。
4. 建立 baseline benchmark 规范。
5. 建立 baseline sanity run 流程。
6. 明确第一批学习任务。
7. 产出进入 M1 的输入文档。

## M0 非目标

本阶段不做：

- INT8/INT4 量化。
- KV backend 抽象。
- Triton kernel 优化。
- 多模型 native 适配。
- 多架构模型适配。
- 大规模 benchmark 结论。

## 任务拆分

### M0-T1: 仓库状态审计

目标：

- 确认实际项目根目录。
- 确认 git 状态。
- 明确哪些文件是已有修改，哪些是本阶段新增。

产出：

- `roadmap/planning/00_current_repo.md` 更新。
- 一条干净的文档 commit。

验收：

- 不混入 runtime 代码改动。
- 能解释为什么当前仓库存在外层目录和 `nano-vllm-V1/` 两套 git 状态。

### M0-T2: 推理主链路梳理

目标：

- 梳理 `LLM.generate -> LLMEngine.step -> Scheduler.schedule -> ModelRunner.run -> Attention.forward`。

产出：

- 当前代码路径图。
- prefill/decode 两条路径说明。

验收：

- 能说明每一步输入输出是什么。
- 能说明 prefill 和 decode 的差异。

### M0-T3: KV Cache 生命周期梳理

目标：

- 梳理 allocate、store、read、copy-on-write、swap、deallocate。

产出：

- KV cache lifecycle 文档。
- 当前 layout 和 slot mapping 说明。

验收：

- 能说明物理 block ID、global slot、block table 的关系。
- 能说明 prefix cache 和 CoW 为什么会影响量化 metadata。

### M0-T4: Baseline Benchmark 规范化设计

目标：

- 设计一个符合 NanoCache-V 要求的 benchmark harness。

产出：

- benchmark 参数列表。
- 输出格式。
- warmup/repeat/synchronize/memory stats 规范。

验收：

- benchmark 设计能区分 prefill 和 decode。
- benchmark 输出能作为后续 M1 profiler 的对照。

### M0-T5: Baseline Sanity Run 计划

目标：

- 为 Qwen3-4B-Instruct-2507 建立最小 sanity run。

产出：

- 运行命令。
- 预期输出。
- 失败排查表。

验收：

- 能跑通一条短 prompt。
- 能跑通 token id input。
- 能确认 CUDA、FlashAttention、Triton 依赖可用。

### M0-T6: 学习任务启动

目标：

- 开始补足实现 M1/M2 所需知识。

学习主题：

- PyTorch CUDA timing 和 memory stats。
- prefill/decode 基本概念。
- current Triton `store_kvcache_kernel`。
- PagedAttention 的 block table 思想。

产出：

- `learnning/profiling_benchmark/` 下至少一篇 timing/memory stats 笔记。
- `learnning/inference/` 下至少一篇 prefill/decode 笔记。
- `learnning/triton/` 下至少一篇 `store_kvcache_kernel` 解析。

## M0 建议任务顺序

1. M0-T1 仓库状态审计。
2. M0-T2 推理主链路梳理。
3. M0-T3 KV Cache 生命周期梳理。
4. M0-T4 Baseline Benchmark 规范化设计。
5. M0-T5 Baseline Sanity Run 计划。
6. M0-T6 学习任务启动。

## M0 完成定义

M0 完成时应满足：

- 当前仓库结构和主路径清楚。
- 当前 KV cache lifecycle 清楚。
- 有 baseline benchmark 设计。
- 有 sanity run 计划。
- 有明确进入 M1 profiler 的接口点。
- 没有对量化性能做未经测量的结论。

## 进入 M1 的条件

只有满足以下条件，才进入 M1：

- baseline run 能跑通，或者明确记录环境 blocker。
- benchmark harness 的设计通过 review。
- 已经知道 profiler hook 应该放在哪些位置。
- 已经明确 profiler 只观测、不改变模型输出。

## 当前下一步

M0 文档侧工作已经基本完成。

当前有两个可选下一步：

1. 处理环境 blocker：
   - 创建 Python 虚拟环境。
   - 安装 torch/triton/transformers/flash-attn。
   - 下载 Qwen3-4B-Instruct-2507。
   - 执行 `11_sanity_run_plan.md` 中的 sanity run。

2. 进入 M1 profiler 设计：
   - 先写 profiler 设计文档。
   - 明确 profiler 数据结构和 hook 点。
   - 暂不修改 runtime，直到设计通过 review。

建议优先处理环境 blocker，因为 baseline sanity run 是 M1 profiler 验证的基础。

## M0 当前状态表

| 任务 | 状态 | 说明 |
|---|---|---|
| M0-T1 仓库状态审计 | pass | 已确认内外两层 git repo，后续以内层 `nano-vllm-V1` 为项目根 |
| M0-T2 推理主链路梳理 | pass | 已补充主链路、prefill/decode 输入输出、上下文张量和 profiler hook 点 |
| M0-T3 KV Cache 生命周期梳理 | pass | 已补充 allocate、slot mapping、store、read、CoW、swap、deallocate 全链路 |
| M0-T4 Baseline Benchmark 规范化设计 | pass | 已完成 benchmark harness 设计，明确 warmup/repeat/sync/memory stats/output format |
| M0-T5 Baseline Sanity Run 计划 | partial | 计划已完成；已迁移共享 venv，仍缺 flash_attn 和本地 Qwen3-4B-Instruct-2507 模型 |
| M0-T6 学习任务启动 | pass | 已补 profiling、inference、Triton 三类首批学习笔记 |

## M0-T1 验收记录

Task:

```text
M0-T1 仓库状态审计
```

Milestone:

```text
M0 Baseline 与文档梳理
```

Changed files:

```text
roadmap/planning/00_current_repo.md
roadmap/planning/08_phase_m0_execution.md
```

What changed:

```text
记录了外层 repo 和内层 repo 的路径、branch、remote、当前 git status。
明确后续开发以 nano-vllm-V1 作为项目根。
补充 M0 当前状态表。
```

What did not change:

```text
没有修改 runtime 代码。
没有修改 benchmark。
没有清理外层 repo 状态。
```

Verification:

```text
运行了外层和内层 git status、git rev-parse、git branch、git remote、git log、git ls-files。
确认内层 repo 当前未跟踪文件只有 roadmap/ 和 learnning/ 文档目录。
```

Not tested:

```text
没有运行 Python 测试或模型推理，因为本任务是仓库状态审计。
```

Risks:

```text
外层 repo 仍处于大量删除 + 未跟踪 nano-vllm-V1 的状态。
后续如果需要整理仓库结构，必须单独规划 repo migration，不和功能开发混合。
```

Next step:

```text
继续 M0-T2，补充 prefill/decode 的输入输出和 profiler hook 点。
```

## M0-T2 验收记录

Task:

```text
M0-T2 推理主链路梳理
```

Milestone:

```text
M0 Baseline 与文档梳理
```

Changed files:

```text
roadmap/planning/00_current_repo.md
roadmap/planning/08_phase_m0_execution.md
```

What changed:

```text
补充了 LLM.generate -> LLMEngine.step -> Scheduler.schedule -> ModelRunner.run -> model forward -> Attention.forward 的主链路。
补充了 prefill/decode 的 input_ids、positions、slot_mapping、block_tables、context_lens、cu_seqlens 等输入输出 shape。
补充了 M1 profiler 的建议 hook 点。
```

What did not change:

```text
没有修改 runtime 代码。
没有实现 profiler。
没有修改 bench.py。
```

Verification:

```text
对照阅读了 nanovllm/engine/llm_engine.py、scheduler.py、model_runner.py、layers/attention.py、models/qwen3.py。
确认文档描述和当前代码路径一致。
```

Not tested:

```text
没有运行模型推理，因为本任务是代码路径梳理。
```

Risks:

```text
Attention.forward 内部更细粒度的每层耗时尚未实际测量。
M1 profiler 需要控制 overhead，不能默认对每层都做重同步计时。
```

Next step:

```text
进入 M0-T3，补齐 KV cache lifecycle：allocate、store、read、CoW、swap、deallocate。
```

## M0-T3 验收记录

Task:

```text
M0-T3 KV Cache 生命周期梳理
```

Milestone:

```text
M0 Baseline 与文档梳理
```

Changed files:

```text
roadmap/planning/00_current_repo.md
roadmap/planning/08_phase_m0_execution.md
```

What changed:

```text
补充 KV cache allocate、global slot mapping、store、read、copy-on-write、swap out/in、deallocate 的完整生命周期。
明确 quantized KV cache 后续必须让 data、metadata、block state 都以 physical block 为核心管理。
标记了 partial block、shared block、swap metadata、CoW metadata 的风险。
```

What did not change:

```text
没有修改 runtime 代码。
没有实现 KV cache backend。
没有实现 profiler。
```

Verification:

```text
对照阅读了 model_runner.py 中 allocate_kv_cache、prepare_prefill、prepare_decode、copy_kv_blocks、swap_blocks。
对照阅读了 block_manager.py 中 allocate、copy_on_write、swap_out、swap_in、deallocate。
对照阅读了 attention.py 中 store_kvcache 和 FlashAttention 调用路径。
```

Not tested:

```text
没有运行模型推理，因为本任务是 lifecycle 文档梳理。
```

Risks:

```text
当前文档是静态代码阅读结果，尚未通过 profiler runtime event 验证。
M1 需要用实际 counters 验证 CoW/swap/prefix cache 事件是否按预期发生。
```

Next step:

```text
进入 M0-T4，设计 baseline benchmark harness，但先不改 bench.py。
```

## M0-T4 验收记录

Task:

```text
M0-T4 Baseline Benchmark 规范化设计
```

Milestone:

```text
M0 Baseline 与文档梳理
```

Changed files:

```text
roadmap/planning/10_baseline_benchmark_design.md
roadmap/planning/08_phase_m0_execution.md
roadmap/All_guide.md
```

What changed:

```text
设计了 baseline benchmark harness 的层级、参数、timing 规则、warmup/repeat 规则、memory stats、输出格式和验收标准。
明确 M0 只设计，不修改 bench.py。
明确 prefill/decode 分离需要 M1 profiler 支撑。
```

What did not change:

```text
没有修改 bench.py。
没有实现 benchmark harness。
没有生成任何性能结论。
```

Verification:

```text
对照 03_test_benchmark_plan.md 的硬性要求检查：warmup、repeat、synchronize、memory stats、shape/config 输出、median/p90/min/max 均已纳入设计。
```

Not tested:

```text
没有运行 benchmark，因为本任务是 benchmark 设计。
```

Risks:

```text
当前 LLMEngine.generate 尚未返回结构化 step metrics。
prefill/decode 分离需要 M1 profiler 接入后才能稳定产出。
```

Next step:

```text
进入 M0-T5，建立 baseline sanity run 计划。
```

## M0-T5 验收记录

Task:

```text
M0-T5 Baseline Sanity Run 计划
```

Milestone:

```text
M0 Baseline 与文档梳理
```

Changed files:

```text
roadmap/planning/11_sanity_run_plan.md
roadmap/planning/08_phase_m0_execution.md
roadmap/All_guide.md
```

What changed:

```text
补充 baseline sanity run 计划，包括环境检查、模型检查、文本 prompt run、token id prompt run、eager/CUDA Graph 对照和失败排查。
记录当前环境 blocker。
```

What did not change:

```text
没有安装依赖。
没有下载模型。
没有运行 nano-vLLM generate。
没有修改 runtime 代码。
```

Verification:

```text
运行了 python3 版本检查。
运行了 torch/triton/transformers/flash_attn import availability 检查。
运行了 torch CUDA availability 检查。
搜索了 /home/xsmccc 下 Qwen3-4B-Instruct-2507 本地模型目录。
```

Result:

```text
python3 exists: /usr/bin/python3, Python 3.12.3
torch: missing
triton: missing
transformers: missing
flash_attn: missing
Qwen3-4B-Instruct-2507 local model dirs: not found
```

Shared venv update:

```text
found venv: /home/xsmccc/SGlang/.venv-sglang-debug
moved to: /home/xsmccc/.venvs/sglang-debug-py312
compat symlink: /home/xsmccc/SGlang/.venv-sglang-debug
project symlink: /home/xsmccc/nano-vllm/.venv
torch: 2.9.1+cu128
cuda_available: True
cuda_device_count: 1
device_0: NVIDIA GeForce RTX 4070 Laptop GPU
triton: found
transformers: found
flash_attn: missing
```

Not tested:

```text
没有运行模型推理，因为共享环境仍缺 flash_attn，且本地模型目录缺失。
```

Risks:

```text
后续安装 flash-attn 可能受 CUDA/PyTorch 版本影响。
Qwen3 模型路径和当前 qwen3.py 的 config 字段兼容性还需实际验证。
```

Next step:

```text
进入 M0-T6，补首批学习笔记；环境准备完成后再执行 sanity run。
```

## M0-T6 验收记录

Task:

```text
M0-T6 学习任务启动
```

Milestone:

```text
M0 Baseline 与文档梳理
```

Changed files:

```text
learnning/profiling_benchmark/cuda_timing_memory_stats.md
learnning/inference/prefill_decode_paged_kv.md
learnning/triton/store_kvcache_kernel.md
roadmap/planning/08_phase_m0_execution.md
```

What changed:

```text
补充 PyTorch CUDA timing 与 memory stats 学习笔记。
补充 prefill/decode 与 paged KV cache 学习笔记。
补充当前 Triton store_kvcache_kernel 解析。
```

What did not change:

```text
没有实现 profiler。
没有实现 Triton 新 kernel。
没有运行 benchmark。
```

Verification:

```text
学习笔记均包含为什么重要、核心概念、当前代码映射、常见坑和小实验。
内容与 M0/M1/M2 的后续任务直接相关。
```

Not tested:

```text
没有运行代码实验，因为当前 Python 环境缺少 torch/triton/transformers/flash_attn。
```

Risks:

```text
Triton kernel 性能判断目前仍是 hypothesis，后续必须通过 profiler/benchmark 验证。
```

Next step:

```text
汇总 M0 状态，确认是否进入 M1 或先处理环境 blocker。
```
