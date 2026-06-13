# 里程碑规划

## M0: Baseline 与文档梳理

交付物：

- 当前仓库结构图。
- KV cache 数据路径说明。
- baseline benchmark 要求。
- 初始 AI 协作规则。

完成标准：

- 新读者能说清楚 KV cache 在哪里分配、写入、读取、复制和 swap。

## M1: KV Cache Profiler

交付物：

- profiler module。
- per-step timing。
- GPU memory summary。
- KV block occupancy。
- prefix cache、CoW、swap counters。

完成标准：

- 跑 benchmark 时能产出结构化 profiler report。
- profiler 不改变模型输出。

## M2: FP Baseline KV Backend

交付物：

- `KVCacheBackend` interface。
- 与当前行为一致的 FP16/BF16 backend。
- 接入 `ModelRunner` 和 `Attention`。

完成标准：

- 原有 generation 行为不变。
- tests 能证明 baseline backend 与原来的裸 tensor path 等价。

## M3: Quantized KV Reference Backend

交付物：

- FP8 storage path。
- INT8 storage path。
- INT4 group-wise storage path。
- dequant-to-scratch path。
- no-silent-fallback checks。

完成标准：

- quant/dequant correctness tests 通过。
- quantized cache dequant 后的 attention output 可以和 FP baseline 对比。

## M4: KIVI-Style Quantization

交付物：

- Key per-channel quantization。
- Value per-token 或 group-wise quantization。
- scale/zero metadata layout。
- error analysis report。

完成标准：

- 每层、每种 tensor 类型都有 error metrics。
- decode output degradation 可测量、可解释。

## M5: Hybrid Recent/Old Policy

交付物：

- block quantization state。
- recent window configuration。
- old block quantization trigger。
- 与 CoW、swap、prefix cache 的兼容。

完成标准：

- recent tokens 保持 FP。
- old full blocks 可以被量化。
- partial writable blocks 永远不会被量化。

## M6: 多模态 Visual Prefix KV

交付物：

- Qwen3.5-2B processor / model config 分析。
- visual token 数量统计。
- visual prefix KV cache memory profiler。
- visual prefix KV reuse 设计。
- visual prefix KV quantization reference path。
- HF / vLLM / SGLang oracle 对齐方案。

完成标准：

- 能解释 image input 如何转成 visual tokens。
- 能统计 visual prefix KV cache 显存占用。
- 能说明 visual prefix reuse 和 quantization 的 correctness 验证方式。

## M7: Compressed KV Microbenchmark

交付物：

- synthetic compressed KV layout。
- decode benchmark。
- decompression overhead report。
- memory bandwidth analysis。

完成标准：

- 能用实测数据解释 compressed KV 的收益和代价。

## M8: Kernel Optimization

交付物：

- Triton quant store。
- Triton dequant load。
- INT4 pack/unpack。
- optional CUDA core implementation。
- optional TileLang/CUTLASS exploration。

完成标准：

- 每个 optimized kernel 都有 correctness tests。
- 每个 optimized kernel 都有与 reference implementation 的 benchmark 对比。
