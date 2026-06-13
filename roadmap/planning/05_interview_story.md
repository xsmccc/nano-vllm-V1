# 秋招项目叙事

## 一句话介绍

NanoCache-V 是一个基于 nano-vLLM-V1 的 KV cache 量化压缩与 profiling 系统，面向长上下文和多模态推理场景。

## 项目动机

长上下文和多模态推理中，显存瓶颈往往不仅来自模型权重，也来自随着 context length、生成长度和 visual tokens 增长的 KV cache。降低 KV cache 显存占用可以提升最大上下文长度、batch size、并发能力，以及 visual prefix cache 的复用效率。

但 KV cache 量化不能只写一个 quant/dequant 函数。它必须和 paged attention、prefix cache、copy-on-write、swap、scheduler、benchmark、profiler，以及多模态 visual prefix 生命周期一起考虑。

## 为什么这是 Infra 项目

这个项目涉及：

- inference engine scheduling。
- paged KV cache layout。
- visual prefix KV cache reuse。
- memory profiling。
- quantization error analysis。
- GPU benchmark discipline。
- Triton/CUDA kernel path。

它不是单纯的模型精度实验。

## 核心技术挑战

1. 当前 FlashAttention cache path 期望 FP-compatible cache。
2. INT4 cache 需要 packing 和 metadata 管理。
3. K 和 V 的量化敏感性不同。
4. Hybrid recent/old policy 必须尊重 block lifecycle。
5. Prefix cache 和 CoW 会复杂化 block ownership。
6. Visual prefix cache 需要处理 image tokens、processor 输出和复用粒度。
7. Benchmark 必须区分 memory saving、quant/dequant overhead 和 attention latency。

## 预期展示结果

- 量化前后的 KV cache memory per token。
- 固定显存下可容纳的最大上下文长度。
- quantization error metrics。
- decode latency comparison。
- profiler timeline。
- no-silent-fallback tests。
- microkernel benchmark。
- visual prefix KV cache profiling / reuse / quantization 结果。

## 项目叙事结构

1. 我从 nano-vLLM-V1 出发，先追踪完整 KV cache 路径。
2. 我在 Qwen3-4B-Instruct-2507 上完成 KV cache profiler、benchmark 规范和 FP baseline。
3. 我抽象出 KV cache backend，让 FP baseline 和 quantized modes 走同一个受控接口。
4. 我实现 reference quantized KV cache，并补 correctness tests。
5. 我实现 KIVI-style K/V asymmetric quantization。
6. 我评估 hybrid recent/old policy。
7. 我将 profiler 和量化策略扩展到 Qwen3.5-2B 的 visual prefix KV cache。
8. 我根据 profiler 证据，用 Triton/CUDA 优化真正的 bottleneck。

## 面试诚实原则

只展示已经实现和测量过的结果。

如果某个模块还在 roadmap 阶段，就明确说 planned，不把计划说成成果。
