# 模型适配路线

## 项目定位

NanoCache-V 是一个以 KV Cache 量化压缩为核心的 LLM/VLM inference 项目。

主线目标：

```text
面向长上下文与多模态推理的 KV Cache profiler、量化、压缩与复用系统
```

模型路线分为两层：

1. Attention KV 底座模型：先在标准 Causal LM 上完成 KV cache 系统。
2. 多模态目标模型：将 KV cache profiler、visual prefix cache reuse 和 quantization 扩展到 VLM 推理。

## 第一阶段模型：Qwen3-4B-Instruct-2507

第一阶段 native 适配模型：

```text
Qwen3-4B-Instruct-2507
```

用途：

- 建立 KV cache profiler。
- 建立 FP16/BF16 baseline。
- 实现 FP8 / INT8 / INT4 KV cache。
- 实现 KIVI-style K/V asymmetric quantization。
- 实现 hybrid recent/old KV policy。
- 建立 long-context benchmark。
- 建立 PyTorch reference、Triton optimized、CUDA optional 的算子后端路线。

选择它作为第一阶段模型，是因为它能隔离 attention KV cache 这个核心问题：

- 标准 Causal LM。
- GQA 结构清晰。
- 原生长上下文。
- 当前 nano-vLLM 的 Qwen3 decoder-only 路径可以承接。
- 不引入 vision encoder 和 processor 链路，便于先把 KV cache 系统做正确。

## 第二阶段模型：Qwen3.5-2B

第二阶段多模态适配目标：

```text
Qwen3.5-2B
```

用途：

- 分析 image-text-to-text 推理链路。
- 统计 visual tokens 数量。
- 统计 visual prefix KV cache 占用。
- 实现 visual prefix KV cache profiler。
- 实现 visual prefix KV cache reuse。
- 实现 visual prefix KV quantization。
- 与 HF / vLLM / SGLang oracle 对齐 correctness。

Qwen3.5-2B 是多模态阶段的主目标，因为它具备 VLM 链路，同时规模适合个人项目迭代。

## 可选压力 Workload：Qwen3-4B-Thinking-2507

可选同架构压力 workload：

```text
Qwen3-4B-Thinking-2507
```

它不作为独立模型适配任务，只作为 Qwen3-4B 路径下的长 decode / reasoning workload。

用途：

- 长输出 decode。
- TPOT 评估。
- KV cache 随生成长度增长的压力测试。

## 为什么先 Qwen3，再 Qwen3.5

NanoCache-V 的核心难点不是“能不能调起一个模型”，而是：

- KV cache layout 是否正确。
- quantized data 和 metadata 是否跟 block lifecycle 对齐。
- prefix cache、CoW、swap 是否能和量化状态共存。
- dequant-to-scratch、quantized store、INT4 pack/unpack 是否有 correctness test。
- profiler 和 benchmark 是否能证明显存与性能变化。

这些问题最好先在标准 attention KV cache 路径中解决。

Qwen3-4B-Instruct-2507 提供稳定的 attention KV 底座；Qwen3.5-2B 则用于把该系统扩展到多模态 visual prefix KV 场景。

## 为什么不直接以 Qwen3.5 / Qwen3.6 作为第一阶段

Qwen3.5 和 Qwen3.6 更接近最终多模态方向，但它们会同时引入多个变量：

- vision encoder。
- multimodal processor。
- image/video special tokens。
- hybrid layer structure。
- visual prefix cache 生命周期。
- 更复杂的 serving/oracle 对齐。

如果第一阶段直接进入这些链路，很难判断问题来自 KV cache 量化，还是来自多模态模型适配。

因此当前路线是：

```text
Qwen3-4B-Instruct-2507:
  解决 KV cache 系统本身

Qwen3.5-2B:
  解决 visual prefix KV cache reuse + quantization
```

Qwen3.6-27B 作为前沿调研对象，不作为当前端到端交付模型。

## 阶段路线

### M0: Baseline

目标：

- 下载并检查 Qwen3-4B-Instruct-2507。
- 跑通最小 sanity run。
- 确认 tokenizer、config、model forward、KV cache lifecycle。

### M1: KV Cache Profiler

目标：

- 在 Qwen3-4B-Instruct-2507 上实现 KV cache profiler。
- 输出 prefill/decode step metrics。
- 输出 KV cache memory summary。
- 为后续 visual prefix KV profiler 复用数据结构。

### M2: FP Baseline Backend

目标：

- 保持 Qwen3-4B-Instruct-2507 行为不变。
- 将当前 FP KV cache 路径抽象为 backend。
- 明确 backend interface 后续如何服务 VLM visual prefix KV。

### M3-M5: Quantized KV Cache

目标：

- 在 Qwen3-4B-Instruct-2507 上完成 FP8/INT8/INT4。
- 完成 KIVI-style 策略。
- 完成 hybrid recent/old policy。
- 输出 correctness、memory saving、latency overhead。

### M6: 多模态 Visual Prefix KV

目标：

- 以 Qwen3.5-2B 为目标模型。
- 使用 HF / vLLM / SGLang 作为 oracle。
- 分析 processor 输出、visual token 数量和 visual prefix KV 占用。
- 设计 visual prefix KV cache reuse。
- 设计 visual prefix KV quantization。

### M7: Compressed KV Microbenchmark

目标：

- 围绕 Qwen3-4B 和 Qwen3.5-2B 的 KV shape 做 compressed KV microbenchmark。
- 分析 latent/compressed KV layout、decompression overhead 和 memory bandwidth。

### M8: Kernel Optimization

目标：

- 对 quantized store、dequant-to-scratch、INT4 pack/unpack 做 Triton kernel。
- CUDA core 版本作为可选对照实现。

## 算子后端路线

### PyTorch Reference

用途：

- quant/dequant correctness。
- metadata layout 验证。
- error analysis。

不用于性能结论。

### Triton Optimized

用途：

- quantized store。
- dequant-to-scratch。
- INT4 pack/unpack。
- memory bandwidth microbenchmark。

这是 NanoCache-V 的主优化后端。

### CUDA Optional

用途：

- 对关键 Triton kernel 做 CUDA core 对照实现。
- 展示底层 kernel 理解。

CUDA 作为加分项，不阻塞主线交付。

### CUTLASS / TileLang

定位：

- 调研和后续扩展。
- 不作为当前阶段必须交付。

## 最终展示口径

NanoCache-V 最终展示应包含两条线：

```text
Qwen3-4B-Instruct-2507:
  KV cache profiler + quantization + benchmark + kernel optimization

Qwen3.5-2B:
  visual prefix KV cache profiling + reuse + quantization
```
