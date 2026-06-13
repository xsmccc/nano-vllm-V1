# 风险表

## R1: FlashAttention 不能直接消费 INT4/INT8 KV

风险：

当前 decode path 使用 `flash_attn_with_kvcache`，它期望 FP-compatible cache layout。

缓解：

- 第一版实现 quantized storage + explicit dequant-to-scratch。
- 明确标注这是 reference path，不是最终 optimized path。
- 只有在 fused 或 optimized kernel 实测后，才能声称 speedup。

## R2: CUDA Graph 兼容性

风险：

动态 quantization metadata 和 scratch buffer 可能破坏 CUDA Graph capture 假设。

缓解：

- 初期 quantized backend 如有必要，明确要求 `enforce_eager=True`。
- 不支持时显式报错。
- 后续再加入 graph-compatible static scratch buffers。

## R3: Hybrid Policy 与 Prefix Cache 冲突

风险：

一个 shared block 可能对某条 sequence 是 old block，但对另一条 sequence 仍然属于 recent block。

缓解：

- 从保守的 block-level quantization 开始。
- 永远不量化 partial writable blocks。
- 在 ownership 规则明确前，不量化 shared blocks。

## R4: Metadata Overhead 吃掉 INT4 收益

风险：

group size 太小会降低误差，但 scale/zero metadata 开销会上升。

缓解：

- benchmark 多个 group size。
- 报告 effective bytes/token，必须包含 metadata。

## R5: Benchmark 噪声

风险：

单次 timing 很容易误导。

缓解：

- 使用 warmup 和 repeats。
- synchronize CUDA。
- 报告 median 和 p90。
- 固定并记录 benchmark configs。

## R6: 硬件可移植性

风险：

Triton、FlashAttention、CUDA kernels 都明显偏 NVIDIA 生态。

缓解：

- NVIDIA path 作为主路径。
- backend/kernel code 隔离。
- 国产 GPU 支持作为后续兼容性研究，不放在初期关键路径。

## R7: AI-Assisted Development Drift

风险：

AI 可能过度声称结果、跳过测试、做大范围重构、隐藏 unsupported behavior。

缓解：

- 执行 `06_ai_collaboration_rules.md`。
- 每个 feature 都要求 tests 和 benchmark notes。
- 所有 unsupported mode 必须显式记录。
