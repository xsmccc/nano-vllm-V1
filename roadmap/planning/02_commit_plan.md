# 小步提交计划

## 基本规则

每次 commit 只改变一个明确行为或一个明确模块。

避免混合：

- formatting 和 logic。
- profiler 和 quantization。
- benchmark rewrite 和 model path 修改。
- 文档和 runtime 行为，除非文档描述的就是同一个小改动。

## 建议 Commit 顺序

1. 添加项目 roadmap 和 planning 文档。
2. 添加 benchmark 规范文档和 baseline benchmark skeleton。
3. 添加 profiler utilities，但先不接入 model execution。
4. 将 profiler 接入 `LLMEngine.step()` 和 `ModelRunner.run()`。
5. 添加 KV cache backend interface。
6. 将当前 FP cache allocation 移入 FP backend。
7. 将当前 FP cache store path 移入 FP backend。
8. 添加 FP backend parity correctness tests。
9. 添加 quantization reference utilities。
10. 添加 quant/dequant unit tests。
11. 添加 INT8 metadata layout。
12. 添加 INT8 dequant-to-scratch path。
13. 添加 INT4 packing utilities。
14. 添加 INT4 unit tests。
15. 添加 hybrid block state。
16. 添加 recent-window policy。
17. 添加 MLA-style microbenchmark。
18. 添加 Triton quant/dequant kernels。

## Commit Message 风格

使用短而事实明确的 commit message：

```text
docs: add NanoCache-V roadmap
profiler: add cuda timing utility
kvcache: introduce backend interface
kvcache: move fp allocation into backend
tests: add int8 kv roundtrip test
bench: add decode kv cache benchmark
```

## 每次 Commit 前记录

需要记录：

- 改了什么。
- 为什么改。
- 怎么测试。
- 已知限制。

如果没有跑测试，必须说明原因。
