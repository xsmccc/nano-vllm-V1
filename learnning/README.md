# 学习笔记

这个目录用于存放 NanoCache-V 实现过程中的技术学习笔记。

当前目录名是 `learnning`。这里先保留用户已经创建的目录名，不擅自改动。项目对外发布前，可以考虑统一改成 `learning`。

## 分类

- `triton/`: Triton 编程模型与 kernel 学习。
- `cuda/`: CUDA core 编程与 memory hierarchy。
- `pytorch/`: PyTorch tensor、dtype、profiling 和 CUDA 行为。
- `inference/`: LLM inference engine 相关概念。
- `kv_cache_quant/`: KV cache 量化技术。
- `papers/`: 论文阅读笔记和实现映射。
- `tilelang/`: TileLang 学习笔记。
- `profiling_benchmark/`: profiling 与 benchmark 方法论。
- `ai_workflow/`: AI-assisted development 规则和 skill 设计。

## 笔记模板

建议每篇学习笔记使用这个结构：

```text
# 主题

## 为什么重要

## 核心概念

## 在 NanoCache-V 中对应哪段代码

## 常见坑

## 可以做的小实验

## 参考资料
```
