# Inference 学习

重点：

- prefill vs decode。
- paged attention。
- continuous batching。
- prefix cache。
- copy-on-write。
- swap。
- CUDA Graph。

NanoCache-V 相关性：

- quantized KV cache 必须接入现有 inference lifecycle，而不是孤立存在。
