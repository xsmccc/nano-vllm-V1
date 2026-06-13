# KV Cache 量化学习

重点：

- FP8 KV cache。
- INT8 KV cache。
- INT4 group-wise KV cache。
- KIVI-style asymmetric K/V quantization。
- hybrid recent/old policy。
- metadata overhead。

核心问题：

在不引入不可接受精度损失和 decode latency overhead 的前提下，KV cache 到底能节省多少显存？
