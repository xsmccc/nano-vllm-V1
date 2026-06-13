# Triton 学习

重点：

- program id 映射。
- block-level parallelism。
- pointer arithmetic。
- masks。
- memory coalescing。
- quant/dequant kernels。
- INT4 pack/unpack。

NanoCache-V 的第一个目标：

- 理解当前 `store_kvcache_kernel`。
- 写 reference dequant kernel。
- 和 PyTorch implementation 做 benchmark 对比。
