# PyTorch CUDA Timing 与 Memory Stats

## 为什么重要

NanoCache-V 的目标之一是做 KV cache profiler 和 benchmark。

如果没有正确的 CUDA timing 和 memory stats，很容易得出错误性能结论。CUDA kernel 是异步执行的，直接用 `time.time()` 或 `perf_counter()` 包住 Python 调用，通常会漏掉 GPU 实际执行时间。

## 核心概念

### CUDA 是异步的

大多数 PyTorch CUDA op 提交 kernel 后会立刻返回，CPU 不会默认等待 GPU 执行完成。

因此端到端 timing 至少要：

```python
torch.cuda.synchronize()
start = time.perf_counter()
...
torch.cuda.synchronize()
elapsed = time.perf_counter() - start
```

### CUDA Event

microbenchmark 更适合使用 CUDA event：

```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
...
end.record()
torch.cuda.synchronize()
elapsed_ms = start.elapsed_time(end)
```

端到端 benchmark 可以先用 wall time + synchronize，因为它包含 scheduler、Python、sampling 等开销。

### Memory Stats

常用接口：

```python
torch.cuda.memory_allocated()
torch.cuda.memory_reserved()
torch.cuda.max_memory_allocated()
torch.cuda.mem_get_info()
torch.cuda.reset_peak_memory_stats()
```

含义：

- `memory_allocated`: 当前 tensor 实际占用。
- `memory_reserved`: PyTorch caching allocator 已向 CUDA 申请的显存。
- `max_memory_allocated`: 峰值 allocated。
- `mem_get_info`: CUDA 层面的 free/total。

## 在 NanoCache-V 中对应哪段代码

当前相关位置：

- `nanovllm/engine/model_runner.py`
  - `warmup_model()`
  - `allocate_kv_cache()`
  - 已使用 `torch.cuda.mem_get_info()` 和 `torch.cuda.memory_stats()`。
- `bench.py`
  - 当前只用 `time.time()`，没有 synchronize、repeat、memory stats。

后续 M1 profiler hook：

- `LLMEngine.step()`
- `ModelRunner.run_model()`
- `ModelRunner.allocate_kv_cache()`
- `Attention.forward()`

## 常见坑

1. 没有 `torch.cuda.synchronize()` 就记录时间。
2. 只跑一次 benchmark。
3. 只报 mean，不报 median/p90。
4. 没有 warmup，把首次 kernel 编译和 cache 初始化算进正式结果。
5. 把 `memory_reserved` 当作实际 tensor 占用。
6. 只记录总吞吐，不区分 prefill 和 decode。

## 可以做的小实验

1. 对同一段 CUDA op，比较有无 synchronize 的 timing 差异。
2. 用 `torch.cuda.reset_peak_memory_stats()` 观察一次 generate 的峰值显存。
3. 在 `allocate_kv_cache()` 前后记录 memory stats，验证 KV cache 真实显存占用。

## 与当前任务关系

M0 阶段只记录方法论。

M1 阶段会把这些接口接入 profiler，但必须控制 profiler overhead，不能每个小 op 都强制 synchronize。
