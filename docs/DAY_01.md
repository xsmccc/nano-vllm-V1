# Day 1 (6/14): Qwen3-VL-8B 架构理解 + 数据流 Trace

## 任务 1.1：加载模型，Trace 前向数据流（4h）

### 1.1a: 下载模型 + 环境验证（30min）
```bash
# 确认模型已缓存，如果没有则下载
python -c "from transformers import Qwen3VLForConditionalGeneration; print('OK')"
```

### 1.1b: 写 trace 脚本（2h）
- 文件: `prism_infer/analysis/trace_qwen3vl.py`
- 输入: 一张测试图 (占位图即可) + 文本 "描述这张图片"
- 输出: 每一步的 tensor shape、dtype、device
- 关键 hook 点:
  - ViT 输入/输出 shape
  - Projector 输入/输出 shape
  - token merge 后的 shape
  - 每层 attention 的 Q/K/V shape
  - M-RoPE position_ids 的 shape

### 1.1c: 手画数据流图（1.5h）
- 在纸上画出完整的数据流
- 标注每个阶段的 shape 变化
- 特别标注 visual token 和 text token 的分叉/合并点
- 拍个照存到 data/ 目录

**验证标准**: 能不看代码，看着数据流图讲清楚 Qwen3-VL 的完整推理流程

---

## 任务 1.2：对比 nano-vllm，列出改动清单（3h）

### 1.2a: prism-infer 源码重读（1h）
重点看这几个文件:
- `prism_infer/engine/model_runner.py` — prepare_prefill, warmup_model
- `prism_infer/layers/attention.py` — Attention.forward, store_kvcache
- `prism_infer/models/qwen3.py` — Qwen3Attention, Qwen3DecoderLayer, Qwen3Model
- `prism_infer/config.py` — Config

### 1.2b: 列出改动清单（1.5h）
清单文件: `docs/CHANGES.md`

每项标注:
- 文件路径
- 改动类型: [新增] / [修改] / [不改]
- 为什么需要改
- 优先级: P0(必须) / P1(重要) / P2(优化)

### 1.2c: 与 Claude Code review 清单（30min）
- 检查是否有遗漏
- 确认优先级排序合理

**验证标准**: 清单完整，每个改动点能一句话说清原因

---

## 任务 0：项目框架 + 环境验证（Day 1 前置，30min）

### 环境检查
- [ ] CUDA 可用
- [ ] torch 2.9.1, transformers 4.57.1
- [ ] Qwen3-VL-8B 模型是否已下载
- [ ] 4090 显存状况
- [ ] flash-attn 是否可用（可选）

### 项目框架检查
- [ ] pyproject.toml 正确
- [ ] prism_infer/ 包结构完整
- [ ] .gitignore 包含 data/ 等大文件

---

## Day 1 产出清单
1. `prism_infer/analysis/trace_qwen3vl.py` — 数据流 trace 脚本
2. 数据流手绘图 (照片)
3. `docs/CHANGES.md` — prism-infer 改动清单
4. 知识库: 1篇 (Qwen3-VL 数据流 / M-RoPE 原理)
