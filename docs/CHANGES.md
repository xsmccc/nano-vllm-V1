# Prism-Infer 改动清单 — 适配 Qwen3-VL-8B

> 基于 2026-06-14 Day 1 数据流 Trace 结果更新

## Qwen3-VL-8B 数据流

```
图片(224×224)                              文本 "描述这张图片"
     │                                           │
     ▼                                           ▼
 processor                                   tokenizer
  pixel_values [256, 1536]                    input_ids [1, 75]
  image_grid_thw [[1, 16, 16]]
     │                                           │
     ▼                                           ▼
 ┌─────────────────────┐                   embed_tokens
 │  Vision Encoder     │                        │
 │  (27 ViT blocks)    │                   [75, 4096]
 │                     │                        │
 │  patch_embed Conv3D │                        │
 │  pos_embed + ViT RoPE                        │
 │  27× VisionBlock    │                        │
 │                     │                        │
 │  Outputs:           │                        │
 │   main   [64,4096]──┼──→ 替换 input_ids 中视觉占位 token
 │   ds[0]  [64,4096]──┼──→ 注入 layer 8
 │   ds[1]  [64,4096]──┼──→ 注入 layer 16
 │   ds[2]  [64,4096]──┼──→ 注入 layer 24
 └─────────────────────┘                        │
                                                ▼
 ┌──────────────────────────────────────────────────┐
 │  Qwen3VLTextModel (36 layers, LLM backbone)      │
 │                                                  │
 │  Layer 0-7:    标准 Transformer (visual tokens   │
 │                + text tokens 混合, M-RoPE)        │
 │  Layer 8 ← ds[0] 注入 (hidden += deepstack[0])   │
 │  Layer 9-15:   标准 Transformer                   │
 │  Layer 16 ← ds[1] 注入                           │
 │  Layer 17-23:  标准 Transformer                   │
 │  Layer 24 ← ds[2] 注入                           │
 │  Layer 25-35:  标准 Transformer                   │
 │                                                  │
 │  M-RoPE: mrope_section=[24,20,20]×2=128         │
 │    - 视觉 token: 3D position (T, H, W)            │
 │    - 文本 token: 1D position (仅 T)               │
 └──────────────────────────────────────────────────┘
     │
     ▼
 lm_head (4096 → 151936)
     │
     ▼
 logits [1, 75, 151936]
```

## 模型参数摘要

| 组件 | 详情 |
|------|------|
| Vision Encoder | 27 ViT blocks, hidden=1152, intermediate=4304 |
| ViT Output | hidden=1152 → merger → 4096 (LLM dim) |
| Merger | LN → Linear(1152→4096) → GELU → Linear(4096→4096) |
| Deepstack Mergers | 3 个额外 merger, 分别在 LLM layer 8/16/24 注入 |
| LLM Backbone | 36 layers, hidden=4096, Q-heads=32, KV-heads=8, head-dim=128 |
| M-RoPE | mrope_section=[24,20,20], interleaved, theta=5M |
| Vocab | 151936 |
| Max Context | 262144 |

## 与现有 prism-infer 的差异

| 现有 (Qwen3, 纯文本) | Qwen3-VL | 结论 |
|------|------|------|
| Qwen3ForCausalLM | Qwen3VLForConditionalGeneration | **需新建** |
| Qwen3Model (28 layers) | Qwen3VLTextModel (36 layers) | **需新建**, 层数不同 |
| Qwen3Attention | Qwen3VLTextAttention | **需新建**, 接口可能不同 |
| Qwen3DecoderLayer | Qwen3VLTextDecoderLayer | **需新建** |
| 1D RoPE | M-RoPE (3D+1D混合) | **需新建** |
| 无 vision | VisionEncoder + 4 mergers | **需新建** |
| 无 deepstack | 视觉特征 4 点注入 | **需新建** |
| QKVParallelLinear | 同名, 功能相同 | **可复用** |
| RMSNorm | 同名, 功能相同 | **可复用** |
| SiLU MLP | 同名, 功能相同 | **可复用** |
| BlockManager | 不变 | **可复用** |
| Scheduler | 不变 | **可复用** |

---

## 改动清单 (更新)

### P0 — 第一阶段：单图推理跑通

#### [新增] `prism_infer/vision/vision_encoder.py`
- **做什么**：Qwen3-VL 的 ViT, 27 个 VisionBlock
- **类名**：`VisionEncoder`
- **输入**：`pixel_values` [B, C×T, H, W]（Conv3d 内部 reshape）
- **输出**：`tuple[main_features, [ds0, ds1, ds2]]`
  - main: [total_patches, 4096]
  - ds0-ds2: 各 [total_patches, 4096]
- **子模块**：
  - `patch_embed` (Conv3d, kernel=(3,16,16), stride=(2,16,16))
  - `pos_embed` (Embedding)
  - `rotary_pos_emb` (Vision-only RoPE)
  - `blocks` (27× VisionBlock, 包含 attention + MLP)
  - `merger` (PatchMerger: LN→Linear→GELU→Linear)
  - `deepstack_merger_list` (3× PatchMerger)

#### [新增] `prism_infer/models/qwen3_vl.py`
- **Qwen3VLForCausalLM** (最外层)
  - `model` (Qwen3VLModel) 包含 `.visual` + `.language_model`
  - `lm_head` (Linear 4096→151936)
  - `packed_modules_mapping` 需包含 vision 前缀映射
- **Qwen3VLModel**
  - `visual` = VisionEncoder
  - `language_model` = Qwen3VLTextModel
- **Qwen3VLTextModel** (36 layers)
  - `embed_tokens` (Embedding)
  - `layers` (36× Qwen3VLTextDecoderLayer)
  - `norm` (RMSNorm)
- **Qwen3VLTextDecoderLayer**
  - `input_layernorm`, `post_attention_layernorm` (RMSNorm)
  - `self_attn` (Qwen3VLTextAttention)
  - `mlp` (Qwen3VLTextMLP)
- **Qwen3VLTextAttention**
  - `q_proj`, `k_proj`, `v_proj` (不再合并 QKV)
  - `o_proj` (Linear)
  - `q_norm`, `k_norm` (RMSNorm, 可选)
  - 与现有 `Qwen3Attention` 的关键区别：**Q/K/V 是分开的线性层，不是合并的 QKVParallelLinear**
- **关键**：forward 需要处理 deepstack 注入 —— 在 layer 8/16/24 处将对应的 deepstack features 加到 hidden_states

#### [新增] `prism_infer/vision/mrope.py`
- M-RoPE: 支持 3D position encoding
- `mrope_section=[24, 20, 20]`：每个 head 的前 24 维用时间位置，中间 20 维用高度位置，后 20 维用宽度位置
- 视觉 token 的 position_ids shape: [total_tokens, 3] (T, H, W)
- 文本 token 的 position_ids shape: [total_tokens, 1] → 只填 T 维

#### [修改] `prism_infer/config.py`
- 新增字段：无需大改，`hf_config` 自动适配 Qwen3VLConfig

#### [修改] `prism_infer/engine/model_runner.py`
- **L25**: 导入路径改为 VL 模型
- **L58**: 根据 config 类型创建 VL 模型
- **`warmup_model()`**：VL 模型 warmup 需要使用文本-only 虚拟序列
- **`prepare_prefill()`**：
  - 新增 `pixel_values`, `image_grid_thw` 的处理
  - 视觉 token 的 position_ids 需要三维
  - 视觉 token 在 input_ids 中为占位符，实际 embedding 由 vision encoder 产生
  - M-RoPE 的 position 计算不同于 1D RoPE
- **`run_model()`**：forward 时传递 `pixel_values` 和 `image_grid_thw`
- **`capture_cudagraph()`**：暂不处理 VL path（decode 阶段无图像输入，且 VL 走 eager）

#### [修改] `prism_infer/utils/loader.py`
- `packed_modules_mapping` 需要包含 VL 模型的映射
- 新增 `visual.*` 权重加载逻辑
- 不再有 QKV 合并层（VL 使用分开的 q_proj/k_proj/v_proj）

---

### P1 — 第二阶段（后续）

- `prism_infer/engine/sequence.py` — 携带图像信息
- `prism_infer/engine/llm_engine.py` — `add_request` 接受图像参数
- 多图/多轮对话支持

### P2 — 第三阶段（后续）

- `prism_infer/layers/attention.py` — 视觉 token KV Cache 压缩策略
- `prism_infer/engine/block_manager.py` — 视觉 token 感知

### 可复用的模块

| 模块 | 原因 |
|------|------|
| `layers/linear.py` | 线性层不变 |
| `layers/layernorm.py` | RMSNorm 不变 |
| `layers/activation.py` | SiLU 不变 |
| `layers/sampler.py` | 采样逻辑不变 |
| `engine/scheduler.py` | 调度逻辑不变 |
| `engine/block_manager.py` | Block 管理不变 |
| `engine/sequence.py` | P0 阶段不改 |
| `utils/context.py` | Context 结构够用 |
| `sampling_params.py` | 不变 |
| `llm.py` | 透传不变 |

---

## 执行顺序（更新）

```
Day 2: vision_encoder.py    — ViT + PatchMerger (4个merger)
Day 3: mrope.py              — 3D RoPE
Day 4: qwen3_vl.py           — 完整模型组装 (含 deepstack 注入)
Day 5: model_runner.py 改造  — prepare_prefill 支持图像
Day 6: loader.py + config.py — 权重加载适配
Day 7: 端到端单图推理验证     — 周总结
```
