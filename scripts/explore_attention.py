"""
Day 1: 加载模型 → 捕获 Attention Weights 和 KV Cache → 验证 Attention Sink 现象

关键问题:
  1. 每一层的 attention score 分布长什么样?
  2. Attention Sink (前几个 token 异常高的 attention score) 在哪几层最明显?
  3. 不同 head 之间的 attention pattern 有多大差异?
  4. KV Cache 在推理时的内存变化情况?

用法: python scripts/01_explore_attention.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer


# ═══════════════════════════════════════════════════════════════
# Hook 系统: 注册 forward hook 来捕获 attention 中间结果
# ═══════════════════════════════════════════════════════════════

class AttentionCapture:
    """在 attention 层注册 hook，捕获:
       - attention_weights: [batch, heads, seqlen_q, seqlen_k] softmax 后的权重
       - key_states: [batch, num_kv_heads, seqlen_k, head_dim] 写入 cache 前的 K
       - value_states: [batch, num_kv_heads, seqlen_k, head_dim] 写入 cache 前的 V
    """

    def __init__(self, model, layer_indices=None):
        self.model = model
        self.attention_weights = {}   # layer_idx → [heads, seqlen_q, seqlen_k]
        self.key_states = {}          # layer_idx → [kv_heads, seqlen, head_dim]
        self.value_states = {}        # layer_idx → [kv_heads, seqlen, head_dim]
        self.hooks = []

        # 找到所有 attention 层
        for i, layer in enumerate(self._get_layers()):
            if layer_indices is not None and i not in layer_indices:
                continue
            hook = layer.self_attn.register_forward_hook(
                self._make_hook(i), with_kwargs=True
            )
            self.hooks.append(hook)

    def _get_layers(self):
        """获取模型的 decoder layers"""
        if hasattr(self.model, 'model'):
            model_body = self.model.model  # Qwen2/CausalLM 模式
        else:
            model_body = self.model
        return model_body.layers

    def _make_hook(self, layer_idx):
        def hook_fn(module, args, kwargs, output):
            # capture attention weights from the attention module
            # output 是 attention 层的输出 hidden_states
            # 但我们更需要的是 attention weights
            # 它们通常在 attention 计算内部，需要用 output_attentions=True 或者更底层的 hook
            pass
        return hook_fn

    def remove(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


class KVBuffer:
    """简易 KV Cache 缓冲区 — 模拟 PagedAttention 中的 KV Cache

    与 nano-vllm block_manager 不同的是，这里用连续张量存储(不做分块)，
    因为 Day 1 的目标只是观察 KV cache 的行为模式，不涉及调度。
    """

    def __init__(self, max_length: int, num_layers: int, num_kv_heads: int, head_dim: int, dtype=torch.bfloat16):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype

        # [2, num_layers, max_length, num_kv_heads, head_dim]
        # 2 = K and V
        self.cache = torch.zeros(2, num_layers, max_length, num_kv_heads, head_dim, dtype=dtype, device='cuda')
        self.current_len = 0

    def append(self, key: torch.Tensor, value: torch.Tensor, layer_idx: int):
        """追加一个 token 的 K/V 到 cache 中"""
        seqlen = key.shape[-2]  # prefill 时可能 > 1
        end = self.current_len + seqlen
        self.cache[0, layer_idx, self.current_len:end] = key.squeeze(0)  # K
        self.cache[1, layer_idx, self.current_len:end] = value.squeeze(0)  # V
        if layer_idx == self.num_layers - 1:
            self.current_len = end


# ═══════════════════════════════════════════════════════════════
# 主脚本
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Qwen/Qwen3-0.6B', help='模型名称或路径')
    parser.add_argument('--prompt', default="今天天气很好，我和朋友一起去公园散步，看到很多美丽的花朵。", help='输入文本')
    parser.add_argument('--output-dir', default='data/explore_01', help='输出目录')
    parser.add_argument('--dtype', default='bfloat16', help='模型精度')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = 'cuda'
    dtype = getattr(torch, args.dtype)

    print(f"=== Day 1: Attention Exploration ===")
    print(f"Model: {args.model}")
    print(f"Prompt: {args.prompt}")
    print()

    # ── 1. 加载模型和分词器 ──
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map='auto',
        trust_remote_code=True,
        attn_implementation='eager',     # ★ 必须 eager: flash_attn/sdpa 不会返回 attention weights
    )
    model.eval()

    # 模型结构信息
    hf_config = model.config
    num_layers = hf_config.num_hidden_layers
    num_heads = hf_config.num_attention_heads
    num_kv_heads = getattr(hf_config, 'num_key_value_heads', num_heads)
    head_dim = getattr(hf_config, 'head_dim', hf_config.hidden_size // num_heads)
    hidden_size = hf_config.hidden_size

    print(f"  Layers: {num_layers}, Q-Heads: {num_heads}, KV-Heads: {num_kv_heads}, Head-Dim: {head_dim}")
    print(f"  Hidden: {hidden_size}, Max-Pos-Emb: {hf_config.max_position_embeddings}")
    print()

    # ── 2. Tokenize + 前向推理 ──
    inputs = tokenizer(args.prompt, return_tensors='pt').to(device)
    seqlen = inputs.input_ids.shape[1]
    print(f"Input sequence length: {seqlen} tokens")
    print(f"Tokens: {inputs.input_ids[0].tolist()}")

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # outputs.attentions: tuple of [num_layers]
    #   每层: [batch, num_heads, seqlen, seqlen]  ← softmax 之后的 attention weights

    attentions = outputs.attentions
    assert attentions is not None, "output_attentions=True 但没有拿到 attention weights"

    print(f"\nCaptured attention weights for {len(attentions)} layers")
    print(f"Shape per layer: {attentions[0].shape}  (batch, heads, seqlen_q, seqlen_k)")

    # ── 3. 分析 Attention Sink 现象 ──
    # 取第 0 个 sample, 按所有 head 平均, 看每个查询位置对每个键位置的关注度

    # 按层和 head 记录统计量
    layer_sink_scores = []  # (layer_idx, sink_token_avg_attention)
    head_sink_matrix = np.zeros((num_layers, num_heads))  # 每层每 head 的 sink score

    for l, attn in enumerate(attentions):
        # attn: [1, num_heads, seqlen, seqlen]
        attn = attn[0]  # [heads, seqlen, seqlen]

        # 对每个 head, 计算所有 query 对前 2 个 token (sink tokens) 的平均 attention
        sink_attention = attn[:, :, :2].mean(dim=[1, 2])  # [heads]
        # 解释: attn[:, :, :2] = 所有 query 位置对第 0,1 个 key 位置的 attention
        #       .mean(dim=1) = 对所有 query 位置取均值
        #       .mean(dim=1) = 对 2 个 sink token 取均值

        # 同时也计算对所有其他 token 的平均 attention
        if seqlen > 2:
            non_sink_attention = attn[:, :, 2:].mean(dim=[1, 2])  # [heads]
        else:
            non_sink_attention = torch.zeros_like(sink_attention)

        head_sink_matrix[l] = sink_attention.float().float().cpu().numpy()
        layer_sink_scores.append((l, sink_attention.mean().item(), non_sink_attention.mean().item()))

    # ── 4. 打印分析结果 ──
    print("\n=== Layer-wise Attention Sink Analysis ===")
    print(f"{'Layer':>5}  {'Sink Attn':>12}  {'Non-Sink Attn':>14}  {'Ratio':>8}")
    print("-" * 45)
    for l, sink, non_sink in layer_sink_scores:
        ratio = sink / non_sink if non_sink > 0 else float('inf')
        print(f"{l:5d}  {sink:12.6f}  {non_sink:14.6f}  {ratio:8.2f}")

    # ── 5. 可视化 ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 5a. 最浅层的 attention pattern
    ax = axes[0, 0]
    first_layer_attn = attentions[0][0].mean(dim=0).float().cpu().numpy()  # 所有 head 平均
    im = ax.imshow(first_layer_attn, cmap='RdYlBu_r', aspect='auto')
    ax.set_title(f'Layer 0 Average Attention Pattern (seqlen={seqlen})')
    ax.set_xlabel('Key Position')
    ax.set_ylabel('Query Position')
    ax.axvline(x=0, color='red', alpha=0.5, linewidth=3, label='Sink tokens (pos 0-1)')
    ax.axvline(x=1, color='red', alpha=0.5, linewidth=3)
    ax.legend()
    plt.colorbar(im, ax=ax)

    # 5b. 最深层的 attention pattern
    ax = axes[0, 1]
    last_layer_attn = attentions[-1][0].mean(dim=0).float().cpu().numpy()
    im = ax.imshow(last_layer_attn, cmap='RdYlBu_r', aspect='auto')
    ax.set_title(f'Layer {num_layers-1} Average Attention Pattern')
    ax.set_xlabel('Key Position')
    ax.set_ylabel('Query Position')
    plt.colorbar(im, ax=ax)

    # 5c. Attention Sink 按层变化
    ax = axes[1, 0]
    layers = [x[0] for x in layer_sink_scores]
    sinks = [x[1] for x in layer_sink_scores]
    non_sinks = [x[2] for x in layer_sink_scores]
    ax.plot(layers, sinks, 'o-', label='Sink tokens (pos 0-1)', linewidth=2)
    ax.plot(layers, non_sinks, 's-', label='Non-sink tokens (pos 2+)', linewidth=2)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Average Attention Weight')
    ax.set_title('Attention Sink vs Layer Depth')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5d. Head-level differences in the first layer
    ax = axes[1, 1]
    first_layer_heads = head_sink_matrix[0]
    ax.bar(range(len(first_layer_heads)), first_layer_heads, alpha=0.7)
    ax.axhline(y=first_layer_heads.mean(), color='red', linestyle='--', label=f'Mean ({first_layer_heads.mean():.4f})')
    ax.set_xlabel('Head Index')
    ax.set_ylabel('Sink Attention')
    ax.set_title(f'Layer 0: Sink Attention per Head')
    ax.legend()

    plt.tight_layout()
    fig.savefig(output_dir / 'attention_analysis.png', dpi=150)
    print(f"\nSaved visualization to {output_dir / 'attention_analysis.png'}")

    # ── 6. 保存数据（供后续分析） ──
    np.savez(
        output_dir / 'attention_data.npz',
        attention_weights=np.stack([a[0].float().cpu().numpy() for a in attentions]),
        head_sink_matrix=head_sink_matrix,
        layer_sink_scores=np.array(layer_sink_scores),
        seqlen=seqlen,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        prompt=args.prompt,
    )
    print(f"Saved attention data to {output_dir / 'attention_data.npz'}")
    print("\n=== Day 1 完成 ===")
    print("结论要点:")
    print("  1. 看 Layer-wise 表格中 Sink/Non-Sink ratio 是否 > 1")
    print("  2. 看第一层 attention 热力图的前两列是否特别亮")
    print("  3. 不同 head 的 sink score 差异大吗?")


if __name__ == '__main__':
    main()
