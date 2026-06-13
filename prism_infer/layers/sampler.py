import torch
from torch import nn


# ============================================================
# Sampler — 从 logits 采样下一个 token
# ============================================================
# 输入: logits [num_seqs, vocab_size] + temperatures [num_seqs]
# 输出: sample_tokens [num_seqs] (每个序列采样一个 token id)
#
# 采样策略: Gumbel-Max trick (等价于 multinomial 采样, 但更适合 GPU 并行)
class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.compile    # 融合整个采样流程为一个 kernel
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        # 1. Temperature scaling: logits / temperature
        #    temperature 越高 → 分布越平 (更随机)
        #    temperature 越低 → 分布越尖 (越确定)
        #    temperature = 0 时应该用 greedy (这里假设不为0)
        #    unsqueeze(dim=1): [num_seqs] → [num_seqs, 1] 广播到 [num_seqs, vocab]
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        # 2. Softmax: logits → 概率分布
        probs = torch.softmax(logits, dim=-1)
        # 3. Gumbel-Max trick 采样:
        #    等价于 torch.multinomial(probs, 1) 但更快
        #    原理: probs / Exp(1) 的 argmax 等价于按 probs 概率采样
        #    - exponential_(1): 原地生成指数分布随机数 (Exp(1))
        #    - clamp_min_(1e-10): 防止除零
        #    - probs / exp_random: 概率高的位置更可能是最大值
        #    - argmax: 取最大值的索引 = 采样结果
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
