import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from prism_infer.utils.context import get_context


# ============================================================
# store_kvcache_kernel — Triton Kernel: 将当前 K/V 写入 KV Cache
# ============================================================
# 每个 program (线程块) 处理一个 token
# slot_mapping 告诉每个 token 应该写到 cache 的哪个 slot
#
# C++ 类比: 这就是一个 CUDA kernel, 但用 Python 写的 (Triton DSL)
# grid = (N,) → N 个线程块, 每个处理一个 token
@triton.jit
def store_kvcache_kernel(
    key_ptr,            # key 张量的指针 (起始地址)
    key_stride,         # key 每行的步长 (= num_heads * head_dim = D)
    value_ptr,          # value 张量的指针
    value_stride,       # value 每行的步长
    k_cache_ptr,        # KV Cache 中 key 部分的指针
    v_cache_ptr,        # KV Cache 中 value 部分的指针
    slot_mapping_ptr,   # slot_mapping 的指针: 每个 token 对应的 cache slot
    D: tl.constexpr,    # num_heads * head_dim (编译时常量)
):
    # tl.program_id(0) → 当前线程块编号 (处理第 idx 个 token)
    # C++ 类比: blockIdx.x
    idx = tl.program_id(0)
    # 读取当前 token 应该写到 cache 的哪个 slot
    slot = tl.load(slot_mapping_ptr + idx)
    # slot == -1 表示 padding token, 跳过
    if slot == -1: return
    # 计算 key/value 在内存中的偏移
    # key 形状 [N, num_heads, head_dim], 连续存储 → 每行 D 个元素
    # tl.arange(0, D) → [0, 1, 2, ..., D-1]  (C++ 类比: iota)
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    # 从 key/value 张量读取当前 token 的数据
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    # 计算 cache 中目标位置的偏移
    # cache 形状 [num_blocks * block_size, D], 每个 slot 占 D 个元素
    cache_offsets = slot * D + tl.arange(0, D)
    # 写入 cache
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    """store_kvcache_kernel 的 Python 入口: 参数校验 + 启动 kernel"""
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim  # 每个 token 的 K/V 数据量
    # 内存布局检查: 最后一维 stride 必须为 1 (连续), head 维 stride = head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    # cache 的 stride(1) = D, 即每个 slot 的数据量
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N  # 每个 token 一个 slot
    # 启动 kernel: grid=(N,) → N 个线程块并行处理 N 个 token
    # C++ 类比: kernel<<<N, 1>>>(args...)
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


# ============================================================
# Attention — 注意力层 (调用 FlashAttention)
# ============================================================
# 不自己实现注意力计算, 而是调用 flash_attn 库的高效实现
# Prefill 用 flash_attn_varlen_func (变长序列拼接)
# Decode 用 flash_attn_with_kvcache (单 token 查 KV Cache)
class Attention(nn.Module):

    def __init__(
        self,
        num_heads,      # Q 的 head 数 (如 28)
        head_dim,       # 每个 head 维度 (如 128)
        scale,          # softmax 缩放因子 (通常 1/sqrt(head_dim))
        num_kv_heads,   # KV 的 head 数 (如 4, GQA 时 < num_heads)
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        # KV Cache 初始化为空张量, model_runner 会在 init_kv_cache 中分配
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        q: [N, num_heads, head_dim]
        k: [N, num_kv_heads, head_dim]
        v: [N, num_kv_heads, head_dim]
        """
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        # 如果 KV Cache 已分配 (非空), 先把当前 K/V 写入 cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache 命中
                # 有 prefix cache 时, 用 cache 中的完整 K/V (包含 prefix 部分)
                k, v = k_cache, v_cache
            # flash_attn_varlen_func: 处理变长序列的 FlashAttention
            # cu_seqlens_q/k: 累计序列长度 (告诉 FA 哪些 token 属于哪个序列)
            # causal=True: 因果掩码 (token 只能看到之前的 token)
            # block_table: 如有 prefix cache, 告诉 FA 怎么访问分块的 KV Cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode 阶段
            # flash_attn_with_kvcache: 单 token 查询 KV Cache
            # q.unsqueeze(1): [N, num_heads, D] → [N, 1, num_heads, D]
            #   FA 要求 q 有 seqlen 维度, decode 时 seqlen=1
            # cache_seqlens: 每个序列的实际长度 (决定查多少历史 KV)
            # block_table: PagedAttention 的块映射表
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
        return o
