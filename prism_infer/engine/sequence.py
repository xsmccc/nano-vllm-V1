from copy import copy              # copy模块: 浅拷贝(复制列表本身, 不复制元素)
from enum import Enum, auto        # Enum: 枚举类(类似C++ enum class); auto: 自动分配值
from itertools import count        # count(): 无限自增迭代器(0,1,2,3,...)

from prism_infer.sampling_params import SamplingParams


# 序列状态枚举: WAITING→RUNNING→FINISHED 三阶段生命周期
class SequenceStatus(Enum):
    WAITING = auto()    # 等待prefill(在scheduler.waiting队列中)
    RUNNING = auto()    # 正在decode(在scheduler.running队列中)
    FINISHED = auto()   # 生成结束(从队列移除)
    SWAPPED = auto()    # KV Cache已换出到CPU内存(等待换回)


class Sequence:
    block_size = 256    # 类变量: KV Cache块大小(所有实例共享)
    counter = count()   # 类变量: 全局自增ID计数器(类似C++ static atomic<int>)

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)          # 全局唯一ID: 0, 1, 2, ...
        self.status = SequenceStatus.WAITING           # 初始状态=等待prefill
        self.token_ids = copy(token_ids)               # 浅拷贝prompt的token列表(值语义, 类似C++ vector拷贝)
        self.last_token = token_ids[-1]                # 最后一个token(序列化优化用)
        self.num_tokens = len(self.token_ids)           # 当前总token数(prompt+生成)
        self.num_prompt_tokens = len(token_ids)         # prompt token数(固定不变)
        self.num_cached_tokens = 0                      # 已在KV Cache中缓存的token数 (Prefix Cache)
        self.num_computed_tokens = 0                      # 已Prefill计算的token数 (Chunked Prefill)
        self.block_table = []                           # 物理块映射表: [block_id_0, block_id_1, ...]
        # 从SamplingParams展开存储(避免序列化时携带整个SamplingParams对象)
        self.temperature = sampling_params.temperature  # 采样温度
        self.max_tokens = sampling_params.max_tokens    # 最大生成token数
        self.ignore_eos = sampling_params.ignore_eos    # 是否忽略EOS

    def __len__(self):           # len(seq) → 总token数
        return self.num_tokens

    def __getitem__(self, key):  # seq[0], seq[5:10] → 访问token_ids
        return self.token_ids[key]

    @property
    def is_finished(self):       # seq.is_finished → bool
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):  # 已生成的token数 = 总数 - prompt数
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):      # 切片取prompt部分: token_ids[:num_prompt]
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):  # 切片取生成部分: token_ids[num_prompt:]
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self):     # 已缓存的完整块数(整除)
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):            # 总共需要的块数(向上取整: (n+BS-1)//BS)
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):  # 最后一个块中的token数(可能不满)
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):              # 取第i个块对应的token子列表(用于hash匹配KV复用)
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    @property
    def is_prefill_finished(self) -> bool:
        """是否已完成所有 Prefill (Chunked Prefill 用)"""
        return self.num_computed_tokens >= self.num_prompt_tokens

    @property
    def remaining_prefill_tokens(self) -> int:
        """还需要 Prefill 多少 token"""
        return max(0, self.num_prompt_tokens - self.num_computed_tokens)

    def append_token(self, token_id: int):  # 追加新生成的token
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    # === 跨进程序列化优化 ===
    # Python pickle序列化对象时自动调用这两个方法
    # 目的: 减少主进程→子进程的数据传输量

    def __getstate__(self):
        """序列化: 决定发送什么数据给子进程
        - Prefill(未生成): 发完整token_ids列表(子进程需要所有prompt做计算)
        - Decode(已生成): 只发last_token(1个int, 子进程已有KV Cache)
        """
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
                self.token_ids if self.num_completion_tokens == 0 else self.last_token)

    def __setstate__(self, state):
        """反序列化: 子进程收到数据后恢复对象
        state就是__getstate__返回的那个元组
        state[:-1] = 前4个值, state[-1] = token_ids(list)或last_token(int)
        """
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1]
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]   # Prefill: state[-1]是完整列表
        else:
            self.last_token = state[-1]  # Decode: state[-1]是一个int
