from dataclasses import dataclass   # @dataclass: 自动生成__init__


@dataclass
class SamplingParams:
    """请求级采样参数: 每条请求可以不同"""
    temperature: float = 1.0        # 采样温度: logits/temperature后softmax, 越大越随机, 越小越确定
    max_tokens: int = 64            # 最多生成多少个token
    ignore_eos: bool = False        # True=遇到EOS也不停, 继续生成到max_tokens(benchmark用)

    def __post_init__(self):
        # 禁止temperature≈0(贪心采样需要argmax, nano-vllm简化省去了)
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
