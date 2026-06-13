# __init__.py: Python包的"门面"文件
# 当用户写 from prism_infer import LLM, SamplingParams 时, Python执行此文件
# 作用: 把内部模块的类"提升"到包顶层, 用户不需要知道具体在哪个文件里
from prism_infer.llm import LLM
from prism_infer.sampling_params import SamplingParams
