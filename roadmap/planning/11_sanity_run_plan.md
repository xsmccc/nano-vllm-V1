# Baseline Sanity Run 计划

## 目标

建立 NanoCache-V 的最小 baseline sanity run 流程。

Sanity run 的目的不是性能测试，而是确认：

- Python 环境可用。
- PyTorch CUDA 可用。
- Triton 可用。
- FlashAttention 可用。
- Transformers/tokenizer 可用。
- 本地模型路径可用。
- nano-vLLM 能完成一次最小 generate。

## 当前环境检查结果

检查时间：M0-T5。

命令：

```text
which python3
python3 --version
```

结果：

```text
/usr/bin/python3
Python 3.12.3
```

命令：

```text
python3 - <<'PY'
import importlib.util
mods = ['torch', 'triton', 'transformers', 'flash_attn']
for m in mods:
    spec = importlib.util.find_spec(m)
    print(f'{m}: {"found" if spec else "missing"}')
PY
```

结果：

```text
torch: missing
triton: missing
transformers: missing
flash_attn: missing
```

命令：

```text
python3 - <<'PY'
import torch
PY
```

结果：

```text
ModuleNotFoundError("No module named 'torch'")
```

本地模型搜索：

```text
find /home/xsmccc -maxdepth 3 -type d -iname '*Qwen3-4B-Instruct-2507*'
```

结果：

```text
未找到 Qwen3-4B-Instruct-2507 本地模型目录
```

当前结论：

```text
sanity run status: blocked
blocker: 当前 Python 环境缺少 torch/triton/transformers/flash_attn，且本地没有 Qwen3-4B-Instruct-2507 模型目录
```

## 共享虚拟环境检查结果

更新记录：已在 WSL 中找到一个已有虚拟环境，并迁移到共享位置。

原始位置：

```text
/home/xsmccc/SGlang/.venv-sglang-debug
```

共享位置：

```text
/home/xsmccc/.venvs/sglang-debug-py312
```

兼容软链接：

```text
/home/xsmccc/SGlang/.venv-sglang-debug -> /home/xsmccc/.venvs/sglang-debug-py312
/home/xsmccc/nano-vllm/.venv -> /home/xsmccc/.venvs/sglang-debug-py312
```

激活方式：

```bash
source /home/xsmccc/.venvs/sglang-debug-py312/bin/activate
```

或在 nano-vLLM 外层目录中：

```bash
source /home/xsmccc/nano-vllm/.venv/bin/activate
```

当前共享环境检查：

```text
python: 3.12.3
torch: 2.9.1+cu128
cuda_available: True
cuda_device_count: 1
device_0: NVIDIA GeForce RTX 4070 Laptop GPU
triton: found
transformers: found
sglang: found
flash_attn: missing
```

当前结论：

```text
shared venv status: partial
remaining blocker: flash_attn missing, local Qwen model missing
```

注意：

```text
该 venv 原本脚本 shebang 指向旧路径 /home/xsmccc/VLLM/.venv-sglang-debug。
迁移后已将 bin scripts 和 pyvenv.cfg 中的旧路径替换为 /home/xsmccc/.venvs/sglang-debug-py312。
```

## 环境准备要求

建议使用独立虚拟环境，避免污染系统 Python。

示例：

```bash
cd /home/xsmccc/nano-vllm/nano-vllm-V1
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

依赖安装需要根据目标机器 CUDA 版本选择。

项目依赖来自 `pyproject.toml`：

```text
torch>=2.4.0
triton>=3.0.0
transformers>=4.51.0
flash-attn
xxhash
```

安装后检查：

```bash
python - <<'PY'
import torch
import triton
import transformers
import flash_attn

print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
print("triton ok")
print("transformers", transformers.__version__)
print("flash_attn ok")
PY
```

验收：

```text
torch.cuda.is_available() == True
torch.cuda.device_count() >= 1
triton import 成功
flash_attn import 成功
transformers import 成功
```

## 模型准备要求

主线模型：

```text
Qwen3-4B-Instruct-2507
```

推荐路径：

```text
/home/xsmccc/huggingface/Qwen3-4B-Instruct-2507
```

模型目录至少需要包含：

```text
config.json
tokenizer.json or tokenizer.model
model weights
```

检查命令：

```bash
python - <<'PY'
from transformers import AutoConfig, AutoTokenizer

model = "/home/xsmccc/huggingface/Qwen3-4B-Instruct-2507"
cfg = AutoConfig.from_pretrained(model)
tok = AutoTokenizer.from_pretrained(model, use_fast=True)

print("model_type", cfg.model_type)
print("hidden_size", cfg.hidden_size)
print("num_hidden_layers", cfg.num_hidden_layers)
print("num_attention_heads", cfg.num_attention_heads)
print("num_key_value_heads", cfg.num_key_value_heads)
print("max_position_embeddings", cfg.max_position_embeddings)
print("torch_dtype", cfg.torch_dtype)
print("eos_token_id", tok.eos_token_id)
PY
```

## Sanity Run 1: 文本 Prompt

目标：

- 验证 tokenizer -> Sequence -> scheduler -> model -> sampler 全链路。

示例脚本：

```python
from nanovllm import LLM, SamplingParams

model = "/home/xsmccc/huggingface/Qwen3-4B-Instruct-2507"
llm = LLM(model, enforce_eager=True, max_model_len=1024)
outputs = llm.generate(
    ["Hello, NanoCache-V."],
    SamplingParams(temperature=0.0, max_tokens=16, ignore_eos=True),
    use_tqdm=False,
)
print(outputs[0]["text"])
print(outputs[0]["token_ids"])
```

验收：

- 不报 CUDA/Triton/FlashAttention/import error。
- 能输出 text。
- 能输出 token_ids。

## Sanity Run 2: Token ID Prompt

目标：

- 绕过 tokenizer，验证 token id input 路径。

示例：

```python
from nanovllm import LLM, SamplingParams

model = "/home/xsmccc/huggingface/Qwen3-4B-Instruct-2507"
llm = LLM(model, enforce_eager=True, max_model_len=1024)
prompt_token_ids = [[1, 2, 3, 4, 5]]
outputs = llm.generate(
    prompt_token_ids,
    SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True),
    use_tqdm=False,
)
print(outputs[0]["token_ids"])
```

验收：

- 能完成 generate。
- 不要求文本语义合理。
- 用于验证 list[int] prompt path。

## Sanity Run 3: CUDA Graph 关闭/开启对照

目标：

- 确认 eager path 可用。
- 后续再确认 CUDA Graph path 可用。

第一阶段：

```text
enforce_eager=True
```

第二阶段：

```text
enforce_eager=False
```

验收：

- M0 至少 eager path 成功。
- CUDA Graph 如果失败，记录 blocker，不阻塞 profiler 设计。

## 常见失败排查

### Python import error

可能原因：

- 没有激活虚拟环境。
- 依赖未安装。

### torch.cuda.is_available() == False

可能原因：

- 当前机器没有 GPU。
- CUDA driver 不可用。
- PyTorch 安装版本不匹配。

### flash_attn import/build error

可能原因：

- CUDA/PyTorch/flash-attn 版本不匹配。
- 编译环境缺失。

### model path assert failed

可能原因：

- `Config.__post_init__` 要求模型路径是本地目录。
- 模型没有下载到指定路径。

### OOM

可能原因：

- `max_model_len` 太大。
- `gpu_memory_utilization` 太高。
- 模型过大。

M0 sanity 建议：

```text
max_model_len=1024
max_num_seqs=8
max_num_batched_tokens=2048
enforce_eager=True
```

## M0-T5 状态

```text
plan: pass
execution: blocked
```

blocker：

```text
当前环境缺少 torch/triton/transformers/flash_attn，且没有本地 Qwen3-4B-Instruct-2507 模型目录。
```

下一步：

```text
完成环境准备后，执行 Sanity Run 1 和 Sanity Run 2。
```
