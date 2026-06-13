# 任务验收流程

## 目标

这份文档定义 NanoCache-V 每个小任务的验收流程。

它解决三个问题：

1. 任务是否真的完成。
2. AI 是否偷懒、过度声称或跳过验证。
3. 学习产出是否沉淀为可复用知识。

## 总体原则

每个任务都必须通过五个 gate：

1. Planning Gate：任务边界清楚。
2. Implementation Gate：实现范围受控。
3. Verification Gate：测试或验证明确。
4. Learning Gate：相关知识有输出。
5. AI Audit Gate：AI 输出可信，没有过度声称。

如果某个 gate 不能通过，任务不能标记为 done，只能标记为 blocked 或 partial。

## Gate 1: Planning Gate

任务开始前必须回答：

- 这个任务属于哪个 milestone。
- 本次只改什么。
- 本次不改什么。
- 需要读哪些文件。
- 预期产出是什么。
- 验收标准是什么。
- 有哪些风险。

通过标准：

- 任务边界可以用一句话描述。
- touched files 可以提前列出。
- 不包含多个互相独立的大目标。

失败信号：

- “顺手把某某也改了”。
- 同时改 benchmark、runtime、kernel、文档。
- 没有明确 done definition。

## Gate 2: Implementation Gate

实现时必须满足：

- 改动范围符合 planning。
- 不修改无关文件。
- 不引入 silent fallback。
- 不隐藏 unsupported path。
- 不把 TODO 当成实现。
- 不用大范围重构掩盖功能实现。

通过标准：

- diff 可以解释为一个清晰行为变化。
- runtime 改动和文档改动边界清楚。
- 每个新配置都有默认值和错误处理。

失败信号：

- 用户请求 A，代码顺手重构 B/C/D。
- quantized mode 不支持时自动走 FP。
- benchmark 里缺少 synchronize 但仍报告性能结论。

## Gate 3: Verification Gate

每个任务完成后必须说明：

- 跑了哪些测试。
- 测试命令是什么。
- 测试结果是什么。
- 没跑哪些测试。
- 为什么没跑。
- 是否需要后续补测。

不同任务的最低验证要求：

| 任务类型 | 最低验证 |
|---|---|
| 纯文档 | 文件存在、内容可读、链接路径正确 |
| benchmark | warmup、repeat、synchronize、memory stats、shape 输出 |
| profiler | 不改变模型输出，能输出结构化结果 |
| backend 抽象 | FP baseline parity test |
| quant/dequant | roundtrip error test |
| attention path | output/logits diff test |
| kernel | reference 对比 + benchmark |

通过标准：

- 测试结果和任务目标相关。
- 未测试项被明确记录。
- 没有把“能运行”说成“正确”。

失败信号：

- 只说“应该没问题”。
- 没有命令、没有输出、没有失败记录。
- 没测性能却声称优化。

## Gate 4: Learning Gate

每个阶段至少要沉淀对应学习笔记。

M0 最低学习输出：

- `learnning/profiling_benchmark/`: PyTorch CUDA timing 与 memory stats。
- `learnning/inference/`: prefill/decode 与 paged KV cache。
- `learnning/triton/`: 当前 `store_kvcache_kernel` 解析。

后续阶段最低学习输出：

| 阶段 | 学习输出 |
|---|---|
| M1 Profiler | CUDA event、torch memory stats、profiling overhead |
| M2 Backend | KV layout、backend interface、FlashAttention cache interface |
| M3 Quant | FP8/INT8/INT4 quantization basics、scale/zero metadata |
| M4 KIVI | KIVI paper note、K/V error sensitivity |
| M5 Hybrid | block lifecycle、recent/old policy、CoW interaction |
| M6 Visual Prefix KV | VLM processor、visual token、visual prefix KV lifecycle |
| M7 Compressed KV | MLA/latent KV note、compressed KV shape microbench |
| M8 Kernel | Triton/CUDA kernel notes、memory traffic analysis |

通过标准：

- 学习笔记能解释“为什么这个知识对当前实现有用”。
- 学习笔记能指向当前仓库的相关代码。
- 不是复制粘贴教程，而是结合项目消化。

失败信号：

- 只有链接，没有总结。
- 只有概念，没有代码映射。
- 学习和当前任务无关。

## Gate 5: AI Audit Gate

每次 AI 辅助完成任务后，需要检查：

- AI 是否先读了相关代码或文档。
- AI 是否明确说明假设。
- AI 是否区分 measured result 和 hypothesis。
- AI 是否报告未运行的测试。
- AI 是否改了无关文件。
- AI 是否声称了没有证据支持的性能结果。
- AI 是否隐藏 unsupported mode。
- AI 是否把 roadmap 写成已完成成果。

通过标准：

- final response 中有 changed files、verification、limitations。
- 没有未经验证的性能或正确性结论。
- 对 blocker 和未完成项表述清楚。

失败信号：

- “已经优化完成”但没有 benchmark。
- “支持 INT4”但没有 no-silent-fallback test。
- “完全兼容”但没有列出模型 config 差异。
- “测试通过”但没有命令或结果。

## Task Done 模板

每个任务结束时，建议用这个格式记录：

```text
Task:

Milestone:

Changed files:

What changed:

What did not change:

Verification:

Not tested:

Risks:

Learning notes:

Next step:
```

## Blocked / Partial 模板

如果任务没有完成，必须记录：

```text
Task:

Status: blocked / partial

What is complete:

What is missing:

Blocker:

Evidence:

Next action:
```

## 与 Skill 的关系

当前阶段先使用文档化 checklist，不立即创建正式 Codex skills。

原因：

- 项目规则还在演化，过早固化 skill 容易变僵。
- 先用 checklist 跑几个任务，观察哪些检查最常用。
- 等 M0/M1 稳定后，再把高频 checklist 提炼成正式 skills。

后续建议创建的正式 skills：

- `nanocache-task-acceptance`
- `nanocache-ai-auditor`
- `nanocache-benchmark-auditor`
- `nanocache-kv-cache-reviewer`
- `nanocache-learning-coach`

## 当前 M0 的验收状态

当前 M0 的验收流程还没有完全完成。

已具备：

- milestone 文档。
- M0 execution plan。
- AI collaboration rules。
- 初始 skill plan。
- 当前代码主链路说明。

仍需补齐：

- M0-T1 仓库状态审计记录。
- M0-T3 KV cache lifecycle 详细文档。
- M0-T4 baseline benchmark harness 设计。
- M0-T5 sanity run 计划。
- M0-T6 学习笔记首批输出。

因此当前阶段状态应标记为：

```text
M0 status: in progress
```
