# AI Workflow and Skill Plan

## Purpose

Use AI skills to improve development quality, not to add runtime complexity.

The goal is to make future AI assistance:

- stricter.
- more test-oriented.
- less likely to overclaim.
- better at explaining infra tradeoffs.
- better at producing learning notes.

## Current Decision

Do not create formal Codex skills immediately.

For M0, use checklist-based workflow first:

- `roadmap/planning/06_ai_collaboration_rules.md`
- `roadmap/planning/09_acceptance_workflow.md`

After M0/M1, convert stable repeated checklists into formal project-specific skills.

## Candidate Formal Skills

### nanocache-task-acceptance

Purpose:

- Check whether a task is really complete.

Checks:

- milestone mapping.
- touched files.
- implementation scope.
- verification result.
- not-tested items.
- risk and next step.

Expected output:

```text
status: pass / partial / blocked
missing items:
required next action:
```

### nanocache-ai-auditor

Purpose:

- Detect AI laziness, overclaiming, unsupported assumptions, and hidden fallback.

Checks:

- Did the AI inspect relevant code?
- Did it separate measured result from hypothesis?
- Did it report tests not run?
- Did it change unrelated files?
- Did it claim performance without benchmark?
- Did it hide unsupported modes?

Expected output:

```text
audit result:
overclaims:
missing evidence:
required correction:
```

### nanocache-benchmark-auditor

Purpose:

- Check benchmark quality before accepting performance claims.

Checks:

- warmup.
- repeated runs.
- CUDA synchronization.
- GPU memory stats.
- median/p90/min/max.
- shape/config printout.
- measured claims only.

Expected output:

```text
benchmark status:
valid claims:
invalid claims:
missing measurement:
```

### nanocache-kv-cache-reviewer

Purpose:

- Review KV cache related code changes.

Checks:

- input/output shape.
- block layout.
- slot mapping.
- metadata lifecycle.
- prefix cache compatibility.
- CoW compatibility.
- swap compatibility.
- no silent fallback.

Expected output:

```text
review result:
shape/layout issues:
lifecycle issues:
fallback risks:
```

### nanocache-learning-coach

Purpose:

- Ensure technical learning is converted into project-specific notes.

Checks:

- topic relevance.
- key concepts.
- mapping to current code.
- common pitfalls.
- small experiments.
- references.

Expected output:

```text
learning note status:
missing sections:
suggested experiment:
```

### nanocache-paper-digester

Purpose:

- Convert papers into implementation plans.

Checks:

- main idea.
- assumptions.
- layout/shape.
- algorithm steps.
- implementation risks.
- benchmark plan.

Expected output:

```text
paper summary:
implementation mapping:
risks:
experiments:
```

## M0 Skill Usage

During M0, use these checklists manually:

- task acceptance: `09_acceptance_workflow.md`
- AI behavior audit: `06_ai_collaboration_rules.md`
- learning output: `learnning/README.md`

M0 is accepted only when:

- M0 tasks have pass/partial/blocked status.
- missing items are explicit.
- learning notes exist for profiling, inference, and Triton basics.

## Rule

These skills are development aids only. They should not be imported by `nanovllm` or required for inference.
