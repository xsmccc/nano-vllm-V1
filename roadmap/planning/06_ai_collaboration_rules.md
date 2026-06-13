# AI Collaboration Rules

This file defines how AI assistants should help build NanoCache-V.

The goal is to make AI useful without allowing it to fake progress, skip verification, or create low-quality code.

## Core Behavior

The AI assistant must:

1. inspect relevant code before proposing implementation.
2. state assumptions explicitly.
3. prefer small changes.
4. avoid unrelated refactors.
5. explain unsupported modes.
6. refuse to invent benchmark data.
7. distinguish measured results from hypotheses.
8. keep code paths explicit.
9. add tests for behavior changes.
10. report commands run and tests not run.

## Forbidden Behavior

The AI assistant must not:

1. silently fallback from quantized KV to FP KV.
2. claim speedup without benchmark evidence.
3. claim correctness without tests or clear manual validation.
4. edit many unrelated modules in one step.
5. hide TODOs inside vague comments.
6. copy paper claims as project results.
7. implement demo-only code and call it a system.
8. remove user changes without permission.

## Required Explanation For Each Implementation

Every non-trivial implementation should document:

- input shape.
- output shape.
- memory layout.
- dtype.
- metadata layout.
- error source.
- performance bottleneck.
- unsupported cases.

## Required Verification Language

Use precise wording:

- "Tested with ..."
- "Not tested because ..."
- "Expected to ..."
- "Measured ..."
- "Hypothesis ..."

Avoid vague wording:

- "should be fine"
- "probably works"
- "optimized"
- "efficient"
- "fast"

## Development Checklist

Before implementation:

- identify touched files.
- identify behavior being changed.
- identify tests needed.

During implementation:

- keep edits scoped.
- do not introduce silent fallback.
- preserve FP baseline behavior unless intentionally changing it.

After implementation:

- run relevant tests or explain why not.
- summarize changed files.
- document risk and next step.

## Benchmark Checklist

Every benchmark must include:

- warmup.
- repeat count.
- `torch.cuda.synchronize()`.
- memory stats.
- shape/config printout.
- median/p90/min/max.

## Skill Usage Plan

We may later create project-specific AI skills for:

- KV cache review.
- benchmark audit.
- Triton learning.
- paper digestion.
- interview story packaging.

These skills are for development workflow quality. They should not become runtime dependencies of NanoCache-V.
