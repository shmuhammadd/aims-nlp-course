# Lecture 08 — Reasoning, test-time compute, and tool-using systems

[Course home](../../README.md) · Week 2, day 3 · 120 minutes

## Learning outcomes

- Distinguish answer verification from explanation plausibility.
- Implement bounded tool dispatch with schema validation.
- Compare sampling and verification under equal budgets.

## Session plan

| Minutes | Activity |
|---|---|
| 0–10 | Retrieval quiz: recall the previous session; day 1 uses the prerequisite diagnostic |
| 10–35 | Concepts and concrete examples from the first half of these notes |
| 35–60 | Board derivation, worked example, and failure analysis |
| 60–65 | Break |
| 65–105 | Guided [notebook tutorial](../tutorials/08_reasoning.ipynb) |
| 105–115 | Practical questions in pairs, then discuss evidence |
| 115–120 | Exit ticket: one result, one limitation, one next experiment |

## Teaching notes

### More inference can change the result
A model can spend extra computation by generating longer responses, sampling several candidates, revising, searching, or calling tools. These have different costs and failure modes. Compare them under a declared token, time, or monetary budget. Greedy accuracy, majority-vote accuracy, and pass@k answer different questions. Pass@k assumes an oracle recognizes success among candidates and is not the same as a deployable selector.

A verbal chain of thought can be useful for solving a problem but is not guaranteed to be a faithful account of internal computation. Score final answers and checkable intermediate objects such as equations or tool outputs. A polished explanation can rationalize a wrong answer. Do not reward length alone.

### Verification changes the pipeline
For deterministic arithmetic, use a calculator. For code, tests can verify some properties but may be incomplete. A verifier has false positives and negatives; evaluate it separately. Self-consistency votes over sampled answers, so correlated mistakes and parsing errors can dominate. Best-of-n selects using a scoring function that may itself be biased.

### A bounded agent loop
Represent a state containing the user task, evidence, proposed action, tool result, and remaining budget. A controller proposes an action; a dispatcher validates the schema and permitted tool name; execution produces an observation; the controller decides whether to continue. Cap the number of steps and return a clear failure if the budget is exhausted. Log actions and outcomes for evaluation.

The lab uses a deterministic controller to make boundaries observable. It is not evidence that an LLM has learned planning. The same dispatcher can later accept model proposals, but correctness must still be measured end to end. Never execute arbitrary Python from a generated string. A calculator with explicit operations is enough for this lesson.

### Tool and context boundaries
A retrieved page, image, or tool response can contain an instruction that conflicts with the task. Treat it as data and preserve the distinction between system policy and external content. Structured output helps parsing but does not establish truth or permission. Evaluate malicious and malformed proposals, duplicate calls, budget exhaustion, and inappropriate actions, alongside ordinary success.

## Worked example

With independent per-sample correctness p=0.4, the chance of at least one correct sample among k=4 is 1−0.6⁴=0.8704. This is an oracle-selection upper bound under an independence assumption. Majority voting can still fail if wrong answers concentrate on one value.

## Guided tutorial

**Question:** Build a schema-checked arithmetic dispatcher and compare candidate voting with an oracle-success calculation.

Open the notebook and predict each result before running it. Spend 5 minutes on setup and data inspection, 15 on the mechanism, 10 on the controlled intervention, and 10 on interpretation. The core uses NumPy and authored/synthetic data; the notebook identifies its limits. Save outputs, configuration, and a short interpretation. Do not report an expected trend as a measured result.

**Real-model or research extension:** Use `extensions/hf_text.py` to generate candidate answers and proposed JSON calls; feed parsed proposals through the same validator, never directly into eval or a shell.

See the [extension guide](../extensions/README.md) for commands, hardware assumptions, and validation status.

## Practical questions

1. Why is pass@k not the accuracy of majority voting?
2. What belongs in a tool schema besides its name?
3. Does a longer reasoning trace establish that a system is more reliable?

## After class

Complete the [exercise sheet](../exercises/08_reasoning.md) in 45–60 minutes. Read the first reference's abstract and method overview before the next session; remaining references are optional depth. For long papers, focus on the objective, one figure/table, and one limitation.

## Readings

- [Self-Consistency Improves Chain of Thought Reasoning](https://arxiv.org/abs/2203.11171)
- [ReAct](https://arxiv.org/abs/2210.03629)
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948)
