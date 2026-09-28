# Lecture 15 — Multimodal system integration, research critique, and project defence

[Course home](../../README.md) · Week 3, day 5 · 120 minutes

## Learning outcomes

- Integrate evidence, language, and visual inputs with auditable provenance.
- Run modality and evidence ablations on a frozen test set.
- Defend conclusions using uncertainty, limitations, and reproducible artefacts.

## Session plan

| Minutes | Activity |
|---|---|
| 0–10 | Retrieval quiz: recall the previous session; day 1 uses the prerequisite diagnostic |
| 10–35 | Concepts and concrete examples from the first half of these notes |
| 35–60 | Board derivation, worked example, and failure analysis |
| 60–65 | Break |
| 65–105 | Guided [notebook tutorial](../tutorials/15_capstone.ipynb) |
| 105–115 | Practical questions in pairs, then discuss evidence |
| 115–120 | Exit ticket: one result, one limitation, one next experiment |

## Teaching notes

### A system is more than the strongest component
Our capstone is a small evidence-grounded assistant or an equally scoped research investigation. A question may require text, an image, speech, or a combination. Define which component is responsible for each operation and preserve evidence IDs throughout. A correct final answer reached through an unsupported path is fragile; a failed answer may originate in perception, retrieval, reasoning, or output formatting.

The integration lab pairs an authored policy snippet with a chart. Some questions require both: the policy says which bar to read and the chart gives its value. This creates a controlled test of complementarity. Removing the policy or image should force abstention for those questions. The deterministic baseline establishes task solvability; optional pretrained extensions test whether a learned model uses both inputs appropriately.

### Ablations answer research questions
Ablate text, image, retrieval, and reasoning budget one at a time while holding examples and decoding settings fixed. Include mismatched evidence to distinguish ignoring a modality from robustly detecting conflict. Use the same item IDs across conditions for paired analysis. Repeatedly tuning against the final test set invalidates its role; use a development set for exploratory diagnosis.

For multimodal hallucination, construct paired questions with true and false premises over the same image. The 2026 KnowHal and ReactBench preprints provide current examples of targeted evaluation design. Read their construction methods and limitations rather than importing a headline score. Explanation text alone is not proof of an internal error cause.

### Reproducible reporting
A result record should include item ID, split, language, modalities, prediction, target, evidence IDs, model/revision, prompt version, seed, and measured cost when available. Use null for an unmeasured cost rather than an invented zero. Save environment versions and exact commands. A new reader should be able to trace a table row to predictions and reconstruct the metric.

### A defensible research argument
Structure the report around a question, a baseline, a controlled change, observations, and limits. “Model A is better” is incomplete without the task, sample, metric, and cost. Negative results can be valuable if they eliminate a plausible explanation or expose a failure. A two-point gain on a tiny sample needs uncertainty and error analysis, not a universal claim.

Each team delivers a six-minute talk and a two-minute defence. State one result, show one failure, and identify the next experiment. Individual contributions and short oral questions ensure that collaboration does not hide gaps in understanding. The capstone rubric rewards reproducibility and reasoning rather than leaderboard position.

## Worked example

A policy selects plot B. A chart shows A=3 and B=7. The full system should return 7 with both source IDs. Removing the policy leaves the selection ambiguous; removing the chart leaves the value unknown. Always returning 7 can pass one example without using either modality.

## Guided tutorial

**Question:** Run a text-plus-chart evidence pipeline under modality ablations, calculate paired uncertainty, and write reproducible result records.

Open the notebook and predict each result before running it. Spend 5 minutes on setup and data inspection, 15 on the mechanism, 10 on the controlled intervention, and 10 on interpretation. The core uses NumPy and authored/synthetic data; the notebook identifies its limits. Save outputs, configuration, and a short interpretation. Do not report an expected trend as a measured result.

**Real-model or research extension:** Choose one real-model extension from earlier days for the final project. Keep the deterministic baseline and identical test protocol so the added model can be evaluated fairly.

See the [extension guide](../extensions/README.md) for commands, hardware assumptions, and validation status.

## Practical questions

1. What experiment distinguishes modality use from a memorized answer?
2. Why are null and zero different for an unmeasured runtime?
3. What makes a negative result useful?

## After class

Complete the [exercise sheet](../exercises/15_capstone.md) in 45–60 minutes. Read the first reference's abstract and method overview before the next session; remaining references are optional depth. For long papers, focus on the objective, one figure/table, and one limitation.

## Readings

- [KnowHal (2026 preprint)](https://arxiv.org/abs/2608.03782)
- [ReactBench (2026 preprint)](https://arxiv.org/abs/2605.29579)
- [Model Cards](https://arxiv.org/abs/1810.03993)
