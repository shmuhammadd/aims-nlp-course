# Lecture 15 — SemEval systems, scientific writing, and project defence

[Course home](../../README.md) · Week 3, day 5 · 120 minutes

## Learning outcomes

- Prepare a SemEval system-description paper with claims traceable to experiments.
- Distinguish development analysis, official evaluation, and post-submission results.
- Transfer controlled-ablation lessons from a multimodal lab to the assigned shared task.

## Session plan

| Minutes | Activity |
|---|---|
| 0–10 | Retrieval quiz: task rules, metric, split use, and submission status |
| 10–35 | SemEval system papers: a research question, method, and evidence |
| 35–60 | Worked ablation, results-table critique, and error-analysis exercise |
| 60–65 | Break |
| 65–105 | Guided [notebook/paper-transfer clinic](../tutorials/15_capstone.ipynb), or project presentations as scheduled |
| 105–115 | Practical questions in pairs, then discuss evidence |
| 115–120 | Exit ticket: one result, one limitation, one next experiment |

## Teaching notes

### A system is more than the strongest component
Our capstone is an instructor-selected SemEval shared task and a system-description paper. Follow the [project brief](../PROJECT.md), [task specification](../SEMEVAL_TASK.md), and [paper guide](../SEMEVAL_PAPER.md). The chart assistant below is a controlled teaching example; its synthetic data do not replace the shared-task evaluation. A question may require text, an image, speech, or a combination. Define which component is responsible for each operation and preserve evidence IDs throughout. A correct final answer reached through an unsupported path is fragile; a failed answer may originate in perception, retrieval, reasoning, or output formatting.

The integration lab pairs an authored policy snippet with a chart. Some questions require both: the policy says which bar to read and the chart gives its value. This creates a controlled test of complementarity. Removing the policy or image should force abstention for those questions. The deterministic baseline establishes task solvability; optional pretrained extensions test whether a learned model uses both inputs appropriately.

### Ablations answer research questions
Ablate text, image, retrieval, and reasoning budget one at a time while holding examples and decoding settings fixed. Include mismatched evidence to distinguish ignoring a modality from robustly detecting conflict. Use the same item IDs across conditions for paired analysis. Repeatedly tuning against the final test set invalidates its role; use a development set for exploratory diagnosis.

For multimodal hallucination, construct paired questions with true and false premises over the same image. The 2026 KnowHal and ReactBench preprints provide current examples of targeted evaluation design. Read their construction methods and limitations rather than importing a headline score. Explanation text alone is not proof of an internal error cause.

### Reproducible reporting
A result record should include item ID, split, language, modalities, prediction, target, evidence IDs, model/revision, prompt version, seed, and measured cost when available. Use null for an unmeasured cost rather than an invented zero. Save environment versions and exact commands. A new reader should be able to trace a table row to predictions and reconstruct the metric.

### From experiment records to a SemEval paper
Use the official task metric and label every table with its split and assigned track. Write the baseline and selection protocol before interpreting the best score. Describe the submitted system separately from later changes. When official labels or scores are unavailable, report measured development results and mark official evaluation as pending. A complete course paper does not require an invented test score.

Use the paper clinic to turn one result table into a paragraph: state the comparison, quantify the observed difference, explain one plausible mechanism, and name one alternative explanation. Inspect at least ten errors across relevant categories where available. The specific task determines whether modality, language, class, retrieval, or prompt ablations are appropriate.

### A defensible research argument
Structure the system-description paper around a question, a baseline, a controlled change, observations, and limits. “Model A is better” is incomplete without the task, sample, metric, and cost. Negative results can be valuable if they eliminate a plausible explanation or expose a failure. A two-point gain on a tiny sample needs uncertainty and error analysis, not a universal claim.

Each team delivers a six-minute talk and a two-minute defence. State one result, show one failure, and identify the next experiment. Individual contributions and short oral questions ensure that collaboration does not hide gaps in understanding. The capstone rubric gives explicit credit to paper quality, evaluation, reproducibility, and individual understanding. Leaderboard position and workshop acceptance do not determine the grade.

## Worked example

A policy selects plot B. A chart shows A=3 and B=7. The full system should return 7 with both source IDs. Removing the policy leaves the selection ambiguous; removing the chart leaves the value unknown. Always returning 7 can pass one example without using either modality.

## Guided tutorial

**Question:** Run a text-plus-chart evidence pipeline under modality ablations, calculate paired uncertainty, and write reproducible result records.

Open the notebook and predict each result before running it. Spend 5 minutes on setup and evidence inspection, 10 on the mechanism/intervention, 5 on interpreting the toy results, and 20 on the SemEval paper-transfer worksheet. The core uses NumPy and authored/synthetic data; the notebook identifies its limits. Save outputs, configuration, and a short interpretation. Do not report an expected trend as a measured result.

**Shared-task transfer:** Identify the official metric and scorer, map your baseline/main system/two ablations onto a split-labelled results table, and draft one evidence-backed paragraph for your SemEval paper. Use the task data and permitted resources; the toy baseline is an illustration only.

See the [extension guide](../extensions/README.md) for commands, hardware assumptions, and validation status.

## Practical questions

1. What experiment distinguishes modality use from a memorized answer?
2. How should the paper report results if the official SemEval evaluation has not opened?
3. What makes a negative result useful?

## After class

Complete the [exercise sheet](../exercises/15_capstone.md) in 45–60 minutes and finalize the separate SemEval paper/package for the course deadline. Record any later official evaluation and paper milestones in the task card.

## Readings

- [SemEval system-paper guide](https://semeval.github.io/system-paper-template.html)
- [Official ACL style files](https://github.com/acl-org/acl-style-files)
- [KnowHal (2026 preprint)](https://arxiv.org/abs/2608.03782)
- [ReactBench (2026 preprint)](https://arxiv.org/abs/2605.29579)
- [Model Cards](https://arxiv.org/abs/1810.03993)
