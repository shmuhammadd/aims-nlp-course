# Lecture 10 — Reliable evaluation, calibration, and responsible reporting

[Course home](../../README.md) · Week 2, day 5 · 120 minutes

## Learning outcomes

- Construct an evaluation protocol separating selection from final testing.
- Compute calibration and selective-prediction metrics.
- Audit model judges, contamination, and subgroup performance.

## Session plan

| Minutes | Activity |
|---|---|
| 0–10 | Retrieval quiz: recall the previous session; day 1 uses the prerequisite diagnostic |
| 10–35 | Concepts and concrete examples from the first half of these notes |
| 35–60 | Board derivation, worked example, and failure analysis |
| 60–65 | Break |
| 65–105 | Guided [notebook tutorial](../tutorials/10_evaluation.ipynb) |
| 105–115 | Practical questions in pairs, then discuss evidence |
| 115–120 | Exit ticket: one result, one limitation, one next experiment |

## Teaching notes

### A benchmark score is a conditional statement
A score describes a checkpoint, prompt/template, dataset version, decoding rule, and metric implementation. Change any of these and the comparison may change. Freeze a development-selected protocol before final testing. Keep the test set out of prompt examples, retrieval indexes, fine-tuning, and model selection. A public benchmark may have entered pretraining; contamination checks reduce uncertainty but rarely prove complete absence.

Choose metrics by task: macro-F1 for imbalanced classification; exact match with declared normalization for short answers; retrieval metrics for evidence selection; word/character error rates for speech; human rubrics for open-ended support and usefulness. Averages hide important errors. Report counts, uncertainty, and meaningful slices.

### Calibration and abstention
For binary prediction probability p and label y, the Brier score is $N^{-1}\sum_i(p_i-y_i)^2$. In multiclass settings, sum across classes and state the convention. Calibration asks whether events assigned probability p occur approximately p of the time. Expected calibration error bins predictions, but results depend on binning and sample size; pair it with reliability tables and proper scores.

Selective prediction answers only when confidence exceeds a threshold. Coverage is answered/total; selective accuracy is correct/answered. Report both. A system can achieve perfect accuracy by answering one easy question, which may not serve users. Choose the threshold on development data, not the final test set. Raw sequence likelihood is not automatically calibrated correctness confidence.

### Judging generated outputs
Model-based judges may show position, verbosity, style, or self-preference biases. Randomize order, use explicit rubrics, blind system identity, and compare with a human-reviewed subset. Measure agreement and inspect disagreement cases. If a model writes both answers and the evaluation rubric, circularity becomes especially concerning. Automatic lexical overlap can penalize correct paraphrases; semantic scores can reward unsupported content.

### Responsible evidence
A model card records intended use, data provenance, evaluation populations, limitations, and operating constraints. A data statement describes collection and representation. For fairness, ask who uses the system and which harms matter; parity of one aggregate metric does not establish fairness. In low-resource settings, even the label definitions may require local interpretation.

Use paired comparisons for two systems on the same items. Distinguish variation from random seeds from variation in the sampled test population. Include failed runs and decoding failures with a clear policy. Report uncertainty without claiming that a tiny classroom dataset establishes production readiness.

## Worked example

For p=[0.9,0.6,0.2] and y=[1,0,0], binary Brier=(0.01+0.36+0.04)/3=0.1367. At confidence threshold 0.8 (max(p,1−p)), two cases are answered and both are correct: coverage 2/3, selective accuracy 1.0.

## Guided tutorial

**Question:** Calculate Brier score, reliability bins, coverage/accuracy tradeoffs, and a simulated judge-order audit.

Open the notebook and predict each result before running it. Spend 5 minutes on setup and data inspection, 15 on the mechanism, 10 on the controlled intervention, and 10 on interpretation. The core uses NumPy and authored/synthetic data; the notebook identifies its limits. Save outputs, configuration, and a short interpretation. Do not report an expected trend as a measured result.

**Real-model or research extension:** Apply the evaluation functions to real outputs from earlier model extensions. Use the included JSONL result schema to preserve IDs and provenance.

See the [extension guide](../extensions/README.md) for commands, hardware assumptions, and validation status.

## Practical questions

1. Can two equally accurate models differ in calibration?
2. Why must an abstention threshold be chosen on development data?
3. What would reveal position bias in a model judge?

## After class

Complete the [exercise sheet](../exercises/10_evaluation.md) in 45–60 minutes. Read the first reference's abstract and method overview before the next session; remaining references are optional depth. For long papers, focus on the objective, one figure/table, and one limitation.

## Readings

- [HELM](https://arxiv.org/abs/2211.09110)
- [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599)
- [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993)
