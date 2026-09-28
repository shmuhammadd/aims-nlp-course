# Reusable reporting templates

[Course home](../README.md) · [Project brief](PROJECT.md)

## Experiment log

| Run ID | Hypothesis | Data/split hash | Model/revision | Change from baseline | Seed | Metric | Cost | Failure/notes |
|---|---|---|---|---|---|---|---|---|
| Fill with a measured run | | | | | | | | |

## Prediction JSONL schema

One JSON object per item per condition. Preserve the same item ID across paired conditions. Required fields:

```json
{"id":"item-001","condition":"full","split":"test","language":"en","modalities":["text","image"],"prediction":"B","target":"B","evidence_ids":["doc-1","image-1"],"model":"your-model-id","revision":"resolved-commit-or-local-version","prompt_version":"v1","seed":42,"latency_ms":null}
```

Use `null` for abstention predictions or unmeasured costs; document the distinction from a literal string answer. A revision should identify an actual artifact, not a mutable `main` branch. Metric normalization and excluded items must be recorded in configuration.

## Data statement

- Task, languages/varieties, modalities, and collection context.
- Sources, access conditions, authorship/consent, and redistribution rules.
- Inclusion/exclusion criteria and counts before/after filtering.
- Annotation instructions, annotator qualifications, disagreement handling.
- Split units, deduplication, and contamination checks.
- Known underrepresentation, sensitive information, and inappropriate uses.
- For synthetic data: generator/version, seed, constraints, and limits on external validity.

## Model/system card

- System components, checkpoints/revisions, and preprocessing.
- Intended users/task and unsupported uses.
- Training/adaptation data, hyperparameters, and selection protocol.
- Evaluation sets, metrics, uncertainty, subgroup counts, and ablations.
- Resource measurements and deployment assumptions.
- Known failures, evidence requirements, and abstention behavior.

## Short report outline

1. **Question:** concrete hypothesis and why it matters (one paragraph).
2. **Data:** provenance, splits, labels, and scope (half page).
3. **Method:** baseline, one main change, and computational budget (half page).
4. **Results:** one table, paired uncertainty, and two ablations (one page).
5. **Analysis:** five errors, alternative explanations, and limitations (half page).
6. **Reproduction:** exact commands and artifact locations (short paragraph).

## Paper discussion card

Citation and version/date; question; objective/equations; data; one central result with table/section; baseline fairness; ablation evidence; limitations; one proposed replication. Mark author-reported results explicitly. Reading a recent preprint does not make its claims settled findings.
