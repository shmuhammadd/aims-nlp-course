# Reusable reporting templates

[Course home](../README.md) · [SemEval project brief](PROJECT.md) · [Paper guide](SEMEVAL_PAPER.md)

## Experiment log

| Run ID | Hypothesis | Data/split hash | Model/revision | Change from baseline | Seed | Metric | Cost | Failure/notes |
|---|---|---|---|---|---|---|---|---|
| Fill with a measured run | | | | | | | | |

## Prediction JSONL schema

This is an internal analysis format, not a SemEval submission schema. Export a separate file matching the selected task’s official format. Use one object per item per condition, preserve IDs across comparisons, and record the task/track in your experiment configuration. Suggested analysis fields:

```json
{"id":"item-001","condition":"full","split":"test","language":"en","modalities":["text","image"],"prediction":"B","target":"B","evidence_ids":["doc-1","image-1"],"model":"your-model-id","revision":"resolved-commit-or-local-version","prompt_version":"v1","seed":42,"latency_ms":null}
```

Use `null` for abstention predictions, unavailable hidden-test targets, or unmeasured costs; document these distinct meanings by field. Do not infer hidden gold labels or force this example schema onto the official submission. A revision should identify an actual artifact, not a mutable `main` branch. Metric normalization and excluded items must be recorded in configuration.

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

## SemEval paper and participation records

Use the [system-description paper worksheet](SEMEVAL_PAPER.md) for the full paper, measured-results table, and day-14 peer review. Submit PDF plus editable source. The course planning target is 4–6 main-text pages excluding references, subject to the selected edition’s rules.

In `participation.md`, record the task/year/track; registration status; evaluation and paper deadlines with timezone and official URLs; frozen system version; prediction-file checksum; validation outcome; and actual submission IDs/receipts when available. If evaluation has not opened, state that fact and record the next action and date. Keep course/development scores separate from official evaluation scores.

## Paper discussion card

Citation and version/date; question; objective/equations; data; one central result with table/section; baseline fairness; ablation evidence; limitations; one proposed replication. Mark author-reported results explicitly. Reading a recent preprint does not make its claims settled findings.
