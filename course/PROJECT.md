# Capstone — a small, defensible language or multimodal research study

[Course home](../README.md) · 45% of the course grade · 6–8 independent hours · pairs or individual work

## Research question and scope

Build a narrowly scoped system and answer one empirical question. Use a baseline, one main change, and at least two controlled ablations. Every project must engage with multimodal evidence: either use at least two input modalities, or make a controlled comparison between a multimodal pathway and a text-only representation of the same evidence. A speech-transcript comparison qualifies; a text classifier alone does not. Do not train a large foundation model.

Choose one pathway:

| Pathway | Minimal baseline | Main comparison | Required diagnostic |
|---|---|---|---|
| Chart/document assistant | Supplied geometric/OCR-style extraction + text rules | Small VLM or alternative evidence representation | Blank image, counterfactual chart, false premise |
| Multilingual image-text retrieval | Handcrafted features or pretrained dual encoder | Prompt language or adaptation | Shuffled captions and hard negatives |
| Speech-grounded question answering | Human transcript + text answer rule | ASR transcript + same answer rule | Critical numbers/names, noise, missing audio |
| Short-video event QA | Fixed-stride frame reader | Denser/event-aware sampling or video model | Reversed order, shifted sampling phase |
| Multimodal campus assistant | Day-15 deterministic text+chart pipeline | Add a learned component or change the evidence policy | Remove each modality and mismatch evidence |

The offline pathway can earn full marks. Its research claims must be about the controlled synthetic task, not natural-world model quality. A real-model pathway should use one small checkpoint and a small permitted dataset rather than an expensive model sweep.

## Milestones

- **Day 3:** choose a partner/pathway and write a 100-word question, user/task definition, and data/compute plan (about 30 minutes).
- **Day 5:** submit a one-page proposal with intended split units, metrics, baseline, and two ablations; instructor checks feasibility (about 45 minutes).
- **Day 8:** produce baseline predictions and audit five failures (about 90 minutes).
- **Day 10:** freeze development-selected settings, final evaluation examples, metric normalization, and exclusion rules (about 45 minutes).
- **Days 11–14:** run the main comparison and ablations; keep a run log including failures (about 2 hours).
- **Day 15:** present one result and one failure; submit code, predictions, and a 3–4-page report excluding references (about 90 minutes writing/preparation).

The time budget assumes a small project. Reduce the data/model scope if setup consumes more than an hour; document the change. An instructor can approve an equivalent narrow research question without requiring new infrastructure.

## Minimum evaluation package

Use separate development and final test examples. For a manual or synthetic project, aim for at least 20 development and 20 final test items with distinct source groups; 40–100 test items are preferable when inexpensive. For speech/video, a justified smaller pilot is acceptable if timing and sample limits are reported explicitly. These sizes support classroom diagnosis, not broad performance claims. Training data, if used, must be a third disjoint split.

Preserve source groups (document, speaker, image family, clip) when splitting. Do not reuse an image with a new caption across train and test without explaining the resulting leakage risk. A synthetic generator must separate source configurations or templates when the claim is about compositional generalization.

Report the baseline, main system, and two ablations on identical item IDs. Include primary metric, coverage/abstention where relevant, sample count, paired uncertainty, a small error taxonomy, and at least five inspected cases. Add a resource table with device, versions, checkpoint/revision, context/output limits, and measured runtime or explicit “not measured.” Larger model size is not a contribution by itself.

## Submission layout

```text
project/
  README.md                 # question, exact setup/run/evaluate commands
  report.pdf or report.md   # 3–4 pages or equivalent concise text
  data_statement.md         # provenance, permitted use, split units, limitations
  model_card.md             # intended use, failure modes, operating conditions
  environment-lock.txt      # actual installed package versions
  config.json               # seeds, model revisions, prompts, budgets
  src/                      # runnable code or notebooks
  predictions.jsonl         # stable IDs for every evaluated condition
  results.csv               # tables computed from predictions
  contributions.md          # individual roles, collaboration and AI disclosure
```

Do not include downloaded model weights, private media, secrets, or restricted data. Supply a retrieval/preparation recipe when redistribution is not permitted. Use [templates](TEMPLATES.md) for reporting. A PDF is optional; Markdown avoids export tooling requirements.

## Rubric (100 points)

| Criterion | Points | Full-credit evidence |
|---|---:|---|
| Question and scope | 10 | Precise, feasible question with a meaningful modality comparison |
| Data and split design | 15 | Provenance, permitted use, disjoint source groups, documented limitations |
| Baseline and method | 20 | Correct runnable baseline, clear change, justified resource choices |
| Evaluation and ablations | 25 | Frozen protocol, paired results, two ablations, uncertainty, error analysis |
| Reproducibility | 15 | Commands, versions/revisions, predictions, seeds, no fabricated measurements |
| Written argument | 10 | Claims supported by evidence; limitations and alternative explanations |
| Defence and contribution | 5 | Clear six-minute presentation and individual understanding |

A broken download does not justify invented results. A well-executed offline study can receive full marks. An unsupported claim of real-world superiority cannot.

## Presentation logistics

Each team gets 6 minutes plus 2 minutes of questions. For up to five teams, use the 40-minute day-15 tutorial slot for presentations and assign the integration notebook as preparation. With more teams, run a poster rotation during that slot and schedule individual two-minute checks during the final independent-work period. Announce the arrangement before week 3; do not silently extend the 30 contact hours.
