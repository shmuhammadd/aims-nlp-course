# Assignment 2 — Week 2 portfolio

[Course home](../../README.md) · [Syllabus](../SYLLABUS.md) · Due at the end of day 10 · 50 points · 15% of the course

## What to submit

| Part | Points | Source |
| --- | ---: | --- |
| Sheet 6: multilingual modelling and data building | 10 | [exercises/06_multilingual.md](../exercises/06_multilingual.md) |
| Sheet 7: CLIP and SigLIP | 10 | [exercises/07_contrastive.md](../exercises/07_contrastive.md) |
| Sheet 8: vision-language models | 10 | [exercises/08_vlm.md](../exercises/08_vlm.md) |
| Sheet 9: speech and ASR | 10 | [exercises/09_speech.md](../exercises/09_speech.md) |
| Sheet 10: benchmarking and evaluation | 10 | [exercises/10_evaluation.md](../exercises/10_evaluation.md) |

Each sheet comes with its notebook, with both "Your turn" tests passing.

## Capstone milestone: one improvement and a frozen plan

Marked under the capstone rubric (system, 20 points; evaluation and analysis, 25 points).

1. **Day 8.** Make the one change you named in week 1. Compare it with the baseline on the same development items, with the official metric and a paired bootstrap interval for the difference. Read ten errors of the new system.
2. **Day 10.** Freeze the plan. After today you may run what the plan lists; you may not add comparisons because a result disappointed you.

Hand in the plan as one page:

- the research question in one sentence;
- the systems you will compare, in a table: baseline, main system, two ablations, and for Task 7 a random-direction control;
- the data split each number will come from;
- the metric, the scorer version and how you will report uncertainty;
- what result would count against your idea;
- your compute budget in GPU-hours and what you will drop if it runs out;
- drafts of the task, data, system and setup sections of the paper.

| Task | Changes that fit a two-week budget |
| --- | --- |
| 12 ArtEmotion-28 | A better fusion than concatenation; per-language heads; a multilingual image-text encoder in place of ViT plus XLM-R. Check the image is used with the decoy test. |
| 6 MMCultureQA | Retrieval of cultural facts before answering; answer-length control for BERTScore; for the spoken track, a cascade against an end-to-end model. |
| 7 CLaS | Clamping in place of adding; a layer sweep; norm-scaled strength; language-specific neurons. |
| 9 Framing | A multilingual encoder trained on all four languages; threshold tuning per label; using text frames to predict image frames. |

A change that does not help still earns full marks if the comparison is fair and the error analysis says something.
