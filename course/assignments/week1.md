# Assignment 1 — Week 1 portfolio

[Course home](../../README.md) · [Syllabus](../SYLLABUS.md) · Due at the end of day 5 · 50 points · 15% of the course

## What to submit

One folder or repository link containing the five items below and the milestone.

| Part | Points | Source |
| --- | ---: | --- |
| Sheet 1: tokens and baselines | 10 | [exercises/01_tokenization.md](../exercises/01_tokenization.md) |
| Sheet 2: the causal Transformer | 10 | [exercises/02_transformers.md](../exercises/02_transformers.md) |
| Sheet 3: pretraining and decoding | 10 | [exercises/03_pretraining.md](../exercises/03_pretraining.md) |
| Sheet 4: prompting, instruction tuning, LoRA | 10 | [exercises/04_adaptation.md](../exercises/04_adaptation.md) |
| Sheet 5: alignment | 10 | [exercises/05_alignment.md](../exercises/05_alignment.md) |

Each sheet comes with its notebook, with both "Your turn" tests passing.

## Capstone milestone: reproduce the baseline

This milestone is marked under the capstone rubric (task protocol, 10 points; system, 20 points). It is due with the portfolio so that nobody starts week 2 without a working system.

1. **Day 1.** Choose one task from the [task cards](../SEMEVAL_TASK.md), one subtask or track, and at most three languages. Register on the task's platform. Task 7 teams request access to the Llama weights.
2. **Day 3.** Fill in the team addendum on the task card: research question, data you may use, the official metric, your compute limit.
3. **Day 5.** Run the organisers' baseline, or the simplest system the task allows, on the development data. Score it with the official scorer.

Hand in a one-page note with:

- the command you ran and the commit of the organisers' code;
- your score beside the organisers' published score, with the split named;
- if the two differ, your best explanation and what you checked;
- ten development errors you read, sorted into two or three kinds;
- the one change you will try in week 2, and why you expect it to help.

| Task | Baseline to reproduce in week 1 |
| --- | --- |
| 12 ArtEmotion-28 | The starter-kit baseline for your subtask. For subtask 1, the text-only run first (published macro-F1 0.550). |
| 6 MMCultureQA | One language track of subtask 2 (written questions) with a small vision-language model or the organisers' starter code, scored with BERTScore F1 on the dev-test leaderboard. |
| 7 CLaS | The prompting baseline from the organisers' repository on one language pair, with the scorer. It is not ranked; it tells you what the scorer returns. |
| 9 Framing | Subtask 1 on one language with a TF-IDF and logistic-regression classifier, or a small encoder, scored with micro-F1 on the data released so far. |

A baseline that scores below the published number is acceptable if the note says why. A baseline that was not run is not.
