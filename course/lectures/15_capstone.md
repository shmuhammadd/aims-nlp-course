# Lecture 15 — Defence and concept check

[Course home](../../README.md) · Week 3, day 5 · 120 minutes · [Slides](../slides/15_capstone.pptx) · [Notebook](../tutorials/15_capstone.ipynb) · [Exercises](../exercises/15_capstone.md) · [Concept check](../CONCEPT_CHECK.md)

## Learning outcomes

- Present one result, one failure and one next experiment in six minutes, and answer questions on them for two.
- Show that a system uses an input by removing it and by swapping in a decoy, on the same items.
- Hand in a package from which a stranger can rerun the system and trace every number in the paper.

## Session plan

Session 15 has no quiz of its own. The individual [concept check](../CONCEPT_CHECK.md) takes its place. Paper writing moved to [day 14](14_paper_clinic.md).

| Minutes | Activity |
|---|---|
| 0–10 | Quiz 14 (paper clinic and peer review) |
| 10–30 | Slides 1–12: three weeks in three slides, the six-minute talk, what examiners ask, the input-use check, the concept-check rules |
| 30–55 | Individual concept check: 25 minutes, 20 points |
| 55–60 | Break |
| 60–100 | Talks: six minutes and two minutes of questions for each of five teams. Slide 13: teams not presenting work through [notebook 15](../tutorials/15_capstone.ipynb) |
| 100–112 | Slides 14–16: the package, the January dates, the closing question. Teams submit. |
| 112–120 | Slide 18: exit ticket and course feedback |

With more than five teams, the others present in a poster rotation or in the final independent-work period, as the [capstone brief](../PROJECT.md) sets out. Announce the order before week 3.

## Teaching notes

### Three weeks, three sentences
Week 1: you can tokenise text, write a decoder block, train it with next-token loss, adapt it with a prompt, instruction tuning or LoRA, and compute a preference loss. Week 2: you can compare scores across languages, compute a contrastive loss, test whether a vision-language model reads the image, score a transcript, and freeze an experiment plan before touching the test set. Week 3: you can steer activations, separate a retrieval failure from an answer failure, work out whether a model fits your GPU, and report a difference with a paired interval.

Today's skill is the last one spoken aloud: defend one result in front of people who ask how you know.

### The six-minute talk
Each team gives a six-minute talk and a two-minute defence. State one result, show one failure, and name the next experiment.

| Time | Slide | Content |
|---|---|---|
| 0:00–1:00 | Task and question | Task, track, languages, one input and output example, your research question |
| 1:00–2:30 | System | The baseline and the one change you made |
| 2:30–4:00 | One result | The table row: metric, split, difference, interval |
| 4:00–5:00 | One failure | The largest error category, its count, one example |
| 5:00–6:00 | Next experiment | The first thing you would run, and the status of your official submission |

Five slides are enough. Every member speaks. Say the split with every number. "Model A is better" is incomplete without the task, the sample, the metric and the cost. A two-point gain on a small sample needs an interval and an error analysis before it supports any wider claim. A negative result is worth presenting when it rules out a plausible explanation or exposes a failure; present it the same way, with the comparison and the interval.

### What examiners ask
- **Which split is that number from?** Give the split, the item count and the run ID.
- **How do you know the gain is not noise?** Give the paired interval, or say that you do not know yet.
- **What does your ablation show?** Say what was removed, what stayed fixed and what moved.
- **What would change your conclusion?** Name one concrete result.
- **What did you do yourself?** Name a file, a run or a section that matches `contributions.md`.
- **Explain this line of your config.** Say what the setting does and why it has that value.

Each member answers at least one question alone, so that a pair cannot hide one member's gaps. "I do not know, and I would check it by running X" earns credit. The defence carries 5 of the capstone's 100 points. The rubric also credits the paper, the evaluation and the reproducible package. Leaderboard rank and workshop acceptance do not affect the grade.

### Is the input used?
A system is more than its strongest component. In a task with text, an image or speech, say which component does which job, and keep evidence IDs through the pipeline. A correct answer reached without the evidence is fragile. A wrong answer may come from perception, retrieval, reasoning or output formatting.

The integration lab pairs a one-line policy with a bar chart. The policy says which bar to read and the chart gives its value, so each question needs both. Ablate one input at a time and hold the items and settings fixed. Use the same item IDs in every condition so that the comparison is paired. In the lab, the full system scores 1.00, and the policy-only and chart-only systems abstain on all 12 cases.

Removal is the easy test, because a system can pass it by abstaining. The harder test keeps the policy and swaps in another case's chart. The lab system then answers all 12 cases, gets all 12 wrong, and still cites two evidence IDs for each. A citation that is present has not been shown to support the answer. Task 12 runs the same test with decoy paintings.

For multimodal hallucination, build paired questions with true and false premises over the same image. The 2026 KnowHal and ReactBench preprints are current examples of this kind of targeted test. Read how they built their items and what limits they state. A model's own explanation of its error is not evidence of the cause.

Tune on development data. A test set that has been used for repeated tuning no longer measures what it was held out to measure.

### Reproducible reporting
A result record holds the item ID, split, language, modalities, prediction, target, evidence IDs, model and revision, prompt version, seed and measured cost. Write `null` for a cost nobody measured; a zero claims a measurement. Save the environment versions and the exact commands. A new reader should be able to trace a table row to the predictions and recompute the metric.

### The concept check
The check lasts 25 minutes, carries 20 points and is 10% of the course grade. It has five questions of four points each, on causal language modelling, adaptation, grounded systems, multimodal reasoning and evaluation. One handwritten reference sheet is allowed. Each question asks for a short calculation and a reason, so show the working.

### Submitting the package
The package is listed in the [capstone brief](../PROJECT.md). Before handing in, check six things:

- `paper/` holds the PDF and the editable source, with the response to your day-14 review.
- `README.md`, `config.json` and `environment-lock.txt` let a stranger train, predict, score and export.
- Every number in the paper has a row in `results.csv` and a run in `experiment_log.md`.
- `submission/` holds organiser-format predictions, validated, or a development-format rehearsal if the test phase has not opened.
- `participation.md` gives the status, the next action and its date. Record a receipt only after a real submission.
- `contributions.md` says who did what and discloses collaboration and AI assistance.

Label every table with its split and track. Describe the submitted system separately from later changes. A missing official score is "pending", and a complete course paper needs no test score.

### After the course

| Date | What happens |
|---|---|
| 10–31 January 2027 | Official evaluation for Tasks 6, 7, 9 and 12 |
| 16 January 2027, 23:59 UTC | Task 12 subtask 3 closes |
| 17 January 2027 | Task 12 test files for subtasks 1 and 2 are released |
| February 2027 (tentative) | System papers due |
| 17–22 August 2027 | SemEval workshop at ACL 2027, Kyoto |

One named person per team submits. Task 7 allows two test submissions and one final submission per track, so tune on development questions. Task 9 teams confirm their dates on the task page, and Task 6 teams ask in the task's Slack channel which platform the test phase uses. Add official scores to the paper only after the organisers release them. The dates come from the [task cards](../SEMEVAL_TASK.md); the organisers' pages govern.

## Worked example

A policy selects plot B. A chart shows A = 3 and B = 7. The full system returns 7 with both source IDs. Without the policy the choice of bar is unknown. Without the chart the value is unknown. A system that always returns 7 passes this one example and uses neither input.

Now keep the policy and swap in a chart with A = 5 and B = 2. A system that reads the chart answers 2. The always-7 system still answers 7. One decoy separates the two systems, and removal alone does not: both could be made to abstain when an input is missing.

Scoring with abstentions: a system answers 9 of 12 questions and gets 7 right. Coverage is $9/12=0.75$, selective accuracy is $7/9=0.78$, and overall accuracy with abstentions counted as wrong is $7/12=0.58$.

## Guided tutorial

**Question:** Does a text-plus-chart system use both inputs, and how do you record the evidence?

The notebook builds 12 authored cases and a deterministic system of about twenty lines. Students run the three modality conditions, compute a paired interval, write one JSON result record, and then write `selective_scores` and `with_decoy` against tests. The cases are authored and say nothing about a real model.

**Shared-task path:** before the talk, name the removal check and the decoy check for your own task. Task 12: the text-only run against the multimodal run, and the organisers' decoy paintings. Task 6: the question without the image, and the image without the question. Task 7: a random direction of the same norm, and ablation of the steering vector. Task 9: subtask 2a against 2b.

## Practical questions

1. What experiment separates a system that uses a modality from one that has memorised the answer?
2. How does the paper report results if the official SemEval evaluation has not opened?
3. What makes a negative result useful?

## After class

Complete the [exercise sheet](../exercises/15_capstone.md) in 45–60 minutes; it is the last sheet of portfolio 3. Write the date of your team's next action in `participation.md`. Send the instructor your official scores when the organisers release them.

## Readings

- [Guidelines for SemEval System Papers](https://semeval.github.io/system-paper-template.html). Reread the results section before you revise in February.
- [ACL style files](https://github.com/acl-org/acl-style-files). The template for the final paper.
- [The Hitchhiker's Guide to Testing Statistical Significance in Natural Language Processing](https://aclanthology.org/P18-1128/) (Dror et al., 2018). For the question "how do you know the gain is not noise?".
- [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993) (Mitchell et al., 2019). What to state about a system's intended use, evaluation and limits.
- [KnowHal](https://arxiv.org/abs/2608.03782) (2026 preprint). Paired positive and negative questions over the same image; read how the pairs were built.
- [ReactBench](https://arxiv.org/abs/2605.29579) (2026 preprint). A benchmark organised by the cause of multimodal hallucination; read the construction and the stated limits.
