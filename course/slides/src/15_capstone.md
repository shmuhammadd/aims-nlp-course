type: title
title: Defence and concept check
subtitle: Day 15 · Week 3
footer: Shamsuddeen Hassan Muhammad | AIMS 2026

---
type: big
title: What can you do today that you could not do three weeks ago?

Notes: Give one minute. Ask for answers that start with a verb: "train", "measure", "steer". The next three slides are the course's own answer, one week each.

---
title: Today has four parts

| Minutes | What happens | Who |
|---|---|---|
| 0–10 | Quiz 14 | Everyone, alone |
| 10–30 | Three weeks in three slides; how to give the talk | Everyone |
| 30–55 | Concept check: 25 minutes, 20 points | Everyone, alone |
| 60–100 | Talks: six minutes, then two minutes of questions | Five teams; the others do the lab |
| 100–120 | Hand in the package; what happens in January | Everyone |

- There is no quiz on today's session. The concept check takes its place.
- The capstone is 45% of the grade and the concept check 10%

Notes: With more than five teams, the remaining teams present in the poster rotation or the independent-work period, as PROJECT.md sets out. Announce the order before class.

---
type: section
title: Part 1: Three weeks
subtitle: What you can now do

---
title: Week 1: you can build and adapt a language model

| Day | You can now |
|---|---|
| 1 Tokenisation | Run a BPE merge by hand and measure a tokeniser's cost in tokens per letter |
| 2 Transformers | Trace shapes through a decoder block and write causal attention |
| 3 Pretraining | Train a next-token model, spot overfitting, and choose a decoding method |
| 4 Adaptation | Choose between a prompt, instruction tuning and LoRA, and count the adapter's weights |
| 5 Alignment | Compute a DPO loss and say how preference training can fail |

- Capstone by day 5: the organisers' baseline reproduced, and the scorer checked

Notes: One sentence per row. Ask one student per row for the number they remember from that day's lab.

---
title: Week 2: you can work across languages and modalities

| Day | You can now |
|---|---|
| 6 Multilingual | Compare micro, macro and per-language scores, and plan data for a language |
| 7 Contrastive | Compute a contrastive loss and recall@k from a similarity matrix |
| 8 Vision-language | Use image interventions to test whether a model reads the image |
| 9 Speech | Turn a waveform into features and compute WER and CER |
| 10 Evaluation | Compute your task's metric by hand and freeze an experiment plan |

- Capstone by day 10: the experiment plan frozen; task, data and method drafted

Notes: Day 8's decoy idea returns in today's lab and in the Task 12 image-use check.

---
title: Week 3: you can look inside, build a system and write it up

| Day | You can now |
|---|---|
| 11 Steering | Find a direction in activations, then add, clamp or ablate it |
| 12 Retrieval and agents | Separate a retrieval failure from an answer failure |
| 13 Efficiency | Work out model memory and whether a quantised model fits your GPU |
| 14 Paper clinic | Report a difference with a paired interval, and review a draft |
| 15 Today | Defend one result in front of people who will ask how you know |

- Capstone today: revised paper, package, talk and defence

Notes: Then the turn: the talk is the same skill as day 14's paragraph, spoken. Comparison, size, mechanism, alternative.

---
type: section
title: Part 2: The talk and the defence
subtitle: Six minutes, then two

---
title: Six minutes hold one result, one failure, one next experiment

| Time | Slide | You say |
|---|---|---|
| 0:00–1:00 | Task and question | Task, track, languages, one example |
| 1:00–2:30 | System | The baseline and your one change |
| 2:30–4:00 | **One result** | Metric, split, difference, interval |
| 4:00–5:00 | **One failure** | Largest error category, count, example |
| 5:00–6:00 | **Next experiment** | What you run first; submission status |

- Five slides. The whole team stands; every member speaks.
- Say the split with every number: "on development"
- Rehearse aloud twice with a timer. The talk is stopped at 6:00.

Notes: The three bold rows are the rubric of the talk. A team with a negative result presents it the same way: the comparison, the interval, what it rules out.

---
title: What examiners ask, and what a good answer holds

| Question | A good answer holds |
|---|---|
| Which split is that number from? | The split, the item count, the run ID |
| How do you know the gain is not noise? | The paired interval, or "we do not know yet" |
| What does your ablation show? | What was removed, what stayed fixed, what moved |
| What would change your conclusion? | One concrete result that would |
| What did you do yourself? | A file, a run or a section, matching `contributions.md` |
| Explain this line of your config. | What the setting does and why it has that value |

- Each member answers at least one question alone
- "I do not know. I would check it by running X" earns credit. A guess does not.
- The defence carries 5 of the capstone's 100 points

Notes: Draw questions from each team's day-14 review: the reviewer's "claim that needs evidence" is the best one. Ask the quietest member first.

---
title: Is the input used? Remove it, then swap in a decoy
subtitle: Lab 15: twelve authored cases, each a policy line and a bar chart
layout: split

| Condition | Accuracy | Coverage |
|---|---|---|
| Both inputs | 1.00 | 1.00 |
| Policy only | 0.00 | 0.00 |
| Chart only | 0.00 | 0.00 |
| Decoy chart | 0.00 | 1.00 |

- With an input missing, the system abstains
- With another case's chart it answers all 12, wrong, and still cites two evidence IDs
- Your task has the same pair of checks. Name them in the talk.

![](figures/15_cases.png)

Notes: From tutorial 15, seed 42. In the first case the policy picks bar B and the chart gives 7. The system is twenty lines of deterministic code and stands for no real model. A system that always answers 7 gets one of these three cases right. Task 12: text-only against multimodal, and the organisers' decoy paintings. Task 7: a random direction of the same norm, and ablation. Task 6: question without image. Task 9: subtask 2a against 2b.

---
title: The concept check: 25 minutes, 20 points, alone

- Five questions, four points each, across the course:
  - causal language modelling
  - adaptation
  - grounded systems
  - multimodal reasoning
  - evaluation
- One handwritten reference sheet is allowed. No devices.
- Each question asks for a short calculation and a reason
  - show the working: a right method with an arithmetic slip keeps most of the marks
- Worth 10% of the course grade

Notes: The paper is CONCEPT_CHECK.md with the numbers changed. Collect the reference sheets with the scripts. Start the break when the last script is in.

---
title: While other teams present: lab 15, 40 minutes

- Open `course/tutorials/15_capstone.ipynb`. NumPy only.
- Sections 1 to 3: run the ablations and read one result record
  - Predict the nine numbers in section 2 before you run it
- Section 4, in pairs: write two functions until the tests pass
  - `selective_scores`: coverage, selective accuracy, overall accuracy
  - `with_decoy`: the same policy with another case's chart
- Then write down the two checks for your own task, ready for your talk
- Listen to at least two talks and write one question for each

Notes: Teams that present in this block do the lab tonight with the exercise sheet.

---
title: The package goes in before you leave

| File | Check |
|---|---|
| `paper/system_description.pdf` and `paper/source/` | Title pattern, split labels, response to your review |
| `README.md`, `config.json`, `environment-lock.txt` | A stranger can train, predict, score and export |
| `results.csv`, `experiment_log.md` | Every number in the paper has a row and a run ID |
| `predictions/`, `submission/` | Organiser format, validated; development rehearsal if the test is not open |
| `participation.md` | Status, next action, date. No receipt unless you submitted. |
| `contributions.md` | Who did what; collaboration and AI assistance |

- Leave restricted data and model weights out. Give the access recipe.

Notes: The full tree, with `task_specification.md`, `data_statement.md` and `src/`, is in PROJECT.md. A missing official score is "pending". The grade does not depend on rank or acceptance.

---
title: The course ends today. The shared task does not.

| Date | What happens |
|---|---|
| 24 December 2026 – 5 January 2027 | AfriGuard test phase; paper due 10 January |
| 10–31 January 2027 | Official SemEval evaluation, Tasks 6, 7, 9 and 12 |
| 16 January, 23:59 UTC | Task 12 subtask 3 closes |
| 17 January | Task 12 test files for subtasks 1 and 2 released |
| February 2027 (tentative) | System papers due |
| 17–22 August 2027 | SemEval workshop at ACL 2027, Kyoto |

- One named person per team submits. Write that name in `participation.md` today.
- Task 7: two test submissions and one final per track. Do not spend them on tuning.
- Add official scores to the paper only after the organisers release them

Notes: Dates were read from the task sites on 16 November 2026 and the organisers' pages govern. Task 9 teams confirm their dates on the task page. Task 6 teams ask in the task's Slack channel for the test-phase platform.

---
type: big
title: Which claim in your paper would you withdraw if one more seed disagreed?

Notes: Take three answers. A team that can name the claim knows where its paper is thin, and that is the first experiment to run before January. A team that says "none" has either very wide intervals or has not looked.

---
title: Readings

### Before you submit in January
- Your task's site and rules, again. The organisers' pages govern.
- SemEval. Guidelines for SemEval System Papers.

### For the revised paper
- Dror et al., 2018. The Hitchhiker's Guide to Testing Statistical Significance in NLP.
- Mitchell et al., 2019. Model Cards for Model Reporting.

### If your task has images
- KnowHal and ReactBench (2026 preprints): targeted tests of multimodal hallucination. Read how the test items were built.

Notes: Links are in the lecture notes.

---
title: Exit ticket

Write three lines and hand them in.

- The one result you defended today, with its number and split
- The question you could not answer, and how you will find the answer
- The date of your team's next action, and who does it

Thank you for three weeks of work. Send your January scores when they arrive.
