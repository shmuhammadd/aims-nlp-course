# Instructor guide

[Course home](../../README.md) · [Syllabus](../SYLLABUS.md) · [Capstone](../PROJECT.md) · [Task cards](../SEMEVAL_TASK.md)

## Two weeks before day 1

- Run `python scripts/check_course.py` and `python scripts/run_notebooks.py` on the teaching machines.
- Open each deck in [course/slides/](../slides/README.md) in PowerPoint and read the speaker notes. The decks were built from text sources and checked in a preview renderer, not in PowerPoint itself, so look once for anything that sits badly.
- Send students the [diagnostic](../PREREQUISITES.md) and the [task cards](../SEMEVAL_TASK.md). Ask Task 7 teams to request the Llama-3.1 licence on Hugging Face now; approval can take days.
- Re-read the five task pages. The cards were written on 16 November 2026 (AfriGuard on 19 November) and organisers change dates and rules. Task 9's dates need confirming.
- Download task data and any model weights to a shared disk. Check that the room's network can reach Codabench, EvalAI and Hugging Face.
- You co-organise Task 12. Tell your co-organisers that your students are entering, keep test labels off any machine the class can reach, and have each Task 12 paper state the connection.

## Each session

- Minutes 0–10: the quiz on the previous day. Mark it in class by swapping papers. Use the commonest wrong answer to open the session.
- Ask for a prediction before every notebook cell is run. The labs are built so that a wrong prediction is visible.
- Do the worked example on the board before showing its slide.
- In the last five minutes collect the exit ticket: one measured result, one limit, one next experiment.
- Keep model installation out of the 40-minute lab. The labs need NumPy only; shared-task work happens outside class and in office hours.

Where the diagnostic shows that the class already knows a week-1 derivation from the DeepMind path, shorten the board work and give the time to the lab and to the teams' baselines.

## Notebooks and solutions

Each notebook ends with two stub cells that begin `# YOUR TURN`, each followed by a test cell. Solutions are in [solutions/](solutions/). `python scripts/run_notebooks.py` runs every notebook with the solutions in place of the stubs, so a passing run means the solutions pass the tests. To edit a notebook as text, use `python scripts/nb_convert.py`.

## Capstone checkpoints

| Day | Check |
| --- | --- |
| 1 | Every student is in a team with a task, a subtask and at most three languages |
| 5 | Each team shows a baseline score from the official scorer ([assignment 1](../assignments/week1.md)) |
| 10 | Each team hands in a frozen plan ([assignment 2](../assignments/week2.md)) |
| 14 | Drafts are exchanged and reviewed in the [paper clinic](../lectures/14_paper_clinic.md) |
| 15 | Papers, packages and defences ([assignment 3](../assignments/week3.md)) |

Give a strong negative result the same marks as a positive one. After the course, the evaluation window is 10–31 January 2027; agree with each team who submits and how you will stay in touch.

## Support and stretch

For a student who is struggling, work the derivation together and ask them to explain each assertion in the notebook before changing code. For a student who is ahead, point to the "going deeper" readings and the [pretrained-model scripts](../extensions/README.md). Do not give marks for compute spent.

Exercise investigations can have more than one good answer. Mark the design and the evidence. Each guide below uses the same anchors: derivation 2, implementation 4, investigation 4.

## Session guides

Each guide has teaching tips, answers to the practical questions, reference notebook outputs, exercise answers and the quiz key.

| Week 1 | Week 2 | Week 3 |
| --- | --- | --- |
| [01 Tokens and baselines](01_tokenization.md) | [06 Multilingual and data building](06_multilingual.md) | [11 Steering](11_steering.md) |
| [02 Causal Transformer](02_transformers.md) | [07 CLIP and SigLIP](07_contrastive.md) | [12 Retrieval, tools, agents](12_rag_agents.md) |
| [03 Pretraining and decoding](03_pretraining.md) | [08 Vision-language models](08_vlm.md) | [13 Efficiency](13_efficiency.md) |
| [04 Prompting, tuning, LoRA](04_adaptation.md) | [09 Speech and ASR](09_speech.md) | [14 Paper clinic](14_paper_clinic.md) |
| [05 Alignment](05_alignment.md) | [10 Benchmarking and evaluation](10_evaluation.md) | [15 Defence](15_capstone.md) |

[Concept-check key](concept_check_key.md) · [Optional video module guide](x1_video_omni.md)

## Answer keys are in this repository

Anyone who can read the repository can read the keys. Keep the repository private during the course, or move this folder out, or change the numbers in each quiz and sheet before using them for marks.
