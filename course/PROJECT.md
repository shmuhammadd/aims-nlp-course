# Capstone — a SemEval-2027 system and its paper

[Course home](../README.md) · 45% of the course · 12–16 hours outside class · pairs, or alone by agreement

## What each team does

Enter **one** SemEval-2027 task: [6, 7, 9 or 12](SEMEVAL_TASK.md). Pick one subtask or track and at most three languages. Build a system, evaluate it with the official metric, package predictions in the official format, and write a system-description paper.

Each team produces:

1. the organisers' baseline, reproduced, and one justified improvement;
2. development experiments with at least two controlled ablations and an error analysis;
3. a prediction file in the official format, checked with the organisers' validator where one exists;
4. a system-description paper with code, configuration and experiment records;
5. a six-minute talk and a two-minute defence, with a record of who did what.

Evaluation runs 10–31 January 2027, after the course. The course mark comes from the development experiments, the paper and a submission that is ready to run. Teams then submit during the evaluation window and add official scores to the paper. Leaderboard rank and paper acceptance do not affect the course mark.

## The task's rules come first

Read the task page before modelling. Record on your [task card](SEMEVAL_TASK.md) which external data, models and APIs are allowed, how many submissions you get, and what you may do with development and test data. One task's rules do not carry over to another. Three that are easy to miss:

- **Task 7** does not rank a target-language instruction in the prompt.
- **Task 12** forbids annotations of the evaluation paintings from any other source, and checks that a multimodal run uses the image.
- **Task 6** test audio is human speech; the training audio is synthetic.

Use the released training and development splits and the official scorer. Fit preprocessing on training data. Choose every setting on development data. If a task has no labelled development set, agree a split of the training data with the instructor and write it down. Leaderboard feedback is not a development set.

Run the main comparison and the ablations on the same development items. Report the official metric first, then per-language or per-class results with counts, and a paired bootstrap interval for each difference you discuss. Read at least ten errors. Record the scorer version, data release, seeds, model revisions and compute.

Label every number as development, official or post-submission. A missing official score is "pending". Never write a rank, a receipt or a result that does not exist.

## Milestones

| Day | Deliverable | Hours |
| --- | --- | ---: |
| 1 | Task, subtask and languages chosen; platform registration | 1 |
| 3 | Task card filled in: research question, permitted resources, metric, compute limit | 1 |
| 5 | Baseline reproduced and scored; ten errors read; paper outline ([assignment 1](assignments/week1.md)) | 2–3 |
| 8 | First improvement compared with the baseline on the same items | 2–3 |
| 10 | Experiment plan frozen; task, data, system and setup sections drafted ([assignment 2](assignments/week2.md)) | 2 |
| 11–13 | Two ablations; error analysis; submission file validated; full draft | 3–4 |
| 14 | Peer review exchanged in the paper clinic | 1 |
| 15 | Revised paper, package, talk and defence ([assignment 3](assignments/week3.md)) | 1–2 |

Keep the system small. A strong linear baseline, a small pretrained encoder, a prompt, a LoRA adapter or one steering method is enough for a good paper. Sweeping many large models is not what is marked.

## The paper

Write a complete paper: 4–6 pages of main text, excluding references, in the ACL style, unless the SemEval-2027 call sets another limit. Follow the [paper guide](SEMEVAL_PAPER.md). The paper explains the task and track, describes the system, states the experimental protocol, reports comparisons and ablations, analyses errors and states limits. Day 14 is a [paper clinic](lectures/14_paper_clinic.md) with a peer review.

## Submission package

```text
project/
  README.md                 # setup, train, predict, score and export commands
  task_card.md              # task, subtask, languages, rules, official URLs
  paper/
    system_description.pdf
    source/                 # LaTeX source and bibliography
  data_statement.md         # where the data came from, how splits were used, limits
  environment-lock.txt      # pip freeze
  config.json               # seeds, model revisions, prompts, budgets
  src/                      # code or notebooks that run
  predictions/              # development predictions
  submission/               # file in the organisers' format
  results.csv               # every number in the paper, with its split
  experiment_log.md         # every configuration tried, failures included
  review_response.md        # the review you received and what you changed
  participation.md          # registration and submission status, next dates
  contributions.md          # who did what; collaboration and AI assistance
```

Do not redistribute task data or model weights; give the commands that fetch them. [Templates](TEMPLATES.md) for the log, the data statement and the result records are provided.

## Rubric (100 points)

| Criterion | Points | Full marks look like |
| --- | ---: | --- |
| Task understanding and data protocol | 10 | Correct subtask, permitted resources, split use and metric |
| Baseline and system | 20 | Baseline reproduced; one justified improvement; compute used sensibly |
| Evaluation and analysis | 25 | Fair comparison, two ablations, uncertainty, errors read and categorised |
| System-description paper | 25 | Complete and clear; every claim matches a recorded result |
| Reproducibility and submission readiness | 15 | Commands run; export validated; sources included |
| Talk and defence | 5 | Each member can explain the system and answer for the results |

A careful negative result can earn full marks. No marks are given for model size, rank or acceptance.

## Talks

With up to five teams, use the 40-minute lab slot on day 15 for six-minute talks and two-minute questions. With more teams, run a poster round and hold individual questions in the final study period. Tell students which format applies before week 3.
