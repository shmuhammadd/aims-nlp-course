# Capstone — SemEval shared-task participation and system-description paper

[Course home](../README.md) · 45% of the course grade · 12–16 independent hours during the course · pairs or individual work

## Required outcome

Develop and evaluate a system for an instructor-selected **SemEval shared task**, prepare predictions in the official submission format, and write a **system-description paper**. The capstone combines shared-task participation with reproducible research and scientific writing. The [task specification](SEMEVAL_TASK.md) will identify the year, task, required subtasks/languages, official resources, and deadlines once the instructor selects them.

All teams work within the selected task and assigned tracks. A text-only SemEval task is eligible; multimodal comparisons are required only when appropriate to that task. The synthetic chart assistant and other course labs are practice exercises, not substitutes for the shared-task capstone. Use the task's actual permitted data, official metric, and evaluation protocol.

Each team must produce:

1. A reproducible baseline and one justified system improvement.
2. Development experiments, at least two controlled ablations, and inspected error cases.
3. A validated prediction file for each assigned subtask/track, using its official format.
4. A system-description paper, code, configuration, and experiment records.
5. A six-minute presentation and two-minute defence, with individual contribution evidence.

Participate in the official evaluation and paper-submission process when the selected edition's schedule permits. The course submission and the external submission are distinct milestones; acceptance and leaderboard rank do not determine the course grade.

## Task rules govern the experiment

Read the official task description before modelling. Record allowed external data, pretrained models/APIs, track restrictions, submission limits, and permitted use of development/test data in the [task specification](SEMEVAL_TASK.md). Do not assume that one SemEval task's rules apply to another.

Use the provided training/development splits and official scorer. Fit preprocessing on training data and select configurations using development results under the task rules. If no labelled development set exists, agree a validation split from training data with the instructor and document it. Preserve source groups where available. Never use hidden test labels or repeated leaderboard feedback as a replacement for sound model selection.

Make the main comparison and ablations on identical development items. Report the official metric as primary, with justified secondary metrics, per-language/class/subtask slices, sample counts, and uncertainty where feasible. Inspect at least ten errors, or all errors if fewer. Document scorer version, dataset release, seeds, model revisions, preprocessing, and compute.

Keep development results, official evaluation results, and post-submission experiments clearly labelled. A missing official test score is “not yet available,” not zero. Do not invent a rank, submission receipt, acceptance decision, or experiment.

## Three-week milestones

| Due | Deliverable | Suggested effort |
|---|---|---:|
| Day 2 | Read the selected task, confirm assigned tracks, data access, and evaluation rules | 1 hour |
| Day 3 | Task card, research question, dataset audit, and compute plan | 1 hour |
| Day 5 | Reproducible baseline, scorer sanity check, paper outline | 2 hours |
| Day 8 | First improvement, development comparison, and inspected errors | 2–3 hours |
| Day 10 | Freeze the main experiment plan; draft task/data/method/setup sections | 2 hours |
| Days 11–13 | Two ablations, error analysis, submission-format validation, full paper draft | 2–4 hours |
| Day 14 | Peer-review exchange and reproducibility check | 1 hour |
| Day 15 | Revised course paper, code, prediction package, results, presentation, and defence | 1–2 hours |

Start writing in week 1 and fill the results section from logged runs. Keep the system small enough for the course: a strong sparse/linear baseline, a small pretrained encoder, prompting, or parameter-efficient adaptation can each support a good paper. Training a foundation model or sweeping many large models is unnecessary. Official evaluation and paper revisions after the course may require additional time; announce those dates separately.

## Course deadline versus SemEval deadlines

**If official evaluation is open:** follow the organiser's registration and prediction-submission procedure, respect submission limits, and retain the submission ID/receipt and the exact submitted system version. Include official scores in the paper only once released.

**If evaluation opens after the course:** submit a complete course paper based on labelled development experiments, a validated development-format rehearsal, and an inference/export command ready for the future test input. State that official test data/results are pending. Submit official predictions during the evaluation window and update the paper afterward. The course grade can be assigned from the completed in-course work; future milestones remain part of the participation plan.

**If a past edition is selected:** describe the work as a retrospective SemEval study. Use the published protocol and report any exposure to released test labels. Do not claim official participation or an official rank. The instructor must explicitly select this route; it is not the default replacement for live participation.

A platform outage or delayed data release should be documented with the instructor. Preserve the validated local package and the submission attempt where applicable. An external delay does not require fabricated evidence and should not penalize otherwise completed course work.

## System-description paper

The course requires a complete paper, not just a proposal or short project summary. Use the [paper-writing guide](SEMEVAL_PAPER.md) and the official ACL style linked there. The course planning target is **4–6 pages of main text, excluding references**, unless the selected edition's rules require another limit. This is a course target, not a statement of SemEval's universal page limit.

The paper should explain the task and chosen tracks, describe the system and resources, specify the experimental protocol, present measured comparisons and ablations, analyze errors, and state limitations. Include a self-contained example and a reproducible link or access recipe for permitted artifacts. List only actual contributions and results. The course paper is assessed whether or not it is later accepted by the workshop.

## Submission package

```text
project/
  README.md                 # task, setup, train, predict, evaluate, export commands
  task_specification.md     # completed course task card with official URLs/rules
  paper/
    system_description.pdf  # complete course paper in the selected ACL style
    source/                 # editable paper source and bibliography
  data_statement.md         # provenance, access, split use, limitations
  environment-lock.txt      # installed versions
  config.json               # seeds, model revisions, prompts, budgets
  src/                      # runnable code or notebooks
  predictions/              # development outputs and permitted official outputs
  submission/               # organiser-format predictions or development rehearsal
  results.csv               # split/track-labelled tables computed from runs
  experiment_log.md         # all compared configurations, including failed runs
  participation.md          # receipts if submitted; otherwise status and next dates
  contributions.md          # individual roles, collaboration and AI disclosure
```

Provide the editable source alongside the PDF. Do not redistribute restricted datasets or include model weights unnecessarily; provide an access/preparation recipe instead. Keep the course's analysis JSONL separate from the official submission file if schemas differ. Validate IDs, row counts, label ranges, required columns, ordering, and output encoding with the organiser's checker where available.

## Rubric (100 points)

| Criterion | Points | Full-credit evidence |
|---|---:|---|
| Task understanding and data protocol | 10 | Correct assigned tracks, allowed resources, split use, official metric |
| Baseline and system development | 20 | Runnable baseline, justified improvement, controlled resource choices |
| Evaluation and analysis | 25 | Measured comparison, two ablations, uncertainty where feasible, inspected errors |
| System-description paper | 25 | Complete, clear, well-cited paper whose claims match the recorded results |
| Reproducibility and submission readiness | 15 | Exact commands/configs, validated export, editable paper, participation evidence appropriate to the schedule |
| Presentation and individual understanding | 5 | Clear defence and documented contributions |

A careful negative result can receive full credit. Grade experimental quality, paper quality, and compliance with the task protocol; do not award marks simply for model size, leaderboard rank, or paper acceptance. If official evaluation is not yet open, assess readiness and development evidence under the same rubric.

## Presentation logistics

For up to five teams, use the 40-minute day-15 tutorial slot for six-minute talks and two-minute questions; assign the integration notebook as preparation. With more teams, run a poster rotation and schedule individual checks in the final independent-work period. Announce the arrangement before week 3 and keep the course to 30 contact hours.
