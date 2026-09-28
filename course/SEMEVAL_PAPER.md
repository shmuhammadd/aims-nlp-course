# Writing the SemEval system-description paper

[Course home](../README.md) · [Capstone](PROJECT.md) · [Task specification](SEMEVAL_TASK.md)

## Purpose and official references

Write a self-contained account of what your team built, how it was evaluated, and what the results support. The official [SemEval system-paper guide](https://semeval.github.io/system-paper-template.html) recommends explaining the task, method, experiment setup, results, and system behavior, including ablations and errors. Use the [official ACL style files](https://github.com/acl-org/acl-style-files) and check the selected edition's current paper call for its specific requirements. These references were checked on 28 September 2026; no task, deadline, or page limit has been selected for this class yet.

## Course writing plan

Aim for 4–6 main-text pages excluding references for course planning, subject to the selected edition's actual limit. This length is a course choice. Start with an editable ACL template, keep a bibliography from day 3, and add each experiment to your log before writing a claim about it. Submit both the PDF and editable source. This guide is a writing scaffold, not a finished scientific paper or a substitute for measured results.

Suggested title pattern: **TeamName at SemEval-YEAR Task N: A Specific Description of Your Approach**. Replace the identifiers only after task selection. A retrospective study should identify itself as retrospective.

## Drafting worksheet

| Section | What to write for this course | Evidence to prepare |
|---|---|---|
| Abstract | Task/track, central approach, strongest supported finding, one scope limit | Verified numbers with split labels; no invented official rank |
| Introduction | Research question and why your comparison matters | Task citation and a short input/output example |
| Task and data | Assigned languages/tracks, labels, split sizes, allowed resources | Completed task card and dataset audit |
| System | Baseline, main change, preprocessing, model/adapter/prompt decisions | Pipeline diagram or pseudocode where useful; checkpoint revisions |
| Experiments | Selection protocol, scorer, hyperparameters, seeds, resource budget | Configuration and run log |
| Results and analysis | Main comparison, two ablations, error categories, uncertainty | Tables generated from saved predictions; at least ten inspected errors if available |
| Limitations and conclusion | What the evidence establishes and what remains untested | Data, language coverage, compute, and evaluation constraints |
| References and supporting material | Proper attribution and details needed to rerun the system | Bibliography, code/access recipe, editable source |

Treat this table as a course worksheet; adapt section organisation to the selected venue's requirements. Do not expand generic Transformer background at the expense of explaining your own system and analysis.

## Results table template

| System | Split / track | Official metric name | Score | Runs / seed | Resource measurement | Status |
|---|---|---|---|---|---|---|
| Baseline | Development / assigned track | Fill from task card | Enter measured value | Record | Measured or not measured | Local experiment |
| Main system | Same development items | Same metric | Enter measured value | Record | Measured or not measured | Local experiment |
| Ablation 1 | Same development items | Same metric | Enter measured value | Record | Measured or not measured | Local experiment |
| Ablation 2 | Same development items | Same metric | Enter measured value | Record | Measured or not measured | Local experiment |
| Submitted system | Official evaluation / assigned track | Official metric | Only once released | Frozen config | Record if measured | Submitted / results pending / released |

If official evaluation has not occurred, omit the official-score row or mark it pending. An organiser-format development rehearsal is not an official submission. Keep post-submission improvements in a separate clearly labelled comparison.

For non-additive metrics such as macro-F1, recompute the full metric for every paired bootstrap resample rather than averaging per-item “F1 contributions.” Respect grouped evaluation units when dependence exists. Do not invent confidence intervals if you only have a public aggregate score without the predictions/labels needed to estimate them.

## Day-14 peer review

Exchange complete drafts. Each reviewer returns 300–500 words covering: one clearly supported claim, one claim needing evidence, one reproducibility gap, one issue in the results/error analysis, and three prioritized revisions. Check every quoted number against a recorded run. Authors submit a short response explaining what they changed or why they retained a decision.

## Before submission

Confirm that the task/track identifiers, author list, citations, metric names, split labels, and tables are correct. Verify that every result is reproducible or explicitly attributed to an external source. Follow the selected edition's rules for page limit, review mode, acknowledgments, and AI-assistance disclosure. Agree authorship and contributions with the people involved; writing a course paper does not establish workshop acceptance.

Retain the exact paper source, system version, prediction file, and receipt for each external submission. If the official deadline follows the course, day 15 still requires a complete development-results paper and a dated plan for evaluation and revision.
