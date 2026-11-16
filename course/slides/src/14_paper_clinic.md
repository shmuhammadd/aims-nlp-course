type: title
title: Paper clinic and peer review
subtitle: Day 14 · Week 3
footer: Shamsuddeen Hassan Muhammad | AIMS 2026

---
type: big
title: Who will read your system paper, and what do they want from it?

Notes: Take answers for one minute. Three readers: the task organisers, who summarise every system in the overview paper; a reviewer, who is usually another participant; and next year's teams, who want to know what to try and what to skip. All three want the method, the numbers and the split they came from. None of them wants a history of the Transformer.

---
title: Today the draft meets its first reader
subtitle: Where day 14 sits in the capstone

| When | What is due |
|---|---|
| Day 10 | Experiment plan frozen; task, data, method and setup drafted |
| Days 11–13 | Two ablations, error analysis, full draft |
| **Day 14** | **Peer review and reproducibility check** |
| Day 15 | Revised paper, package, six-minute talk and defence |
| 10–31 January 2027 | Official evaluation for Tasks 6, 7, 9 and 12 |
| February 2027 | System papers due (tentative) |

- Slides take 30 minutes today. You write and review for the rest.

Notes: Dates are from the task cards in SEMEVAL_TASK.md. Task 12 subtask 3 closes earlier, on 16 January 2027.

---
type: section
title: Part 1: Write
subtitle: Each section has one job

---
title: Each section answers one question
subtitle: Section names from the official SemEval system-paper guide

| Section | The reader asks | You show |
|---|---|---|
| Abstract | Should I read this? | Task, approach, one finding, one limit |
| Introduction | What did you try and learn? | Task and languages, strategy, main result, code link |
| Background | What goes in and out? | One input and output example, data sizes, your track |
| System overview | Could I rebuild it? | Baseline, your change, extra resources |
| Experimental setup | Could I rerun it? | Split use, tuning, seeds, versions, metric |
| Results | What works, and where does it fail? | Main score, ablations, error analysis |
| Conclusion | What is still open? | Limits and the next experiment |

Notes: The guide is at semeval.github.io/system-paper-template.html. It says the paper must stand on its own and that the system overview is usually the longest section. It also says a citation is enough for a widely used model: do not re-derive attention. A best-paper award goes to system papers with strong analysis.

---
title: The title names the team, the task and the method

### Pattern
**Team at SemEval-2027 Task N: what you did**

| | Title |
|---|---|
| Weak | Emotion Recognition with Transformers |
| Weak | Our Submission to the ArtEmotion Shared Task |
| Better | TeamName at SemEval-2027 Task 12: Late Fusion of XLM-R and ViT for Emotion in Paintings |

- Use the official ACL style files: `acl.sty` and the LaTeX template
- Course target: 4 to 6 pages of main text. Check the SemEval-2027 call for the real limit.
- Hand in the PDF and the editable source

Notes: The "better" title is an invented pattern, built from the organisers' baseline design (XLM-R plus ViT with concatenation). The style files are at github.com/acl-org/acl-style-files. Check the SemEval-2027 paper call for the page limit and for review or final mode.

---
title: An abstract written this week has no rank to report

Four sentences, in this order:

- **Task and track.** SemEval-2027 Task 12, subtask 1, Hausa, Yoruba and Swahili.
- **Approach.** What you built, in one clause a stranger could repeat.
- **Finding.** Your strongest measured result, with metric and split: "macro-F1 0.61 on development".
- **Limit.** One thing the evidence does not cover: one seed, three of 28 languages.

Evaluation runs 10 to 31 January 2027. Until scores are released, the abstract reports development results and says so. Add the rank in February, beside the finding and not in place of it.

Notes: The 0.61 is a placeholder to show the form, not a result. The SemEval guide asks for the rank and for what you discovered. An abstract whose only finding is "we ranked 7th" tells next year's team nothing they can use.

---
title: Four faults in student system papers

| Fault | What the reader sees | Fix |
|---|---|---|
| Rank-only abstract | "We placed 7th of 30" | State the method and one measured finding |
| Generic background | A page on attention and BERT | Cite it in one line. Spend the page on your system. |
| Unlabelled splits | "F1 = 0.61" | Name the metric, split, track and language in every caption |
| Numbers with no run | A score nobody can trace | Each number has a run ID in `experiment_log.md` |

- Mark your own draft now: one tick per fault you find

Notes: Give two minutes for the marking. Most drafts have at least two. The fourth fault is the serious one: a reviewer who cannot trace one number stops trusting the others.

---
title: One table becomes one paragraph in four moves

| Move | The sentence says |
|---|---|
| 1. Comparison | Which two rows, on which split, with which metric |
| 2. Size | The difference and its interval |
| 3. Mechanism | One reason the change could cause the difference |
| 4. Alternative | One other explanation, and the experiment that would rule it out |

- Move 2 decides how much of move 3 you may write. If the interval contains zero, say the systems are not separated and stop explaining.
- Move 4 is where reviewers look first

Notes: Write the four words on the board and leave them there for the sprint. The next two slides apply them to one invented paragraph.

---
title: Before: no claim a reader can check
subtitle: Invented teaching example for a Task 12 system

"Our multimodal system achieves strong performance on the task. Adding visual features clearly shows that images are important for emotion recognition in art. Table 2 shows the results. The model struggles with some emotions."

- Which systems are compared? On which split?
- "Strong" and "clearly" stand where the numbers should be
- The table is announced and never read
- "Some emotions": which, and how often?

Notes: Ask the class to find the faults before you reveal the bullets. Nobody wrote this paragraph. It is assembled from sentences that appear in many first drafts.

---
title: After: the same result in four moves
subtitle: Invented example. The two scores are the organisers' published baselines.

"On development data, the multimodal run (XLM-R plus ViT, concatenated) scored macro-F1 0.535 against 0.550 for the text-only run (Table 2). The difference is −0.015, with a 95% paired-bootstrap interval of −0.045 to +0.014, so the two runs are not separated. The caption may already carry what the annotator felt, which would leave the image little to add. An under-tuned fusion layer would give the same gap: both runs used one learning rate and one seed. Table 3 repeats the comparison with three seeds."

- Comparison, size, mechanism, alternative: find each one

Notes: The scores 0.535 and 0.550 are the organisers' subtask 1 baselines from the task card. The split label, the mechanism and the tuning details are invented for teaching. The interval is from notebook 14, whose synthetic predictions score 0.549 and 0.535.

---
title: Sprint: 20 minutes on your own draft

- Minutes 0 to 8: rewrite your abstract as four sentences
  - task and track, approach, finding with metric and split, one limit
- Minutes 8 to 20: rewrite the paragraph under your main results table in four moves
  - leave the interval blank if you have not computed it. The lab fills it.
- Open `experiment_log.md` beside the draft. Copy numbers from the log, not from memory.
- Stuck? Write the comparison sentence only, then the next one

Notes: Walk the room. Read abstracts over shoulders and ask one question each: "which split is that number from?" Stop the sprint on time.

---
type: section
title: Part 2: Measure
subtitle: Is the difference real?

---
title: The table keeps development and official scores apart
subtitle: Rows printed by notebook 14 from synthetic predictions

| System | Split | Metric | Score | Diff. to A [95% CI] | Runs |
|---|---|---|---|---|---|
| A: text-only | dev | macro-F1 | 0.549 | n/a | 1 run, seed 42 |
| B: multimodal | dev | macro-F1 | 0.535 | −0.014 [−0.045, +0.014] | 1 run, seed 42 |
| B: multimodal | official test | macro-F1 | pending | n/a | frozen config |

- Every row names its split. The official row says "pending" until scores are released.
- All development rows use the same items
- Anything you run after submitting goes in a separate, labelled table
- Print the rows from code. Do not type scores into LaTeX.

Notes: The template in SEMEVAL_PAPER.md has two more columns, for resource use and status. A missing official score is "pending", never zero and never a development number.

---
title: Macro-F1 has no per-item score to average

$$
F_1^{(k)} = \frac{2\,TP_k}{2\,TP_k + FP_k + FN_k} \qquad \mathrm{macro\ F_1} = \frac{1}{K}\sum_{k=1}^{K} F_1^{(k)}
$$

Eight items, gold 1 1 1 1 0 0 0 0.

| System | Predictions | F1, class 1 | F1, class 0 | Macro-F1 |
|---|---|---|---|---|
| A | 1 1 1 0 0 0 0 1 | 6/8 = 0.750 | 6/8 = 0.750 | 0.750 |
| B | 1 1 0 0 0 0 0 0 | 4/6 = 0.667 | 8/10 = 0.800 | 0.733 |

- Both systems get 6 of 8 right. Accuracy ties and macro-F1 does not.
- An interval for macro-F1 must recompute the whole metric on each resample

Notes: Do system B on the board. Class 1: TP 2, FP 0, FN 2. Class 0: TP 4, FP 2, FN 0. Task 12 ranks macro-F1 per language and then averaged; Task 9 ranks micro-F1.

---
title: Paired bootstrap by hand: three resamples

Draw eight item numbers with replacement. Score **both** systems on that draw.

| Items drawn | Macro-F1 A | Macro-F1 B | B − A |
|---|---|---|---|
| All eight, once each | 0.750 | 0.733 | −0.017 |
| 1 1 3 4 4 6 7 8 | 0.619 | 0.619 | 0.000 |
| 2 3 3 4 5 5 7 8 | 0.750 | 0.564 | −0.186 |
| 1 2 3 4 4 4 6 8 | 0.467 | 0.500 | +0.033 |

- Repeat 1,000 times. Sort the differences. The 25th and 975th values are the 95% interval.
- Paired means one draw for both systems. Hard items then hurt both.
- Eight items give a useless interval. The method is the same at 2,000.

Notes: Work the third row with the class. Gold is 1 1 1 1 0 0 0 0. System B on that draw predicts 1 0 0 0 0 0 0 0: class 1 has TP 1 and FN 3, so 2/5; class 0 has TP 4 and FP 3, so 8/11. The mean is 0.564. The method is in Koehn 2004 and Berg-Kirkpatrick et al. 2012.

---
title: On 2,000 items a 0.014 gap is inside the noise
subtitle: Lab result: 1,000 resamples of two synthetic systems
layout: stack

![](figures/14_bootstrap_hist.png)

- Paired interval −0.045 to +0.014. It contains zero, so the paper says "not separated".

Notes: Tutorial 14, seed 42. In the toy, B was built to be right about one point more often than A, and it measured 0.014 lower. A paragraph explaining why "the image hurts" would explain noise. The grey histogram is the unpaired mistake: its interval is 0.074 wide against 0.059.

---
title: Four times the items roughly halves the interval
subtitle: Nested subsets of one 8,000-item synthetic pool, same two systems
layout: stack

![](figures/14_interval_vs_n.png)

- At 250 items the interval excludes zero with a gap of +0.083. At 8,000 the gap is +0.025.

Notes: Computed in 14_paper_clinic_figures.py with seed 42. This is a new pool, so its 2,000-item point (+0.021) differs from the lab's 2,000 items (−0.014): two development sets of the same size disagree in sign. Ask how many items the teams' own development sets have, per language.

---
title: McNemar counts only the disagreements

- b = items right only under condition 1. c = items right only under condition 2.
- If the conditions were equal, each disagreement would be a fair coin flip

$$
p = \sum_{k=b}^{b+c} \binom{b+c}{k} \left(\frac{1}{2}\right)^{b+c}
$$

- By hand, b = 7 and c = 1: p = (8 + 1) / 256 = 0.035
- Task 12 uses this test. A multimodal run must beat its own decoy-image run at p < 0.05.
- Lab: b = 70, c = 39 gives p = 0.0019. The run passes.

Notes: One-sided, exact. C(8,7) = 8 and C(8,8) = 1. McNemar suits accuracy on paired items. For a difference in macro-F1 use the bootstrap. Dror et al. 2018 has a table of which test fits which metric.

---
title: Error analysis is a table with counts
layout: split

- Ten pasted examples show that errors exist
- A table shows where they are
  - choose categories: confusion pair, language, caption length
  - count every development error into one category
  - then read ten examples from the largest row
- Lab: awe and contentment make up 26% of system B's 888 errors
- Ask a speaker of the language to read the ten. Is the system wrong, or the label?

![](figures/14_confusions.png)

Notes: Synthetic errors from tutorial 14. The confusions were built in, so the figure shows the form of the analysis and nothing about paintings. PROJECT.md asks for at least ten inspected errors. The SemEval guide suggests a confusion matrix or hand-tagging a small sample.

---
title: Lab: 25 minutes

- Open `course/tutorials/14_paper_clinic.ipynb`. NumPy only.
- Sections 1 to 3, together: two systems, macro-F1, the paired bootstrap
  - Predict whether the interval contains zero before you run the cell
- Sections 4 to 6, in pairs: image-use counts, the error table, the results row
- "Your turn" is for tonight: `paired_bootstrap` and `mcnemar_exact`
- If you have development predictions for two systems, load them in place of section 1 and fill the blank in your paragraph

Notes: Teams without saved predictions use the synthetic data today and rerun on their own files tonight.

---
type: section
title: Part 3: Review
subtitle: Read a draft the way a stranger will

---
title: A review of 300 to 500 words has five parts

| Part | What you write |
|---|---|
| A supported claim | Quote it. Name the table row that supports it. |
| A claim that needs evidence | Quote it. Say what run or number is missing. |
| A reproducibility gap | The first thing you could not rerun: seed, split, version, prompt |
| A results or error-analysis issue | A split label, an interval, an uncounted error category |
| Three revisions, in order | The most useful change first |

- Check every number you quote against the authors' `experiment_log.md`
- Review the paper. Say "the draft does not state", not "you forgot".
- Leave the grammar alone unless it hides the meaning

Notes: The five parts are the day-14 exchange in SEMEVAL_PAPER.md. The supported claim comes first because it tells the authors what to keep.

---
title: Answer every point: changed, or kept and why

| Reviewer wrote | Authors answer |
|---|---|
| "Table 2 does not name the split." | Changed. Caption now reads "development set, subtask 1, Hausa". |
| "The gain of 0.02 has no interval." | Changed. Added a paired bootstrap; the interval contains zero, and section 5 now says so. |
| "Add a second pretrained encoder." | Kept. Outside our compute limit. Added to limitations. |

- One line per point, in the reviewer's order
- "Kept" is a valid answer when you give the reason
- Hand in the response with the revised paper on day 15

Notes: The rows are invented to show the form. Authors who answer every point in order get shorter second reviews.

---
title: Disclose AI help, and open every reference you cite

- ACL policy (2023): checking spelling and grammar needs no disclosure. Say where a model produced text or ideas. A model cannot be an author.
- The ACL Rolling Review checklist asks whether you used AI assistants in research, coding or writing
- In this course: record it in `contributions.md`, with the tool and what it did
- You answer for every sentence and every line of code in the defence tomorrow
- Open each paper you cite. Language models invent references that look real.
- Cite the task overview paper, the dataset, and each model and library you used

Notes: Sources: the ACL 2023 Policy on AI Writing Assistance and item E1 of the Responsible NLP Research checklist. SemEval-2027 may set its own rule; check the paper call. Task 12 teams also state the instructor's role as a co-organiser.

---
title: Peer review: 25 minutes

- Minutes 0 to 3: swap drafts with the team you were paired with. Swap logs too.
- Minutes 3 to 18: read once, then write the five parts
  - trace two numbers from the paper to the log
- Minutes 18 to 23: hand the review back and talk it through. Authors listen first.
- Minutes 23 to 25: authors write their three revisions for tonight
- The full review, 300 to 500 words, is due tonight with the exercise sheet

Notes: Pair teams on different tasks where you can: a reader who does not know the task finds the gaps in the task description. Keep time strictly.

---
type: big
title: Your paired interval for the main result contains zero. What does the abstract say?

Notes: It reports the comparison and says the two systems were not separated on this development set. Then it leads with what the paper did establish: an ablation, an error pattern, a cost saving. A careful negative result can earn full marks under the capstone rubric. It does not say "comparable performance" as if that were a win, and it does not drop the interval.

---
title: Readings

### Before the defence (20 minutes)
- SemEval. Guidelines for SemEval System Papers. Read all of it; it is one page.

### For your results section
- Dror et al., 2018. The Hitchhiker's Guide to Testing Statistical Significance in NLP.
- Berg-Kirkpatrick et al., 2012. An Empirical Investigation of Statistical Significance in NLP.
- Dodge et al., 2019. Show Your Work: Improved Reporting of Experimental Results.

### For reviewing and disclosure
- ACL Rolling Review. Reviewer Guidelines.
- ACL 2023 Policy on AI Writing Assistance.

Notes: Links are in the lecture notes.

---
title: Exit ticket

Write three lines and hand them in.

- The one claim in your paper that your reviewer could not trace to a run
- The interval for your main comparison, or the command you will run tonight to get it
- Your first revision, and who on the team does it

Quiz 14 opens the next session: six questions, ten minutes.
