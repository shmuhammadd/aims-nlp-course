# Lecture 14 — Paper clinic and peer review

[Course home](../../README.md) · Week 3, day 4 · 120 minutes · [Slides](../slides/14_paper_clinic.pptx) · [Notebook](../tutorials/14_paper_clinic.ipynb) · [Exercises](../exercises/14_paper_clinic.md) · [Quiz](../quizzes/14_paper_clinic.md)

## Learning outcomes

- Turn one results table into a paragraph that states the comparison, the size of the difference with an interval, one mechanism and one alternative explanation.
- Compute a paired bootstrap interval for a difference in macro-F1, and an exact McNemar p-value from two counts.
- Write a 300–500-word review of another team's draft, and answer a review of your own.

## Session plan

Bring a full draft of your system paper and your `experiment_log.md`. This session is a workshop: 30 minutes of slides, 70 minutes of writing, measuring and reviewing.

| Minutes | Activity |
|---|---|
| 0–10 | Quiz 13 (efficient inference and quantisation) |
| 10–25 | Slides 1–11: the reader, the job of each section, the title, the abstract, four faults, the four-move paragraph |
| 25–45 | Slide 12: drafting sprint on your own abstract and results paragraph |
| 45–55 | Slides 13–20: the results table, macro-F1, the paired bootstrap, McNemar, error tables |
| 55–60 | Break |
| 60–85 | Slide 21 and [notebook 14](../tutorials/14_paper_clinic.ipynb): sections 1–3 together, sections 4–6 in pairs |
| 85–90 | Slides 22–25: the five-part review, the response, AI disclosure |
| 90–115 | Slide 26: peer-review exchange between paired teams |
| 115–120 | Slides 27 and 29: the closing question and the exit ticket |

## Teaching notes

### Who reads a system paper
A SemEval system paper has three readers. The task organisers summarise every system in the task overview paper. A reviewer, usually another participant, checks the draft. Next year's teams read it to decide what to try. All three want the method, the numbers and the split each number came from. The official [SemEval guide](https://semeval.github.io/system-paper-template.html) says the paper must stand on its own, so do not assume the reader knows the task. The guide also says there is a best-paper award for system papers with strong analysis.

### The job of each section
The guide names the sections. Each one answers a question the reader has.

| Section | The reader asks | You show |
|---|---|---|
| Abstract | Should I read this? | Task and track, approach, one measured finding, one limit |
| Introduction | What did you try and what did you learn? | The task and its languages, your strategy, the main result, a link to the code |
| Background (task and data) | What goes in and what comes out? | One input and output example, the data and its sizes, the track you entered |
| System overview | Could I rebuild it? | The baseline, your change, resources beyond the training data |
| Experimental setup | Could I rerun it? | How each split was used, tuning, seeds, library versions, the metric |
| Results | What works, and where does it fail? | The main score, ablations, error analysis |
| Conclusion and limitations | What is still open? | What the evidence does not cover, and the next experiment |

The system overview is usually the longest section. A citation is enough for a widely used model: the guide says so for logistic regression and LSTMs, and the same holds for the Transformer. A page of generic background takes space from your own system.

Use the official [ACL style files](https://github.com/acl-org/acl-style-files). The course target is 4–6 pages of main text, not counting references. Check the SemEval-2027 paper call for the page limit and for whether to compile in review or final mode.

### The title and the abstract
The title follows one pattern: **Team at SemEval-2027 Task N: what you did**. "Emotion Recognition with Transformers" names neither the team, the task nor the method. "TeamName at SemEval-2027 Task 12: Late Fusion of XLM-R and ViT for Emotion in Paintings" names all three.

Write the abstract as four sentences: the task and track, the approach, the strongest measured finding with its metric and split, and one limit. Evaluation for Tasks 6, 7, 9 and 12 runs 10–31 January 2027, so an abstract written this week has no rank to report. It reports a development result and says so. In February, add the rank beside the finding. An abstract whose only finding is a rank gives next year's team nothing to use.

### One table, one paragraph, four moves
1. **Comparison.** Which two rows, on which split, with which metric.
2. **Size.** The difference and its interval.
3. **Mechanism.** One reason the change could cause the difference.
4. **Alternative.** One other explanation, and the experiment that would rule it out.

Move 2 limits move 3. If the interval contains zero, say that the two systems are not separated and keep the explanation short.

The worked example below applies the four moves to an invented paragraph.

### A results table that keeps splits apart
Every row names its system, split, metric, score and the runs behind it. All development rows use the same items. The official row says "pending" until the organisers release scores; it never holds a development number, and a missing score is never zero. Experiments run after the submission go in a separate, labelled table. The full template, with columns for resource use and status, is in the [paper guide](../SEMEVAL_PAPER.md). Print the rows from code so that each number has a run behind it.

### Macro-F1 is not a mean over items
For class $k$, $F_1^{(k)}=2TP_k/(2TP_k+FP_k+FN_k)$, and macro-F1 is the mean of $F_1^{(k)}$ over the $K$ classes. Accuracy is a mean of per-item scores. Macro-F1 is a ratio of counts for each class, then a mean over classes, so no per-item score exists to average. In the lab, system A scores 0.556 on the first 1,000 items and 0.542 on the second 1,000; the mean of the halves is 0.5489 and the score of the whole set is 0.5490. An interval for macro-F1 has to recompute the whole metric on each resample.

Task 12 ranks macro-F1 computed per language and then averaged. Task 9 ranks micro-F1. Task 6 ranks BERTScore F1, which is a mean of per-item scores.

### The paired bootstrap
Draw $n$ item indices with replacement from the $n$ development items. Score both systems on that one draw and record the difference. Repeat 1,000 times. The 2.5% and 97.5% quantiles of the differences are the 95% interval (Koehn, 2004; Berg-Kirkpatrick et al., 2012).

"Paired" means one draw for both systems. Two systems fail on many of the same items, and a shared draw keeps that. In the lab the paired interval is 0.059 wide. Drawing separately for each system gives 0.074.

The lab's two synthetic systems score 0.549 and 0.535 on 2,000 items, and the interval for the difference is −0.045 to +0.014. It contains zero. The toy was built with system B right about one point more often than A, and B measured 0.014 lower. A paragraph explaining why B is worse would explain noise.

The interval shrinks slowly. On nested subsets of one 8,000-item synthetic pool, the width is 0.164 at 250 items, 0.078 at 1,000 and 0.029 at 8,000: four times the items roughly halves it. At 250 items that pool shows a gap of +0.083 with an interval that excludes zero; at 8,000 the gap is +0.025.

Resample the unit that was sampled. If several annotations share one painting, or several paragraphs share one article, resample paintings or articles. If you have only a public aggregate score and no predictions, you cannot compute an interval, and you should not invent one.

### McNemar's test
For accuracy on paired items, count the disagreements. Let $b$ be the items right only under the first condition and $c$ the items right only under the second. If the conditions were equal, each disagreement would be a fair coin flip, so the exact one-sided p-value is

$$p=\sum_{k=b}^{b+c}\binom{b+c}{k}\left(\tfrac12\right)^{b+c}.$$

With $b=7$ and $c=1$, $p=(8+1)/256=0.035$. Task 12 uses this test for its image-use check: a multimodal run must do better with the right painting than with a decoy, at $p<0.05$. In the lab, $b=70$ and $c=39$ give $p=0.0019$. McNemar's test fits accuracy. For a difference in macro-F1, use the bootstrap. Dror et al. (2018) list which test fits which metric.

### Error analysis with counts
Ten pasted examples show that errors exist. A table shows where they are. Choose categories (confusion pair, language, caption length, question type), count every development error into one category, and report the counts in one table. Then read ten examples from the largest row, with a speaker of the language, and ask whether the system or the label is wrong. In the lab, five of 56 confusion pairs hold 57% of system B's 888 errors, and awe and contentment alone hold 26%. Those confusions were built into the toy.

### Common faults in student system papers
- **A rank-only abstract.** State the method and one measured finding.
- **Generic Transformer background.** Cite it in one line.
- **Unlabelled splits.** "F1 = 0.61" is unreadable without the metric, split, track and language. Put all four in every caption.
- **Numbers with no run behind them.** Each number in the paper has a run ID in `experiment_log.md`.

### Citing, and disclosing AI assistance
Cite the task overview paper, the dataset, and each pretrained model and library you used. Open every paper you cite: language models produce references that look real and do not exist.

The [ACL 2023 policy](https://2023.aclweb.org/blog/ACL-2023-policy/) treats a spelling or grammar check like any other writing tool, with no disclosure needed. Where a model produced text or ideas, the authors say so, check the content and take responsibility for it. A model cannot be an author. The ACL Rolling Review [Responsible NLP Research checklist](https://aclrollingreview.org/responsibleNLPresearch/) asks whether AI assistants were used in research, coding or writing. SemEval-2027 may set its own rule, so check the paper call. In this course, record the tool and what it did in `contributions.md`. In tomorrow's defence you answer for every sentence and every line of code.

Task 12 teams also state in the paper that the course instructor is a co-organiser of the task.

### Reviewing a draft
The day-14 exchange is set out in the [paper guide](../SEMEVAL_PAPER.md). A review is 300–500 words and has five parts:

1. One claim the paper supports. Quote it and name the table row.
2. One claim that needs evidence. Quote it and say which run or number is missing.
3. One reproducibility gap: the first thing you could not rerun (a seed, a split, a version, a prompt).
4. One issue in the results or the error analysis.
5. Three revisions, the most useful first.

Check every number you quote against the authors' log. Write about the draft: "the caption does not name the split". Leave grammar alone unless it hides the meaning.

### Answering a review
Answer every point in the reviewer's order, one line each: "changed", with what the paper now says, or "kept", with the reason. "Kept, outside our compute limit, added to limitations" is a complete answer. Hand in the response with the revised paper on day 15.

## Worked example

**An invented teaching example.** The paragraph below describes a Task 12 system that no team built. The two scores are the organisers' published subtask 1 baselines (0.535 multimodal, 0.550 text-only macro-F1). The split label, the mechanism and the tuning details are invented. The interval is taken from notebook 14, whose synthetic predictions score 0.549 and 0.535.

**Before.**

> Our multimodal system achieves strong performance on the task. Adding visual features clearly shows that images are important for emotion recognition in art. Table 2 shows the results. The model struggles with some emotions.

The paragraph names no systems, no split and no number. "Strong" and "clearly" stand where the evidence should be. The table is announced and never read. "Some emotions" could be counted and is not.

**After.**

> On development data, the multimodal run (XLM-R plus ViT, concatenated) scored macro-F1 0.535 against 0.550 for the text-only run (Table 2). The difference is −0.015, with a 95% paired-bootstrap interval of −0.045 to +0.014, so the two runs are not separated. The caption may already carry what the annotator felt, which would leave the image little to add. An under-tuned fusion layer would give the same gap: both runs used one learning rate and one seed. Table 3 repeats the comparison with three seeds.

Sentence 1 is the comparison. Sentence 2 is the size. Sentence 3 is one mechanism. Sentences 4 and 5 are the alternative and the experiment that tests it.

**The bootstrap by hand.** Eight items have gold labels 1 1 1 1 0 0 0 0. System A predicts 1 1 1 0 0 0 0 1 and system B predicts 1 1 0 0 0 0 0 0. Both get six right.

| Items drawn | Macro-F1 A | Macro-F1 B | B − A |
|---|---|---|---|
| All eight, once each | 0.750 | 0.733 | −0.017 |
| 1 1 3 4 4 6 7 8 | 0.619 | 0.619 | 0.000 |
| 2 3 3 4 5 5 7 8 | 0.750 | 0.564 | −0.186 |
| 1 2 3 4 4 4 6 8 | 0.467 | 0.500 | +0.033 |

For B on the full set, class 1 has TP 2, FP 0, FN 2, so $F_1=4/6$; class 0 has TP 4, FP 2, FN 0, so $F_1=8/10$; the mean is 0.733. On the third row B predicts 1 0 0 0 0 0 0 0 against gold 1 1 1 1 0 0 0 0: class 1 gives $2/5$, class 0 gives $8/11$, and the mean is 0.564.

## Guided tutorial

**Question:** Two systems were scored on the same 2,000 items. Is the difference in macro-F1 larger than the noise, and how do you report it?

The notebook builds synthetic predictions for a text-only run and a multimodal run over the eight Task 12 emotions. Students compute macro-F1 from the confusion matrix, run a paired bootstrap for the difference, count the disagreements for the image-use check, print an error table and print a results row in Markdown and LaTeX. They then write `paired_bootstrap` and `mcnemar_exact` against tests. The predictions are synthetic and say nothing about a real system.

**Shared-task path:** load your own development predictions for the baseline and the main system in place of section 1 and rerun the notebook. Use your task's metric: the per-language mean of macro-F1 for Task 12, micro-F1 for Task 9, the mean of per-item BERTScore F1 for Task 6, per-item compliance and relevance for Task 7.

## Practical questions

1. Your paired interval for the main comparison contains zero. Write one sentence the paper may say and one it may not.
2. Several annotations in your development set describe the same painting. What do you resample, and does the interval get wider or narrower?
3. A reviewer says a claim needs evidence, and the experiment would take two days you do not have. What do you write in the response and in the paper?

## After class

Complete the [exercise sheet](../exercises/14_paper_clinic.md) in 45–60 minutes. Send your 300–500-word review to the authors and the instructor tonight. Revise your paper and write the response for day 15. [Quiz 14](../quizzes/14_paper_clinic.md) opens day 15.

## Readings

**Required**
- [Guidelines for SemEval System Papers](https://semeval.github.io/system-paper-template.html). One page: the section list this session follows. Read it with your draft open.

**For your results section**
- [The Hitchhiker's Guide to Testing Statistical Significance in Natural Language Processing](https://aclanthology.org/P18-1128/) (Dror et al., 2018). Which test fits which metric; read the decision table first.
- [An Empirical Investigation of Statistical Significance in NLP](https://aclanthology.org/D12-1091/) (Berg-Kirkpatrick et al., 2012). The paired bootstrap for metrics such as F1 and BLEU.
- [Statistical Significance Tests for Machine Translation Evaluation](https://aclanthology.org/W04-3250/) (Koehn, 2004). The short original account of bootstrap resampling for a system comparison.
- [Show Your Work: Improved Reporting of Experimental Results](https://arxiv.org/abs/1909.03004) (Dodge et al., 2019). Why a results row needs the tuning budget behind it.

**For formatting, reviewing and disclosure**
- [ACL style files](https://github.com/acl-org/acl-style-files). The LaTeX and Word templates.
- [ARR Reviewer Guidelines](https://aclrollingreview.org/reviewerguidelines). How ACL reviewers are asked to read a paper; skim the list of common review faults.
- [ACL 2023 Policy on AI Writing Assistance](https://2023.aclweb.org/blog/ACL-2023-policy/). What to disclose and why a model cannot be an author.
