# Lecture 14 — instructor guide

Keep this file private if the cohort can see the repository. Change the numbers in quiz question 5 and exercise 1 before reusing them for marks.

## Teaching the session

- Pair the teams for review before class, across tasks where you can. A reader who does not know the task finds the holes in the task description. Ask each team to bring a printed or shared draft and its `experiment_log.md`.
- Keep the slides to 30 minutes. Part 1 is 11 slides in 15 minutes: spend the time on the two paragraph slides and let the section table stand as a reference.
- Write the four moves on the board (comparison, size, mechanism, alternative) and leave them up for the whole session.
- Show the "before" paragraph and ask the class for its faults before revealing the bullets. Say that it is invented and that the two scores in the "after" version are the organisers' baselines.
- Do one bootstrap resample on the board with the eight-item example: row 3 of the table. Students who compute one resample by hand stop averaging per-item scores.
- During the sprint and the exchange, walk the room and ask one question per team: "which run produced this number?"
- Common confusion: an interval that contains zero is read as "the systems are equal". The interval −0.045 to +0.014 allows B to be 4.5 points worse. The closing question is there for this.
- Common confusion: students resample each system separately. The grey histogram on slide 17 shows the cost.
- Common confusion: McNemar is applied to macro-F1. It tests paired right-or-wrong outcomes.
- If a team has no full draft, they review as normal and use the sprint to write the results section. Record it; the day-14 milestone in PROJECT.md is the exchange.

## Practical questions

**1.** May say: "B scored 0.014 lower than A on the development set (95% paired interval −0.045 to +0.014); the two are not separated." May not say: "adding the image hurts", or "A and B perform equally", since the interval allows a gap of 0.045.

**2.** Resample paintings, and take all the annotations of each drawn painting. Annotations of one painting are correlated, so the interval usually gets wider. An item-level interval would be too narrow.

**3.** Response: "Kept. The experiment needs two days of compute we do not have. We have weakened the claim and added the experiment to limitations." Paper: restate the claim as what was measured, and name the missing experiment in the limitations.

## Notebook

Solutions for the two stub cells are in [solutions/14_paper_clinic.py](solutions/14_paper_clinic.py). `python scripts/run_notebooks.py --only 14_paper_clinic` runs the notebook with those solutions in place of the stubs.

Reference outputs (seed 42): both right 783, both wrong 537, only A right 351, only B right 329; macro-F1 0.549 (A) and 0.535 (B), accuracy 0.567 and 0.556; halves of A 0.556 and 0.542; paired interval −0.045 to +0.014 (width 0.059), unpaired −0.050 to +0.023 (width 0.074); B beats A in 14.5% of resamples; decoy macro-F1 0.519, b = 70, c = 39, labels unchanged 0.924; 888 errors, top pair amusement → excitement 138, top five pairs 504 (56.8%); `paired_bootstrap` with seed 0 gives −0.042 to +0.015 for macro-F1 and −0.035 to +0.014 for accuracy; McNemar p = 0.0019.

The discussion cell repeats practical questions 1 and 2. Its third question (two labels often confused): the system may lack the feature that separates them, or annotators may not separate them either, which shows up as low agreement on those items.

## Exercise sheet

**Exercise 1.** a. A: class 1 has TP 2, FP 1, FN 1, so F1 = 4/6 = 0.667; class 0 has TP 4, FP 1, FN 1, so F1 = 8/10 = 0.800; macro-F1 0.733; accuracy 6/8 = 0.75. B: class 1 has TP 3, FP 2, FN 0, so 6/8 = 0.750; class 0 has TP 3, FP 0, FN 2, so 6/8 = 0.750; macro-F1 0.750; accuracy 0.75. Equal accuracy, B − A = +0.017 in macro-F1. b. The draw has gold 1 1 1 0 0 0 0 0. A predicts 1 1 1 0 0 0 0 0, macro-F1 1.000. B predicts 1 1 1 1 1 0 0 0, macro-F1 0.750. B − A = −0.250, the opposite sign to the full set. One point for part a, one for part b.

**Exercise 2.** Reference values with the stated seeds: per-language macro-F1 for A 0.554, 0.576, 0.541, 0.523 and for B 0.535, 0.511, 0.543, 0.543 (506, 500, 476 and 518 items); averaged scores 0.548 and 0.533; stratified interval for B − A about −0.045 to +0.015 when the indices are drawn with `rng.choice(group, len(group))` per language in order. Accept any interval within 0.01 of these ends. Two points for working code, one for the assertion, one for a run that reproduces. Good answers note that the per-language scores swing by up to 0.065 between the systems on about 500 items each, which is the same noise seen at a smaller size.

**Exercise 3.** One point each for: an interval from 1,000 paired resamples on identical items, with the resampled unit named; an error table with at least three counted categories; a paragraph with all four moves whose explanation is no stronger than the interval allows; run IDs for every number. A team using the synthetic data scores at most 3. Give no marks for numbers that do not appear in an output or a log.

Marking anchors. Derivation (2): result 1, working 1. Implementation (4): working code 2, test 1, reproducible run 1. Investigation (4): controlled design 1, measured results 1, explanation 1, limitation 1.

## Peer review

Mark the review as complete or incomplete for the day-14 milestone: five parts present, 300–500 words, at least two numbers traced to the log. The quality of the paper is marked on day 15 under the capstone rubric. Read the three prioritised revisions of each review before the defence; they make good questions.

## Quiz 14 key

1. B. 2. B. 3. C. 4. C.

5. a. b = 6, c = 1, so n = 7. p = [C(7,6) + C(7,7)] / 2^7 = (7 + 1)/128 = 0.0625. One point for the set-up, one for the value. b. No: 0.0625 > 0.05. Seven disagreements cannot pass unless all seven favour the right painting (1/128 = 0.008).

6. Any three, one point each: the metric and the split (and track or language); that both scores come from the same items; an interval or test for the difference of 0.03; the number of runs or seeds; one mechanism; one alternative explanation; the run or table the numbers come from. Also accept "remove 'clearly'" if a reason is given.
