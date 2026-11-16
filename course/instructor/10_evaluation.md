# Lecture 10 — instructor guide

Keep this file private if the cohort can see the repository. Change the numbers in quiz questions 5 and 6 and exercise 1 before reusing them for marks.

## Teaching the session

- Open with the 71.2 against 70.4 question and write the class's objections on the board. Tick each one off as its slide arrives.
- Give teams two minutes on the "four things" slide to fill the rows for their own task. Most cannot state the protocol. Send them to the task card.
- Do the BERTScore example on the board as a 2 × 3 grid: circle the best cell in each row for recall, then the best in each column for precision.
- Do micro and macro F1 from the counts table before showing the figure. Ask what happens if the system never predicts the rare frames: micro 0.714, macro 0.267.
- Common confusion: students read ECE as an error rate. It is a gap between confidence and accuracy, and both lab models have the same accuracy.
- Common confusion: McNemar's test is run on all items. Only the disagreements enter.
- The outcome table on the safety slide uses invented counts apart from the two refusal rates. Say so aloud.
- During the lab, read each team's frozen plan. Refuse a plan with no rule for choosing between runs.

## Practical questions

**1.** Yes. Calibration depends on the probabilities and accuracy only on which side of 0.5 they fall. In the lab both models score 0.732 and their ECE is 0.009 and 0.130.

**2.** A threshold chosen on the test items is fitted to them, so the reported coverage and selective accuracy are too good. It is the same selection effect as the 50-system leaderboard.

**3.** Show every pair in both orders and count verdicts that change with the order. In the lab 3 of 6 change.

**4.** The score of a deliberately wrong answer of the same type on the same question, the encoder and settings behind the scorer, and a native speaker's reading of a sample. Exact match beside BERTScore helps on one-word answers.

## Notebook

Solutions for the two stub cells are in [solutions/10_evaluation.py](solutions/10_evaluation.py). `python scripts/run_notebooks.py --only 10_evaluation` runs the notebook with those solutions in place of the stubs.

Reference outputs (seed 42): accuracy 0.7 and Brier 0.174 on the ten predictions; ECE 0.175 with five bins; coverage 0.3 and selective accuracy 1.0 at threshold 0.9. Simulated models: accuracy 0.732 for both, Brier 0.178 and 0.198, ECE 0.009 and 0.130; overconfident model at threshold 0.99 has coverage 0.22 and selective accuracy 0.908. Judge verdicts change on 0.5 of pairs; bootstrap interval 0.4 to 1.0. Leaderboard winner 0.771, fresh 0.699, standard error 0.032. McNemar p = 0.0207, 0.2517 and 0.0284. Micro-F1 0.735, macro-F1 0.511. Greedy F1: 1.00, 0.51, 0.68 and 0.03. Within-painting r: 0.206 and 1.000. Your-turn A prints micro 0.667 and macro 0.444; B prints 5.32 and 0.72 bits.

## Exercise sheet

**Exercise 1.** a. Brier $=(0.04+0.49+0.36+0.01)/4=0.225$. Predictions are 1, 1, 0, 0 against labels 1, 0, 1, 0, so accuracy is 0.5. Confidences are 0.8, 0.7, 0.6, 0.9; items 1 and 4 pass 0.75 and both are right: coverage 0.5, selective accuracy 1.0. b. Recall takes row maxima: $(0.9+0.7+0.5)/3=0.70$. Precision takes column maxima: $(0.9+0.7)/2=0.80$. $F=2(0.7)(0.8)/1.5=0.747$. One point for each part.

**Exercise 2.** Reference ECE for the overconfident model, bins over 0.5 to 1:

| Predictions | 5 bins | 10 bins | 20 bins |
|---|---|---|---|
| All 5,000 | 0.130 | 0.130 | 0.130 |
| First 100 | 0.242 | 0.274 | 0.290 |

On 5,000 predictions the bin count hardly matters. On 100 the estimate is about twice as large and grows with the number of bins, because small bins add noise that the absolute value turns into bias. Full marks need the table, the assertion, empty bins reported with count 0, and that observation.

**Exercise 3.** Expected findings by task. Task 6: expect the copied question and the fixed sentence to score well above zero, because raw BERTScore sits in a narrow band; exact match exposes them. Task 7: the fixed target-language sentence scores full compliance; relevance exposes it. Task 9: predicting only the most frequent frame keeps micro-F1 respectable and macro-F1 low. Task 12: the majority emotion gives low macro-F1; a constant 0.5 gives exactly 1 bit of log loss; a constant per painting has no defined correlation. Accept any measured table with a correct explanation of the metric's arithmetic.

Marking anchors. Derivation (2): one point per part, with working. Implementation (4): working code 2, test 1, reproducible run 1. Investigation (4): controlled design 1, measured results 1, explanation 1, limitation 1. Give no marks for numbers that do not appear in an output.

## Quiz 10 key

1. B. 2. C. 3. C. 4. D.

5. a. $F_A=2(30)/(60+10+10)=0.75$; $F_B=2(5)/(10+5+15)=0.333$. b. Macro $=(0.75+0.333)/2=0.542$. c. Pooled TP 35, FP 15, FN 25: micro $=70/(70+15+25)=0.636$.

6. a. Recall, row maxima: $(0.8+0.6)/2=0.70$. Precision, column maxima: $(0.8+0.6+0.3)/3=0.567$. (F1 is 0.626; not asked.) One point each. b. Any one of: each extra candidate token must match a reference token and a poor match lowers precision; a wrong answer of the same type has an embedding close to the reference; raw scores sit in a narrow band, so an unrelated answer still scores high.
