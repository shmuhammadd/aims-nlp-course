# Lecture 06 — instructor guide

Keep this file private if the cohort can see the repository. Change the numbers in quiz questions 5 and 6 and exercise 1 before reusing them for marks.

## Teaching the session

- The first hour is full: 15 minutes on models, 15 on measurement, 20 on building data. If time runs short, cut the five-pipelines table (slide 6) to one minute and keep the kappa derivation.
- Do the 90/100 and 3/10 example on the board before showing slide 10. Then ask for a second system with the same overall score and a better Hausa score.
- Derive kappa on the board from the 2×2 table. Write the row and column totals first, then $p_e$ as the sum of the products of the label rates. Then show the rare-class table and let the class compute $p_e=0.887$.
- Part 3 is your own work. Tell the class what took longest in AfriSenti and AfriHate, and one guideline that had to be rewritten after a pilot. The slides state only what the abstracts state, so the first-hand detail has to come from you.
- Common confusion: macro-F1 over classes and the mean over languages are mixed up. Task 12 does both: macro over classes within a language, then the mean over languages.
- Common confusion: a narrow bootstrap interval is read as proof the test set is representative.
- Common confusion: 92% agreement is taken as good annotation whatever the class balance.
- Common confusion: kappa is read as accuracy. Two annotators can agree and both be wrong.
- The Hausa phrase on slide 5's notes, "Na gode sosai", means thank you very much. Check any further examples you add yourself.

## Practical questions

**1.** The model sees the same few examples many times and can memorise them. Look at unique coverage and at the development curve for that language.

**2.** No. Translated text and naturally code-switched text come from different distributions; test on text people wrote.

**3.** Class balance and language balance are separate. Macro-F1 handles the first within one test set. Report it per language, with support and an interval, to handle the second.

**4.** The class balance and kappa, kappa by slice (for example code-switched posts), the disagreements read with the annotators, and a revised guideline. At 92% agreement with a rare class, kappa can be near 0.3.

## Notebook

Solutions for the two stub cells are in [solutions/06_multilingual.py](solutions/06_multilingual.py). `python scripts/run_notebooks.py --only 06_multilingual` runs the notebook with those solutions in place of the stubs.

Reference outputs (seed 42): both systems 0.845 overall; equal-language 0.60 for A and 0.78 for B. Paired interval over all items $[-0.045, 0.055]$ around 0.0; Hausa only 0.4 with $[0.1, 0.7]$; resampling authors $[-0.091, 0.073]$. Six-item example: accuracy 0.667, macro-F1 0.472. Sampling at $\alpha=0.5$: 0.706, 0.223, 0.071 with repeats 0.71, 2.23, 7.06; at $\alpha=0$ repeats 0.33, 3.33, 33.3. Kappa: balanced table $p_o=0.700$, $p_e=0.500$, $\kappa=0.400$; rare class 0.920, 0.887, 0.291. Simulated annotators: kappa 0.577 on all 600 tweets, 0.738 on 424 in one language, 0.189 on 176 code-switched; all agree 0.607, two of three 0.360, no majority 0.033; one annotator 0.842, majority 0.952. Your-turn A prints 0.593; B prints ha 0.472, yo 0.733, mean 0.603.

## Exercise sheet

**Exercise 1.** a. Overall $56/75=0.7467$; equal-language $(0.9+0.4+0.6)/3=0.6333$. b. $p_o=(30+55)/100=0.85$. A says "offensive" 40 times and B 35 times, so $p_e=0.40\times0.35+0.60\times0.65=0.14+0.39=0.53$ and $\kappa=(0.85-0.53)/0.47=0.68$. One point for each part.

**Exercise 2.** Reference with the given cell and the leave-out rule: system A scores 0.800, 0.671, 0.427 (mean 0.633); system B scores 0.762, 0.733, 0.738 (mean 0.745). A paired interval for B − A in the mean over languages, from one reference run, is about $[0.03, 0.20]$; students' intervals differ in the second decimal with their own resampling order. The interval excludes zero, and the gain comes from "yo" with 30 items, while A is ahead on "ha". A good answer says B is better on the language average, says that the claim rests on 30 items, and reports each language. The rule for an absent class must be stated; either rule is acceptable if it is applied to both systems. Working code 2, paired resampling within language 1, stated rule and claim 1.

**Exercise 3.** a. Full marks need counts from the team's files, not from the task card, and the majority-class score computed with the task metric. b. Look for independent labelling, the three kappas with the 30-item table, and causes of disagreement that name something concrete, such as two emotions that overlap or a caption that needs the image. Kappa from 30 items is noisy; a student who says so earns the limitation mark. c. Assess language variety, who wrote and labelled the data, the licence as stated by the organisers, and one limit. For Task 12 the paper must also state that the course instructor co-organises the task.

Marking anchors. Derivation (2): result 1, working 1. Implementation (4): working code 2, verification 1, reproducible run 1. Investigation (4): counts and baseline 1, pilot kappas 1, causes of disagreement 1, data statement and limitation 1. Give no marks for numbers that do not appear in an output or a file.

## Quiz 6 key

1. C. 2. B. 3. B. 4. C.

5. a. $p_o=(20+50)/100=0.70$. b. A says positive 30 times and B 40 times: $p_e=0.3\times0.4+0.7\times0.6=0.12+0.42=0.54$. c. $\kappa=(0.70-0.54)/(1-0.54)=0.16/0.46=0.35$.

6. a. $200/300=0.667$. b. $(0.80+0.60+0.20)/3=0.533$. c. Equal-language accuracy, or both, with the per-language counts and scores beside it; the overall figure is dominated by English and hides Yoruba at 0.20. Accept either headline number if the per-language table is required.
