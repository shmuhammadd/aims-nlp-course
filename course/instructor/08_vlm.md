# Lecture 8 — instructor guide

Keep this file private if the cohort can see the repository. Change the numbers in quiz question 5 and exercise 1 before reusing them for marks.

## Teaching the session

- Open with the question slide and take a vote. Return to it at the decoy slide.
- Write the loss on the board, cross out $z_{\text{image}}$, and ask whether the loss stays low for "What colour is the sky?". This is the argument for every test in part 2.
- Do the McNemar sum on the board before showing the figure: $\binom{16}{12}=1820$, $\binom{16}{13}=560$, $\binom{16}{14}=120$, $\binom{16}{15}=16$, $\binom{16}{16}=1$; total 2,517; $2^{16}=65{,}536$; $p=0.038$. Then ask what changes if two rows move from $b$ to $c$.
- Common confusion: students compare the two accuracies with an unpaired test. Point at the 2×2 table. The 84 agreeing rows say nothing about the image.
- Common confusion: a system that passes the decoy test is taken to be correct. It only shows dependence on the image. `parser` after a 20-pixel shift depends on the image and is wrong.
- The architecture table is from the three papers. LLaVA-1.5 uses a two-layer MLP; say so if asked.
- The SmolVLM error is one recorded run. Do not let the class generalise from it.
- Conflict of interest: the instructor co-organises Task 12. Describe the decoy check only as the task page does.
- Safety block: this is the image-input block of the safety thread. `course/SAFETY.md` lists it under session 12, the old number; it is now session 8. Use only the benign instruction.

## Practical questions

**1.** It measures how much of the score comes from the question, the caption or the label distribution. In the lab, `prior` scores 0.467 with no image at all. Accuracy with the real image cannot show that.

**2.** Yes. `prior` answers B and is right on 46.7% of charts without reading one. A real model can guess, recall a common value or infer the answer from the question.

**3.** The parser has the chart's layout written into it: bar columns at 12, 40 and 68 and 4 pixels per unit. A 20-pixel shift makes it answer C with heights 0, 3, 7. A change of style, colour or scale breaks it too. A VLM has no such knowledge and no such brittleness.

## Notebook

Solutions for the two stub cells are in [solutions/08_vlm.py](solutions/08_vlm.py). `python scripts/run_notebooks.py --only 08_vlm` runs the notebook with those solutions in place of the stubs.

Reference outputs (seed 42): heights 3, 7, 5; answers B, None, A for original, blank and counterfactual; shifted heights 0, 3, 7 and answer C; patch counts 576 and 1,152. Three systems, real / decoy / abstention on blank: `prior` 0.467 / 0.467 / 0.000; `weak` 0.447 / 0.387 / 0.000; `parser` 1.000 / 0.333 / 1.000. Decoy test: lecture table $b=12$, $c=4$, $p=0.0384$; `prior` 0, 0, 1; `weak` 76, 58, 0.071; `parser` 200, 0, $6.2\times10^{-61}$. Same label on real and decoy: 1.000, 0.393, 0.333. Tokens at patch 14: 256, 1,024 and 4,096 for sides 224, 448 and 896.

The answer to the question after test A: `weak` reads the chart 15% of the time, so its expected gain is about 0.15 × (1 − 1/3) = 0.10 in accuracy. Students find the size needed in exercise 2c.

## Exercise sheet

**Exercise 1.** a. $24\times24=576$ and $48\times24=1152$; after 2×2 pooling, 144 and 288. Real counts differ through resizing or cropping to a fixed size, tiling, padding and special tokens. b. $n=11$; $p=(\binom{11}{9}+\binom{11}{10}+\binom{11}{11})/2^{11}=(55+11+1)/2048=0.033$. It passes. One point each.

**Exercise 2.** a. `np.mean(real_labels == decoy_labels)` and a comparison with 0.99; `prior` gives 1.000 and `parser` 0.333. b and c. Values depend on how the student threads the random generator, so mark against the student's own output. Expect the p-value to fall as the probability rises and as the number of questions grows; at 0.15 most runs pass by 600 to 1,200 questions. Working code 2, assertions 1, reproducible table 1.

**Exercise 3.** A. Full marks need real and decoy accuracy, $b$, $c$, $p$, the unchanged-label share and a verdict that follows from them. A system can fail both rules, as `weak` does. B. Look for honest reporting of a small gap and five analysed items. C. The table must show `prior` failing abstention on false-premise questions and `parser` abstaining only if the student adds a rule for a missing bar. Do not treat all refusals as successes. Give no credit for model results that were not run.

Marking anchors. Derivation (2): result 1, working 1. Implementation (4): working code 2, test 1, reproducible run 1. Investigation (4): controlled design 1, measured results 1, explanation 1, limitation 1. Give no marks for numbers that do not appear in an output.

## Quiz 8 key

1. A. 2. B. 3. C. 4. D.

5. a. Real: $(20+6)/40=0.65$. Decoy: $(20+1)/40=0.525$. b. $b=6$, $c=1$, $n=7$: $p=(\binom{7}{6}+\binom{7}{7})/2^7=(7+1)/128=0.0625$. c. It does not pass, although accuracy differs by 12.5 points. Only 7 rows are discordant. Use more rows, or choose decoys that differ more from the real image, so that more rows become discordant.

6. a. Correctness: wrong, since there is no bar D and so no true difference. Support: none, the image holds no evidence for the answer. Abstention: failed, the model should have said the chart has no bar D (2 for three sound phrases, 1 for two). b. Swap or change the heights and ask a valid question such as "Which bar is tallest?"; a model that reads heights changes its answer. A blank image also earns the mark.
