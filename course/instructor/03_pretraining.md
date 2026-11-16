# Lecture 03 — instructor guide

Keep this file private if the cohort can see the repository. Change the numbers in quiz questions 5 and 6 and exercise 1 before reusing them for marks.

## Teaching the session

- Open with the question slide. Most students say "the first one" or "the more likely one". Leave it open and return to it on slide 27.
- Derive the gradient on the board: $L=-z_y+\log\sum_j e^{z_j}$, and the derivative of the log-sum-exp is the softmax. Then do the one-step worked example with logits $(\ln5,\ln3,\ln2)$.
- Part 2 is the place to save time. Data engineering returns on day 6 and efficiency on day 13. Keep $6ND$ and the Chinchilla table; shorten mixture of experts if the clock is at 33 minutes.
- Give decoding its full 20 to 25 minutes. Do the $T=0.5$ row on the board before showing the table: square, add, divide.
- Ask the class to predict greedy decoding of the bigram model before the lab. Few expect a loop.
- The configuration block on slide 26 is a sketch of what to record. It is not the syntax of a library.
- Common confusion: temperature is thought to change which token is most likely. It never changes the order.
- Common confusion: top-p is thought to keep tokens with probability above $p$. It keeps the smallest set whose total reaches $p$.
- Common confusion: beam search is assumed to give better text because it gives higher probability. The lab's "the lay" is the counter-example.
- Common confusion: the best held-out loss is reported as the result. It was used for selection.

## Practical questions

**1.** Any two of: overfitting, where training loss falls and held-out loss rises; distribution mismatch, where the training text differs from the task's text; objective mismatch, where next-token prediction is not what the task scores.

**2.** The larger model can score by recalling the duplicated documents, so the comparison overstates what size buys. Remove the overlap, or report scores on the clean subset, before comparing.

**3.** It has one token of context, no depth and no attention. It isolates the objective, the gradient and the loop, and tells you nothing about what a large model can do.

**4.** No. The two numbers differ by the decoding rule and by sampling noise. The temperature-1.0 result is one draw unless it is a mean over seeds. Rerun one system under the other's settings before comparing.

## Notebook

Solutions for the two stub cells are in [solutions/03_pretraining.py](solutions/03_pretraining.py). `python scripts/run_notebooks.py --only 03_pretraining` runs the notebook with those solutions in place of the stubs.

Reference outputs (seed 42):

- Section 1: vocabulary 22, 93 training pairs, initial development loss 3.095 ($\ln22=3.091$).
- Section 2: at step 300, training loss 1.093 and development loss 1.077.
- Section 3: sample "tmodie clage la"; $6ND=1.2\times10^{18}$.
- Section 4: held-out loss 2.415 at step 40, 2.417 at step 80, 2.840 at step 400; training loss 0.797 at step 400.
- Section 5: $T=0.5$ gives 0.734, 0.183, 0.066, 0.014, 0.003 (1.15 bits); $T=2$ gives 0.348, 0.246, 0.191, 0.130, 0.085 (2.17 bits); top-k 2 gives 0.667, 0.333 (0.92 bits); the model's entropy is 1.83 bits.
- Section 6: greedy "the lab lab lab …" at −31.76; with penalty 1.3 "the lab s clay" at −11.21; beam width 3 "the lay" at −5.20.
- Section 7: top-p 0.8 gives 0.556, 0.278, 0.167 (1.42 bits) and keeps 3 tokens of the peaked distribution and 4 of the flat one. Best held-out loss 2.394 at step 220 for learning rate 0.5, 2.395 at step 60 for 2.0, 2.420 at step 20 for 8.0.

## Exercise sheet

**Exercise 1.** a. $L=-z_y+\log\sum_k e^{z_k}$, so $\partial L/\partial z_j=p_j-\mathbf 1[j=y]$. The sum over $j$ is $\sum_jp_j-1=0$. Check: $(0.6, 0.3, -0.9)$, sum 0. b. $6\times8.03\times10^9\times5\times10^8=2.41\times10^{19}$ operations; at $10^{13}$ per second that is $2.41\times10^6$ seconds, or 27.9 days. Left out: attention over long contexts, the optimiser, data loading, idle time, evaluation runs, and the gap between peak and sustained throughput. One point for each part.

**Exercise 2.** Full marks need the four-method table, the temperature-0.01 assertion and a reasoned choice. Expected pattern: greedy gives one string, the loop from the lab, which never ends; lower temperature gives longer runs of the common words and fewer endings; temperature 1.0 gives the most varied strings and the lowest log-probability per character. Accept any numbers the student's output supports. A student who seeds once outside the loop and reports 200 identical strings has not sampled.

**Exercise 3.** Marks are for sourced facts and a concrete contamination plan. From the task cards: the Task 12 baseline is XLM-R plus ViT (masked objective for the text encoder), and its rule forbids annotations of evaluation paintings from other sources, including the original ArtELingo release; Task 7 ranks `meta-llama/Llama-3.1-8B-Instruct` and `CohereLabs/tiny-aya-global`, both causal. "Not found, looked in the repository README and the baseline script" is an acceptable answer for a setting. An unsourced parameter count or setting earns no mark.

Marking anchors. Derivation (2): result 1, working 1. Implementation (4): working code 2, test 1, reproducible run 1. Investigation (4): model facts with sources 1, settings or label read-out 1, contamination sources 1, plan and rule 1. Give no marks for numbers that do not appear in an output or a cited source.

## Quiz 3 key

1. B. 2. C: $6\times10^9\times10^{10}$. 3. B. 4. C.

5. a. $(0.4, 0.3)/0.7=(4/7, 3/7, 0, 0)=(0.571, 0.429, 0, 0)$. b. Cumulative mass 0.4, 0.7, 0.9: three tokens are needed to reach 0.75, giving $(4/9, 3/9, 2/9, 0)=(0.444, 0.333, 0.222, 0)$. c. Squares 0.16, 0.09, 0.04, 0.01, sum 0.30: $(16/30, 9/30, 4/30, 1/30)=(0.533, 0.300, 0.133, 0.033)$.

6. a. $(0.6, 0.3, -0.9)$. b. Zero: the probabilities sum to one and so does the one-hot target. c. Keep the weights from step 60. Report the final score on a test set that was not used to choose the step.
