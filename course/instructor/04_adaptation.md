# Lecture 04 — instructor guide

Keep this file private if the cohort can see the repository. Change the numbers in quiz questions 5 and 6 and exercise 1b before reusing them for marks.

## Teaching the session

- Open with the 500-captions question and take a vote. Vote again at the decision table (slide 20).
- Part 1 has 20 minutes. Spend most of it on the order figure and the development-against-test table. Ask the class to predict the spread before showing the figure.
- Derive the two LoRA gradients on the board from $Y=X(W+sAB)$. Then set $B=0$ and ask which gradient vanishes. Then set both to zero.
- Count parameters for 4096×4096 at rank 8 with the class: $8\times8192=65{,}536$.
- Common confusion: "response-only" is read as "remove the prompt". The prompt stays in the input; only its labels are masked.
- Common confusion: LoRA is taken to cut memory by the same 99.6% as it cuts trainable weights. The base weights and the activations are still there.
- Common confusion: a prompt is treated as free of tuning. Any prompt chosen by looking at scores has been tuned on that data.
- The toy in-context classifier has recency built in by hand. Say so once. It shows what the bias does to a score. It is no evidence that a given model has the bias.

## Practical questions

**1.** With both factors at zero, $\partial L/\partial A=sGB^\top=0$ and $\partial L/\partial B=sA^\top G=0$, so nothing trains. One random factor gives the other a non-zero gradient, and a zero factor keeps $W'=W$ at the start.

**2.** No. The model needs the prompt as context. Keep the tokens and set their labels to −100.

**3.** The base model ID and revision, the tokenizer and chat template, the adapter config and weights, software versions and generation settings.

**4.** Report the test score of the chosen ordering, once, and say that 24 orderings were tried on 40 development items. Expect the test score to be lower than 0.825: in the lab it is 0.795. One development item is 2.5 points.

## Notebook

Solutions for the two stub cells are in [solutions/04_adaptation.py](solutions/04_adaptation.py). `python scripts/run_notebooks.py --only 04_adaptation` runs the notebook with those solutions in place of the stubs.

Reference outputs (seed 42): base test MSE 0.0541 with 40 adapter and 96 full parameters; test MSE 0.0539 at step 0, 0.0041 at step 100 and $1.3\times10^{-9}$ at step 400; labels `[-100 -100 -100 20 21 2 -100 -100]`; $\gamma=1$ gives 0.714 for all 24 orderings; $\gamma=0.5$ gives minimum 0.483, mean 0.584, maximum 0.795, with the share answered +1 between 0.002 and 1.000; the ordering chosen on development data is (1, 3, 0, 2) with labels +1, −1, +1, −1, development accuracy 0.825 and test accuracy 0.795; the worst ordering is (2, 3, 0, 1) with labels −1, −1, +1, +1. Your-turn A prints supervised positions `[3 5 2]`; B prints a share of 0.00390625.

The rank question after section 2: a rank-1 adapter fits the larger singular direction of the true change (singular values 0.608 and 0.327) and leaves the smaller one, with held-out MSE 0.015.

## Exercise sheet

**Exercise 1.** a. Let $E=Y-T$ and $G_Y=2E/(n\,d_{out})$. Then $G=\partial L/\partial W'=X^\top G_Y$, $\partial L/\partial A=sGB^\top=sX^\top G_YB^\top$ and $\partial L/\partial B=sA^\top G=sA^\top X^\top G_Y$. With $B=0$ the first is zero. The second is $sA^\top X^\top G_Y$, non-zero when $A$ is random and the base model has error. b. $16\times(768+3072)=61{,}440$. The matrix has $768\times3072=2{,}359{,}296$ entries, so the share is 2.60%. One point for each part.

**Exercise 2.** Reference held-out MSE, with the adapter's $A$ drawn from a separate generator so that the data are the same for every rank; students' rank-1 numbers will match to two digits and the others in order of magnitude.

| Seed | Base | Rank 1 (20) | Rank 2 (40) | Rank 4 (80) | Rank 8 (160) |
|---|---|---|---|---|---|
| 42 | 0.0541 | 0.0154 | below $10^{-8}$ | below $10^{-8}$ | below $10^{-8}$ |
| 43 | 0.0290 | 0.0124 | below $10^{-6}$ | below $10^{-7}$ | below $10^{-8}$ |
| 44 | 0.0429 | 0.0057 | below $10^{-6}$ | below $10^{-6}$ | below $10^{-7}$ |

Rank 1 cannot fit a rank-2 change. Ranks 2, 4 and 8 all fit it; rank 2 does so with the fewest parameters. Rank 8 has 160 parameters against 96 in the full matrix, so it is larger than what it adapts. Full marks need the table from the student's own run, the assertion, and that observation. A student who lets `r` change the true rank as well has not kept the comparison controlled: at most 2 of 4.

**Exercise 3.** a. Reference:

| $\gamma$ | Minimum | Mean | Maximum |
|---|---|---|---|
| 1 | 0.714 | 0.714 | 0.714 |
| 0.8 | 0.571 | 0.696 | 0.788 |
| 0.5 | 0.483 | 0.584 | 0.795 |
| 0.2 | 0.483 | 0.534 | 0.751 |

The spread grows as $\gamma$ falls. At $\gamma=0.2$ the last demonstration decides almost every answer. The worst orderings end with two demonstrations of the same label; the best alternate the labels. b. Look for: a closed label list taken from the task card, balanced demonstrations from the training split and not from development or test data, a parsing rule written before any run, invalid replies scored as wrong, and loss on the response and end token only.

Marking anchors. Derivation (2): result 1, working 1. Implementation (4): working code 2, assertion 1, reproducible run 1. Investigation (4): measured table 1, pattern stated 1, prompt and parsing rule 1, correct loss positions 1. Give no marks for numbers that do not appear in an output.

## Quiz 4 key

1. B. 2. B. 3. C. 4. C.

5. a. $4\times(1024+2048)=12{,}288$. b. $1024\times2048=2{,}097{,}152$; $12{,}288/2{,}097{,}152=0.586\%$. c. $s=\alpha/r=8/4=2$; with $B=0$ the layer computes $XW$, the base model's output.

6. a. `[-100, -100, -100, -100, 30, 31, 2, -100]`. One point for masking the four prompt positions and the padding, one for keeping 30, 31 and 2. b. It never learns to emit the end token, so its answers do not stop.
