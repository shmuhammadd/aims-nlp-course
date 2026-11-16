# Lecture 7 — instructor guide

Keep this file private if the cohort can see the repository. Change the numbers in quiz question 5 and exercise 1 before reusing them for marks.

## Teaching the session

- Open with the task table and ask each team to name its row. Task 7 teams have no images; tell them the loss and the frozen-feature recipe return on day 9 (speech encoders) and day 12 (retrieval).
- Do row 2 of the 3×3 example on the board before showing the table: $e+e^2+e^3=2.718+7.389+20.086=30.193$, $\log 30.193=3.408$, minus 2 gives 1.408. Then ask which direction will have the smaller loss and why.
- Draw the two towers and the $N\times N$ grid. Shade the diagonal. Students who see the grid find the row and column softmax easy.
- Ask the class to predict the step-0 loss in the lab. Most say $\log 5=1.609$. The run prints 2.54, because normalised random embeddings give random cosines and the temperature multiplies them by 5.
- Common confusion: a softmax over prompts is read as a probability that the class is present. Ask what the model returns for a photo of a goat when the prompts are "cat" and "dog".
- Common confusion: students read the lab's correct blue-right match as composition. Show the margin (0.14) and the seed count (3 of 20).
- The zero-shot and frozen-feature code slides are sketches. The course has not run them on a pretrained model.

## Practical questions

**1.** The dot product of unnormalised vectors grows with their length, so a long vector matches everything. After normalisation the score is a cosine between −1 and 1 and the temperature alone sets the logit scale. Guard against zero-length vectors.

**2.** It treats the second correct caption as a negative and pushes it away from the image. Remove duplicates from the batch, or mark all correct cells as positives and average their log-probabilities. In evaluation, count a hit when any relevant item is in the top k.

**3.** No. The softmax sums to 1 over the prompts that were written, whatever the image shows. The value moves with the temperature, the wording and the number of prompts. Calibrate on labelled development data before reading it as a probability.

## Notebook

Solutions for the two stub cells are in [solutions/07_contrastive.py](solutions/07_contrastive.py). `python scripts/run_notebooks.py --only 07_contrastive` runs the notebook with those solutions in place of the stubs.

Reference outputs (seed 42): loss 2.5377 at step 0 and 0.0763 at step 200; image-to-text top-1 1.0 and text-to-image top-1 0.833; blue-right row 1.67 for its own caption against 1.53 for green-right; column "blue right" 2.36 for the blue-left image. Worked matrix: image-to-text terms 0.1698, 1.4076, 0.0659 (mean 0.5478); text-to-image terms 0.1698, 0.5514, 0.3266 (mean 0.3493); symmetric loss 0.4485; sigmoid loss 2.8561 at $b=0$ and 1.2134 at $b=-2$. Frozen features: 1.000 on seen combinations and 0.750 on blue-right; raw pixels 1.000 and 0.033. Recall table at σ = 0.3: image-to-text recall@1 0.655 and recall@5 0.895; recall@1 on the worked matrix is (0.667, 1.0).

## Exercise sheet

**Exercise 1.** a. Rows: $\log(1+e^{-2})=0.1269$ and $\log 2=0.6931$, mean 0.4100. Columns: $(2,1)$ gives $\log(1+e^{-1})=0.3133$ and $(0,1)$ gives $0.3133$, mean 0.3133. Loss 0.3616. b. For row $i$, $\partial(-\log P^{\text{row}}_{ii})/\partial S_{ij}=P^{\text{row}}_{ij}-\delta_{ij}$; the column term is the same along columns; each mean carries $1/N$ and the average carries $1/2$. With $P^{\text{row}}=\begin{pmatrix}0.881&0.119\\0.5&0.5\end{pmatrix}$ and $P^{\text{col}}=\begin{pmatrix}0.731&0.269\\0.269&0.731\end{pmatrix}$ the gradient is $\begin{pmatrix}-0.097&0.097\\0.192&-0.192\end{pmatrix}$. One point for the loss, one for the derivation and the entries.

**Exercise 2.** a. `-(log_sigmoid((2*np.eye(N)-1)*(S+b))).sum()/N`. b. Final loss 0.307 against 0.076. The two red-left rows have identical logits, so each gives at most probability 1/2 to its own caption and contributes at least $\log 2=0.693$ in each direction. Two of six rows at 0.693 put a floor of $2\times0.693/6=0.231$ under the loss. c. Averaging the log-probabilities of the two positives gives 0.307 again: the two columns are identical, so the average of two equal log-probabilities is the same number, still capped at $\log(1/2)$. The log of the summed probability gives 0.076, because the two positives can now share a probability of 1. Reference values use seed 42, 201 steps and learning rate 0.1. Working code 2, the assertion in (a) 1, a reproducible run with the calculation 1.

**Exercise 3.** A. Full marks need a baseline that names encoder, features, classifier, metric and split; a clear line between what was run and what was planned; and per-language tokeniser counts if no model was run. Do not reward invented scores. B. With seeds 0 to 19 and `W` drawn first from `default_rng(seed)`, image-to-text is correct for blue-right on 3 seeds and text-to-image on 3. Chance with six candidates is 1/6, or 3.3 of 20. The single run showed nothing about composition. Good changes: add pixel noise or small shifts so that every pixel gets some gradient; use an encoder that shares weights across positions, such as patches with a shared projection. Marks follow the anchors below.

Marking anchors. Derivation (2): result 1, working 1. Implementation (4): working code 2, test 1, reproducible run 1. Investigation (4): controlled design 1, measured results 1, explanation 1, limitation 1. Give no marks for numbers that do not appear in an output.

## Quiz 7 key

1. B. 2. A. 3. C. 4. D.

5. a. Row 1: $\log(e^2+e^0)-2=\log(1+e^{-2})=0.127$. Row 2: $\log(e^3+e^1)-1=\log(1+e^{2})=2.127$. Image-to-text loss $(0.127+2.127)/2=1.127$. b. Column 1 holds $(2,3)$ with the match at 2: $\log(e^2+e^3)-2=\log(1+e)=1.313$. Column 2 holds $(0,1)$ with the match at 1: $\log(1+e)-1=0.313$. Text-to-image loss $0.813$. Symmetric loss $(1.127+0.813)/2=0.970$. c. Image-to-text: row 1 right, row 2 wrong (3 > 1), so 0.5. Text-to-image: caption 1 retrieves image 2 (wrong), caption 2 retrieves image 2 (right), so 0.5.

6. a. Each photo's other caption is treated as a negative, so the loss lowers the similarity of two image-caption pairs that are correct (1). b. Remove duplicate captions from a batch, or mark both cells as positives (1). c. Count a hit when any relevant item is in the top k, and state that rule (1).
