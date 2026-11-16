# Exercise sheet 7 — Image-text representations: CLIP and SigLIP

[Course home](../../README.md) · [Lecture notes](../lectures/07_contrastive.md) · [Notebook](../tutorials/07_contrastive.ipynb)

Budget: 45–60 minutes. This sheet is worth 10 points. Submit your working, your changed notebook, the numbers you measured and 150–250 words on what they show.

## 1. Derive (2 points)

Two images and two captions have the logit matrix $S=\begin{pmatrix}2&0\\1&1\end{pmatrix}$, with rows for images and matched pairs on the diagonal.

a. Compute the two image-to-text terms, the two text-to-image terms and the symmetric loss.
b. Show that $\partial L/\partial S=(P^{\text{row}}-I+P^{\text{col}}-I)/2N$, where $P^{\text{row}}$ and $P^{\text{col}}$ are the row-wise and column-wise softmax of $S$. Evaluate all four entries for this $S$.

## 2. Implement (4 points)

In notebook 7:

a. Write `sigmoid_loss(S, b)` for the SigLIP loss with scale 1 and bias `b`. Assert that it returns 2.8561 at `b = 0` and 1.2134 at `b = -2` on the worked matrix `S3`, to four decimals.
b. Add a false negative. Append a second copy of the red-left image and caption to the training batch, so the batch has six pairs and two are identical. Train with the same seed, steps and learning rate. Report the final loss beside the original 0.076 and explain the difference with one calculation.
c. Change the loss so that both red-left captions count as positives for both red-left images. Do it twice: average the log-probabilities of the positives, then take the log of their summed probability. Report the final loss for each and explain why one of them changes nothing.

## 3. Investigate (4 points)

Choose one.

**A. Your task's images (Tasks 6, 9 and 12).** Take 50 development items from your task. Write down, before running anything, a frozen-encoder baseline for it: the encoder, the features you would save, the classifier or ranker on top, the metric from your task card and the split. Then run it if you have the compute, or run only the tokeniser check: tokenise the 50 captions or questions with the text encoder's tokeniser and report the mean number of pieces per word for each language you plan to enter. State which of your numbers are measured.

**B. Seeds (any team).** The lab's blue-right image picks the right caption with seed 42. Rerun sections 1–3 for seeds 0 to 19 and count how often image-to-text and text-to-image are correct for blue-right. Compare each count with chance and say what the lab's single run did and did not show. Then propose one change to the training images that would give the encoder evidence about blue-right without showing it a blue-right image, and test it.

## Submission checklist

- Seed, NumPy version and every changed setting are recorded.
- Each number in your write-up appears in a notebook output.
- You say which results are measured and which are your explanation.
- Collaboration and any AI assistance are disclosed. You can explain every line you submit.
