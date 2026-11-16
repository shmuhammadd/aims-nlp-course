# Quiz 7 — Image-text representations: CLIP and SigLIP

[Course home](../../README.md) · [Lecture notes](../lectures/07_contrastive.md) · Sat at the start of day 8 · 10 minutes · 10 points · closed book

**1. (1 point)** A CLIP-style model is trained with a batch of 8 image-caption pairs. In the image-to-text loss, how many negatives does each image have?

- A. 1
- B. 7
- C. 8
- D. 56

**2. (1 point)** What does lowering the temperature $\tau$ do to the softmax over one row of the similarity matrix?

- A. It makes the distribution sharper, so the largest logit takes more of the probability.
- B. It makes the distribution flatter.
- C. It changes which caption has the highest score.
- D. It normalises the embeddings to length 1.

**3. (1 point)** How does the SigLIP loss differ from the CLIP loss?

- A. It uses one encoder for both images and text.
- B. It needs no negative pairs.
- C. It scores each image-caption pair as its own yes/no problem, with no softmax across the batch.
- D. It removes the temperature and the normalisation.

**4. (1 point)** A team saves frozen CLIP features for its paintings and fits a logistic regression on them. Which weights change during training?

- A. The image encoder's.
- B. The image encoder's and the text encoder's.
- C. The patch projection only.
- D. The logistic regression's only.

**5. (3 points)** Two images and two captions have the logit matrix $S=\begin{pmatrix}2&0\\3&1\end{pmatrix}$, with rows for images and matched pairs on the diagonal. Use $\log(1+e^{-2})=0.127$, $\log(1+e^{2})=2.127$ and $\log(1+e)=1.313$.

a. Compute the image-to-text loss. (1)
b. Compute the text-to-image loss and the symmetric loss. (1)
c. Give recall@1 in each direction. (1)

**6. (3 points)** A training batch holds two photographs of the same market. One has the caption "a market" and the other "people at a market".

a. What does the diagonal-only contrastive loss do to these two pairs? (1)
b. Give one change to training that removes the problem. (1)
c. How should recall@k be computed on a test set where several captions fit one image? (1)
