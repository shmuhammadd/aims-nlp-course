# Quiz 8 — Vision-language models

[Course home](../../README.md) · [Lecture notes](../lectures/08_vlm.md) · Sat at the start of day 9 · 10 minutes · 10 points · closed book

**1. (1 point)** In a LLaVA-style model, what does the projector do?

- A. It maps patch features from the vision encoder to the width of the language decoder's embeddings.
- B. It turns the question into patches.
- C. It computes the contrastive loss between the image and the caption.
- D. It selects 32 learned queries that summarise the image.

**2. (1 point)** A model encodes a 448×448 image with 14×14 patches. The side is doubled to 896. The number of patches is multiplied by

- A. 2
- B. 4
- C. 8
- D. 16

**3. (1 point)** A VLM answers "B" to "Which bar is tallest?" when shown a blank white image. What does this show?

- A. The model reads charts well.
- B. The image encoder is broken.
- C. The answer can come from a prior, so a correct "B" on a real chart is weak evidence of reading.
- D. The question is a false-premise question.

**4. (1 point)** In the decoy test, which rows decide the McNemar p-value?

- A. All rows.
- B. Rows that are right with both images.
- C. Rows that are wrong with both images.
- D. Rows that are right with one image and wrong with the other.

**5. (3 points)** A system is scored on 40 rows. It is right with both the real image and the decoy on 20 rows, right only with the real image on 6, right only with the decoy on 1, and wrong with both on 13.

a. Give its accuracy with the real image and with the decoy. (1)
b. Compute the one-sided exact McNemar p-value. Note that $2^7=128$. (1)
c. Does the run pass the image-use check at $p<0.05$? What would you change about the test set to learn more? (1)

**6. (3 points)** A chart shows three bars, A, B and C. The question is "How much taller is bar D than bar A?" and the model answers "2 units".

a. Score the answer on correctness, support and abstention, with one phrase for each. (2)
b. Name one image intervention that would tell you whether the model reads bar heights at all. (1)
