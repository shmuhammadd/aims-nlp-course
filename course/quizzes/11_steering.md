# Quiz 11 — Interpretability and cross-lingual steering

[Course home](../../README.md) · [Lecture notes](../lectures/11_steering.md) · Sat at the start of day 12 · 10 minutes · 10 points · closed book

**1. (1 point)** A linear probe reaches 98% accuracy for language identity at layer 6. What does this show?

- A. The model uses language identity at layer 6.
- B. Language identity can be recovered from layer-6 activations.
- C. Steering at layer 6 will change the output language.
- D. Layer 6 is where language identity is first computed.

**2. (1 point)** In a difference-of-means steering vector, why use parallel sentences in the two languages?

- A. They make the vector longer.
- B. They are needed for the forward hook to run.
- C. The content cancels in the subtraction and the language remains.
- D. They remove the need for a development set.

**3. (1 point)** Task 7 does not rank one of these methods. Which?

- A. Adding a vector to the residual stream at layer 12.
- B. Switching on a set of language-specific neurons.
- C. Writing "Answer in Swahili" in the system prompt.
- D. Turning up a sparse-autoencoder feature.

**4. (1 point)** A team raises α until every answer is in the target language, then stops. Which measurement did they skip?

- A. Language compliance.
- B. Relevance to the question.
- C. The norm of the steering vector.
- D. Probe accuracy.

**5. (3 points)** Let $h=(1,3)$, the source mean $(2,0)$ and the target mean $(0,2)$.

a. Compute $v$ and the projection of $h$ on $\hat v$. (1)
b. Compute $h'$ after adding with $\alpha=0.5$. (1)
c. Compute $h'$ after ablating $\hat v$. (1)

**6. (3 points)** Your additive steering gives compliance 1.00 and relevance 0.40 at the strength you chose. Name two changes to the method that could raise relevance (2), and say on which data you would choose between them (1).
