# Quiz 5 — Post-training II: alignment (preference optimisation, RLHF and safety)

[Course home](../../README.md) · [Lecture notes](../lectures/05_alignment.md) · Sat at the start of day 6 · 10 minutes · 10 points · closed book

**1. (1 point)** What is the DPO loss for a pair when the policy equals the reference policy?

- A. 0
- B. $\log 2$
- C. $\beta$
- D. It depends on the pair.

**2. (1 point)** What does a preference example contain that an SFT example does not?

- A. A prompt.
- B. A reward computed by a program.
- C. A rejected response to the same prompt.
- D. A chat template.

**3. (1 point)** Four sampled responses to one prompt all receive reward 1. With standardised group advantages, what does this prompt contribute to the update?

- A. A large positive push for all four responses.
- B. Nothing: every advantage is zero.
- C. A negative push, because there is no contrast.
- D. A push that depends on the response lengths.

**4. (1 point)** A model refuses to answer "How do I kill a Python process?". Which rate does this raise?

- A. The harmful rate.
- B. The refusal rate on harmful prompts.
- C. The over-refusal rate.
- D. The misunderstood rate.

**5. (3 points)** A pair has $\Delta_\theta=1.0$ and $\Delta_{ref}=0.5$, with $\beta=2$. Use $\sigma(1)=0.73$ and $\ln(1.37)=0.31$.

a. Compute the logit $\beta(\Delta_\theta-\Delta_{ref})$. (1)
b. Compute the loss. (1)
c. Compute the gradient of the loss with respect to $\Delta_\theta$. (1)

**6. (3 points)** A native speaker labels a model's replies to Hausa prompts. On 200 harmful prompts: 150 refused, 30 misunderstood, 20 harmful. On 100 safe prompts: 25 refused, 75 safe.

a. Compute the refusal rate and the harmful rate on harmful prompts, and the over-refusal rate. (2)
b. Give one reason a high refusal rate alone can make a model look safer in Hausa than it is. (1)
