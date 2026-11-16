# Quiz 4 — Post-training I: prompting, instruction tuning and LoRA

[Course home](../../README.md) · [Lecture notes](../lectures/04_adaptation.md) · Sat at the start of day 5 · 10 minutes · 10 points · closed book

**1. (1 point)** In response-only supervised fine-tuning, what happens to the prompt tokens?

- A. They are removed from the input.
- B. They stay in the input and their labels are set to −100.
- C. They stay in the input and their labels are set to the padding ID.
- D. They are moved after the response.

**2. (1 point)** A LoRA adapter starts with $A$ random and $B=0$. At the first training step:

- A. Both gradients are zero.
- B. The gradient of $A$ is zero and the gradient of $B$ is not.
- C. The gradient of $B$ is zero and the gradient of $A$ is not.
- D. The model's output already differs from the base model's.

**3. (1 point)** A team tries 30 prompts on the test set and reports the best score. What is wrong with the number?

- A. Nothing, if all 30 prompts are listed.
- B. It is too low, because prompts interfere with each other.
- C. It is too high, because it is the best of 30 noisy scores on the data it is reported on.
- D. It cannot be compared with a fine-tuned model.

**4. (1 point)** The same four demonstrations in a different order give a different few-shot accuracy. Which explanation fits the evidence from the lecture?

- A. The model's weights change with each prompt.
- B. The tokenizer splits the demonstrations differently.
- C. The model leans towards the labels nearest the end of the prompt.
- D. Longer prompts are always less accurate.

**5. (3 points)** A rank-4 adapter sits on a 1024×2048 matrix, with $\alpha=8$.

a. How many parameters does the adapter train? (1)
b. What percentage of the matrix is that? (1)
c. What is the scale $s$, and what does the layer compute at step 0? (1)

**6. (3 points)** One training row has input IDs `[5, 6, 7, 8, 30, 31, 2, 0]`. The prompt is the first four tokens, 2 is the end-of-sequence token and 0 is padding.

a. Write the labels for response-only fine-tuning. (2)
b. What does the tuned model do wrong if the label for token 2 is also −100? (1)
