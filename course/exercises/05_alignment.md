# Exercise sheet 05 — Post-training II: alignment (preference optimisation, RLHF and safety)

[Course home](../../README.md) · [Lecture notes](../lectures/05_alignment.md) · [Notebook](../tutorials/05_alignment.ipynb)

Budget: 45–60 minutes. This sheet is worth 10 points. Submit your working, your changed notebook, the numbers you measured and 150–250 words on what they show. The optional model extension is not needed for full marks.

## 1. Derive (2 points)

a. Differentiate $L=-\log\sigma(\beta(\Delta_\theta-\Delta_{ref}))$ with respect to $\Delta_\theta$. Evaluate the derivative at $\Delta_\theta=\Delta_{ref}$.
b. Compute $L$ for $\Delta_\theta=0.9$, $\Delta_{ref}=0.4$ and $\beta=1$.

## 2. Implement (4 points)

In section 2 of notebook 05, change the annotators to eight who prefer A and two who prefer B. Train to convergence for $\beta$ = 0.5, 1, 2 and 5, and say how you checked convergence. Report the final $\theta$, the preference probability $\sigma(\beta\theta)$ and the response probability $\sigma(\theta)$ in one table. Add an assertion that the preference probability is within 0.01 of 0.8 for every $\beta$. Explain in two sentences what $\beta$ controls.

## 3. Investigate (4 points)

a. In section 4, train the third preference set with 0, 10, 50 and 200 of the safe prompts that sound risky, keeping the other two kinds at 200 each. Tabulate the refusal rate on harmful prompts and both over-refusal rates. State what each added batch of prompts buys and what it costs.

b. Write five safe prompts that sound risky, in an African language you speak, with an English gloss for each. Follow the XSTest idea: the wording sounds dangerous and the request is harmless. Say who would label a model's replies, with which four labels, and how you would compute the over-refusal rate. Do not write harmful prompts.

## Submission checklist

- Seed, NumPy version and every changed setting are recorded.
- Each number in your write-up appears in a notebook output.
- You say which results are measured and which are your explanation.
- Collaboration and any AI assistance are disclosed. You can explain every line you submit.
