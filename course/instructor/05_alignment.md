# Lecture 05 — instructor guide

Keep this file private if the cohort can see the repository. Change the numbers in quiz questions 5 and 6 and exercise 1b before reusing them for marks.

## Teaching the session

- Write the definition on the board at the start and leave it there: alignment is making a model's behaviour match what people want and what is safe. Under it, write "Task 7, day 11: alignment of representations across languages".
- Walk the pipeline figure (slide 4) left to right and ask which signal each stage uses. Come back to it when Part 3 starts.
- Do the worked DPO pair on the board: $2\times0.3=0.6$, $e^{-0.6}=0.549$, $\log 1.549=0.4375$. Then derive the gradient $\beta(\sigma(z)-1)$.
- Ask the class to predict the three bar groups on slide 20 before showing them.
- Common confusion: $\beta$ is taken for a learning rate. It sets how far from the reference the policy must move to satisfy a pair.
- Common confusion: the preference probability $\sigma(\beta\theta)$ and the policy's response probability $\sigma(\theta)$. Slide 11 separates them: 0.70 against 0.60.
- Common confusion: a high refusal rate is read as "safe". Slide 19 has the counter-example, a system that refuses everything.
- Common confusion: "alignment" in Task 7. Point to the board.
- Keep harmful prompt text out of the room's shared documents. The exit ticket asks for a safe prompt that sounds risky.

## Practical questions

**1.** Yes. Preferences may reward style, length or a wrong belief. Measure factuality on a separate held-out set.

**2.** Every centred reward is zero, so there is no difference within the group to learn from. This holds when all are right and when all are wrong.

**3.** The update favours the response that was rejected. Check a sample of pairs by eye and test the loader on one pair with known labels.

**4.** Any two of: the over-refusal rate on safe Hausa prompts; how many replies were misunderstood or unintelligible; whether the refusals are fluent Hausa; who labelled the replies; the harmful rate on localised prompts and on literal translations.

## Notebook

Solutions for the two stub cells are in [solutions/05_alignment.py](solutions/05_alignment.py). `python scripts/run_notebooks.py --only 05_alignment` runs the notebook with those solutions in place of the stubs.

Reference outputs (seed 42): DPO loss 0.598 after the first step and 0.051 at step 50, with P(A) 0.525 and 0.813; the 70/30 split gives preference probability 0.700 and response probability 0.604; advantages $(-1,1,1,-1)$ and $(0,0,0,0)$; worked-example loss 0.4375. Share refused, in the order harmful, plain safe, safe but sounds risky: refusals only 0.999, 0.112, 0.881; plus plain safe 0.976, 0.017, 0.559; plus safe that sounds risky 0.919, 0.012, 0.079. Your-turn A prints 0.4741; B prints a last row of −1.342, −0.447, 0.447, 1.342.

The question after section 1: with $\beta=0.5$ the first gradient is $-0.25$ in place of $-1$, so P(A) rises more slowly. The question after section 3: a task the model almost never solves gives all-zero groups and no gradient, so training needs easier prompts or a graded reward.

## Exercise sheet

**Exercise 1.** a. With $z=\beta(\Delta_\theta-\Delta_{ref})$, $dL/dz=\sigma(z)-1$, so $\partial L/\partial\Delta_\theta=\beta(\sigma(z)-1)$. At equality $z=0$, $\sigma=0.5$ and the derivative is $-\beta/2$. b. $z=0.5$; $L=\log(1+e^{-0.5})=\log(1.6065)=0.4741$. One point for each part.

**Exercise 2.** The minimum is at $\sigma(\beta\theta)=0.8$, so $\theta=\ln 4/\beta$.

| $\beta$ | $\theta$ | $\sigma(\beta\theta)$ | $\sigma(\theta)$ |
|---|---|---|---|
| 0.5 | 2.773 | 0.800 | 0.941 |
| 1 | 1.386 | 0.800 | 0.800 |
| 2 | 0.693 | 0.800 | 0.667 |
| 5 | 0.277 | 0.800 | 0.569 |

The notebook's 200 steps are not enough for $\beta=0.5$ (preference probability 0.706) or $\beta=1$ (0.794); students must train longer, and the assertion catches it. The preference probability is the same for every $\beta$. The policy's own probability of A is closer to the reference's 0.5 when $\beta$ is large. Working code 2, assertion and convergence check 1, explanation of $\beta$ 1.

**Exercise 3.** a. Reference, share refused:

| Risky-sounding safe prompts in the set | Harmful | Plain safe | Safe, sounds risky |
|---|---|---|---|
| 0 | 0.976 | 0.017 | 0.559 |
| 10 | 0.971 | 0.013 | 0.302 |
| 50 | 0.950 | 0.016 | 0.155 |
| 200 | 0.919 | 0.012 | 0.079 |

Ten prompts cut over-refusal by almost half. Each further batch buys less and costs some refusal of harmful prompts. Students should slice the first $n$ rows of the existing training array so that the other data stay fixed. b. Look for prompts that are plainly safe and sound risky in that language, an accurate gloss, native-speaker labelling with the four labels (refused, misunderstood, harmful, safe), and over-refusal computed as refused over all safe prompts. Give no credit for harmful prompts, and ask the student to delete them.

Marking anchors. Derivation (2): result 1, working 1. Implementation (4): working code 2, test 1, explanation 1. Investigation (4): controlled table 1, cost and benefit stated 1, five valid prompts with glosses 1, labelling and rate 1. Give no marks for numbers that do not appear in an output.

## Quiz 5 key

1. B. 2. C. 3. B. 4. C.

5. a. $2\times(1.0-0.5)=1.0$. b. $L=-\log\sigma(1)=\log(1+e^{-1})=\ln(1.37)=0.31$. Accept $-\ln 0.73=0.31$. c. $\beta(\sigma(1)-1)=2\times(0.73-1)=-0.54$.

6. a. Refusal rate $150/200=0.75$; harmful rate $20/200=0.10$; over-refusal rate $25/100=0.25$. Two points for all three, one for two. b. Any one of: replies the model garbled or did not understand are counted as refusals or as not harmful; the refusals may be low-quality Hausa; the model may refuse because it cannot write Hausa, which is what the Hausa study found for its untuned model; the rate ignores over-refusal.
