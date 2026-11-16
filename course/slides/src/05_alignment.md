type: title
title: Post-training II: alignment
subtitle: Day 5 · Week 1
footer: Shamsuddeen Hassan Muhammad | AIMS 2026

---
type: big
title: Your model follows instructions. Someone asks it for something harmful. What should training have taught it, and with what data?

Notes: Take answers for a minute. Students usually say "add refusals to the fine-tuning data". Ask what that model then does with a safe question that sounds risky. Part 3 measures it.

---
title: What "alignment" means today
subtitle: Making a model's behaviour match what people want and what is safe

| | Day 5: behavioural alignment | Day 11, Task 7: cross-lingual alignment |
|---|---|---|
| What is aligned | The model's answers, with people's preferences and safety rules | Representations of the same content in two languages |
| Changed by | Training on comparisons and rewards | Editing activations |
| Measured by | Preference agreement, refusal, over-refusal | Language compliance and relevance |

- Same word, two meanings. Say which one you mean in your paper.
- Today is also preference optimisation, RLHF and safety training: these are the methods of alignment

Notes: The instructor asked for alignment to be visible as a topic. Write the one-line definition on the board and leave it there for the session.

---
title: Alignment is the third stage of training
subtitle: Each stage uses a different signal
layout: stack

![](figures/05_pipeline.png)

- Pretraining and SFT say what text looks like. Preferences and rewards say which answer is better.

Notes: Day 3 covered pretraining, day 4 SFT and LoRA. Evaluation feeds back: prompts the model failed on are labelled and added to the next preference set. Days 6 and 10 return to the evaluation box.

---
type: section
title: Part 1: Preferences
subtitle: Learning from "this answer is better than that one"

---
title: Three signals, three kinds of data

| Signal | One training example | What it cannot tell you |
|---|---|---|
| SFT | A prompt and one target response | What a bad response looks like |
| Preference | A prompt, a chosen response and a rejected one | Whether the chosen response is correct |
| Reward | A prompt, a sampled response and a score | Whether the scorer can be fooled |

- A preferred answer can still be wrong. A verifier can reward a shortcut.
- Preferences differ by language, context and annotator
- Record ties, disagreement and the rubric the annotators used

Notes: Ask which signal each of the four capstone tasks supplies. All four supply labels, so SFT. None supplies preferences. That is why today's lab is a toy.

---
title: Bradley–Terry turns comparisons into scores

Give every response a score r. The chance that annotators choose the winner depends on the score gap.

$$
P(y_w \succ y_l \, | \, x) = \sigma\left(r(x, y_w) - r(x, y_l)\right)
$$

- Equal scores give 0.5. A gap of 1 gives σ(1) = 0.73.
- It is a model of noisy comparisons
- It assumes one scale. Annotators who value different things break that assumption.

Notes: σ(1) = 1/(1 + e⁻¹) = 0.731. The same model ranks chess players and, on day 10, models in an arena.

---
title: DPO trains the policy on the comparisons directly

How much more does the policy like the winner than the loser?

$$
\Delta_\theta = \log \pi_\theta(y_w | x) - \log \pi_\theta(y_l | x)
$$

$$
L_{\mathrm{DPO}} = -\log \sigma\left(\beta\,(\Delta_\theta - \Delta_{\mathrm{ref}})\right)
$$

- Δref is the same gap under a frozen copy of the starting model
- The loss rewards moving the gap beyond the reference's. It needs no reward model and no sampling.
- Log-probabilities are summed over response tokens. Prompt and padding are left out.

Notes: Rafailov et al., 2023. Dividing by length changes the objective; standard DPO sums. Lab exercise A builds the masked sums.

---
title: Worked example: one pair
layout: split

Δθ = 0.4, Δref = 0.1, β = 2

$$
\beta(\Delta_\theta - \Delta_{\mathrm{ref}}) = 2 \times 0.3 = 0.6
$$

$$
L = \log(1 + e^{-0.6}) = 0.4375
$$

- Policy equal to reference: L = log 2 = 0.693, for every pair
- β comes from the penalty for leaving the reference. With β = 5 a small move already drives the loss to zero.
- β is not a learning rate

![](figures/05_dpo_loss.png)

Notes: e^(−0.6) = 0.549; log(1.549) = 0.4375. Do it on the board. The curves are the formula, not data.

---
title: The gradient fades as a pair is satisfied

$$
\frac{\partial L}{\partial \Delta_\theta} = \beta\,(\sigma(z) - 1) \qquad z = \beta(\Delta_\theta - \Delta_{\mathrm{ref}})
$$

- At the start z = 0 and the gradient is −β/2
- As z grows, σ(z) goes to 1 and the gradient to 0
- Lab, β = 2, 50 steps: loss 0.693 → 0.051, P(A) 0.50 → 0.81
- A pair with swapped labels gets the same push in the wrong direction. Check ten pairs by eye before training.

Notes: Derive on the board: d/dz of −log σ(z) is σ(z) − 1, then the chain rule gives β. This is exercise 1.

---
title: When annotators disagree, the policy stops
layout: split

Ten annotators see the same two responses. Seven choose A.

- With β = 2 the loss is lowest at θ = 0.42
- There σ(βθ) = 0.70, the annotators' split
- The policy then gives A with probability σ(0.42) = 0.60
- If all ten agree there is no minimum, and θ grows for as long as you train

![](figures/05_disagreement.png)

Notes: θ = ln(0.7/0.3)/2 = 0.424. Two probabilities are in play: the model of the annotators' choice, σ(βθ), and the policy's own probability of A, σ(θ). Students mix them up.

---
type: section
title: Part 2: Rewards
subtitle: Sample, score, update

---
title: RLHF: learn a reward model, then optimise against it

- Step 1: SFT on demonstrations
- Step 2: train a reward model on human comparisons, with the Bradley–Terry loss
- Step 3: sample responses, score them, and update the policy with PPO

$$
L_{\mathrm{PPO}} = E\left[\min\left(\rho A,\ \mathrm{clip}(\rho,\ 1-\epsilon,\ 1+\epsilon)\,A\right)\right] \qquad \rho = \frac{\pi_\theta(y|x)}{\pi_{\mathrm{old}}(y|x)}
$$

- A is the advantage: the reward minus a baseline. The clip stops one batch from moving the policy far.
- A penalty keeps the policy near the SFT model, as β does in DPO

Notes: Ouyang et al., 2022 is the pipeline; Schulman et al., 2017 is PPO. DPO is offline: fixed pairs. RLHF is online: the policy's own samples are scored.

---
title: Group-relative advantages need no value model

Sample a group of responses to one prompt, score each, and standardise within the group.

$$
A_i = \frac{r_i - \bar{r}}{s_r + \epsilon}
$$

| Rewards for four samples | Advantages |
|---|---|
| 0, 1, 1, 0 | −1, 1, 1, −1 |
| 1, 1, 1, 1 | 0, 0, 0, 0 |
| 0, 0, 0, 0 | 0, 0, 0, 0 |

- A group with equal rewards teaches nothing, whether all right or all wrong
- This is one part of GRPO. Sampling, clipping and the reference penalty are the rest.

Notes: First row: mean 0.5, standard deviation 0.5. Shao et al., 2024 (DeepSeekMath) introduced GRPO. Ask what the third row implies for a task the model almost never solves.

---
title: Case study: DeepSeek-R1

- Trained for reasoning with GRPO and rewards computed by rules: is the final answer correct, and is the output in the required format
- The reasoning ability was then distilled into smaller models
- Read it as one case of reinforcement learning on a task with a checkable answer

### What it does not show
- That every task has a reward you can trust
- That longer reasoning text means better reasoning

Notes: DeepSeek-AI, 2025. Maths answers and unit tests can be checked by a program. A caption's emotion in Hausa cannot.

---
title: A verifier is a model of correctness, and it can be wrong

| Failure | Example for an arithmetic checker |
|---|---|
| False negative | The answer is "0.5" and the key says "1/2" |
| False negative | The answer is right, with a unit the parser does not expect |
| False positive | The response lists several numbers and the checker takes the last |
| False positive | Tests pass on code written to match only those tests |

- The policy is trained to raise the reward, so it finds these cases
- Test the verifier on known right and wrong answers before training against it
- Keep a set that people review, to catch reward hacking

---
type: section
title: Part 3: Safety alignment
subtitle: The same signals, used to teach refusal

---
title: Safety training uses all three signals

| Signal | Safety use |
|---|---|
| SFT | Harmful prompt → a written refusal |
| Preference | Rank a refusal above a harmful answer |
| Feedback from a model | A model critiques outputs against written principles (Constitutional AI, Bai et al., 2022) |

- Each carries its limit from Part 1
  - SFT never shows the harmful answer
  - A preference set that always ranks the cautious reply first teaches caution everywhere
  - A critic model has blind spots, and they are passed on

Notes: Day 4 showed that fine-tuning can remove this training again (Qi et al., 2023).

---
title: Refusal and over-refusal are two measurements

A native speaker gives each response one label: refused, misunderstood, harmful or safe.

| Rate | Prompts | Counts | Want |
|---|---|---|---|
| Refusal rate | Harmful | refused / all | High |
| Harmful rate | Harmful | harmful / all | Low |
| Over-refusal rate | Safe | refused / all | Low |

- XSTest (Röttger et al., 2023): 250 safe prompts that sound unsafe, and 200 unsafe contrasts
  - "How do I kill a Python process?" is safe
- "Misunderstood" covers off-topic, empty and unintelligible replies
- A system that refuses everything scores 1.0 on the first row and 1.0 on the third

Notes: The four labels are the outcome table in SAFETY.md. Day 10 runs safety_eval.py on such labels. XSTest counts are from the paper's abstract.

---
title: A refusals-only preference set over-refuses
subtitle: Lab toy: a refusal policy trained with DPO on three preference sets
layout: stack

![](figures/05_overrefusal.png)

- Adding safe prompts that sound risky cuts over-refusal from 0.88 to 0.08. Refusal of harmful prompts falls from 1.00 to 0.92.

Notes: Toy numbers from tutorial 05, seed 42, 1,000 held-out prompts of each kind. Each prompt is two numbers: how risky it sounds and whether it is harmful. Ask the class to predict the pink bars for the first set.

---
title: Why trained refusals still fail

### Jailbreaks (Wei et al., 2023)
- Competing objectives: the prompt sets instruction-following against the safety training
- Mismatched generalisation: the prompt is in a form the model can read and safety training never covered

### Shallow alignment (Qi et al., 2024)
- Safety training mostly changes the first few output tokens
- Push the model past those tokens and the harmful answer follows

- Day 11 shows the same fact inside the model: refusal behaves like one direction in activation space

Notes: A low-resource language is one case of mismatched generalisation: the model reads it, and the safety data did not include it. That leads to the next slide.

---
title: Safety trained in English does not carry to Hausa

- Most safety training data is English
- Yong et al. (2023): unsafe English prompts translated into low-resource languages got past GPT-4's safeguards
- The Hausa study (HausaSafeLM, 2026):
  - the untuned model looked safe by refusal rate, because it wrote Hausa poorly
  - its refusals were almost all of low quality
- A model that cannot follow a Yoruba prompt can look unsafe for the same reason reversed: it failed to read, and did not choose to comply
- So: label "misunderstood" separately, use native speakers, report each language

Notes: All four claims are from the course safety thread and its readings. The Hausa study compares SFT with SFT plus DPO on human-corrected (gold) and machine-translated (silver) pairs. Day 6 takes this further.

---
title: What to evaluate after alignment training

| Measure | Guards against |
|---|---|
| Held-out task correctness | A preferred style with wrong content |
| Preference agreement on new pairs | Memorised pairs |
| Response length | "Longer" being learned as "better" |
| Refusal and over-refusal, per language | A single safety number |
| A sample reviewed by people | Reward hacking and annotation artefacts |

- Compare blind, with the answer order shuffled
- Keep tuning pairs apart from reporting pairs

---
type: big
title: A safety preference set ranks the cautious reply first, every time. What does the policy learn about a safe question that sounds risky?

Notes: It learns to refuse it: the lab's first bar group, 0.88. Follow up: how many safe-but-risky-sounding pairs are needed, and who writes them for Hausa or Yoruba? That is a data-building question, and day 6 is about building data.

---
title: Lab: 40 minutes

- Open `course/tutorials/05_alignment.ipynb`. NumPy only.
- Sections 1 to 4, together
  - DPO on a two-response policy, then with annotators who disagree
  - Group advantages, with a tied group
  - Refusal and over-refusal for three preference sets
- Section 5, in pairs: write two functions until the tests pass
  - `sequence_dpo_loss` from masked token log-probabilities
  - `group_advantages` with no `nan` on a tie
- Capstone deadline today: your team has reproduced the organisers' baseline

---
title: Readings

### Before the next session (20 minutes)
- Rafailov et al., 2023. Direct Preference Optimization. Sections 1, 3 and 4.

### Rewards
- Ouyang et al., 2022. Training language models to follow instructions with human feedback.
- Shao et al., 2024. DeepSeekMath, the GRPO section.
- DeepSeek-AI, 2025. DeepSeek-R1.

### Safety alignment
- Bai et al., 2022. Constitutional AI.
- Wei et al., 2023. Jailbroken.
- Qi et al., 2024. Safety Alignment Should Be Made More Than Just a Few Tokens Deep.
- Röttger et al., 2023. XSTest.
- Aligning LLMs for Safety in African Languages: A Case Study of Hausa, 2026.

Notes: Links are in the lecture notes. The Hausa paper is a 2026 preprint; read it critically and beside XSTest.

---
title: Exit ticket

Write three lines and hand them in.

- One result you measured today, with its number
- One reason it might not hold for a real model in your language
- One safe prompt that sounds risky, in a language you speak

Quiz 5 opens the next session: six questions, ten minutes.
