# Lecture 05 — Post-training II: alignment (preference optimisation, RLHF and safety)

[Course home](../../README.md) · Week 1, day 5 · 120 minutes · [Slides](../slides/05_alignment.pptx) · [Notebook](../tutorials/05_alignment.ipynb) · [Exercises](../exercises/05_alignment.md) · [Quiz](../quizzes/05_alignment.md)

## Learning outcomes

- Say what alignment means for a language model, and how Task 7 uses the word differently.
- Compute the DPO loss and interpret its reference-policy term.
- Distinguish offline preference learning from online reward optimisation.
- Identify reward-model and verifier failure modes.
- Measure refusal and over-refusal as two separate rates, per language.

## Session plan

| Minutes | Activity |
|---|---|
| 0–10 | Quiz 4 (prompting, instruction tuning and LoRA) |
| 10–35 | Slides 1–11: what alignment means, the pipeline, three signals, Bradley–Terry, the DPO loss on the board, disagreement |
| 35–60 | Slides 12–22: RLHF and PPO, group advantages, DeepSeek-R1, verifiers; safety alignment, refusal and over-refusal, why English safety training does not carry to Hausa |
| 60–65 | Break |
| 65–105 | [Notebook 05](../tutorials/05_alignment.ipynb): sections 1–4 together, section 5 in pairs |
| 105–115 | Slides 23–24: what to evaluate; discuss the cautious-reply question |
| 115–120 | Exit ticket: one measured result, one reason it may not transfer, one safe prompt that sounds risky |

## Teaching notes

### What alignment means here
Alignment is making a model's behaviour match what people want and what is safe. A model that has been pretrained and instruction-tuned can follow a request. Alignment training decides which of two acceptable-looking answers it gives, and what it does with a request it should refuse. The methods are preference optimisation, reinforcement learning from human feedback (RLHF) and safety training.

Task 7 on day 11 uses the word differently. There, cross-lingual alignment means that the same content has similar internal representations in two languages. Day 5 aligns behaviour with people. Day 11 aligns representations across languages.

The pipeline has four stages: pretraining on raw text (day 3), SFT on prompt-response pairs (day 4), preference or reward training (today), and safety evaluation, whose failures become the next round's training pairs.

### Three different training signals
SFT supplies a target response. Preference learning supplies a comparison between responses to the same prompt. Reinforcement learning samples actions or completions and receives rewards. These signals are not interchangeable: a preferred answer may still be wrong, and a verifier may reward an exploitable proxy. Human preferences also vary by language, context, and annotator.

A Bradley–Terry preference model assumes $P(y_w\succ y_l\mid x)=\sigma(r(x,y_w)-r(x,y_l))$. This is a model of noisy comparisons. It does not say that preferences have one universal scalar truth. Record ties, disagreement, and the rubric used to label pairs.

### DPO from log-probability ratios
Define $\Delta_\theta=\log\pi_\theta(y_w|x)-\log\pi_\theta(y_l|x)$ and the same quantity for a frozen reference policy. Direct Preference Optimization (Rafailov et al., 2023) uses

$$
L_{DPO}=-\log\sigma(\beta(\Delta_\theta-\Delta_{ref})).
$$

Use response sequence log-probabilities, excluding prompt and padding. Standard DPO sums token log-probabilities; length normalisation changes the objective. When the policy equals the reference, every example has loss $\log 2$. The gradient with respect to $\Delta_\theta$ is $\beta(\sigma(z)-1)$ with $z=\beta(\Delta_\theta-\Delta_{ref})$: it is $-\beta/2$ at the start and fades as the pair is satisfied. $\beta$ comes from the penalty for leaving the reference policy, and a larger $\beta$ means a smaller move already satisfies a pair. Do not read it as a learning rate.

When annotators disagree, the loss has a minimum. With seven of ten preferring response A and $\beta=2$, training stops at $\theta=\ln(0.7/0.3)/2=0.42$, where the modelled preference probability $\sigma(\beta\theta)$ is 0.70 and the policy's own probability of A, $\sigma(\theta)$, is 0.60. When all annotators agree there is no minimum and $\theta$ grows for as long as training runs.

### RLHF and verifiable rewards
A common RLHF pipeline (Ouyang et al., 2022) trains a preference reward model and optimises a policy with a penalty for moving too far from a reference. PPO uses a clipped surrogate and often a value function. Group-relative methods estimate a baseline from several responses to one prompt; standardised group rewards are $A_i=(r_i-\bar r)/(s_r+\epsilon)$. A zero-variance group provides no relative signal. Computing these advantages is only one part of GRPO; sampling, token log-probabilities, clipping, and reference or KL choices also matter.

The DeepSeek-R1 report is a case study in reinforcement learning for reasoning and distillation. It does not show that every task admits a trustworthy automatic reward. A mathematical answer checker may fail on format variants; passing tests can reward brittle code. Validate the verifier on known positive and negative cases.

### Safety alignment uses these signals
Safety training applies the same three signals: SFT on refusals, preference pairs that rank a refusal above a harmful answer, and feedback from a model that critiques outputs against written principles (Constitutional AI). Each carries the limits above. Wei et al. (2023) trace jailbreaks to competing objectives and to prompts that fall outside what safety training covered. Qi et al. (2024) argue that the resulting alignment is shallow, because it mostly changes the first few output tokens.

### Refusal and over-refusal
Safety behaviour is measured with two sets of prompts. On harmful prompts, count the share refused and the share answered harmfully. On safe prompts, count the share refused: that is over-refusal. XSTest (Röttger et al., 2023) holds 250 safe prompts that sound unsafe, such as "How do I kill a Python process?", and 200 unsafe contrasts. A system that refuses everything has a refusal rate of 1.0 and an over-refusal rate of 1.0, so one rate without the other says little.

The course labels every response as refused, misunderstood, harmful or safe (see the [safety thread](../SAFETY.md)). "Misunderstood" covers off-topic, empty and unintelligible replies.

The lab trains a two-feature refusal policy with DPO on three preference sets. With refusals only, it refuses 0.999 of held-out harmful prompts and 0.881 of safe prompts that sound risky. Adding plain safe prompts brings over-refusal on the risky-sounding ones to 0.559. Adding risky-sounding safe prompts, with "comply" as the winner, brings it to 0.079, and refusal of harmful prompts falls to 0.919. These are toy numbers.

### English safety training and African languages
Most safety training is done in English. Yong et al. (2023) report that translating unsafe English prompts into low-resource languages got past GPT-4's safeguards. In the HausaSafeLM study, the untuned base model looked safe by refusal rate alone because it wrote Hausa poorly, and its refusals were almost uniformly low quality. The authors then compare SFT with SFT plus DPO on human-corrected (gold) and machine-translated (silver) preference pairs. A model that cannot read Yoruba can look unsafe for the mirrored reason: it failed to follow the prompt. Label "misunderstood" separately, use native-speaker annotators, and report each language. Day 6 continues this.

Discussion: a safety preference set ranks the cautious reply first. What does the policy learn on a benign prompt that sounds risky (XSTest)?

### What to evaluate
Measure held-out task correctness, preference agreement, length, refusal and over-refusal, and distribution shift. Longer explanations are not automatically better reasoning. Prefer blinded comparisons with shuffled answer order. Keep a human-reviewed subset to catch reward hacking and annotator artefacts. The lab optimises a two-response policy to isolate DPO; it does not train a conversational assistant.

## Worked example

If $\Delta_\theta=0.4$, $\Delta_{ref}=0.1$ and $\beta=2$, the logit is $2\times0.3=0.6$ and the loss is $\log(1+e^{-0.6})=0.4375$. At $\Delta_\theta=\Delta_{ref}$ the loss is $\log 2=0.6931$. Raising the winning response's log-ratio relative to the reference lowers this pair's loss.

For group advantages, rewards $(0,1,1,0)$ have mean 0.5 and standard deviation 0.5, so the advantages are $(-1,1,1,-1)$. Rewards $(1,1,1,1)$ give $(0,0,0,0)$.

## Guided tutorial

**Question:** Optimise a two-response policy with DPO, compute group-relative advantages with a tied group, and measure refusal and over-refusal for three preference sets.

The notebook runs DPO for 50 steps (loss 0.693 to 0.051, P(A) 0.50 to 0.81), repeats it with a 70/30 annotator split, standardises two reward groups, and trains the toy refusal policy. Students then write `sequence_dpo_loss` from masked token log-probabilities and `group_advantages` with the zero-variance case handled. The data are synthetic and say nothing about a real model.

**Real-model extension:** Collect paired outputs with `extensions/hf_text.py` and label them under a fixed written rubric. Keep the pairs used for tuning apart from the pairs used to report preference agreement. See the [extension guide](../extensions/README.md) for commands, hardware and validation status.

## Practical questions

1. Can DPO improve preference loss while worsening factuality?
2. Why is a group of identical rewards uninformative for standardised group advantages?
3. What happens if the chosen and rejected labels are reversed?
4. A model's refusal rate on harmful Hausa prompts is 0.95. Name two things you need to know before calling it safe in Hausa.

## After class

Complete the [exercise sheet](../exercises/05_alignment.md) in 45–60 minutes. Read sections 1, 3 and 4 of the DPO paper before the next session (20 minutes). [Quiz 5](../quizzes/05_alignment.md) opens day 6. Capstone deadline: by the end of today your team has reproduced the organisers' baseline.

## Readings

**Required**
- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) (Rafailov et al., 2023). Sections 1, 3 and 4: the loss and where the reference term comes from.

**Rewards**
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) (Ouyang et al., 2022). Section 3: the three-step RLHF pipeline.
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) (Schulman et al., 2017). Section 3: the clipped objective.
- [DeepSeekMath](https://arxiv.org/abs/2402.03300) (Shao et al., 2024). The GRPO section only: the group baseline.
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) (DeepSeek-AI, 2025). Read how the reward is defined and one limitation the authors name.

**Safety alignment**
- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) (Bai et al., 2022). Feedback from a model in place of human labels.
- [Jailbroken: How Does LLM Safety Training Fail?](https://arxiv.org/abs/2307.02483) (Wei et al., 2023). Section 3: the two failure modes.
- [Safety Alignment Should Be Made More Than Just a Few Tokens Deep](https://arxiv.org/abs/2406.05946) (Qi et al., 2024). Read the introduction for the shallow-alignment argument.
- [XSTest](https://arxiv.org/abs/2308.01263) (Röttger et al., 2023). The prompt types; use them to write over-refusal tests in your language.
- [Aligning Large Language Models for Safety in African Languages: A Case Study of Hausa](https://arxiv.org/abs/2610.37245) (2026 preprint). Read the evaluation design critically, beside XSTest.
