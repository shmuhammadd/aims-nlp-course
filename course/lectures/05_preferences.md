# Lecture 05 — Preference optimization and reinforcement learning for reasoning

[Course home](../../README.md) · Week 1, day 5 · 120 minutes

## Learning outcomes

- Compute DPO loss and interpret its reference-policy term.
- Distinguish offline preference learning from online reward optimization.
- Identify reward-model and verifier failure modes.

## Session plan

| Minutes | Activity |
|---|---|
| 0–10 | Retrieval quiz: recall the previous session; day 1 uses the prerequisite diagnostic |
| 10–35 | Concepts and concrete examples from the first half of these notes |
| 35–60 | Board derivation, worked example, and failure analysis |
| 60–65 | Break |
| 65–105 | Guided [notebook tutorial](../tutorials/05_preferences.ipynb) |
| 105–115 | Practical questions in pairs, then discuss evidence |
| 115–120 | Exit ticket: one result, one limitation, one next experiment |

## Teaching notes

### Three different training signals
SFT supplies a target response. Preference learning supplies a comparison between responses to the same prompt. Reinforcement learning samples actions or completions and receives rewards. These signals are not interchangeable: a preferred answer may still be wrong, and a verifier may reward an exploitable proxy. Human preferences also vary by language, context, and annotator.

A Bradley–Terry preference model assumes $P(y_w\succ y_l\mid x)=\sigma(r(x,y_w)-r(x,y_l))$. This is a model of noisy comparisons, not a statement that preferences have one universal scalar truth. Record ties, disagreement, and the rubric used to label pairs.

### DPO from log-probability ratios
Define $\Delta_\theta=\log\pi_\theta(y_w|x)-\log\pi_\theta(y_l|x)$ and the same quantity for a frozen reference policy. Direct Preference Optimization uses

$$
L_{DPO}=-\log\sigma(\beta(\Delta_\theta-\Delta_{ref})).
$$

Use response sequence log-probabilities, excluding prompt and padding. Standard DPO sums token log-probabilities; length normalization changes the objective. When policy equals reference, every example has loss log(2). Larger beta changes scaling and the implied reference regularization relationship; do not interpret it merely as a learning rate.

### RLHF and verifiable rewards
A common RLHF pipeline trains a preference reward model and optimizes a policy with a penalty for moving too far from a reference. PPO uses a clipped surrogate and often a value function. Group-relative methods estimate a baseline from several responses to one prompt; standardized group rewards are $A_i=(r_i-\bar r)/(s_r+\epsilon)$. A zero-variance group provides no relative signal. Computing these advantages is only one part of GRPO; sampling, token log probabilities, clipping, and reference/KL choices also matter.

The DeepSeek-R1 report is a case study in reinforcement learning for reasoning and distillation, not proof that every task admits a trustworthy automatic reward. A mathematical answer checker may fail on format variants; passing tests can reward brittle code. Validate the verifier on known positive and negative cases.

### What to evaluate
Measure held-out task correctness, preference agreement, length, refusal appropriateness, and distribution shift. Longer explanations are not automatically better reasoning. Prefer blinded comparisons with shuffled answer order. Keep a human-reviewed subset to catch reward hacking and annotator artefacts. The lab optimizes a two-response policy to isolate DPO; it does not train a conversational assistant.

## Worked example

If Δθ=0.4, Δref=0.1 and β=2, the logit is 0.6 and loss≈0.4375. At Δθ=Δref the loss is 0.6931. Increasing the winning response ratio relative to the reference decreases this pair loss.

## Guided tutorial

**Question:** Optimize a two-response policy with DPO; calculate group-relative advantages and expose a zero-variance reward group.

Open the notebook and predict each result before running it. Spend 5 minutes on setup and data inspection, 15 on the mechanism, 10 on the controlled intervention, and 10 on interpretation. The core uses NumPy and authored/synthetic data; the notebook identifies its limits. Save outputs, configuration, and a short interpretation. Do not report an expected trend as a measured result.

**Real-model or research extension:** Collect paired outputs from `extensions/hf_text.py` under a fixed human rubric; do not use the same examples to tune and report preference agreement.

See the [extension guide](../extensions/README.md) for commands, hardware assumptions, and validation status.

## Practical questions

1. Can DPO improve preference loss while worsening factuality?
2. Why is a group of identical rewards uninformative for standardized group advantages?
3. What happens if the chosen/rejected labels are reversed?

## After class

Complete the [exercise sheet](../exercises/05_preferences.md) in 45–60 minutes. Read the first reference's abstract and method overview before the next session; remaining references are optional depth. For long papers, focus on the objective, one figure/table, and one limitation.

## Readings

- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- [DeepSeekMath (GRPO)](https://arxiv.org/abs/2402.03300)
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948)
