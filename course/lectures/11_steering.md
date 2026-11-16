# Lecture 11 — Interpretability and cross-lingual steering

[Course home](../../README.md) · Week 3, day 1 · 120 minutes · [Slides](../slides/11_steering.pptx) · [Notebook](../tutorials/11_steering.ipynb) · [Exercises](../exercises/11_steering.md) · [Quiz](../quizzes/11_steering.md)

## Learning outcomes

- Find a feature direction in activations with a difference of means, and test it against a random direction.
- Apply three interventions (add, clamp, ablate) and say what each one shows.
- Measure a steering method on compliance and relevance, with the layer and strength chosen on development data.

## Session plan

| Minutes | Activity |
|---|---|
| 0–10 | Quiz 10 (benchmarking and evaluation) |
| 10–35 | Slides 1–9: the Task 7 problem, the residual stream, directions, probes |
| 35–60 | Slides 10–17: add, clamp, ablate; the two-dimensional example on the board; the trade-off |
| 60–65 | Break |
| 65–105 | [Notebook 11](../tutorials/11_steering.ipynb): sections 1–4 together, section 5 in pairs |
| 105–115 | Slides 18–23: measurement, the paper table, compute, the safety note; discuss the layer-2 question |
| 115–120 | Exit ticket: one measured result, one reason it may not transfer, one next experiment |

## Teaching notes

### The problem
SemEval-2027 Task 7 (CLaS) gives a question in one language and asks the model to answer in another. A target-language instruction in the prompt is not ranked, and there is no training budget. Teams must change the model's activations at inference time. Two models are ranked: `meta-llama/Llama-3.1-8B-Instruct` and `CohereLabs/tiny-aya-global`. Track 2 has Swahili as a target, the only African language in the task. Answers are scored on language compliance and on relevance to the question.

Days 2 to 5 changed weights. Today changes activations. The word "alignment" also changes meaning: on day 5 it meant safety and preference training, and in Task 7 it means the same content landing in the same place across languages.

### The residual stream
A decoder layer adds to a running vector: $x_{l+1}=x_l+\mathrm{Attn}_l(x_l)+\mathrm{MLP}_l(x_l)$. Each token position has one such vector, 4,096 numbers wide in Llama-3.1-8B. Because layers add and do not overwrite, we can stop between layers, read the vector, and write a changed one back.

### Features as directions
The linear representation hypothesis says a property such as "this text is Swahili" is a direction $v$ in activation space, and that more of the property means a larger projection on $v$. Word-vector arithmetic was early evidence. Arditi et al. (2024) found one direction that mediates refusal in 13 open chat models. Tang et al. (2024) found small sets of language-specific neurons. The hypothesis is convenient and partly true. Test it for each feature you care about.

The cheapest way to find a direction is a difference of means. Run parallel sentences in two languages through the model, stop at layer $l$, and subtract: $v_l=\mu_l^{\text{target}}-\mu_l^{\text{source}}$. Parallel text cancels the content and leaves the language. State whether you pooled the last token or the mean over tokens.

### Probes are correlational
A linear probe is a classifier trained on activations. High accuracy shows the information can be recovered. It does not show that later layers read it. Report a random direction and a shuffled-label control task (Hewitt and Liang, 2019) beside every probe. To show that the model uses a direction, intervene on it.

### Three interventions
- **Add:** $h'=h+\alpha v$. Every example moves the same distance.
- **Clamp:** set the projection on $\hat v$ to the target language's mean projection and leave the rest of $h$ alone.
- **Ablate:** $h'=h-(h\cdot\hat v)\hat v$. If the behaviour disappears, the model relied on the direction.

Activation patching, which copies an activation from another run, belongs to the same family and localises where information lives (Zhang and Nanda, 2023).

In a real model the edit is a forward hook on one layer. The hook fires at every generated token, so an additive push is applied again at each step.

### The price of steering
In the lab, compliance reaches 0.99 at $\alpha=1$ with relevance at 1.00. At $\alpha=3$ relevance is 0.35. Two things go wrong. The direction is not clean: in the lab it leans 0.6 towards one topic, so a large push drags every answer to that topic. And the norm grows: mean $|h|$ is 11.9 at $\alpha=4$ against 4.6 unsteered, which later layers never saw in training. Fixes to try one at a time: clamp, scale $\alpha$ by the layer's activation norm, steer one layer, steer only the first few generated tokens.

There is no rule for the layer or the strength. Sweep both on development questions. Wendler et al. (2024) report that Llama-2 routes through an English-like representation in its middle layers, which makes early and late layers the first candidates for language steering. Treat that as a hypothesis for Llama-3.1.

### Measuring a steering method
Compliance is scored by language identification, which is weak on answers of two or three words, on mixed text and between close relatives. Relevance needs a judge model or a person, and a judge may read the target language badly. A system that prints the same fluent Swahili sentence for every question scores 1.0 on compliance and near 0 on relevance, so report both for every language pair. The course combines them with a harmonic mean. The task page names both criteria and does not publish a formula.

A paper needs six rows on the same questions and seeds: no steering; the prompt instruction, marked as not ranked; difference of means at the best layer and strength; one change of your own; a random direction of the same norm; and ablation.

### Compute and access
Llama-3.1-8B has 8.03 billion weights: 16.1 GB at 16 bits, which does not fit a 15 GB T4, and about 5.5 GB at 4 bits, which does. A vector found on the 4-bit model may not transfer to the 16-bit model the organisers run, so check the method first on the smaller model, where both precisions fit. The Llama weights are licence-gated on Hugging Face; request access on day 1 of the task.

### The same tool removes safety training
The bonus track asks for refusals in the requested language, which is steering towards a refusal direction. The same arithmetic with a minus sign removes refusals from an open model. In this course, steer language and refusal-on only, and keep harmful prompts out of the repository. See the [safety thread](../SAFETY.md).

## Worked example

Take $h=(2,1)$, an English mean of $(1,0)$ and a Swahili mean of $(0,1)$. Then $v=(-1,1)$ and $\hat v=(-1,1)/\sqrt2$. The projection of $h$ on $\hat v$ is $(-2+1)/\sqrt2=-0.71$.

| Operation | Result | Projection on $\hat v$ |
|---|---|---|
| Add, $\alpha=1.5$ | $(0.5, 2.5)$ | $1.41$ |
| Clamp to the Swahili mean projection, $0.71$ | $(1, 2)$ | $0.71$ |
| Ablate | $(1.5, 1.5)$ | $0$ |

Adding overshoots the Swahili mean. Clamping lands on it. Ablating leaves the point halfway between the two languages.

## Guided tutorial

**Question:** Find a language direction in a toy residual stream, steer English activations into Swahili, and measure what steering costs.

The notebook builds 32-dimensional synthetic activations from a topic direction, a language direction and noise. Students compute the difference of means (cosine 0.997 with the true direction), compare it with a random direction, sweep $\alpha$, ablate, and then write `steering_vector` and `clamp` against tests. The data are synthetic and say nothing about a real model.

**Shared-task path:** Task 7 teams clone the organisers' repository, <https://github.com/d-gurgurov/CLaS-SemEval-2027>, which holds FLORES-200 development sentences, the CLaS-Bench validation set, a prompting baseline and the scorer. The course has not run that code; see [validation](../VALIDATION.md).

## Practical questions

1. A probe finds language identity at 99% in layer 2, but steering at layer 2 changes nothing. Give two explanations and one experiment that separates them.
2. Clamping needs no $\alpha$. What did you choose instead, and on which data?
3. Name two ways a real model's later layers could undo a steering edit.

## After class

Complete the [exercise sheet](../exercises/11_steering.md) in 45–60 minutes. Read sections 1–3 of Arditi et al. before the next session (20 minutes). [Quiz 11](../quizzes/11_steering.md) opens day 12.

## Readings

**Required**
- [Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717) (Arditi et al., 2024). Sections 1–3: difference of means, then addition and ablation on 13 models.

**For Task 7 teams**
- [CLaS-Bench: A Cross-Lingual Alignment and Steering Benchmark](https://aclanthology.org/2026.findings-acl.1086/) (Gurgurov et al., 2026). The benchmark behind the task; read the evaluation section first.
- [Language-Specific Neurons](https://arxiv.org/abs/2402.16438) (Tang et al., 2024). A neuron-level way to switch output language.
- [Do Llamas Work in English?](https://arxiv.org/abs/2402.10588) (Wendler et al., 2024). Where language lives across layers.
- [Steering Llama 2 via Contrastive Activation Addition](https://arxiv.org/abs/2312.06681) (Rimsky et al., 2023). Layer and strength sweeps on a chat model.

**Going deeper**
- [Steering Language Models With Activation Engineering](https://arxiv.org/abs/2308.10248) (Turner et al., 2023).
- [The Linear Representation Hypothesis and the Geometry of Large Language Models](https://arxiv.org/abs/2311.03658) (Park et al., 2023).
- [Designing and Interpreting Probes with Control Tasks](https://arxiv.org/abs/1909.03368) (Hewitt and Liang, 2019).
- [Towards Best Practices of Activation Patching in Language Models](https://arxiv.org/abs/2309.16042) (Zhang and Nanda, 2023).
- [Sparse Autoencoders Find Highly Interpretable Features in Language Models](https://arxiv.org/abs/2309.08600) (Cunningham et al., 2023).
