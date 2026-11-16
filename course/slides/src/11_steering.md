type: title
title: Interpretability and cross-lingual steering
subtitle: Day 11 · Week 3
footer: Shamsuddeen Hassan Muhammad | AIMS 2026

---
type: big
title: Can we change the language a model answers in without touching the prompt or the weights?

Notes: Let the class answer for a minute. Most will say "fine-tune" or "tell it in the prompt". Task 7 forbids the prompt and gives you no training budget. That leaves the activations.

---
title: Task 7 asks for exactly this
subtitle: CLaS: Cross-Lingual Alignment and Steering, SemEval-2027

| Track | Source → target | Pairs |
|---|---|---|
| 1 High-resource | 10 languages → the same 10 | 90 |
| 2 Low-resource | 10 languages → cs, el, fa, vi, uk, mt, no, **sw**, th, tr | 100 |
| 3 Many-to-one | 20 languages → ja, ru or de | 19 per target |
| Bonus | Refuse in the requested language | 100 |

- Two models are ranked: `Llama-3.1-8B-Instruct` and `tiny-aya-global`
- You may edit activations, neurons, heads or SAE features at inference time
- A target-language instruction in the prompt is **not** ranked
- Answers are scored on language compliance **and** relevance to the question

Notes: Swahili is the only African language, in track 2. Evaluation runs 10 to 31 January 2027. Each team gets two test submissions and one final submission per track.

---
title: Where today sits
subtitle: Days 2 to 5 changed the weights. Today we change the activations.

| Lever | What changes | Cost | Day |
|---|---|---|---|
| Prompt | Input tokens | None | 4 |
| LoRA, SFT | A few weights | Training run | 4 |
| DPO, RL | Weights, by preference | Training run and labels | 5 |
| **Steering** | **Activations at one layer** | **A few hundred forward passes** | **11** |

- "Alignment" on day 5 meant safety and preferences
- "Alignment" in Task 7 means the same content landing in the same place across languages

---
type: section
title: Part 1: Read
subtitle: What is in the residual stream?

---
title: The residual stream is a running sum
subtitle: Every layer reads the same vector and adds to it

$$
x_{l+1} = x_l + \mathrm{Attn}_l(x_l) + \mathrm{MLP}_l(x_l)
$$

- One vector per token position, of width d (4,096 in Llama-3.1-8B)
- Layers do not overwrite it. They add to it.
- So we can stop at any layer l, **read** x, and **write** a new x before the model continues
- This is the day-2 decoder block. Nothing new has been added to the model.

Notes: Draw three layers on the board as boxes hanging off one horizontal line. Mark the point between layers where a hook reads and writes.

---
title: The working hypothesis: features are directions
subtitle: The linear representation hypothesis

- A property such as "this text is Swahili" is a direction v in activation space
- More of the property means a larger projection on v
- Evidence, in order of age:
  - 2013: word vectors, where king − man + woman lands near queen
  - 2024: one direction mediates refusal in 13 open chat models (Arditi et al.)
  - 2024: a small set of language-specific neurons switches output language (Tang et al.)
- It is a hypothesis. It fails for features that are not linear, and you must test it for your feature.

Notes: Park et al. 2023 give the formal statement. Say plainly that the hypothesis is convenient and partly true, and that today's lab shows how to test it for one feature.

---
title: Find the direction: difference of means
layout: split

Take the same sentences in two languages. Run both through the model. Stop at layer l.

$$
v_l = \mu_l^{\mathrm{target}} - \mu_l^{\mathrm{source}}
$$

- μ is the mean activation over sentences
- Use the last token, or the mean over tokens, and say which
- Parallel text cancels the content and leaves the language
- Task 7 supplies FLORES-200 sentences for all 20 languages

![](figures/11_projection_hist.png)

Notes: The histogram is the lab's toy: 400 held-out activations projected on the unit vector. In the lab the cosine between v and the true direction is 0.997.

---
title: A probe shows the information is there, not that it is used
subtitle: Reading is correlational

- A linear probe is a classifier trained on activations: here, English or Swahili
- High probe accuracy has two readings:
  - the model computes language identity and uses it
  - the information is recoverable, and nothing downstream reads it
- Controls that belong in your paper:
  - a random direction (the lab gets 60% against 95% for v)
  - a control task with shuffled labels (Hewitt and Liang, 2019)
- To show the model **uses** a direction, change it and watch the output

---
type: section
title: Part 2: Write
subtitle: Three interventions on one direction

---
title: Add, clamp, ablate
layout: split

### Add
h′ = h + α·v. Push every example the same distance.

### Clamp
Set the projection on v̂ to the target language's mean. Leave the rest of h alone.

### Ablate
h′ = h − (h·v̂)·v̂. Remove the direction. Tests whether the model needs it.

![](figures/11_geometry.png)

Notes: Activation patching is the fourth member of this family: copy the activation from a different run. It localises where information lives. We do not need it for Task 7, but Zhang and Nanda 2023 is the reference.

---
title: Worked example in two dimensions

Take h = (2, 1), the English mean (1, 0) and the Swahili mean (0, 1).

$$
v = (0,1) - (1,0) = (-1, 1) \qquad \hat{v} = (-1,1)/\sqrt{2}
$$

| Operation | Result | Projection on v̂ |
|---|---|---|
| Start | (2, 1) | −0.71 |
| Add, α = 1.5 | (0.5, 2.5) | 1.41 |
| Clamp to the Swahili mean (0.71) | (1, 2) | 0.71 |
| Ablate | (1.5, 1.5) | 0 |

- Adding overshoots the Swahili mean. Clamping lands on it. Ablating sits between the two languages.

Notes: Do this on the board before showing the table. Projection of h is (−2+1)/√2. After adding 1.5v the vector is (0.5, 2.5) and the projection is 2/√2.

---
title: In a real model the edit is a forward hook

```
v = mean_acts[target][layer] - mean_acts[source][layer]

def steer(module, inputs, output):
    hidden = output[0]                  # (batch, tokens, d)
    return (hidden + alpha * v,) + output[1:]

handle = model.model.layers[layer].register_forward_hook(steer)
text = generate(model, question)        # every new token is steered
handle.remove()                         # always remove the hook
```

- The hook runs at every generated token, so the push is applied again and again
- Start from the organisers' baseline code. This sketch shows the idea and has not been run in this course.

Notes: The repository is github.com/d-gurgurov/CLaS-SemEval-2027. It has the data download, a prompting baseline and the scorer.

---
title: Steering has a price
subtitle: Lab result: 400 held-out English activations pushed towards Swahili
layout: stack

![](figures/11_tradeoff.png)

- At α = 1 compliance is 0.99 and relevance is 1.00. At α = 3 relevance is 0.35.

Notes: These are the toy numbers from tutorial 11, seed 42. They show the shape, not a measurement of Llama. Ask the class to predict the pink curve before showing it.

---
title: Why relevance falls, and three fixes

- The direction is not clean. In the lab it leans 0.6 towards one topic, so a large push drags every answer to that topic.
- The norm grows. At α = 4 the mean |h| is 11.9 against 4.6 unsteered. Later layers never saw vectors that long.

### Fixes to try, one at a time
- Clamp the projection instead of adding (lab: compliance 0.98, relevance 0.99, no α to tune)
- Scale α by the activation norm at that layer
- Steer one layer, or only the first few generated tokens

---
title: Which layer, and how strong?

- There is no rule. Sweep both on development questions.
- A grid of 8 layers × 6 strengths is 48 runs per language pair. Track 2 has 100 pairs.
  - Tune on a few pairs, then check that the choice transfers
- What published work found, as a starting point:
  - Refusal: middle layers work best (Arditi et al., 2024)
  - Language: models route through an English-like middle (Wendler et al., 2024), so early and late layers are the candidates
- Record layer, α, pooling and the number of sentences behind every vector

Notes: Wendler et al. study Llama-2 with the logit lens. Their claim is about a latent language close to English in middle layers. Treat it as a hypothesis for Llama-3.1 and test it.

---
title: Finer tools: neurons and sparse features

### Language-specific neurons (Tang et al., 2024)
- Score each MLP neuron by how unevenly it fires across languages
- A small set per language matters: switch it off and that language degrades; switch it on and the output language changes

### Sparse autoencoders (Cunningham et al., 2023)
- Learn a wide dictionary so that each activation is a sum of a few features
- Steer by turning one feature up
- Cost: training or finding an SAE for your model and layer; features depend on the dictionary you trained

---
type: section
title: Part 3: Measure
subtitle: A steering method is a claim. Test it.

---
title: Two scores, and both can mislead

| | Compliance | Relevance |
|---|---|---|
| Asks | Is the answer in the target language? | Does it answer the question? |
| Measured by | Language identification | A judge model or a human |
| Fails when | The answer is short, mixed, or in a close relative | The judge reads the target language badly |

- A system that prints one fluent Swahili sentence for every question scores 1.0 on compliance and near 0 on relevance
- Report both, per language pair. The course combines them with a harmonic mean; the task page does not publish its formula.
- Check 20 outputs by eye in a language someone in the room reads

Notes: Language identification for Swahili is reliable on full sentences. It is weaker on answers of two or three words, which is what short questions produce.

---
title: The experiment your paper needs

| Row | What it shows |
|---|---|
| No steering | The floor: the model answers in the source language |
| Prompt instruction (not ranked) | What the easy method reaches |
| Difference of means, best layer and α | Your baseline |
| One change: clamp, norm scaling or neurons | Your contribution |
| Random direction of the same norm | The effect is the direction, not the size of the push |
| Ablation of v | The model relies on the direction |

- Same questions and same seeds in every row
- Choose layer and α on development data. You have two test submissions and one final per track.

---
title: Compute: what fits on a free GPU

- Llama-3.1-8B has 8.03 billion weights
  - 16-bit: 8.03 × 2 = 16.1 GB. Does not fit a 15 GB T4.
  - 4-bit: about 5.5 GB with overhead. Fits, with room for activations.
- A vector from the 4-bit model may not transfer to the 16-bit model the organisers run. Check on the smaller model first, where you can run both.
- The Llama weights are licence-gated on Hugging Face. Request access today.
- Storing activations: 500 sentences × 4,096 numbers × 4 bytes = 8.2 MB per layer. Keep the means, not the tokens.

Notes: Day 13 covers quantisation properly. For now the point is to plan the request for model access and to prototype on tiny-aya-global.

---
title: The same tool removes safety training

- The bonus track asks for refusals in the requested language: add a refusal direction
- The same arithmetic with a minus sign removes refusals from an open model
- Day 5 showed that safety training changes few early tokens. This is the mechanism view of that result.
- In this course:
  - steer language and refusal-on, never refusal-off
  - keep harmful prompts out of the repository, as in the safety thread

Notes: Link back to the day-5 discussion of shallow alignment and to SAFETY.md.

---
type: big
title: A probe finds language at 99% in layer 2. Steering at layer 2 changes nothing. What do you conclude?

Notes: Two acceptable answers. First, later layers recompute language from the tokens and overwrite the edit. Second, the probe found information the model does not use at that layer. An experiment that separates them: steer at every layer from 2 upward, or patch layer 2 from a Swahili run and check whether the output moves.

---
title: Lab: 40 minutes

- Open `course/tutorials/11_steering.ipynb`. NumPy only.
- Sections 1 to 4, together: build the toy stream, find v, sweep α, ablate
  - Predict each output before you run the cell
- Section 5, in pairs: write two functions until the tests pass
  - `steering_vector` for any pair of labels
  - `clamp` to a target projection
- Task 7 teams: clone the organisers' repository after class and run the prompting baseline on one language pair

---
title: Readings

### Before the next session (20 minutes)
- Arditi et al., 2024. Refusal in Language Models Is Mediated by a Single Direction. Sections 1 to 3.

### For Task 7 teams
- Gurgurov et al., 2026. CLaS-Bench. The benchmark behind the task.
- Tang et al., 2024. Language-Specific Neurons.
- Wendler et al., 2024. Do Llamas Work in English?

### Going deeper
- Turner et al., 2023. Steering Language Models With Activation Engineering.
- Hewitt and Liang, 2019. Designing and Interpreting Probes with Control Tasks.
- Zhang and Nanda, 2023. Best practices of activation patching.

Notes: Links are in the lecture notes.

---
title: Exit ticket

Write three lines and hand them in.

- One result you measured today, with its number
- One reason that result might not hold in Llama-3.1-8B
- The first experiment you would run tomorrow

Quiz 11 opens the next session: six questions, ten minutes.
