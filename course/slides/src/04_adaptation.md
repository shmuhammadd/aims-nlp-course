type: title
title: Post-training I: prompting, instruction tuning and LoRA
subtitle: Day 4 · Week 1
footer: Shamsuddeen Hassan Muhammad | AIMS 2026

---
type: big
title: You have 500 labelled Hausa captions and one free GPU. Do you prompt, train an adapter, or fine-tune everything?

Notes: Take a show of hands for each of the three. Do not resolve it. The decision table near the end answers it, and the class votes again there.

---
title: Where today sits
subtitle: Day 3 trained a model on raw text. Today we make it do a task.

| Lever | What changes | Needs | Today |
|---|---|---|---|
| Prompt | Input tokens | A few examples, forward passes | Part 1 |
| Instruction tuning (SFT) | Weights, on prompt-response pairs | Labelled pairs, a training run | Part 2 |
| LoRA | A small added matrix | The same pairs, less memory | Part 3 |

- Task 12 and Task 9 can each be attempted three ways: an encoder classifier, a prompt, or a LoRA adapter
- Day 5 adds a fourth lever: training on preferences and rewards

Notes: Task 12 is emotion for paintings with captions in 28 languages. Task 9 is frame labels for news paragraphs. Both are classification, so all three routes apply.

---
type: section
title: Part 1: Prompt
subtitle: No weights change. The input does the work.

---
title: Zero-shot gives the task, few-shot adds examples

```
Choose the emotion the caption expresses.
Answer with one word from: amusement, awe, contentment,
excitement, anger, disgust, fear, sadness.

Caption: <demonstration caption 1>
Emotion: awe

Caption: <demonstration caption 2>
Emotion: sadness

Caption: <the caption to label>
Emotion:
```

- Remove the two demonstrations and the prompt is zero-shot
- The model's weights are the same for every task. Brown et al. (2020) called this in-context learning.

Notes: The eight labels are the Task 12 subtask 1 label set. Ask what else the prompt fixes without saying so: the language of the instruction, the label words, the separator, the number and order of demonstrations.

---
title: The same four demonstrations, 24 orders
subtitle: Lab toy: a similarity vote in which later demonstrations weigh more
layout: stack

![](figures/04_order.png)

- With equal weights every order scores 0.714. With recency weights the range is 0.483 to 0.795.

Notes: Toy numbers from tutorial 04, seed 42, 2,000 test queries. The toy has one property of a real model built in by hand: a lean towards the labels at the end of the prompt. Ask the class to predict the pink curve before showing it.

---
title: Why order and choice move the score

- Recency: the model leans towards the labels nearest the end of the prompt (Zhao et al., 2021)
- Majority label: three "awe" demonstrations out of four pull answers towards "awe" (Zhao et al., 2021)
- Order: the same demonstrations reordered can fall from near the best score to near chance (Lu et al., 2021)
- In the lab, the worst order ends with both +1 demonstrations and answers +1 for every query

### What to do
- Balance the labels and alternate them
- Try several orders and report the spread
- Fix the demonstrations before you look at test data

Notes: Lu et al. tested GPT-2 and GPT-3 sizes and found the effect at every size. Zhao et al. also name a third bias, towards tokens that are common in pretraining text.

---
title: The prompt is a hyperparameter
subtitle: Choose it on development data, as you would a learning rate

| In the lab toy | Accuracy |
|---|---|
| Best of 24 orders, on the 40 development queries | 0.825 |
| That same order, on 2,000 test queries | 0.795 |
| Mean over all 24 orders, test | 0.584 |

- The development score is the best of 24 noisy numbers, so it flatters the prompt
- Report the test score once, and say how many prompts you tried
- Save the full prompt text, the demonstrations and their order with every result
- Task 12 and Task 9 limit test submissions. Tune on your own development split.

Notes: 0.825 against 0.795 is the same selection effect as picking the best of many seeds. With 40 queries one query is 2.5 points.

---
title: Ask for output a program can read

- A classifier returns one of 8 labels. A language model returns text.
- Say what to return: one word from a closed list, or JSON with named fields
- Map the reply to a label with a rule you wrote before the run
- Count replies that fit no label, and score them as wrong

| Reply | Parsed as |
|---|---|
| `awe` | awe |
| `Awe.` | awe, after lower-casing and stripping |
| `The caption expresses wonder` | invalid |

- Report the invalid rate beside macro-F1

Notes: Dropping invalid replies from the denominator raises the score without improving the system. Some libraries can restrict decoding to the label tokens; day 3 covered the decoding step this changes.

---
type: section
title: Part 2: Instruction tuning
subtitle: Supervised fine-tuning on prompt-response pairs

---
title: SFT is next-token training on the response

$$
L(\theta) = -\sum_{t \in \mathrm{response}} \log p_\theta(y_t \, | \, \mathrm{prompt},\ y_{<t})
$$

- The loss is the day-3 loss. The data changed: prompt-response pairs in place of raw text.
- A chat template turns roles into tokens: system, user, assistant, end of turn
- Train and test with the same template. The same words under another template are a different input.
- Split by source before you write paraphrases, so one question cannot sit in both train and test

Notes: Draw one training pair on the board as a single row of tokens with the role markers. Wei et al. (2021), the FLAN paper, is the reference for instruction tuning across many tasks.

---
title: Only the response carries loss
subtitle: The label −100 tells the loss to skip a position
layout: stack

![](figures/04_label_mask.png)

- The model still reads every prompt token. Three of twelve positions are scored.

Notes: Ina kwana is the Hausa morning greeting. Ask why the end token is supervised: without it the tuned model never learns to stop.

---
title: Four ways to build the labels wrong

| Mistake | What you see |
|---|---|
| Shift the labels yourself, and the library shifts again | Loss falls slowly; outputs are off by one token |
| Mask the end token | Answers never stop |
| Leave padding unmasked | Loss looks low; the model learned to predict padding |
| Train with one template, test with another | Good loss, poor answers |

- Print the token IDs and labels of three examples before every training run
- In multi-turn data, decide whether every assistant turn or only the last one is supervised, and write it down

Notes: The first row catches most students. Causal language-model classes in common libraries shift labels internally.

---
type: section
title: Part 3: LoRA
subtitle: Train a small matrix beside a frozen one

---
title: LoRA adds a low-rank change to a frozen matrix

$$
W' = W + sAB \qquad A:\ d_{in} \times r \qquad B:\ r \times d_{out} \qquad s = \alpha / r
$$

- W is frozen. Only A and B train.
- This course writes inputs as row vectors, so the layer computes Y = X(W + sAB)
- Many libraries store the transpose. Check shapes before you copy a formula.
- At the start A is random and B is zero, so W′ = W and the model's outputs are unchanged

Notes: Hu et al. (2021) write the update as BA with column vectors. The algebra is the same.

---
title: Worked example: count the parameters
layout: split

One 4096 × 4096 projection, rank 8.

$$
4096 \times 4096 = 16{,}777{,}216
$$

$$
r(d_{in} + d_{out}) = 8 \times 8192 = 65{,}536
$$

- The adapter is 0.39% of the matrix
- At 4 bytes each: 64 MB against 0.25 MB
- Doubling the rank doubles the count

![](figures/04_param_share.png)

Notes: 65,536 / 16,777,216 = 0.00390625. 16,777,216 × 4 bytes = 64 MiB; 65,536 × 4 bytes = 0.25 MiB. Biases and other adapted layers add to both.

---
title: Why B starts at zero and A does not

Write G for the gradient of the loss with respect to W′.

$$
\frac{\partial L}{\partial A} = s\,G\,B^\top \qquad\qquad \frac{\partial L}{\partial B} = s\,A^\top G
$$

- B = 0 makes the first gradient of A zero. A random A makes the first gradient of B non-zero.
- After one step B is non-zero, and A starts to move
- With both at zero, both gradients are zero and nothing ever trains
- The lab asserts both facts at step 0

Notes: Derive this on the board from Y = X(W + sAB) and mean squared error. The exercise sheet asks for the derivation with the X factors written out.

---
title: The adapter recovers a rank-2 change
subtitle: Lab: frozen 12 × 8 base, target is base plus a rank-2 matrix
layout: split

- Frozen base: held-out error 0.054
- Rank 2, 40 numbers: error falls to about 10⁻⁹
- Rank 1, 20 numbers: error stops at 0.015
- The base matrix is identical before and after, and the lab asserts it
- A linear toy. It shows the algebra, and says nothing about how much rank a language task needs.

![](figures/04_lora_fit.png)

Notes: The rank-1 curve is in the figure script and on the exercise sheet, not in the notebook. Rank 1 can fit only the larger of the two singular directions of the true change.

---
title: QLoRA: a 4-bit base with 16-bit adapters

- Store the frozen base in 4 bits per weight
- Keep the LoRA matrices at higher precision and train only those
- Gradients pass through the frozen 4-bit weights to reach the adapters (Dettmers et al., 2023)

### What LoRA and QLoRA do not remove
- The forward pass still stores activations for every layer
- The base weights still sit in memory, at 4 bits or 16
- Quantisation adds error. Measure the 4-bit base before you train on it.

Notes: Day 13 covers quantisation. The saving from LoRA is in gradients and optimiser state for the trainable weights, and in the size of what you store per task.

---
title: Prompt, LoRA or full fine-tune?
subtitle: Rules of thumb. Your development set decides.

| | Prompt | LoRA | Full fine-tune |
|---|---|---|---|
| Labelled examples | None to a few dozen | Hundreds to thousands | Thousands and more |
| Compute | Forward passes | One GPU, frozen base | Gradients and optimiser state for every weight |
| Stored per task | A text file | A few megabytes | A full copy of the model |
| Main risk | Score moves with wording and order | Overfits small data; can remove refusals | The same, and the model forgets more |

- Start with the cheapest column that beats your baseline

Notes: Vote again on the opening question. With 500 captions and a T4, a defensible plan is a prompt baseline first, then LoRA on a small model or a fine-tuned encoder.

---
title: Three routes into Task 12 and Task 9

| Route | Task 12 subtask 1 | Task 9 subtask 1 |
|---|---|---|
| Encoder | XLM-R on captions: the organisers' text-only baseline, macro-F1 0.550 | A multilingual encoder with 14 sigmoid outputs |
| Prompt | Label list plus balanced demonstrations in the caption's language | Frame definitions plus demonstrations; parse a list |
| LoRA | Instruction pairs: caption → emotion word | Paragraph → list of frames |

- Task 12 has not yet published its guidance on LLMs and extra data. Check before you build on a prompted model.
- Task 9 is multi-label: a paragraph can carry no frame or several

Notes: Numbers and rules are from the course task cards, read on 16 November 2026. The Task 12 multimodal baseline is 0.535, below text-only. Day 6 covers the encoder route.

---
title: The comparison your paper needs

| Row | What it shows |
|---|---|
| Majority class | The floor |
| Base model, zero-shot prompt | What you get for nothing |
| Few-shot prompt, order chosen on development data | What the cheap method reaches |
| LoRA adapter | Your trained system |

- Same development and test split in every row
- Search the learning rate on development data, a few values
- Record trainable parameters, time, peak memory and the invalid-output rate
- A low loss on eight training examples shows the pipeline runs. It does not show the model generalises.

---
title: Fine-tuning can remove refusals

- Qi et al. (2023) fine-tuned aligned chat models and measured their safety behaviour afterwards
- Fine-tuning on benign data weakened it. The users did not intend that.
- Your adapter is a fine-tune

### The check for this course
- Fix one set of refusal checks before training
- Run it on the base checkpoint and on the adapted one
- Report counts for both, in every language you evaluate
- Keep prompt and response text out of the repository. Commit IDs and labels.

Notes: See the safety thread, SAFETY.md. Day 5 explains why the refusals are shallow enough to lose. Discussion prompt: which setting would you vary first, rank or learning rate, and why?

---
type: big
title: Your few-shot prompt scores 0.61 and your LoRA adapter 0.63 on 200 development items. Which do you submit?

Notes: One item is half a point, so a 2-point gap is 4 items. Good answers ask for the spread over prompt orders and training seeds, the invalid-output rate, cost per prediction, and whether the task rules allow the prompted model. Day 6 gives the paired bootstrap for this comparison.

---
title: Lab: 40 minutes

- Open `course/tutorials/04_adaptation.ipynb`. NumPy only.
- Sections 1 to 4, together
  - Train a rank-2 adapter and check the base matrix is unchanged
  - Build response-only labels
  - Reorder four demonstrations and choose an order on development queries
- Section 5, in pairs: write three functions until the tests pass
  - `response_only_labels` for a padded batch
  - `lora_param_count` and `merge_lora`
- Predict each output before you run the cell

---
title: Readings

### Before the next session (20 minutes)
- Hu et al., 2021. LoRA. Sections 1 and 4.1: the update and its initialisation.

### Prompting
- Brown et al., 2020. Language Models are Few-Shot Learners. Section 2.
- Lu et al., 2021. Fantastically Ordered Prompts. Sections 1 to 3.
- Zhao et al., 2021. Calibrate Before Use. Section 4.

### Adapting and its risks
- Dettmers et al., 2023. QLoRA.
- Qi et al., 2023. Fine-tuning Aligned Language Models Compromises Safety.

Notes: Links are in the lecture notes.

---
title: Exit ticket

Write three lines and hand them in.

- One result you measured today, with its number
- One reason it might not hold for a real model on your task
- The route your team will try first: encoder, prompt or adapter

Quiz 4 opens the next session: six questions, ten minutes.
