type: title
title: Pretraining and decoding
subtitle: Day 3 · Week 1
footer: Shamsuddeen Hassan Muhammad | AIMS 2026

---
type: big
title: One model, one prompt, two runs, two different answers. Which one is the model's answer?

Notes: Take answers for one minute. Neither is. The model gives a probability distribution over the next token, and a separate procedure turns it into text. The last third of today is about that procedure. Come back to this slide at the end.

---
title: Where today sits
subtitle: Yesterday we built the block. Today we fill it with weights and get text out.

| Day | Question | Your task meets it when |
|---|---|---|
| 2 | What does a decoder compute? | You read a model's configuration |
| **3** | **Where do the weights come from? How do logits become text?** | **You pick a pretrained model and a decoding setting** |
| 4 | How do we adapt a pretrained model cheaply? | You fine-tune or prompt the baseline |
| 5 | How do we shape its behaviour? | You read the model's refusals and preferences |

- Nobody pretrains a large model in this course. Every team **chooses** one.
- Tasks 6 and 7 score generated text, so the decoding setting changes the score

Notes: Quiz 2 has just been sat. The choice of pretrained model is a choice of objective, data and size, which is Parts 1 and 2.

---
type: section
title: Part 1: Objectives and the loop
subtitle: What the model is paid to do

---
title: Three ways to make text predict itself

| Objective | The model sees | It predicts | Used for | Example |
|---|---|---|---|---|
| Causal | The prefix | The next token | Generation | GPT, Llama |
| Masked | The text with some tokens hidden | The hidden tokens | Classification, retrieval | BERT, XLM-R |
| Denoising | A corrupted text | The original text | Translation, summarising | BART |

- BERT hides 15% of the tokens and reads both left and right context
- The Task 12 baseline uses XLM-R, a masked model. Task 7 ranks Llama-3.1-8B, a causal model.
- Pick the objective that matches what you need at the output: a label, a vector or a text

Notes: BERT is Devlin et al. 2018; BART is Lewis et al. 2019. The Task 12 baseline is XLM-R plus ViT with concatenation, from the task card.

---
title: The causal objective rewards prediction, and only that

$$
L = -\frac{1}{N} \sum_{t} \log p_\theta(x_t \mid x_{<t})
$$

- N is the number of predicted tokens in the batch
- The model is paid for predicting its training text, with that text's gaps and errors
- A language with little text in the corpus contributes few terms to the sum
- Nothing in L mentions truth, instructions or safety. Days 4 and 5 add those.
- Day 1 said the same thing from the other side: a low loss is not a useful system

Notes: This is the NLL of day 1 divided by N. Ask what happens to Hausa if it is 0.1% of the tokens: 0.1% of the gradient.

---
title: The gradient is prediction minus target
subtitle: The lab's model: one row of logits for each previous character

$$
p_j = \mathrm{softmax}(W_{i,:})_j \qquad L = -\log p_y
$$
$$
\frac{\partial L}{\partial W_{i,j}} = p_j - 1[j = y]
$$

- i is the previous character, y the character that followed
- The target's logit is pushed up by 1 − p_y. Every other logit is pushed down by its own p_j.
- The row sums to zero: p sums to 1 and so does the one-hot target
- The same formula sits at the top of every language model. Backpropagation carries it down.

Notes: Derive it on the board: L = −z_y + log Σ exp(z_j), and the derivative of log-sum-exp is the softmax. The lab checks the formula against a finite difference.

---
title: Worked example: one gradient step

One row has logits (ln 5, ln 3, ln 2). The next character was number 1. Learning rate 1.

| | Token 0 | Token 1 (target) | Token 2 |
|---|---|---|---|
| Probability p | 0.5 | 0.3 | 0.2 |
| Gradient p − target | 0.5 | −0.7 | 0.2 |
| New logit | 1.109 | 1.799 | 0.493 |
| New probability | 0.283 | 0.564 | 0.153 |

- Loss before: −ln 0.3 = 1.204. Loss after: −ln 0.564 = 0.573.
- The gradient row sums to 0.5 − 0.7 + 0.2 = 0
- One example moved the target from 0.3 to 0.564. A thousand repeats of it would move it to nearly 1.

Notes: New logits are 1.609 − 0.5, 1.099 + 0.7 and 0.693 − 0.2. Exponentials: 3.03, 6.04, 1.64, sum 10.71. The last bullet sets up memorisation and the next slide.

---
title: The loop, and where documents end

```
for step in range(steps):
    p = softmax(W[x])                       # x: previous tokens
    p[np.arange(len(y)), y] -= 1            # prediction minus target
    grad = np.zeros_like(W)
    np.add.at(grad, x, p / len(y))          # sum over repeated contexts
    W -= lr * grad
    if step % every == 0:
        held = loss(W, dev_x, dev_y)        # no update on these
```

- Shift inputs and targets by one, once, inside each document
- Give every document its own start and end symbol. A pair that runs from the end of one document into the next teaches a transition that no text contains.
- Seed the initial weights. Evaluate on documents the updates never touch.

Notes: np.add.at matters: with plain indexed assignment, pairs that share a previous character overwrite each other and only one gradient survives.

---
title: Training loss keeps falling. Held-out loss turns.
layout: split

- A small causal Transformer, 28,256 weights, trained on four sentences for 100 steps
- Training loss falls to 0.09
- Development loss is lowest at step 60, at 1.25, then rises to 1.41
- Keep the weights from the turn, not the last ones
- The checkpoint was chosen on development data, so report the final score on a test set
- The lab shows the same turn in a bigram model with two training sentences

![](figures/03_curves.png)

Notes: The run is stored in course/validation/tiny_decoder_run.json and comes from extensions/train_tiny_decoder.py with seed 42. It is a toy corpus. In the lab the held-out loss goes from 2.42 at step 40 to 2.84 at step 400.

---
type: section
title: Part 2: Data and scale
subtitle: What goes in, and what it costs

---
title: Every filter changes who is in the data

| Step | What it does | What it can break |
|---|---|---|
| Provenance and licence | Records where text came from and whether you may use it | Nothing. Skipping it breaks your paper. |
| Language identification | Routes text to a language | Short, code-switched or closely related text is mislabelled and lost |
| Quality filter | Drops text a classifier scores low | A filter tuned on English drops good text in other languages |
| Deduplication | Removes repeated documents | Near-duplicates survive exact matching |
| Contamination check | Removes evaluation items | Paraphrases and translations of test items survive |

- Count tokens per language and domain **before and after** each step, and publish the table

Notes: Code-switching between Hausa and English, or Swahili and English, is ordinary on social media. A filter that keeps only confidently single-language text removes it. Day 6 builds data for African languages and returns to this table.

---
title: Contamination makes a model look better than it is

- If test documents are in the pretraining data, the model can score by recall
- A larger model can memorise more, so contamination can favour the larger of two models
  - a scaling comparison then measures memory, not generalisation
- You cannot inspect the pretraining data of most models. You can still check:
  - whether the benchmark was public before the model's training cut-off
  - whether the model completes a test item from its first half
- Task 12 has a rule for this: annotations of the evaluation paintings from any other source, including the original ArtELingo release, may not be used

Notes: The Task 12 rule is on the task card. Ask each team what the public sources of their task's evaluation items are.

---
title: Compute: six operations per weight per token
layout: stack

$$
C \approx 6ND
$$

![](figures/03_compute.png)

- N = 100 million weights, D = 2 billion tokens: C = 1.2 × 10¹⁸. At a sustained 10 TFLOP/s that is 120,000 s, or 33.3 hours.

Notes: Two operations per weight for the forward pass and four for the backward pass. The estimate ignores attention over long contexts, embeddings, the optimiser and idle hardware. A device's peak rate is not its sustained rate. The figure is the formula; no hardware was measured.

---
title: Chinchilla: a fixed budget buys weights or tokens
subtitle: Hoffmann et al., 2022

| Model | Weights N | Tokens D | Tokens per weight | 6ND |
|---|---|---|---|---|
| Gopher | 280 billion | 300 billion | 1.1 | 5.0 × 10²³ |
| Chinchilla | 70 billion | 1.4 trillion | 20 | 5.9 × 10²³ |

- About the same compute. The smaller model, trained on more tokens, scored higher.
- Their finding: for the best loss at a fixed budget, grow weights and tokens in equal proportion
- It was measured for one architecture and one kind of data. It is not a law that every model needs 20 tokens per weight.
- A smaller model is also cheaper every time someone uses it

Notes: The 6ND column is computed here from the two published sizes. For low-resource languages the binding limit is often D: there may not be 20 tokens of text per weight to train on.

---
title: Mixture of experts: many weights, few used per token

- Replace one MLP per block with several expert MLPs and a small router
- The router sends each token to one expert, or to a few
- **Total** weights set the storage. **Active** weights per token set much of the compute.
- What can go wrong:
  - the router sends most tokens to a few experts, so the rest learn nothing
  - experts on different devices cost communication
- When a model card gives two sizes, ask which one is active

Notes: Switch Transformers (Fedus et al., 2021) route each token to a single expert and add a load-balancing loss. Day 2 showed the MLP holds most of a block's weights, which is why the MLP is the part that gets split.

---
title: Continued pretraining moves a model towards your text

- Take a pretrained model and keep training it with the same objective on your domain or language
- It needs unlabelled text only, which a low-resource language may have when it lacks labels
- It can forget: scores on the original domains fall while the target score rises
- Report two numbers, one on the target and one on general text, before and after
- Estimate the cost with 6ND before you start. The exercise sheet does this for an 8-billion-weight model.

Notes: Gururangan et al. 2020, Don't Stop Pretraining, is the reference for domain- and task-adaptive pretraining. Day 4 covers cheaper adaptation with LoRA; day 6 covers language adaptation.

---
type: section
title: Part 3: Decoding
subtitle: From a distribution to a text

---
title: The model gives a distribution. Something else picks the token.
subtitle: After "Darasa linaanza" (Swahili: "the class starts")

| Next word | leo | kesho | sasa | hapa | tena |
|---|---|---|---|---|---|
| Meaning | today | tomorrow | now | here | again |
| Probability | 0.50 | 0.25 | 0.15 | 0.07 | 0.03 |

- **Greedy** takes the most probable token: leo, every time
- **Sampling** draws a token at random with these probabilities: leo half the time
- Every other method reshapes this distribution and then samples from it
- The probabilities are invented for the example. The lab uses the same five numbers.

Notes: All five continuations are grammatical Swahili. Entropy of this distribution is 1.83 bits.

---
title: Temperature sharpens or flattens
layout: split

$$
p_i(T) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
$$

- Divide the logits by T before the softmax
- T < 1 sharpens. As T goes to 0 it becomes greedy.
- T = 1 is the model's own distribution
- T > 1 flattens, towards uniform
- The order of the tokens never changes

![](figures/03_temperature.png)

Notes: In terms of probabilities, p_i(T) is proportional to p_i raised to the power 1/T. So T = 0.5 squares and T = 2 takes square roots.

---
title: Worked example: T = 0.5 squares the probabilities

| | leo | kesho | sasa | hapa | tena | Sum |
|---|---|---|---|---|---|---|
| p | 0.50 | 0.25 | 0.15 | 0.07 | 0.03 | 1 |
| p² | 0.2500 | 0.0625 | 0.0225 | 0.0049 | 0.0009 | 0.3408 |
| p² / 0.3408 | 0.734 | 0.183 | 0.066 | 0.014 | 0.003 | 1 |
| T = 2: √p, normalised | 0.348 | 0.246 | 0.191 | 0.130 | 0.085 | 1 |

- At T = 0.5 the entropy falls from 1.83 to 1.15 bits. At T = 2 it rises to 2.17.
- At T = 2 the least likely word is drawn about 1 time in 12, up from 1 in 33

Notes: Do the p² row on the board. The square roots are 0.707, 0.5, 0.387, 0.265, 0.173, sum 2.032. All numbers match notebook section 5.

---
title: Top-k and top-p cut the tail
layout: split

- The tail holds many tokens with small probabilities. Together they are drawn often, and they are where text goes wrong.
- **Top-k**: keep the k most probable tokens, renormalise
  - k = 2 keeps leo and kesho: 0.667 and 0.333
- **Top-p** (nucleus): keep the smallest set whose probabilities sum to at least p, renormalise
  - p = 0.8 needs 0.50 + 0.25 + 0.15 = 0.90, three words: 0.556, 0.278, 0.167
- Both set the removed tokens to exactly zero

![](figures/03_truncation.png)

Notes: Nucleus sampling is from Holtzman et al. 2019. Their argument: the tail of the distribution is unreliable, and truncating it keeps diversity while removing most incoherent choices.

---
title: Top-p follows the model's confidence

| Distribution | Probabilities | Kept by top-k, k = 2 | Kept by top-p, p = 0.8 |
|---|---|---|---|
| Peaked | 0.50, 0.25, 0.15, 0.07, 0.03 | 2 | 3 |
| Flat | 0.2, 0.2, 0.2, 0.2, 0.2 | 2 | 4 |
| One sure answer | 0.97, 0.01, 0.01, 0.005, 0.005 | 2 | 1 |

- Top-k keeps the same number everywhere, even when one answer is nearly certain
- Top-p keeps few tokens when the model is sure and many when it is not
- The methods combine. A common order is temperature first, then the cut.

Notes: The first two rows are printed by the lab's test cell. Third row: 0.97 alone exceeds 0.8, so one token is kept. With T = 2 and then p = 0.8 the example keeps four words.

---
title: Beam search looks for the most probable whole string
subtitle: Lab section 6: decoding the trained character model

| Method | Output | Log-probability |
|---|---|---|
| Greedy | the lab lab lab lab lab lab … | −31.76, and no end in 40 characters |
| Greedy with repetition penalty 1.3 | the lab s clay | −11.21 |
| Beam search, width 3 | the lay | −5.20 |

- Beam search keeps the best few partial strings at each step, and returns the best finished one
- It found a far more probable string than greedy. The string is in none of the training sentences.
- It prefers short outputs: each extra token multiplies in a probability below 1
  - real systems divide the score by a function of the length
- Use it when there is one right output, as in translation. For open-ended text it gives bland, repetitive output.

Notes: The bigram model is a toy with one character of memory. The length preference of beam search is a property of the method and appears in large models too.

---
title: Repetition: the most probable path can be a loop

- Greedy decoding took "the lab " back to "lab " for ever
- Holtzman et al. (2019) found the same in large models: decoding that maximises probability gives bland, repetitive text, unlike what people write
- Three remedies, each with a cost
  - sample with top-p: breaks loops, and makes runs differ
  - a repetition penalty lowers the logits of tokens already produced: also punishes names and terms that should repeat
  - forbid repeated n-grams: blunt, and breaks quotations
- Answers in Tasks 6 and 7 are short. Check whether your baseline needs any of these before you add one.

Notes: The repetition penalty is from the CTRL paper, Keskar et al. 2019. In a short factual answer a repeated word is often correct.

---
title: Decoding settings are part of the configuration

```
generation:
  do_sample: false        # greedy
  temperature: 1.0
  top_p: 1.0
  top_k: 0
  num_beams: 1
  repetition_penalty: 1.0
  max_new_tokens: 64
  seed: 42
```

- Changing any line changes the outputs and the score, with the same weights
- With sampling on, one run is one draw. Report the mean over several seeds.
- Record these beside the model revision and the prompt. Libraries change their defaults.
- Compare two systems under the same decoding settings, or say that you did not

Notes: The block is a sketch of what to record, not the syntax of a particular library. Many libraries load generation defaults from a file shipped with the model, so two models can decode differently with the same code.

---
type: big
title: Two teams test the same model on the same data. One uses greedy decoding, the other T = 1. Can you compare their scores?

Notes: No. The scores differ by the decoding method and by sampling noise, and you cannot tell those from a real difference without rerunning one under the other's settings. The T = 1 score is also one draw unless they averaged over seeds. This answers the opening question: the model's answer is a distribution, and a reported output is a model plus a decoding rule.

---
title: Lab: 40 minutes

- Open `course/tutorials/03_pretraining.ipynb`. NumPy only.
- Sections 1 to 4, together: train the bigram model, check the gradient, halve the data and watch held-out loss
- Sections 5 and 6, together: temperature and top-k on the five-word distribution; greedy and beam search
  - Work out the T = 0.5 row on paper first
- Section 7, in pairs: write two functions until the tests pass
  - `top_p_filter(probs, p)`
  - `train_with_selection`, which returns the best checkpoint and not the last
- Teams: find the generation settings in your task's baseline code

---
title: Readings

### Before the next session (20 minutes)
- Holtzman et al., 2019. The Curious Case of Neural Text Degeneration. Sections 1 to 3.

### Scale and data
- Hoffmann et al., 2022. Training Compute-Optimal Large Language Models. Read the abstract and the main results figure.
- Gururangan et al., 2020. Don't Stop Pretraining.

### Going deeper
- Kaplan et al., 2020. Scaling Laws for Neural Language Models.
- Fedus et al., 2021. Switch Transformers.
- Devlin et al., 2018. BERT. Lewis et al., 2019. BART.

Notes: Links are in the lecture notes.

---
title: Exit ticket

Write three lines and hand them in.

- The held-out loss and step of the checkpoint you selected in the lab
- The decoding settings of your task's baseline, or "not found yet"
- One filter that could remove your language's text from a pretraining corpus

Quiz 3 opens the next session: six questions, ten minutes.
