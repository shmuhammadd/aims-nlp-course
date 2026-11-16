# Lecture 03 — Pretraining: data, objectives, scaling; decoding and sampling

[Course home](../../README.md) · Week 1, day 3 · 120 minutes · [Slides](../slides/03_pretraining.pptx) · [Notebook](../tutorials/03_pretraining.ipynb) · [Exercises](../exercises/03_pretraining.md) · [Quiz](../quizzes/03_pretraining.md)

## Learning outcomes

- Train a next-token model with a correctly shifted objective and a gradient you have checked.
- Select a checkpoint on held-out loss, and say why the selected score is not a test score.
- Estimate training compute with $6ND$ and reason about how a data pipeline changes what a model learns.
- Apply greedy decoding, temperature, top-k, top-p and beam search to a distribution, and record decoding settings as part of an experiment.

## Session plan

| Minutes | Activity |
|---|---|
| 0–10 | Quiz 2 (the causal Transformer) |
| 10–35 | Slides 1–17: objectives, the bigram gradient on the board, the loop, data engineering, $6ND$, Chinchilla, mixture of experts, continued pretraining |
| 35–60 | Slides 18–26: decoding. One distribution reshaped by temperature, top-k and top-p; beam search; repetition; settings as configuration |
| 60–65 | Break |
| 65–105 | [Notebook 3](../tutorials/03_pretraining.ipynb): sections 1–6 together, section 7 in pairs |
| 105–115 | Slide 27: the two-teams question, then the practical questions in pairs |
| 115–120 | Slides 28–30 and the exit ticket: the checkpoint you selected, your baseline's decoding settings, one filter that could remove your language |

## Teaching notes

### What the objective rewards
Causal pretraining minimises $L=-\frac1N\sum_t\log p_\theta(x_t\mid x_{<t})$. The model is rewarded for predicting its training text, including that text's omissions and errors. Masked language modelling hides some tokens (15% in BERT) and predicts them from context on both sides. Encoder-decoder denoising, as in BART, corrupts a text and reconstructs it. Choose the objective by what you need at the output: a causal model for generation, a masked model for labels and vectors, a denoising model for conditional generation such as translation. The Task 12 baseline uses XLM-R, a masked model; Task 7 ranks Llama-3.1-8B, a causal one.

### A training derivation you can check
The notebook's model stores one logit vector $W_{i,:}$ for each preceding character $i$. Its prediction is $p_j=\mathrm{softmax}(W_{i,:})_j$. For target $y$, $\partial L/\partial W_{i,j}=p_j-\mathbf 1[j=y]$. Each gradient row sums to zero. Average the gradients over examples and apply gradient descent. This neural bigram model has no long-range context; its value is that the whole train-and-evaluate loop fits on one screen.

Initialise from a seed. Shift inputs and targets once, inside each document. Compute the training loss, update the weights, then evaluate without updates on separate documents. Do not concatenate documents without boundary symbols: a pair that runs from the end of one document into the start of the next teaches a transition no text contains.

Repeated passes drive training loss down while held-out loss rises. A measured 100-step run of a 28,256-weight causal Transformer on the course corpus is stored in [validation/tiny_decoder_run.json](../validation/tiny_decoder_run.json): training loss falls to 0.09, and development loss is lowest at step 60 (1.25) and rises to 1.41. Keep the checkpoint from the turn. That choice used the development set, so the final score comes from a test set. These are toy-corpus measurements.

### Data engineering is part of modelling
A pretraining pipeline needs provenance, permission to use the data, language identification, quality filters, deduplication and contamination checks. Every filter changes who is represented. An English-oriented quality classifier can discard useful code-switched text, and language identification is weakest on short, mixed or closely related text. Record token volume before and after each filter, for each language and domain. Deduplication reduces the pressure to memorise and leaves near-duplicates in place.

Contamination means evaluation items appear in the training data. A contaminated model can score by recall, and a larger model can memorise more, so a scaling comparison on contaminated data can measure memory. Task 12 has a rule against one route: annotations of the evaluation paintings from any other source, including the original ArtELingo release, may not be used.

### Scaling under constraints
A dense Transformer needs roughly $C\approx6ND$ floating-point operations to train, for $N$ parameters and $D$ training tokens. Attention over long contexts, embeddings, the optimiser and hardware overhead are left out. It is an estimate and not a promise about running time.

The best split of a compute budget between parameters and tokens depends on the loss regime and on how often the model will be used. The Chinchilla study (Hoffmann et al., 2022) challenged the practice of training large models on few tokens: Chinchilla, with 70 billion parameters and 1.4 trillion tokens, outperformed Gopher, with 280 billion parameters and 300 billion tokens, at about the same compute ($6ND$ gives $5.9\times10^{23}$ and $5.0\times10^{23}$). The paper finds that parameters and tokens should grow in equal proportion in its setting. That is 20 tokens per parameter for Chinchilla, and it is not a fixed ratio for every architecture or application.

A mixture-of-experts model routes each token to a subset of feed-forward experts. Total parameters set the storage; active parameters describe much of the compute. Load balancing, communication, routing stability and expert capacity still matter.

Continued pretraining on a target domain or language can help, and it needs only unlabelled text (Gururangan et al., 2020). It can also forget earlier capabilities, so compare target and general-domain evaluations before and after.

### Decoding: from a distribution to a text
A model outputs a distribution over the next token. A decoding rule turns it into a token. Take five Swahili continuations of "Darasa linaanza" with invented probabilities: *leo* 0.50, *kesho* 0.25, *sasa* 0.15, *hapa* 0.07, *tena* 0.03.

- **Greedy** takes the most probable token every time: *leo*.
- **Temperature** divides the logits by $T$ before the softmax, which raises each probability to the power $1/T$ and renormalises. $T<1$ sharpens, $T>1$ flattens, and the order of tokens does not change. As $T\to0$ it becomes greedy.
- **Top-k** keeps the $k$ most probable tokens and renormalises. With $k=2$: 0.667 and 0.333.
- **Top-p**, or nucleus sampling (Holtzman et al., 2019), keeps the smallest set of most probable tokens whose mass reaches $p$ and renormalises. With $p=0.8$ the set is three words (mass 0.90): 0.556, 0.278, 0.167. On a flat distribution of five tokens at 0.2 each, the same $p$ keeps four. Top-p follows the model's confidence; top-k does not.
- **Beam search** keeps the best few partial sequences at each step and returns the finished sequence with the highest total log-probability. It prefers short outputs, because each extra token multiplies in a probability below one, so real systems normalise the score by length.

Decoding that maximises probability tends to give bland and repetitive text in open-ended generation (Holtzman et al., 2019). In the notebook, greedy decoding of the bigram model loops on "the lab lab lab" and never reaches the end symbol in 40 characters. Beam search with width 3 returns "the lay" at log-probability −5.20, far above greedy's −31.76, and the string is in none of the training sentences. A repetition penalty (Keskar et al., 2019) lowers the logits of tokens already produced; it breaks loops and also punishes names and terms that should repeat.

### Decoding settings are configuration
Temperature, top-p, top-k, beam width, repetition penalty, maximum length and the sampling seed all change the output of fixed weights. Record them beside the model revision and the prompt. With sampling on, one run is one draw, so report a mean over seeds. Two systems compared under different decoding settings have not been compared. Tasks 6 and 7 score generated text, so this applies to them directly.

## Worked example

**Compute.** With $N=100$ million parameters and $D=2$ billion tokens, $6ND\approx1.2\times10^{18}$ operations. At a sustained 10 TFLOP/s that is 120,000 seconds, or 33.3 hours, before the omitted costs. A device's peak throughput is not its sustained throughput.

**Decoding.** Temperature 0.5 squares the five probabilities and renormalises.

| | leo | kesho | sasa | hapa | tena | Sum |
|---|---|---|---|---|---|---|
| $p$ | 0.50 | 0.25 | 0.15 | 0.07 | 0.03 | 1 |
| $p^2$ | 0.2500 | 0.0625 | 0.0225 | 0.0049 | 0.0009 | 0.3408 |
| $T=0.5$ | 0.734 | 0.183 | 0.066 | 0.014 | 0.003 | 1 |
| $T=2$ | 0.348 | 0.246 | 0.191 | 0.130 | 0.085 | 1 |
| Top-k, $k=2$ | 0.667 | 0.333 | 0 | 0 | 0 | 1 |
| Top-p, $p=0.8$ | 0.556 | 0.278 | 0.167 | 0 | 0 | 1 |

The entropy is 1.83 bits for the model's distribution, 1.15 at $T=0.5$, 2.17 at $T=2$, 0.92 for top-k and 1.42 for top-p.

## Guided tutorial

**Question:** Which checkpoint and which decoding rule would you report, and on what evidence?

The notebook trains a character bigram model on four sentences with an explicit gradient, checks the gradient by finite differences, then retrains on two sentences to show held-out loss turning upward (2.42 at step 40, 2.84 at step 400). It reshapes the five-word distribution with temperature and top-k, and decodes the trained model with greedy search, a repetition penalty and beam search. Students write `top_p_filter` and `train_with_selection` against tests. The corpus is four authored sentences and the results are illustrations.

**Shared-task path:** each team finds the generation settings in its task's baseline code and records them on the exercise sheet.

**Real model:** `extensions/train_tiny_decoder.py` trains a small randomly initialised causal Transformer on the same corpus; the stored run above came from it. See the [extension guide](../extensions/README.md) for commands and validation status.

## Practical questions

1. Training perplexity keeps improving while the system gets worse at its task. Give two different reasons.
2. Some test documents also appear in the training data of the larger of two models. What happens to a scaling comparison?
3. Why is a randomly initialised bigram model not a small copy of a large language model?
4. Two teams report results for the same model on the same test set, one with greedy decoding and one with temperature 1.0. Can their numbers be compared?

## After class

Complete the [exercise sheet](../exercises/03_pretraining.md) in 45–60 minutes. Read sections 1–3 of Holtzman et al. before the next session (20 minutes). [Quiz 3](../quizzes/03_pretraining.md) opens day 4.

## Readings

**Required**
- [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) (Holtzman et al., 2019). Sections 1–3: why maximising probability gives repetitive text, and nucleus sampling.

**Scale and data**
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) (Hoffmann et al., 2022). The Chinchilla study; read the abstract and the main results figure.
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) (Kaplan et al., 2020). The source of the compute estimate used here.
- [Don't Stop Pretraining](https://arxiv.org/abs/2004.10964) (Gururangan et al., 2020). Continued pretraining on domain and task text.
- [Switch Transformers](https://arxiv.org/abs/2101.03961) (Fedus et al., 2021). Mixture of experts with one expert per token.

**Going deeper**
- [BERT](https://arxiv.org/abs/1810.04805) (Devlin et al., 2018). The masked objective.
- [BART](https://arxiv.org/abs/1910.13461) (Lewis et al., 2019). The denoising objective.
- [CTRL](https://arxiv.org/abs/1909.05858) (Keskar et al., 2019). The repetition penalty is in the section on sampling.
- [Diverse Beam Search](https://arxiv.org/abs/1610.02424) (Vijayakumar et al., 2016). What beam search returns and how to make its outputs differ.
