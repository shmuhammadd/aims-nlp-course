# Lecture 04 — Post-training I: prompting, instruction tuning and LoRA

[Course home](../../README.md) · Week 1, day 4 · 120 minutes · [Slides](../slides/04_adaptation.pptx) · [Notebook](../tutorials/04_adaptation.ipynb) · [Exercises](../exercises/04_adaptation.md) · [Quiz](../quizzes/04_adaptation.md)

## Learning outcomes

- Write a zero-shot and a few-shot prompt, choose between prompts on development data, and parse the output into labels.
- Build response-only labels for supervised fine-tuning.
- Derive and train a low-rank weight update, and count its parameters.
- Choose between a prompt, a LoRA adapter and full fine-tuning for a given amount of data and compute.

## Session plan

| Minutes | Activity |
|---|---|
| 0–10 | Quiz 3 (pretraining and decoding) |
| 10–30 | Slides 1–9: prompting and in-context learning, demonstration order, the prompt as a hyperparameter, structured output |
| 30–60 | Slides 10–19: SFT and chat templates, response-only labels, LoRA algebra on the board, QLoRA |
| 60–65 | Break |
| 65–105 | [Notebook 04](../tutorials/04_adaptation.ipynb): sections 1–4 together, section 5 in pairs |
| 105–115 | Slides 20–24: the decision table, routes into Tasks 12 and 9, the comparison table, safety after adaptation |
| 115–120 | Exit ticket: one measured result, one reason it may not transfer, the route your team will try first |

## Teaching notes

### Prompting and in-context learning
A zero-shot prompt states the task and gives the input. A few-shot prompt adds $k$ labelled demonstrations before the input. No weight changes; Brown et al. (2020) called this in-context learning. For Task 12 subtask 1 the prompt lists the eight emotion words, shows a few captions with their labels, and ends with the caption to label.

A prompt fixes more than its wording. It fixes the language of the instruction, the label words, the separator, and the number, choice and order of the demonstrations. Each of these moves the score.

### Demonstration choice and order
Lu et al. (2021) reordered the same demonstrations and saw accuracy range from near the best reported score to near chance, at every model size they tried. Zhao et al. (2021) named two causes. The model leans towards the label that is most frequent among the demonstrations, and towards the labels nearest the end of the prompt.

The lab builds a toy with the second property: a similarity vote over four demonstrations in which position $i$ has weight $\gamma^{k-1-i}$. With $\gamma=1$ all 24 orderings score 0.714. With $\gamma=0.5$ they score between 0.483 and 0.795, and the worst ordering, which ends with both +1 demonstrations, answers +1 for every query. Balance the labels, alternate them, try several orders and report the spread.

### The prompt is a hyperparameter
Choose the prompt on development data, as you would a learning rate. In the lab the best of 24 orderings scores 0.825 on 40 development queries and 0.795 on 2,000 test queries. The development score is the best of 24 noisy numbers, so it flatters the prompt. Report the test score once, state how many prompts you tried, and save the full prompt text with the demonstrations in order.

### Structured output
A language model returns text, and a scorer needs a label. Ask for one word from a closed list, or for JSON with named fields. Write the rule that maps a reply to a label before the run. Count the replies that fit no label, score them as wrong, and report the invalid rate beside the task metric.

### From text continuation to instruction following
Supervised fine-tuning (SFT) maximises the likelihood of the desired response given the prompt. A chat template turns role boundaries into special tokens. A template mismatch between training and inference can change behaviour even when the words are identical. Split by task or source before producing prompt variants, so that paraphrases do not leak across the split.

For response-only SFT, prompt and padding positions get the ignored label, −100 in common libraries, and the assistant's response positions contribute to the causal objective. The model still attends to prompt tokens. The model implementation usually shifts labels internally; do not shift them a second time. Supervise the end-of-sequence token, or the tuned model never learns to stop. For multi-turn data, decide whether every assistant span or only the final response is supervised. Inspect the token IDs and labels of real examples before training.

### LoRA algebra
For a frozen matrix $W\in\mathbb R^{d_{in}\times d_{out}}$, write $W'=W+sAB$, with $A\in\mathbb R^{d_{in}\times r}$, $B\in\mathbb R^{r\times d_{out}}$, and typically $s=\alpha/r$. This course uses row-vector inputs, so $Y=X(W+sAB)$. Many libraries store the transposed weight; check the convention before copying shapes.

Only $r(d_{in}+d_{out})$ adapter parameters train. With $G=\partial L/\partial W'$, the gradients are $\partial L/\partial A=sGB^\top$ and $\partial L/\partial B=sA^\top G$. With random $A$ and zero $B$, the initial function matches the base model, the first gradient of $A$ is zero and the first gradient of $B$ is not. With both factors at zero, both gradients are zero and nothing trains. After training, $W+sAB$ can be merged into one matrix, so inference costs the same as the base model.

LoRA removes the gradients and optimiser state for the frozen weights. It does not remove the base weights or the activation memory.

### Quantisation and adaptation
QLoRA (Dettmers et al., 2023) keeps the base frozen in 4-bit form and trains adapters at higher precision, with gradients passing through the quantised weights. It does not train all parameters in 4-bit arithmetic. Quantisation introduces approximation error, and kernel and device support decide whether a run is feasible. The NumPy lab learns a low-rank residual in a linear task, where every update is visible. `extensions/hf_lora.py` is the optional real-model script for response-only LoRA training; the [extension guide](../extensions/README.md) gives its validation status.

### When to prompt, when to adapt

| | Prompt | LoRA | Full fine-tune |
|---|---|---|---|
| Labelled examples | None to a few dozen | Hundreds to thousands | Thousands and more |
| Compute | Forward passes | One GPU, frozen base | Gradients and optimiser state for every weight |
| Stored per task | A text file | A few megabytes | A full copy of the model |
| Main risk | Score moves with wording and order | Overfits small data; can remove refusals | The same, and the model forgets more |

These are rules of thumb. Start with the cheapest column that beats your baseline on development data.

### The capstone link
Task 12 subtask 1 and Task 9 subtask 1 are classification tasks, so each can be attempted with an encoder classifier, a prompt or a LoRA adapter. For Task 12 the organisers' text-only baseline is XLM-R with macro-F1 0.550; a prompt needs the eight-label list and balanced demonstrations; an adapter needs caption-to-emotion instruction pairs. Task 9 is multi-label over 14 frames, so the encoder needs 14 sigmoid outputs and a prompt must return a list. Task 12 has not yet published its guidance on LLMs and extra data; check before building on a prompted model.

### Experimental discipline
Compare the majority class, the unadapted checkpoint with a zero-shot prompt, a few-shot prompt and the adapter under the same held-out protocol. Use a small learning-rate search on development data. Track trainable parameters, peak memory where measurable, time, the invalid-output rate and task quality. A better fit on eight training examples shows that the pipeline runs. It does not show generalisation. Inspect failures by language and by prompt style.

### Safety after adaptation
Fine-tuning can remove refusals the base model had. Qi et al. (2023) report that fine-tuning an aligned model on benign data can weaken its safety behaviour. After the LoRA lab, run one fixed set of refusal checks on the base and adapted checkpoints and report counts for each. Discussion: which setting would you vary first, rank or learning rate, and why? See the [safety thread](../SAFETY.md).

## Worked example

For a 4096×4096 projection with rank 8, full fine-tuning updates $4096^2=16{,}777{,}216$ weights. The adapter updates $8\times(4096+4096)=65{,}536$, which is 0.390625% of the matrix. At 4 bytes a number that is 64 MB against 0.25 MB. Biases and other adapted layers add parameters. A reduction in trainable weights does not give the same percentage reduction in total memory.

## Guided tutorial

**Question:** Fit a low-rank adapter to a frozen linear model, mark the tokens that carry loss, and measure how much the order of four demonstrations changes a toy in-context classifier.

The notebook trains a rank-2 adapter on a frozen 12×8 matrix (held-out error 0.054 before, about $10^{-9}$ after), builds response-only labels, and scores all 24 orderings of four demonstrations. Students then write `response_only_labels`, `lora_param_count` and `merge_lora` against tests. The data are synthetic and say nothing about a real model.

**Real-model extension:** `extensions/hf_lora.py` runs a small SFT experiment. Inspect the printed trainable parameters and token masks before reading the loss. See the [extension guide](../extensions/README.md) for commands, hardware and validation status.

## Practical questions

1. Why initialise only one LoRA factor to zero?
2. Should prompt tokens be removed from the input for response-only SFT?
3. What must be saved alongside an adapter to reproduce inference?
4. You tried 24 prompt orderings and the best scores 0.825 on 40 development items. What do you report, and what do you expect on test?

## After class

Complete the [exercise sheet](../exercises/04_adaptation.md) in 45–60 minutes. Read sections 1 and 4.1 of the LoRA paper before the next session (20 minutes). [Quiz 4](../quizzes/04_adaptation.md) opens day 5.

## Readings

**Required**
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) (Hu et al., 2021). Sections 1 and 4.1: the update, its initialisation and the parameter count.

**Prompting**
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) (Brown et al., 2020). Section 2 defines zero-shot, one-shot and few-shot prompting.
- [Fantastically Ordered Prompts and Where to Find Them](https://arxiv.org/abs/2104.08786) (Lu et al., 2021). Sections 1–3: the same demonstrations in different orders.
- [Calibrate Before Use](https://arxiv.org/abs/2102.09690) (Zhao et al., 2021). Section 4: majority-label and recency bias.

**Adapting**
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) (Dettmers et al., 2023). Read the abstract and section 3 for what is quantised and what trains.
- [Finetuned Language Models Are Zero-Shot Learners](https://arxiv.org/abs/2109.01652) (Wei et al., 2021). Instruction tuning across many tasks; read the introduction and figure 1.
- [PEFT quick tour](https://huggingface.co/docs/peft/quicktour). The library most teams will use for adapters.

**Safety**
- [Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To!](https://arxiv.org/abs/2310.03693) (Qi et al., 2023). Read the abstract and the benign fine-tuning experiment.
