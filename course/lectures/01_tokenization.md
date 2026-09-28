# Lecture 01 — Language models, tokens, and experimental baselines

[Course home](../../README.md) · Week 1, day 1 · 120 minutes

## Learning outcomes

- Distinguish next-token prediction from downstream usefulness.
- Measure tokenization cost across scripts without conflating it with language quality.
- Design a leakage-resistant baseline and train/development/test split.

## Session plan

| Minutes | Activity |
|---|---|
| 0–10 | Retrieval quiz: recall the previous session; day 1 uses the prerequisite diagnostic |
| 10–35 | Concepts and concrete examples from the first half of these notes |
| 35–60 | Board derivation, worked example, and failure analysis |
| 60–65 | Break |
| 65–105 | Guided [notebook tutorial](../tutorials/01_tokenization.ipynb) |
| 105–115 | Practical questions in pairs, then discuss evidence |
| 115–120 | Exit ticket: one result, one limitation, one next experiment |

## Teaching notes

### From an NLP task to a scientific question
A language model assigns a distribution to sequences. Autoregressive factorization is
$p(x_{1:T})=\prod_{t=1}^{T}p(x_t\mid x_{<t})$. A useful assistant additionally needs instruction following, access to evidence, and task-specific evaluation. Lower language-model loss does not by itself establish factuality or usefulness. Begin a project by specifying a user, input, output, unit of evaluation, and failure cost.

### Tokens are an engineering choice
Whitespace, character, UTF-8 byte, and learned subword tokenizers allocate different numbers of positions to the same string. Unicode code points are not always visible characters: a combining accent can occupy a separate code point. NFC normalization can join some equivalent sequences, but indiscriminate normalization can erase useful distinctions. Byte-level coverage avoids an unknown-character vocabulary problem; it does not make every language equally efficient.

Byte-pair encoding repeatedly merges frequent adjacent symbols. Unigram tokenization instead selects a probabilistic segmentation from a candidate vocabulary. Both learn from a data distribution; underrepresented orthographies may fragment more. Report token counts per word and per UTF-8 byte, with a stated tokenizer, language, and domain. Neither metric alone measures linguistic complexity.

### Loss and comparability
Negative log-likelihood (NLL) is $-\sum_t\log p(x_t\mid x_{<t})$. Perplexity is $\exp(\mathrm{NLL}/T)$, with natural logarithms. Token perplexities across different vocabularies are generally not directly comparable. Bits per byte uses total NLL divided by $\log(2)$ and the original byte count, provided the likelihood and boundary conventions are comparable.

### Splits precede modelling
Split by the source of dependence: document, author, speaker, or image, as appropriate. Deduplicate before assigning splits, fit vocabulary and statistics on training data, select configurations on development data, and open the test set only after freezing the method. A random row split can put paraphrases or frames of the same video in both train and test. A majority or unigram baseline reveals whether a sophisticated system adds value.

## Worked example

A held-out sequence with token probabilities 0.5, 0.25, 0.25 has NLL 3.4657 and perplexity 3.1748. Doubling the number of segmentation units changes the denominator, so a lower token perplexity after retokenization is not evidence of a better model.

## Guided tutorial

**Question:** Compare byte, character, and whitespace tokenization; fit a smoothed character bigram baseline; measure held-out perplexity.

Open the notebook and predict each result before running it. Spend 5 minutes on setup and data inspection, 15 on the mechanism, 10 on the controlled intervention, and 10 on interpretation. The core uses NumPy and authored/synthetic data; the notebook identifies its limits. Save outputs, configuration, and a short interpretation. Do not report an expected trend as a measured result.

**Real-model or research extension:** Use the real tokenizer in `extensions/hf_text.py` to audit the same examples. Record the checkpoint revision; model inference is optional.

See the [extension guide](../extensions/README.md) for commands, hardware assumptions, and validation status.

## Practical questions

1. A tokenizer uses twice as many tokens for Hausa as English on your sample. What can and cannot be concluded?
2. Why is fitting the vocabulary before splitting a form of leakage?
3. How should two sentences from the same source document be split?

## After class

Complete the [exercise sheet](../exercises/01_tokenization.md) in 45–60 minutes. Read the first reference's abstract and method overview before the next session; remaining references are optional depth. For long papers, focus on the objective, one figure/table, and one limitation.

## Readings

- [Speech and Language Processing, online book](https://web.stanford.edu/~jurafsky/slp3/)
- [AfriSenti (2023)](https://aclanthology.org/2023.emnlp-main.862/)
