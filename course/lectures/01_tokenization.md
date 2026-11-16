# Lecture 01 — Language models, tokens and baselines; the four shared tasks

[Course home](../../README.md) · Week 1, day 1 · 120 minutes · [Slides](../slides/01_tokenization.pptx) · [Notebook](../tutorials/01_tokenization.ipynb) · [Exercises](../exercises/01_tokenization.md) · [Quiz](../quizzes/01_tokenization.md)

## Learning outcomes

- Choose one SemEval-2027 task with your team and name its metric, its data and its rules.
- Run one byte-pair-encoding merge by hand, and measure what a tokeniser costs in tokens per letter.
- Convert a negative log-likelihood into perplexity and bits per byte, and say which of the three can be compared across tokenisers.
- Split data by its source of dependence and report a baseline beside every result.

## Session plan

| Minutes | Activity |
|---|---|
| 0–10 | [Prerequisite diagnostic](../PREREQUISITES.md): students mark their own answers; go over questions 2, 4 and 6 |
| 10–25 | Slides 1–7: the capstone. Four tasks, one per team, and today's action |
| 25–45 | Slides 8–16: tokens. Units, Unicode, BPE by hand on the board, what tokenisers cost |
| 45–60 | Slides 17–24: NLL, perplexity and bits per byte; splits, leaks and baselines |
| 60–65 | Break |
| 65–105 | [Notebook 1](../tutorials/01_tokenization.ipynb): sections 1–5 together, section 6 in pairs. Collect each team's task choice |
| 105–115 | Slide 25: the Hausa token-count question, then the practical questions in pairs |
| 115–120 | Slides 26–28 and the exit ticket: task and languages, one measured number, one possible leak |

## Teaching notes

### The capstone starts today
Every team enters one of four SemEval-2027 tasks, one subtask or track, and at most three languages. The [task cards](../SEMEVAL_TASK.md) hold the facts; the organisers' pages govern.

| Task | You build | African languages | Metric | Starting point |
|---|---|---|---|---|
| 12 ArtEmotion-28 | Emotion and sentiment for paintings in 28 languages | Hausa, Igbo, Yoruba, Swahili, Kinyarwanda, Emakhuwa, isiZulu, isiXhosa, Setswana, isiNdebele, Darija | Macro-F1 (subtask 1), log loss (2), Pearson correlation (3) | Starter notebook and three baselines |
| 6 MMCultureQA | Image plus spoken or written question → short answer | Swahili, Amharic; Oromo and Somali announced | BERTScore F1 | Dev-test leaderboard on EvalAI |
| 7 CLaS | Steer an LLM to answer in a target language by editing activations | Swahili (target, track 2) | Language compliance and relevance | Prompting baseline and scorer |
| 9 Multimodal framing | Frame labels for news paragraphs and images | None | Micro-F1 | Trial data |

Task 12 is the default for a team with no strong preference: it has the most African languages, a 130 MB dataset and working baselines. The instructor co-organises Task 12. Students receive no test labels and no information that other teams lack, and each paper states the connection.

The day-1 action has three steps. Pick a task and write it on the class sheet. Clone its starter material tonight. Read the rules page from top to bottom and note which data and models are allowed. Task 7 teams also request access to the licence-gated Llama-3.1-8B weights today. By day 5 each team has run the organisers' baseline and matched its score. Evaluation runs 10–31 January 2027.

### A language model and its notation
A language model assigns a probability to a sequence. The chain rule gives the autoregressive factorisation $p(x_{1:T})=\prod_{t=1}^{T}p(x_t\mid x_{<t})$, and the model supplies each factor. Students built and fine-tuned such a model in the DeepMind path, so fix the notation and move on.

A useful assistant needs more than a low next-token loss: it must follow instructions, reach evidence and be scored on its task. Lower language-model loss does not establish that a system is factual or useful. Begin a project by naming the user, the input, the output, the unit of evaluation and the cost of an error.

### Tokens are an engineering choice
Whitespace words, characters, UTF-8 bytes and learned subwords give different lengths for the same string. "The class starts today." is 4 words, 23 code points and 23 bytes. The Swahili "Darasa linaanza leo." is 3 words, because Swahili puts subject, tense and verb in one word. The Amharic greeting ሰላም is 3 code points and 9 bytes.

A Unicode code point is not always a visible character. "é" is one code point (U+00E9) or two ("e" plus the combining accent U+0301). NFC normalisation joins such sequences where a joined form exists. Yoruba ọ̀ has no single code point, so NFC leaves it as ọ plus a grave mark, and the greeting "Ẹ káàárọ̀" stays at 9 code points. Normalise once, before splitting, and report the form. Careless normalisation erases distinctions that carry meaning, such as Yoruba tone marks.

Byte-level coverage removes the unknown-character problem. It does not make every language equally cheap: a byte model spends three positions on each Ge'ez letter and two on Hausa ɓ, ɗ and ƙ.

### Byte-pair encoding, by hand
Byte-pair encoding (Sennrich et al., 2015) starts with one symbol per character. It counts every adjacent pair of symbols in the training words, merges the most frequent pair into a new symbol, records the merge and repeats until the vocabulary is full. New text is tokenised by replaying the merges in order.

Take four Swahili verbs: *ninasoma* (I am reading), *anasema* (she is saying), *unalala* (you are sleeping), *tunakula* (we are eating). They hold 30 symbols.

| Step | Top pair counts | Corpus after merging the top pair | Symbols |
|---|---|---|---|
| 0 | | n i n a s o m a · a n a s e m a · u n a l a l a · t u n a k u l a | 30 |
| 1 | (n,a) 4, (l,a) 3 | n i na s o m a · a na s e m a · u na l a l a · t u na k u l a | 26 |
| 2 | (l,a) 3, others 2 | n i na s o m a · a na s e m a · u na la la · t u na k u la | 23 |
| 3 | (m,a), (na,s), (u,na) tie at 2 | a tie-break rule decides | |

The first merge is *na*, the present-tense marker, found by counting alone. The second, *la*, is half of *lala* and the end of *kula*: BPE follows frequency and has no notion of morphology. Step 3 is a tie, and every implementation fixes a rule for ties. The notebook takes the pair that sorts first.

Unigram tokenisation (Kudo, 2018) works in the other direction. It starts from a large candidate vocabulary, gives each piece a probability, and prunes the pieces whose removal costs the corpus likelihood least. It tokenises by choosing the segmentation with the highest probability. SentencePiece implements both methods.

### Why tokenisers cost more for African languages
Both methods learn from a corpus, and a language with little text in that corpus earns few merges. In the notebook a tokeniser trained on three English sentences covers a held-out English sentence in 6 tokens for 14 letters and needs 13 tokens for the 15 letters of "darasi ya fara yau". That is a toy and shows the mechanism only.

Petrov et al. (2023) measured production tokenisers on the same text translated into many languages. Token counts differed by up to 15 times. The gap remained for tokenisers trained to be multilingual. Character-level and byte-level models differed by over 4 times for some language pairs. The cost falls on the speaker: more money per request, more latency and less text in the context window.

Three causes can be checked on your own data: little training text in the language, a script outside ASCII, and long words. Report token counts per word and per UTF-8 byte, with the tokeniser, language and domain stated. A high token count measures cost. It does not measure how hard the language is or how accurate the model is.

### Loss and comparability
Negative log-likelihood is $\mathrm{NLL}=-\sum_t\log p(x_t\mid x_{<t})$, with natural logarithms. Perplexity is $\exp(\mathrm{NLL}/T)$. Bits per byte is $\mathrm{NLL}/(\log 2\cdot\text{bytes})$, where the bytes are those of the original text.

Perplexity divides by the token count $T$, and $T$ depends on the tokeniser. Token perplexities from different vocabularies cannot be compared. In the notebook one model with one total NLL of 33.17 nats has perplexity 6.31 per character prediction and 3,994 per word. Bits per byte uses a denominator no tokeniser can change, provided both models score the whole text, including the same boundary symbols.

### Splits come before modelling
Split by the source of dependence: the article for news paragraphs, the painting for captions, the author for tweets, the speaker for speech clips. Task 12 has 173,745 annotations of 1,658 paintings, about 100 per painting, so a random split by row would put every painting on both sides.

Work in this order. Remove duplicates. Assign splits. Fit the vocabulary, the normaliser and every other data-dependent step on training rows. Choose settings on development data. Open the test set once, after the method is frozen. In the notebook, adding the development document to the bigram counts lowers its perplexity from 6.31 to 4.75 with no change to the method.

### Baselines
A result needs a number beside it. Report a majority-class or uniform baseline, one cheap model, the organisers' baseline as you reproduced it, and your system with its new part removed. In the notebook the ladder is uniform 17, unigram 14.49, bigram 6.31. On Task 12 subtask 1 the organisers' multimodal baseline scores macro-F1 0.535 and their text-only baseline 0.550: the image made that system worse, and only the text-only row shows it.

## Worked example

A model gives three held-out tokens probabilities 0.5, 0.25 and 0.25. The text is 12 bytes long.

| Quantity | Working | Value |
|---|---|---|
| NLL in nats | $\ln 2+\ln 4+\ln 4=5\ln 2$ | 3.4657 |
| NLL in bits | $1+2+2$ | 5 |
| Perplexity | $\exp(3.4657/3)$ | 3.1748 |
| Bits per byte | $5/12$ | 0.4167 |

Cut the same text into 6 tokens with the same total NLL. Perplexity becomes $\exp(3.4657/6)=1.78$ and bits per byte stays 0.4167. A lower token perplexity after re-tokenising is not evidence of a better model.

## Guided tutorial

**Question:** How much do the unit, the split and the baseline change a reported number?

The notebook counts eight strings in words, code points and bytes; counts BPE pairs on the four Swahili verbs; fits a smoothed character bigram on three training documents; sweeps the smoothing strength on two development documents; and then leaks the development document into training. Students write `bpe_merge_step` and `bits_per_byte` against tests. The texts are a few sentences written for the course, and the numbers illustrate the method only.

**Shared-task path:** each team opens its task's training file and counts rows per label and per language. The exercise sheet asks for a majority baseline and a duplicate check on that file.

**Real tokeniser:** `extensions/hf_text.py` loads a production tokeniser to audit the same strings. See the [extension guide](../extensions/README.md) for commands and validation status.

## Practical questions

1. A tokeniser uses twice as many tokens for Hausa as for English on your sample. What can you conclude, and what can you not?
2. Why is fitting the vocabulary before splitting the data a leak?
3. Two sentences come from the same news article. Which splits may they go in?

## After class

Complete the [exercise sheet](../exercises/01_tokenization.md) in 45–60 minutes. Clone your task's starter material and read its rules page. Read sections 1–3 of Petrov et al. before the next session (20 minutes). [Quiz 1](../quizzes/01_tokenization.md) opens day 2.

## Readings

**Required**
- [Language Model Tokenizers Introduce Unfairness Between Languages](https://arxiv.org/abs/2305.15425) (Petrov et al., 2023). Sections 1–3: the token-count audit across languages, on production tokenisers.

**For your task**
- [SemEval-2027 task cards](../SEMEVAL_TASK.md). Read your task's card, then its rules page.

**Going deeper**
- [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) (Sennrich et al., 2015). The BPE paper; it gives the merge algorithm as a short Python listing.
- [SentencePiece](https://arxiv.org/abs/1808.06226) (Kudo and Richardson, 2018). The library that trains BPE and unigram tokenisers from raw text.
- [Subword Regularization](https://arxiv.org/abs/1804.10959) (Kudo, 2018). The unigram tokeniser and sampling over segmentations.
- [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/) (Jurafsky and Martin). The chapters on tokenisation and n-gram language models, for perplexity and smoothing.
- [AfriSenti](https://aclanthology.org/2023.emnlp-main.862/) (Muhammad et al., 2023). Sentiment data in 14 African languages; read the data-collection section.
