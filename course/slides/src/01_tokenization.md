type: title
title: Language models, tokens and baselines
subtitle: Day 1 · Week 1
footer: Shamsuddeen Hassan Muhammad | AIMS 2026

---
type: big
title: Your team scores macro-F1 0.54 on Task 12. Is that good?

Notes: Take answers for one minute. Nobody can say without a baseline. The organisers' baseline for subtask 1 is 0.535 with the image and 0.550 with the caption alone, so 0.54 sits between two baselines. Today is about the numbers you need beside your own number.

---
title: Three weeks, one paper
subtitle: Every session feeds the shared-task system your team submits

| Week | Sessions | Your team hands in |
|---|---|---|
| 1 Foundations and post-training | Tokens, Transformer, pretraining, LoRA, alignment | Day 5: the organisers' baseline, reproduced |
| 2 Languages and modalities | Multilingual data, CLIP, VLMs, speech, evaluation | Day 10: experiment plan, frozen |
| 3 Inside the model, systems, paper | Steering, retrieval and agents, efficiency, paper clinic | Day 15: paper, package, defence |

- Each day: 10-minute quiz, 50 minutes of slides, 40-minute lab, one exercise sheet
- Today has no quiz. We mark the prerequisite diagnostic instead.

Notes: The deadlines are the class fields in SEMEVAL_TASK.md. Day 14 is the peer-review exchange.

---
type: section
title: Part 1: The capstone
subtitle: One shared task per team

---
title: Five tasks, and your team enters one
subtitle: SemEval-2027 shared task (evaluation 10 to 31 January 2027) and AfricaNLP 2027 shared task (test phase to 5 January).

| Task | You build | African languages | Starting point |
|---|---|---|---|
| 12 ArtEmotion-28 | Emotion and sentiment for paintings in 28 languages | Hausa, Igbo, Yoruba, Swahili and 7 more | Starter notebook, three baselines |
| 6 MMCultureQA | Image plus spoken or written question → short answer | Swahili, Amharic; Oromo and Somali announced | Dev-test leaderboard on EvalAI |
| 7 CLaS | Steer an LLM to answer in a target language | Swahili, as a target | Prompting baseline and scorer |
| 9 Multimodal framing | Frame labels for news paragraphs and images | None | Trial data |
| AfriGuard | Refuse harmful prompts in 10 African languages | Amharic, Hausa, Igbo, Oromo and 6 more | Starter notebook on Colab |

Notes: Read from the task websites on 16 November 2026. The other seven languages in Task 12 are Kinyarwanda, Emakhuwa, isiZulu, isiXhosa, Setswana, isiNdebele and Darija. Task 9 covers Bulgarian, English, Portuguese and Russian. AfriGuard is the AfricaNLP 2027 task; the other four are SemEval-2027. The organisers' pages govern.

---
title: Each task is scored differently
subtitle: Know your metric before you write a line of code

| Task | Metric | Where you submit |
|---|---|---|
| 12, subtask 1 | Macro-F1 per language, then averaged | Codabench |
| 12, subtask 2 | Log loss in bits per annotator | Codabench |
| 6 | BERTScore F1 | EvalAI (dev-test) |
| 7 | Language compliance and relevance | Code sent to the organisers |
| 9 | Micro-F1 over 14 frames | Codabench |

- One task, one subtask or track, at most three languages
- No strong preference? Take Task 12: most African languages, a 130 MB dataset, working baselines.
- I co-organise Task 12. You get no test labels and nothing other teams lack. Your paper states the link.

Notes: Task 12 subtask 3 is ranked on within-painting Pearson correlation. Task 7 gives two test submissions and one final submission per track. The Task 6 test-phase platform was not stated on its page.

---
title: Today's action: pick, clone, read
subtitle: Before you leave the room

- **Pick** a task with your partner and write it on the class sheet
- **Clone** its starter material tonight
  - Task 12: `github.com/artemotion-28/artemotion-28`
  - Task 7: `github.com/d-gurgurov/CLaS-SemEval-2027`
  - Tasks 6 and 9: the data links on the task sites
- **Read** the rules page from top to bottom. Note what data and which models are allowed.
- Task 7 teams: request access to the Llama-3.1-8B weights today. The licence gate takes time.
- Day 5 deadline: you have run the organisers' baseline and matched its score

Notes: Task sites are listed in SEMEVAL_TASK.md. Walk the room during the lab and check that every pair has written a task down.

---
type: section
title: Part 2: Tokens
subtitle: What the model counts, and what that costs

---
title: A language model predicts the next token
subtitle: You built one in the DeepMind path. Here is the notation we will use.

$$
p(x_{1:T}) = \prod_{t=1}^{T} p(x_t \mid x_{<t})
$$

- The chain rule makes this exact. The model only has to supply each factor.
- x is a sequence of **tokens**. Someone chose what a token is.
- A low loss on next-token prediction does not make an assistant factual or useful
- Start a project by naming five things: the user, the input, the output, the unit you score, and what an error costs

Notes: Keep this brisk. Students fine-tuned a small language model in the DeepMind path. Ask one student to name the five things for Task 12 subtask 1: a museum visitor, a painting and a caption, one of eight emotions, one annotation, a wrong emotion label.

---
title: The same string has four lengths
subtitle: Notebook section 1

| Text | Words | Code points | UTF-8 bytes |
|---|---|---|---|
| The class starts today. | 4 | 23 | 23 |
| Darasi ya fara yau. (Hausa) | 4 | 19 | 19 |
| Darasa linaanza leo. (Swahili) | 3 | 20 | 20 |
| ƙasa (Hausa, "country") | 1 | 4 | 5 |
| Ẹ káàárọ̀ (Yoruba, "good morning") | 2 | 9 | 17 |
| ሰላም (Amharic, "hello") | 1 | 3 | 9 |

- Swahili packs "the class starts" into two words. A word count penalises nobody and compares nothing.
- State the unit every time you report a length or a cost

Notes: The first three rows say "the class starts today". Ask which unit a byte-level model pays for the Amharic greeting: 9 positions for 3 letters.

---
title: One letter can be two code points
layout: split

- "é" can be one code point (U+00E9) or "e" plus a combining accent (U+0301)
- They look the same and compare as different strings
- **NFC** normalisation joins them where a joined form exists
- Yoruba ọ̀ has no single code point. NFC leaves it as ọ plus a grave mark.
- Normalise once, before you split the data, and say which form you used
- Careless normalisation removes tone marks, and tone marks carry meaning in Yoruba

![](figures/01_bytes.png)

Notes: The figure uses the notebook's strings. Amharic letters take 3 bytes each in UTF-8. Hausa in the standard Latin alphabet takes 1 byte a letter except the hooked letters ɓ, ɗ, ƙ, which take 2.

---
title: BPE merges the most frequent pair, again and again
subtitle: Byte-pair encoding (Sennrich et al., 2015)

```
symbols = one per character
repeat until the vocabulary is full:
    count every adjacent pair of symbols in the training words
    merge the most frequent pair into one new symbol
    record the merge
```

- To tokenise new text, replay the recorded merges in order
- Frequent words end up as one token. Rare words fall back to pieces.
- The merges come from the training text. Text unlike it gets few merges.

Notes: Byte-level BPE, as in GPT-2, starts from the 256 byte values, so no string is out of vocabulary. Coverage is not the same as efficiency.

---
title: Worked example: two merges by hand
subtitle: Four Swahili verbs: ninasoma, anasema, unalala, tunakula

| Step | Top pair counts | Corpus after merging the top pair | Symbols |
|---|---|---|---|
| 0 | | n i n a s o m a · a n a s e m a · u n a l a l a · t u n a k u l a | 30 |
| 1 | (n,a) 4, (l,a) 3 | n i na s o m a · a na s e m a · u na l a l a · t u na k u l a | 26 |
| 2 | (l,a) 3, others 2 | n i na s o m a · a na s e m a · u na la la · t u na k u la | 23 |
| 3 | three pairs tie at 2 | you need a rule | |

- "na" is the Swahili present-tense marker. BPE found it by counting.
- Step 3 is a tie between (m,a), (na,s) and (u,na). The lab takes the pair that sorts first.

Notes: Do this on the board first. The verbs mean I am reading, she is saying, you are sleeping, we are eating. Count pairs across all four words: n-a occurs once in each. Merging removes one symbol per occurrence: 30 − 4 = 26, then 26 − 3 = 23.

---
title: Four ways to cut text
subtitle: Each trades vocabulary size against sequence length

| Tokeniser | Vocabulary | Sequence | Unknown words |
|---|---|---|---|
| Whitespace words | Huge, open | Short | Many |
| Characters | Hundreds to thousands | Long | Rare |
| UTF-8 bytes | 256 | Longest | None |
| BPE | You choose, often 30k to 250k | Medium | None, if it starts from bytes |
| Unigram | You choose | Medium | None, with a character fallback |

- **Unigram** (Kudo, 2018) starts from a large candidate vocabulary, gives each piece a probability, and prunes the pieces the corpus likelihood misses least
- BPE builds up by merging. Unigram cuts down by pruning. SentencePiece implements both.

Notes: Unigram picks the segmentation with the highest product of piece probabilities, and can sample other segmentations as regularisation. That is the subject of the Kudo 2018 paper.

---
title: A tokeniser trained on English spells Hausa out
subtitle: Lab result: merges learned on three English sentences
layout: stack

![](figures/01_bpe_cost.png)

- After 20 merges: English 6 tokens for 14 letters, Hausa 13 for 15, Swahili 16 for 17

Notes: Toy numbers from notebook section 6, three training sentences. They show the mechanism and not its size. The only merge that helps Hausa and Swahili is a-r, learned from "starts".

---
title: Why African languages cost more tokens
subtitle: Petrov et al., 2023, measured production tokenisers

- The same text in different languages can differ up to **15 times** in token count
- The gap remains for tokenisers trained to be multilingual
- Character and byte models also differ, by over 4 times for some language pairs

### Three causes you can check on your own data
- Little training text in the language, so few of its words earned merges
- Scripts outside ASCII: 2 bytes a letter for ɓ, ɗ, ƙ; 3 for Ge'ez
- Long words: Swahili and isiZulu put subject, tense and verb in one word

### What it costs
- More money per request, more latency, less text in the context window

Notes: The three findings are from the paper's abstract. The paper compares languages on parallel text. Do not quote a ratio for Hausa or Swahili from memory: measure it on your task data with the tokeniser of the model you will use.

---
type: section
title: Part 3: Measure
subtitle: What number, on what data, against what?

---
title: Three numbers from one sum
subtitle: Natural logarithms unless we say bits

$$
\mathrm{NLL} = -\sum_{t=1}^{T} \log p(x_t \mid x_{<t})
$$
$$
\mathrm{PPL} = \exp(\mathrm{NLL}/T) \qquad \mathrm{bits\ per\ byte} = \frac{\mathrm{NLL}}{\log 2 \cdot \mathrm{bytes}}
$$

- NLL is a total. It grows with the length of the text.
- Perplexity divides by the number of tokens T. Change the tokeniser and T changes.
- Bits per byte divides by the bytes of the original text. No tokeniser can change that.

Notes: Perplexity is the size of a fair die that would be as hard to predict. A uniform model over V symbols has perplexity V.

---
title: Worked example: three tokens, twelve bytes

A model gives the three held-out tokens probabilities 0.5, 0.25 and 0.25. The text is 12 bytes long.

| Quantity | Working | Value |
|---|---|---|
| NLL in nats | ln 2 + ln 4 + ln 4 = 5 ln 2 | 3.466 |
| NLL in bits | 1 + 2 + 2 | 5 |
| Perplexity | exp(3.466 / 3) | 3.175 |
| Bits per byte | 5 / 12 | 0.417 |

- Now cut the same text into 6 tokens with the same total NLL
  - Perplexity becomes exp(3.466 / 6) = 1.78. Bits per byte stays 0.417.
- A lower token perplexity after re-tokenising is not a better model

Notes: Do the first table on the board. Then ask what happens to each row when the token count doubles and the total stays the same.

---
title: The unit changes the number
subtitle: Lab section 5: one model, one total NLL of 33.17 nats

| Divide the NLL by | Count | Perplexity |
|---|---|---|
| Character predictions | 18 | 6.31 |
| Words | 4 | 3,994 |

- Same model, same text, same total
- Two papers that report perplexity with different tokenisers have not been compared
- Compare language models across tokenisers in bits per byte, or on the task metric
- For your shared task the organisers fix the metric. Use their scorer and no other.

Notes: 33.17 / 18 = 1.843 and exp(1.843) = 6.31. 33.17 / 4 = 8.29 and exp(8.29) = 3,994. The development document is "the lab ends soon".

---
title: Split by whatever makes two rows alike
subtitle: Two rows from the same source are not independent

| Data | Keep together in one split |
|---|---|
| News paragraphs (Task 9) | The article |
| Captions of paintings (Task 12) | The painting |
| Questions about images (Task 6) | The image |
| Tweets | The author |
| Speech clips | The speaker |

- Order of work: remove duplicates, then split, then fit anything that learns from data
- A vocabulary, a normaliser and a threshold all learn from data. Fit them on training rows only.
- Use the organisers' split when they give one. Check it for repeats yourself.

Notes: Task 12 has 173,745 annotations of 1,658 paintings, so each painting appears about 100 times. A random split by row puts every painting in both training and validation.

---
title: A leak improves the number and not the model
layout: split

- The bigram model beats the unigram, which beats uniform: 6.31, 14.49, 17
- Add the development document to the counts and perplexity falls to 4.75
- Nothing about the method changed
- Leaks in real projects are quieter:
  - the same tweet, retweeted, in both splits
  - a tokeniser or vocabulary fitted on all the data
  - a learning rate chosen on test accuracy

![](figures/01_ladder.png)

Notes: Notebook sections 3 and 5, alpha 0.5. The development document is 17 characters, so this is an illustration. Diagnostic question 4 asked why choosing a learning rate on test accuracy spoils the test estimate.

---
title: Your number needs a number beside it
subtitle: The baselines your paper will report

| Baseline | What it tells you |
|---|---|
| Majority class or uniform | The floor. Is the metric hiding class imbalance? |
| A cheap model: unigram, bag of words | How much the easy signal gives |
| The organisers' baseline, reproduced | That your pipeline and scorer are right |
| Your system with one part removed | That the part you added does the work |

- Task 12, subtask 1: the organisers' multimodal baseline scores macro-F1 0.535. Their text-only baseline scores 0.550.
- Adding the image made that baseline worse. A table without the text-only row would hide it.

Notes: Both baseline figures are from the Task 12 card. This answers the opening question: 0.54 is between the two baselines.

---
title: Choose on development data, report on test once
layout: split

- Smoothing strength α is a setting, like a learning rate
- Development text with nothing new: the smallest α wins
- Development text with two unseen transitions: α = 0.1 wins, and 0.01 is worse
- The α you pick was fitted to the development set. Its score there flatters you.
- Open the test set after the method is frozen
- Task 7 enforces this: two test submissions and one final per track

![](figures/01_smoothing.png)

Notes: Notebook section 4. At alpha 0.01 the perplexities are 2.90 and 6.10; at 0.1 they are 3.76 and 5.44.

---
type: big
title: A tokeniser uses twice as many tokens for Hausa as for English on your sample. What can you conclude?

Notes: You can conclude that Hausa costs twice as much on this sample with this tokeniser: money, latency and context. You cannot conclude that the model is less accurate on Hausa, or that Hausa is a harder language. Ask what would make the comparison fair: parallel text, the same domain, more than one sample.

---
title: Lab: 40 minutes

- Open `course/tutorials/01_tokenization.ipynb`. NumPy only.
- Sections 1 to 5, together: count units, count BPE pairs, fit the baseline ladder, sweep α, cause a leak
  - Predict each output before you run the cell
- Section 6, in pairs: write two functions until the tests pass
  - `bpe_merge_step`, with the tie-break rule
  - `bits_per_byte` from token log-probabilities
- While you work, I come round for your team's task choice

---
title: Readings

### Before the next session (20 minutes)
- Petrov et al., 2023. Language Model Tokenizers Introduce Unfairness Between Languages. Sections 1 to 3.

### For your task
- The rules page of your shared task, and its card in `course/SEMEVAL_TASK.md`

### Going deeper
- Sennrich et al., 2015. Neural Machine Translation of Rare Words with Subword Units. The BPE paper.
- Kudo and Richardson, 2018. SentencePiece.
- Jurafsky and Martin. Speech and Language Processing, the chapters on tokenisation and n-gram models.

Notes: Links are in the lecture notes.

---
title: Exit ticket

Write three lines and hand them in.

- Your team's task, subtask and languages
- One number you measured today, with its unit
- One way a leak could enter your task's data

Quiz 1 opens the next session: six questions, ten minutes.
