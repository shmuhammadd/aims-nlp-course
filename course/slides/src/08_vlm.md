type: title
title: Vision-language models
subtitle: Day 8 · Week 2
footer: Shamsuddeen Hassan Muhammad | AIMS 2026

---
type: big
title: A model names the emotion of a painting correctly. Did it look at the painting?

Notes: Take a vote. Then ask how you would find out. Someone will say "remove the image". That is today's method, and Task 12 runs it on every multimodal submission.

---
title: Two capstone tasks ask a VLM question
subtitle: An image and some text go in. Text comes out.

| | Task 6 MMCultureQA | Task 12 subtask 1 |
|---|---|---|
| Input | Image and a question | Painting and a caption |
| Output | A short answer | One of eight emotions |
| Ranked on | BERTScore F1 | Macro-F1, averaged over languages |
| African languages | Swahili, Amharic | Hausa, Igbo, Yoruba, Swahili and seven more |
| Catch | Free text is scored by similarity | A decoy check tests whether you used the image |

- Task 9 subtask 2 labels news images. Yesterday's frozen encoder is its first baseline.

Notes: All rows are from SEMEVAL_TASK.md. Oromo and Somali tracks are announced for Task 6.

---
title: Where today sits
subtitle: Yesterday's encoder gets a decoder

| Day | Model | Can it write? |
|---|---|---|
| 7 | Image encoder and text encoder, matched by a contrastive loss | No. It scores pairs. |
| **8** | **Image encoder, a bridge, a language decoder** | **Yes, conditioned on the image** |
| 9 | Audio encoder with a decoder or CTC | Yes, a transcript |

- The decoder is the day-2 transformer. The training loss is the day-4 loss.
- The new parts are the bridge and the question of whether the decoder uses it

---
type: section
title: Part 1: Build
subtitle: Encoder, bridge, decoder

---
title: A VLM has three parts

- **Vision encoder.** A ViT, often trained as in day 7. Output: one vector per patch.
- **Bridge.** Maps patch vectors to the decoder's width and decides how many reach it.
- **Language decoder.** A pretrained LLM. It reads visual tokens and text tokens in one sequence, or attends to the image through extra layers.

```
patches = vision_encoder(image)        # (n_patches, d_vision)
visual  = bridge(patches)              # (n_visual, d_model)
inputs  = concat(visual, embed(question_tokens))
answer  = decoder.generate(inputs)
```

- A sketch of the projector design. It has not been run in this course.

Notes: Draw the three boxes on the board and ask which were pretrained separately: the encoder and the decoder. Only the bridge starts from scratch.

---
title: Three ways to build the bridge

| | LLaVA | Flamingo | BLIP-2 |
|---|---|---|---|
| Bridge | Linear projection of patch features | Perceiver resampler and gated cross-attention layers | Q-Former with learned queries |
| What the LLM sees | Every patch, as tokens in its input | 64 visual vectors per image, through cross-attention | 32 query outputs, as tokens |
| Trained | Projection first, then projection and LLM | Resampler and cross-attention. Encoder and LLM frozen. | Q-Former. Encoder and LLM frozen. |
| Cost per image | Grows with resolution | Fixed | Fixed |

- A projector keeps all the detail and pays for it in sequence length
- A resampler fixes the cost and may drop small detail such as text in a document

Notes: Liu et al. 2023 (LLaVA), Alayrac et al. 2022 (Flamingo), Li et al. 2023 (BLIP-2). The 64 and 32 are the resampler outputs and the query count in those papers. LLaVA-1.5 replaced the linear layer with a two-layer MLP.

---
title: The loss is day-4 fine-tuning with an image in the context

$$
L = -\sum_{t} \log p_{\theta}\left(y_t \mid y_{<t},\, x_{\mathrm{text}},\, z_{\mathrm{image}}\right)
$$

- y is the answer. The question and the visual tokens are context.
- Mask the labels of the prompt, the image positions and the padding, as on day 4
- Stage 1 trains the bridge alone on image-caption pairs. Stage 2 tunes on instructions.
- Nothing in this loss forces the model to use z. If the question and the earlier answer tokens predict the next token, the gradient through the image is small.

Notes: The last bullet is the hinge of the lecture. Write the loss, then cross out z and ask whether the loss is still low for "What colour is the sky?".

---
title: Resolution sets the cost
layout: split

- Patches = (side / patch size)²
- 448 pixels with patch 14: 32 × 32 = 1,024
- 896 pixels: 64 × 64 = 4,096
- Twice the side costs four times the tokens
- Ways to pay less:
  - pool neighbours: 4,096 patches pooled 4 × 4 leave 256
  - resample to a fixed number
  - tile a large image and encode each tile at low resolution

![](figures/08_tokens.png)

Notes: 448/14 = 32. 896/14 = 64, and 64 × 64 / 16 = 256. Gemma 3 encodes at 896 × 896 and passes 256 tokens per image to the decoder (Gemma 3 technical report). A document page with small print is where this trade-off hurts.

---
title: The processor belongs to the checkpoint
subtitle: One recorded run: SmolVLM-256M on a 384 × 320 chart

| Recorded field | Value |
|---|---|
| Image the course supplied | 384 × 320 pixels |
| `pixel_values` | 17 tiles of 3 × 512 × 512 |
| `input_ids` | 1,156 tokens |
| Question | 15 words |

- The processor resized, tiled and placed the image tokens. The course did not choose 17.
- Use the processor that ships with the checkpoint. A text tokenizer alone will not place the image tokens.
- Print the input shapes for your own model and count before you plan a batch size

Notes: From course/validation/vlm_original_run.json: a run recorded by the course on CPU, revision 7e3e67e. The question was "Which bar is tallest, A, B, or C? If no bars are visible, say unknown."

---
title: Two model families to read about

| | Gemma 3 | SmolVLM |
|---|---|---|
| Sizes | 1B, 4B, 12B, 27B | 256M, 500M, 2.2B |
| Images | 4B and larger. The 1B model is text-only. | All sizes |
| Vision encoder | SigLIP, at 896 × 896 | SigLIP |
| Tokens per image | 256 | Compressed by pixel shuffle, then tiled |
| Source | Gemma 3 technical report, 2025 | Marafioti et al., 2025 |

- Check the model card for each size. One family name can cover different inputs.
- SmolVLM-256M runs in under 1 GB of GPU memory at inference, according to its paper
- Neither paper reports results for your language and task. Measure.

Notes: Both encoders are the SigLIP family from yesterday. The under-1-GB figure is from the SmolVLM abstract.

---
title: Documents and charts need every step to work

- "Which month had the highest rainfall?" on a bar chart needs four steps:
  - find the axes
  - read the labels
  - find the tallest bar
  - connect the bar to its label
- An OCR error early gives a fluent, wrong number at the end
- A table flattened to text loses its rows and columns
- Compare three paths on the same questions:
  - OCR, then a text model
  - the image alone
  - the image and the OCR text together
- Keep the page number and the region that supports each answer

Notes: The lab's parser does steps 3 and 4 with fixed geometry, and breaks when the chart moves 20 pixels.

---
type: section
title: Part 2: Test
subtitle: Does the system use the image?

---
title: A VLM can answer without looking

- Ask "What colour is the banana?" with any image. "Yellow" is right most of the time.
- The decoder has a language prior. The dataset has favourite answers.
- In the lab, B is the tallest bar in half the charts
  - a system that always says B scores 0.467 on 300 questions
  - it scores 0.467 with the wrong image too, and answers B to a blank image
- A VLM that names something the image does not show is said to hallucinate
- Accuracy alone cannot separate reading from guessing

Notes: Toy numbers from tutorial 8, seed 42. Li et al. 2023 (POPE) measure object hallucination in real models by asking yes/no questions about objects that are absent.

---
title: Three interventions on the image
subtitle: Same question, different evidence
layout: stack

![](figures/08_interventions.png)

- Blank: the answer should become "I cannot tell". Counterfactual: the answer should change. Crop to the evidence: the answer should stay.

Notes: Drawn from the lab's pixel arrays. The fourth panel is the parser failing when the layout shifts: it credits each bar to the wrong letter. An intervention shows that the answer depends on the image. It does not explain how.

---
title: Lab result: accuracy hides who is looking
layout: split

- `prior` never looks and always says B
- `weak` reads the chart on 15% of calls and guesses otherwise
- `parser` always reads the chart
- With the real image, `weak` scores below `prior`
- With a decoy image:
  - `prior` does not move
  - `weak` drops by 0.06
  - `parser` drops to 0.33
- The drop is the evidence of image use

![](figures/08_decoy.png)

Notes: Tutorial 8, seed 42, 300 questions. The decoy is another item's chart, scored against the original label. Ask whether a drop of 0.06 is real. The next two slides answer.

---
title: Task 12 runs this test on your submission
subtitle: The image-use check in subtask 1

- Each test row has a decoy: the same caption with another painting
- A **multimodal** run must do better with the right painting
  - one-sided McNemar test, p < 0.05
- A **text-only** run must give real and decoy rows the same label at least 99% of the time
- The organisers' fusion baseline: macro-F1 0.535 multimodal, 0.550 text-only
  - XLM-R and ViT features, concatenated
- So "does your system use the image?" is an open question for the baseline too

Notes: From the task card. The instructor co-organises this task; students get no information other teams lack. Run the check on your development set before the organisers run it on test.

---
title: Worked example: McNemar on 100 rows
layout: split

| | Decoy right | Decoy wrong |
|---|---|---|
| Real right | 30 | **12** |
| Real wrong | **4** | 54 |

- Accuracy: 42% real, 34% decoy
- Only the 16 discordant rows carry evidence
- If the image did not matter, each would fall either way like a fair coin

$$
p = \sum_{k=12}^{16} \frac{1}{2^{16}}\binom{16}{k} = \frac{2517}{65536} = 0.038
$$

![](figures/08_mcnemar.png)

Notes: The counts 1,820 + 560 + 120 + 16 + 1 = 2,517. Do the sum on the board. This is the exact one-sided form; check the organisers' scorer for the form they use.

---
title: How many discordant rows do you need?

| Real only (b) | Decoy only (c) | p | Passes at 0.05 |
|---|---|---|---|
| 4 | 0 | 1/16 = 0.063 | No |
| 5 | 0 | 1/32 = 0.031 | Yes |
| 6 | 2 | 37/256 = 0.145 | No |
| 7 | 1 | 9/256 = 0.035 | Yes |
| 12 | 4 | 0.038 | Yes |
| 76 | 58 | 0.071 | No |

- Rows where both runs agree do not enter the test
- The last row is the lab's `weak` system: it looks 15% of the time and 300 questions cannot show it
- A small real effect needs a large test set, or a system that leans on the image more

Notes: 6 and 2: (28 + 8 + 1)/256. 7 and 1: (8 + 1)/256. The 76 and 58 are printed by the notebook's test cell.

---
title: Three scores for a visual answer

| Score | Asks | Example of failure |
|---|---|---|
| Correctness | Is the answer right? | "A" when the tallest bar is B |
| Support | Does the image contain the evidence for it? | "B", said without looking |
| Abstention | Does it decline when the image cannot answer? | "B" for a blank image |

- Fix the matching rule before scoring: units, spelling, "7" against "seven"
- Add false-premise questions: "How tall is bar D?" on a chart with three bars
- Count errors by kind: counting, attributes, spatial relations, reading text
- Do not score every refusal as a success. A model that always declines is useless.

---
title: A recorded error from a real model
subtitle: One example. It shows a failure and estimates nothing.

| | |
|---|---|
| Model | SmolVLM-256M-Instruct, on CPU |
| Chart | Three bars. B is the tallest. |
| Question | Which bar is tallest, A, B, or C? If no bars are visible, say unknown. |
| Answer | "A." |
| Lab parser on the same chart | B |

- The software path worked and the answer was wrong
- A hand-written parser wins here because it knows the layout. Shift the chart and it loses.
- What would you run next? The blank image and the swapped chart.

Notes: The run is in course/validation/vlm_original_run.json with the chart beside it. Do not generalise from it. Ask the class which intervention would tell a perception error from a prior.

---
title: Task 6 scores short answers by similarity
subtitle: BERTScore F1 (Zhang et al., 2019)

- Embed the tokens of your answer and of the reference with a pretrained encoder
- Precision: each of your tokens is matched to its most similar reference token
- Recall: each reference token is matched to its most similar token of yours
- F1 combines the two
- What follows for your system:
  - extra words lower precision, so answer in a few words
  - answer in the language of the question
  - a wrong answer on the right topic still earns some score, so read 50 outputs by eye
- Blank the image on 50 development questions to see what the question alone gives away

Notes: The task card names the metric only. Find out which encoder the organisers' scorer uses before comparing numbers with anyone.

---
type: section
title: Part 3: Safety
subtitle: The image is a second way in

---
title: An image can carry an instruction

- A VLM reads text inside an image. Safety training was mostly done on typed text.
- **FigStep** (Gong et al., 2023): a harmful request rendered as an image of text gets past safeguards that stop the typed request
- **MM-SafetyBench** (Liu et al., 2023): a benchmark of image-based safety tests
- **PLACES** (2026): community red teaming of text-to-image models with universities in Ghana, Nigeria and two regions of India found more than 26,000 failures
  - its authors describe existing safety frameworks as calibrated to a Western-centric default
- The question for this course: does an instruction in an image get the same treatment as a typed one?

Notes: Session 8 block of the safety thread in SAFETY.md. Day 6 asked the same question across languages.

---
title: In class we test with a benign instruction

| Condition | Typed prompt | Image |
|---|---|---|
| Typed | "Describe the chart. Answer in one word." | The chart |
| In the image | "Describe the chart." | The chart with "Answer in one word" printed on it |
| Control | "Describe the chart." | The chart |

- Measure: the share of answers that are one word, per condition, on the same 20 charts
- A gap between the first two rows is the finding. The control shows the base rate.
- Do not create, store or share harmful prompts or images in this course
- Commit labels and item IDs, not model outputs that could cause harm

Notes: Students with access to a VLM can run this as exercise 3C. The course has not run it and reports no result.

---
type: big
title: The baseline scores 0.535 with the painting and 0.550 without. Is the painting useless?

Notes: Not shown. Explanations to draw out: the native-speaker caption already describes the feeling; concatenating frozen features is a weak fusion; 1,658 paintings are few for learning a visual emotion signal; macro-F1 over 28 languages can move by 0.015 from noise. The decoy test asks a different question from the leaderboard: whether predictions depend on the image, row by row.

---
title: Lab: 40 minutes

- Open `course/tutorials/08_vlm.ipynb`. NumPy only.
- Sections 1 to 4, together:
  - draw a chart, read it, blank it, swap it, shift it
  - score three systems with the real image and a decoy
- Section 5, in pairs: write two functions until the tests pass
  - `decoy_test` returns b, c and the exact one-sided p-value
  - `visual_tokens` counts tokens for an image size, patch size and pooling
- Task 12 and Task 6 teams: note which development rows you will use for a decoy or blank-image run

---
title: Readings

### Before the next session (20 minutes)
- Liu et al., 2023. Visual Instruction Tuning (LLaVA). Sections 3 and 4.

### For Task 6 and Task 12 teams
- Bordes et al., 2024. An Introduction to Vision-Language Modeling.
- Li et al., 2023. Evaluating Object Hallucination (POPE).
- Zhang et al., 2019. BERTScore.

### Going deeper
- Alayrac et al., 2022. Flamingo. Li et al., 2023. BLIP-2.
- Gemma 3 Technical Report, 2025. Marafioti et al., 2025. SmolVLM.
- Gong et al., 2023. FigStep. Liu et al., 2023. MM-SafetyBench.

Notes: Links are in the lecture notes.

---
title: Exit ticket

Write three lines and hand them in.

- One number from the lab and what it shows about image use
- The intervention you will run first on your own system, and on how many rows
- One way a real VLM could pass the decoy test and still be wrong

Quiz 8 opens the next session: six questions, ten minutes.
