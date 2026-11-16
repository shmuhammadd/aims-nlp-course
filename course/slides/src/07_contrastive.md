type: title
title: Image-text representations: CLIP and SigLIP
subtitle: Day 7 · Week 2
footer: Shamsuddeen Hassan Muhammad | AIMS 2026

---
type: big
title: How do you put a painting and a Hausa caption in the same vector space?

Notes: Take answers for a minute. Most students propose one model that reads both. Today's answer is two models and one loss that pulls matched pairs together.

---
title: Three of the four capstone tasks have images
subtitle: Today is the first multimodal day

| Task | The image | What you need from it |
|---|---|---|
| 12 ArtEmotion-28 | 1,658 paintings | Emotion, with a caption (subtask 1) or without (subtask 3) |
| 6 MMCultureQA | A photo with a question | A short answer in the question's language |
| 9 Framing | The image of a news article | Zero or more of 14 frames (subtask 2) |
| 7 CLaS | None | Text only |

- Every image system starts with an encoder that turns pixels into vectors
- Today: how that encoder is trained, and how to use it without training it again

Notes: All facts are from the task cards in SEMEVAL_TASK.md. Ask each team which row is theirs.

---
title: Where today sits
subtitle: Week 2 adds languages, then modalities, then measurement

| Day | Session | What it adds |
|---|---|---|
| 6 | Multilingual modelling | Many languages, text only |
| **7** | **CLIP and SigLIP** | **An image encoder and a text encoder that agree** |
| 8 | Vision-language models | A decoder that writes about the image |
| 9 | Speech | Audio in, text out |
| 10 | Evaluation | Whether any of it works |

- Tomorrow's models take today's image encoder as their first stage

---
type: section
title: Part 1: Align
subtitle: Two encoders and one loss

---
title: Two encoders, one comparison space
subtitle: A dual encoder

$$
u = \frac{f(I)}{\|f(I)\|} \qquad v = \frac{g(t)}{\|g(t)\|} \qquad \mathrm{sim}(I,t) = u^{\top} v
$$

- f is an image encoder. g is a text encoder. They share no weights.
- Both end in a projection to the same width (512 in CLIP ViT-B/32)
- Dividing by the length makes the dot product a cosine, between −1 and 1
- Images and captions are encoded separately, so you can encode a million images once and search them with any new caption

Notes: Draw two towers on the board with one similarity cell joining them. Contrast with a cross-encoder, which must run once per pair. The 512 is the CLIP ViT-B/32 embedding width in Radford et al. 2021.

---
title: An image becomes a sequence of patches
subtitle: The vision transformer (Dosovitskiy et al., 2020)
layout: split

- Cut the image into square patches
- Flatten each patch and project it to the model width
- Add a position embedding to each
- Run the day-2 transformer over the sequence
- A 224-pixel image:
  - patch 32 gives 49 tokens
  - patch 16 gives 196 tokens
- Smaller patches see finer detail and cost four times the tokens

![](figures/07_patches.png)

Notes: 224/32 = 7 and 7 × 7 = 49. 224/16 = 14 and 14 × 14 = 196. CLIP adds one class token whose output becomes the image embedding. ViT-B/32 is the encoder in the Task 12 subtask 3 baseline.

---
title: A batch gives a similarity matrix
subtitle: N images and their N captions

$$
S_{ij} = \frac{u_i^{\top} v_j}{\tau}
$$

- Row i is image i against every caption. Column j is caption j against every image.
- The N matched pairs sit on the diagonal
- The other N² − N cells are negatives that cost nothing to collect
- τ is the temperature. CLIP learns it, starting from 0.07.
- CLIP trained with batches of 32,768 pairs, so each image met 32,767 wrong captions per step

Notes: The batch size and the 0.07 are from Radford et al. 2021, section 2. Ask how many negatives a batch of 8 on a laptop gives: 7 per image.

---
title: The loss is cross-entropy along rows and columns

$$
L_{\mathrm{i2t}} = -\frac{1}{N}\sum_{i} \log \frac{e^{S_{ii}}}{\sum_{j} e^{S_{ij}}}
\qquad
L_{\mathrm{t2i}} = -\frac{1}{N}\sum_{j} \log \frac{e^{S_{jj}}}{\sum_{i} e^{S_{ij}}}
$$

$$
L = \frac{1}{2}\left(L_{\mathrm{i2t}} + L_{\mathrm{t2i}}\right)
$$

- Image-to-text: each row is a classification problem with N classes
- Text-to-image: the same along each column
- If every logit is equal, L = log N. For N = 3 that is 1.099.
- The gradient with respect to S is (row softmax − I + column softmax − I) / 2N

Notes: This is the day-3 cross-entropy with the caption index as the label. Students derive the gradient in exercise 1.

---
title: Worked example: a 3×3 loss by hand
subtitle: Logits after dividing by the temperature

| | Caption 1 | Caption 2 | Caption 3 | Row term |
|---|---|---|---|---|
| Image 1 | **3** | 1 | 0 | 0.170 |
| Image 2 | 1 | **2** | 3 | 1.408 |
| Image 3 | 0 | 1 | **4** | 0.066 |
| Column term | 0.170 | 0.551 | 0.327 | |

- Row 2: log(e + e² + e³) − 2 = 3.408 − 2 = 1.408
- Image-to-text mean 0.548. Text-to-image mean 0.349. Loss 0.449.
- Image 2 prefers caption 3, so its row carries most of the loss
- Caption 2 still ranks image 2 first, so the column term is smaller

Notes: Do row 2 on the board: e = 2.718, e² = 7.389, e³ = 20.086, sum 30.193, log 3.408. The notebook prints 0.5478, 0.3493 and 0.4485. Ask which image is mismatched before computing anything.

---
title: Temperature sets how sharp the choice is
layout: split

One image scores cosine 0.8 with its own caption, 0.6 with a near miss and 0.1 with an unrelated one.

| τ | Probability of own caption |
|---|---|
| 1 | 0.43 |
| 0.2 | 0.72 |
| 0.07 | 0.95 |

- A cosine cannot exceed 1, so at τ = 1 the loss barely separates right from nearly right
- A small τ puts the gradient on the hardest negatives

![](figures/07_temperature.png)

Notes: The probabilities are computed in the figure script: softmax of (0.8, 0.6, 0.1)/τ. In the lab τ is fixed at 0.2, so no logit can exceed 5 and the loss levels off at 0.076.

---
title: A negative in the batch may be a correct match

- Two photos of the same market get the captions "a market" and "people at a market"
- The loss treats each photo's other caption as wrong and pushes it away
- Larger batches and narrower domains make this more common
  - 1,658 paintings and eight emotions: many captions fit many paintings
- What to do:
  - remove duplicate captions from a batch
  - mark several positives per row and average their log-probabilities
  - score retrieval against a set of relevant items, not one index

Notes: The 1,658 is the Task 12 painting count. The worked example had a mild version of this: caption 3 fits image 2 better than its own caption does.

---
title: SigLIP asks one yes/no question per pair
subtitle: Zhai et al., 2023

$$
L = -\frac{1}{N}\sum_{i}\sum_{j} \log \sigma\left(z_{ij}\,(t\,u_i^{\top} v_j + b)\right)
\qquad z_{ij} = +1 \;\mathrm{if}\; i=j,\; -1 \;\mathrm{otherwise}
$$

- Each cell is its own binary problem. Nothing is normalised across the batch.
- N positives face N² − N negatives, so a learned bias b starts at −10
- On the worked matrix: loss 2.856 with b = 0, and 1.213 with b = −2
- The paper reports that the sigmoid loss does better than the softmax loss at small batch sizes

Notes: The two loss values are printed in notebook section 4. The scale t starts at 10 and b at −10 in the paper. The first claim to test in your own work is the last bullet: small batches are what a free GPU gives you.

---
title: SigLIP 2 changed several things at once
subtitle: Tschannen et al., 2025: a case study in reading a model paper

| What changed | In SigLIP 2 |
|---|---|
| Loss | Sigmoid loss plus a captioning decoder and self-supervised losses |
| Data | Multilingual image-text pairs |
| Input | Variants that keep the image's aspect ratio |

- The title claims three gains: semantic understanding, localisation, dense features
- When a paper changes loss, data and input together, ask which table removes one change at a time
- Credit a gain to a component only where an ablation supports it

Notes: The paper's title is "SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features". Have students find the ablation table as the after-class reading task.

---
type: section
title: Part 2: Use
subtitle: Three things to do with a trained encoder

---
title: Zero-shot classification is retrieval over prompts

```
prompts = [f"a painting that evokes {e}" for e in emotions]
v = normalise(text_encoder(prompts))      # (8, d), once
u = normalise(image_encoder(image))       # (d,)
scores = u @ v.T / tau
label = emotions[scores.argmax()]
```

- No training examples. The class names are the classifier.
- The wording of the prompt changes the answer, so choose it on development data
- softmax(scores) sums to 1 over the prompts you wrote. It is not a calibrated probability.
- A sketch: the course has not run this code on a pretrained model

Notes: The eight emotions are the Task 12 label set. Ask what happens if the true emotion is not in the list: the model still picks one.

---
title: Retrieval: report recall@k in both directions

- Recall@k: the share of queries whose match is in the top k
- On the worked matrix, k = 1:
  - image-to-text: rows 1 and 3 are right, row 2 is wrong: 2/3
  - text-to-image: all three columns are right: 3/3
- The two directions differ, so report both
- State the gallery size. A random ranker scores k/N.
- With several correct items per query, count a hit if any one is in the top k, and say so

Notes: Column 3 holds 0, 3, 4, so caption 3 still retrieves image 3. Students write recall_at_k in the lab.

---
title: A frozen encoder is a feature extractor
subtitle: The cheapest image baseline for your task

```
features = image_encoder(images)        # run once, save to disk
clf = LogisticRegression().fit(features[train], y[train])
score = metric(y[dev], clf.predict(features[dev]))
```

- Task 12 subtask 3: the organisers' baseline is frozen CLIP ViT-B/32 features with a logistic regression. Correlation 0.45.
- Task 9 subtask 2: frame labels for news images. Start the same way, with one classifier per frame.
- 1,658 paintings × 512 numbers × 4 bytes = 3.4 MB of features
- No GPU is needed after the features are saved

Notes: 1,658 × 512 × 4 = 3,395,584 bytes. The 0.45 is from the task card. Lab section 5 does this with the toy encoder: 5 frozen features match 192 raw pixels on images like the training set.

---
type: section
title: Part 3: Limits
subtitle: What the encoder did not learn

---
title: The lab holds out one combination
subtitle: Trained on five colour-position pairs. Blue right was never shown.
layout: split

- Image-to-text top-1: 6 of 6
- Text-to-image top-1: 5 of 6
- The blue-right image wins by 1.67 against 1.53. Trained images win by more than 1.
- The caption "blue right" retrieves the blue-left image
- The pixels for blue right had zero input in training, so their weights are still random
- Over seeds 0 to 19, blue right is correct 3 times. Chance is 1 in 6.

![](figures/07_similarity.png)

Notes: Toy numbers from tutorial 7, seed 42. The seed count is in the instructor guide. The point: one correct held-out example is not evidence of composition.

---
title: Contrastive training can ignore word order
subtitle: Yuksekgonul et al., 2022

- The loss asks only that the right caption beats the other captions in the batch
- If no caption in the batch differs by word order alone, the model never needs word order
- The paper tests this: shuffle the words of a caption and see whether the score changes
- Expect weak spots in:
  - counting ("three goats" against "four goats")
  - relations ("the cup on the book" against "the book on the cup")
  - small text inside the image
- Test these on your own data before you rely on the encoder for them

Notes: The paper is "When and why vision-language models behave like bags-of-words, and what to do about it?". Describe the finding without numbers.

---
title: What happens to a caption in Hausa or Yoruba?

- CLIP's text encoder was trained on captions collected with English queries
- SigLIP 2 is trained on multilingual data, as its title says. That does not tell you how it does on your language.
- Two things you can check in ten minutes:
  - tokenise a caption and count the pieces. Try "shuɗi" (Hausa: blue), with its hooked ɗ.
  - take 50 image-caption pairs in English and in your language and compare recall@1
- Yoruba web text is often written without tone marks. Test with and without them.
- The Task 12 baseline sidesteps the question: XLM-R reads the caption, ViT reads the painting

Notes: The CLIP data collection is described in section 2.2 of Radford et al. 2021: 400 million pairs from queries built on English Wikipedia words. The XLM-R plus ViT baseline is from the task card.

---
title: A weaker text tower shows up in recall@1 first
subtitle: Lab toy: the same 200 images, caption vectors with added noise
layout: stack

![](figures/07_recall_noise.png)

- In the notebook run at σ = 0.3, recall@1 is 0.655 and recall@5 is 0.895

Notes: The curve is computed in the figure script with seed 42 on a finer grid than the notebook table, so values differ slightly from the notebook's. It stands in for a language the text tower embeds less precisely. It is a toy and measures no real model.

---
type: big
title: Your multimodal run scores below your text-only run. Give three explanations.

Notes: This happened to the Task 12 organisers: 0.535 multimodal against 0.550 text-only. Good answers: the caption already states the emotion; concatenation adds parameters without enough data; the image encoder was trained on photographs and captions, not on emotion in paintings; the image features were not normalised to the scale of the text features. Tomorrow's decoy test asks whether a system uses the image at all.

---
title: Lab: 40 minutes

- Open `course/tutorials/07_contrastive.ipynb`. NumPy only.
- Sections 1 to 5, together:
  - train the alignment and read the held-out row and column
  - check the 3×3 loss against your hand calculation
  - fit a logistic regression on frozen features
- Section 6, in pairs: write two functions until the tests pass
  - `contrastive_loss(S)`
  - `recall_at_k(S, k)` in both directions
- Predict each output before you run the cell

---
title: Readings

### Before the next session (20 minutes)
- Radford et al., 2021. CLIP. Sections 1 and 2, and figure 1.

### For teams with images
- Zhai et al., 2023. Sigmoid Loss for Language Image Pre-Training.
- Tschannen et al., 2025. SigLIP 2. Find the ablation tables.
- Dosovitskiy et al., 2020. An Image is Worth 16x16 Words.

### Going deeper
- Yuksekgonul et al., 2022. When and why vision-language models behave like bags-of-words.
- Zhai et al., 2021. LiT: locked-image text tuning.

Notes: Links are in the lecture notes.

---
title: Exit ticket

Write three lines and hand them in.

- One number you measured today and what it shows
- One reason the toy result may not hold for a pretrained encoder
- The first image baseline you will run on your task, or why your task needs none

Quiz 7 opens the next session: six questions, ten minutes.
