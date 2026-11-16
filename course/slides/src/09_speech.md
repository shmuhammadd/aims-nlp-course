type: title
title: Speech and multilingual ASR
subtitle: Day 9 · Week 2
footer: Shamsuddeen Hassan Muhammad | AIMS 2026

---
type: big
title: A machine read your training questions aloud. People will read the test questions. What breaks?

Notes: This is the situation in Task 6 subtask 1. Take answers: accents, noise, hesitation, speed, microphones. Keep the list on the board and return to it at the lab result.

---
title: Task 6 subtask 1: a spoken question about an image
subtitle: MMCultureQA, SemEval-2027

| | |
|---|---|
| Input | An image and a spoken question |
| Output | A short text answer |
| Tracks | One per language. Kiswahili and Amharic are open; Afaan Oromo and Somali are announced. |
| Data | Up to 10,000 training and 1,000 development items per track |
| Audio | Synthetic in train and dev. Human recordings at test. |
| Ranked on | BERTScore F1 |

- Subtask 2 is the same task with a written question
- Two decisions for a subtask 1 team: cascade or end-to-end, and how to prepare for human audio

Notes: All rows are from SEMEVAL_TASK.md. The rules on external data and LLM APIs were not stated on the task page; teams should ask in the task's Slack channel.

---
title: Where today sits
subtitle: The third input type of week 2

| Day | Input | Encoder output |
|---|---|---|
| 6 | Text in many languages | One vector per subword |
| 7, 8 | Image | One vector per patch |
| **9** | **Audio** | **One vector per 10 or 20 milliseconds** |
| 10 | Any of them | Evaluation |

- The pattern from day 8 repeats: an encoder, then a decoder or a classifier on top
- New today: the input is far longer than the output, and nobody tells you which part matches which word

---
type: section
title: Part 1: Signal
subtitle: From air pressure to frames

---
title: Audio is a list of numbers and a sample rate

- A microphone measures pressure. Sampling keeps 16,000 measurements per second for most speech models.
- A rate of 16 kHz can represent frequencies up to 8 kHz
- The sample rate is not stored in the samples. Get it wrong and everything shifts.
- Lab: a 440 Hz tone recorded for 1 second at 48 kHz

| Treatment | Duration | Pitch |
|---|---|---|
| Read as 48 kHz | 1.0 s | 440 Hz |
| Relabelled as 16 kHz | 3.0 s | 146.7 Hz |
| Resampled to 16 kHz | 1.0 s | 440 Hz |

- Check the rate, the number of channels, clipping and silence before any modelling

Notes: Notebook section 2. Relabelling makes speech slow and low. A model trained at 16 kHz will fail without any error message. Resample with a library function, which filters before it drops samples.

---
title: Short windows turn a waveform into frames
subtitle: The short-time Fourier transform
layout: stack

![](figures/09_stft.png)

- A 25 ms window every 10 ms gives 100 frames per second. Each frame says how much of each frequency is present.

Notes: Lab numbers: 1 second, frame 400 samples, hop 160 samples, 98 full frames, 201 bins from 0 to 8 kHz, 40 Hz apart. The window fades the edges so that cutting the signal does not add false frequencies.

---
title: Mel and log: what a speech model reads

- **Mel filters** pool the 201 frequency bins into fewer channels, narrow at low frequencies and wide at high ones
- **Log** compresses loud and quiet into a similar range
- Whisper's input (Radford et al., 2022):
  - 16 kHz audio in 30-second windows
  - 25 ms frames every 10 ms, 80 mel channels
  - 30 s × 100 frames = 3,000 frames
  - two convolution layers halve that to 1,500 encoder positions
- The lab stops at the log-magnitude STFT and has no mel step

Notes: wav2vec 2.0 skips hand-built features: a convolutional encoder reads the raw waveform and outputs one vector about every 20 ms.

---
type: section
title: Part 2: Models
subtitle: Two ways to get text out of frames

---
title: Frames outnumber characters, and the alignment is hidden

- A 3-second question is 300 frames
- Its transcript is perhaps 8 words, or 40 characters
- The training data gives audio and text. It does not say which frames belong to which character.
- Two answers:
  - **CTC**: let every frame vote for a character or a blank, and sum over all the ways the votes can spell the text
  - **Encoder-decoder**: let a text decoder attend over the frames and write one token at a time

Notes: Draw 12 frames and the word "sannu" on the board and ask the class to assign frames to letters. Every student gives a different answer. All are valid alignments.

---
title: CTC needs a blank and a collapse rule
subtitle: Merge repeated symbols, then remove blanks

| Frame outputs | After merging repeats | After removing blanks |
|---|---|---|
| s s a _ n n _ n u u | s a _ n _ n u | sannu |
| s s a n n n u u | s a n u | sanu |
| _ s _ a n _ n u _ _ | _ s _ a n _ n u _ | sannu |

- The blank separates the two n's of "sannu" (Hausa: hello). Without it they merge.
- Many frame sequences give the same text
- Order matters: merge first, then delete blanks

Notes: Do row 1 on the board. Then ask what "a a _ a b b _" gives: aab. Students write this function in the lab.

---
title: CTC adds up every path that spells the text
layout: split

$$
p(y \mid x) = \sum_{\pi \,:\, \mathrm{collapse}(\pi) = y} \;\prod_{t=1}^{T} p_t(\pi_t)
$$

- Lab: 4 frames, 3 symbols, 81 paths
- Best path: 0.8 × 0.6 × 0.7 × 0.5 = 0.168
- 15 paths collapse to "ab"
- Their total is 0.604
- Each frame is predicted on its own, given the audio. CTC has no language model unless you add one at decoding.

![](figures/09_ctc.png)

Notes: The 15 and the 0.604 are printed by the notebook's test cell. Training uses a dynamic programme over the frames, the forward algorithm, and never lists the paths. Graves et al. 2006.

---
title: Encoder-decoder ASR: Whisper
subtitle: Radford et al., 2022

- An audio encoder reads the mel frames. A text decoder attends to them and writes subword tokens.
- The decoder conditions on what it has written, so it is a language model of transcripts
- Trained with supervision on 680,000 hours of audio with transcripts from the web
  - 117,000 of those hours cover 96 languages other than English
- A language token and a task token at the start of the output choose the language and transcription or translation
- The paper reports its own failure modes on long audio: repeat loops, and text that was never spoken

Notes: The hours are from the Whisper abstract and section 2. The failure modes are in its limitations section. The paper also shows that error rate per language falls as the hours of training audio for that language rise.

---
title: Self-supervised speech: wav2vec 2.0
subtitle: Baevski et al., 2020

- Most languages have far more untranscribed audio than transcribed audio
- Pretraining: mask spans of the encoder's latent frames and train the model to pick the true latent among distractors. A contrastive loss, as on day 7.
- Fine-tuning: add a linear layer and train with CTC on a small transcribed set
- The paper's headline: 10 minutes of transcribed English with 53,000 hours of untranscribed audio gives WER 4.8 on clean test speech and 8.2 on the harder set
- Multilingual descendants:
  - XLS-R: pretrained on speech in 128 languages
  - MMS: one ASR model for 1,107 languages, built on readings of religious texts

Notes: The 4.8 and 8.2 are LibriSpeech test-clean and test-other in the wav2vec 2.0 abstract. XLS-R is Babu et al. 2021; MMS is Pratap et al. 2023.

---
title: Which family for your language?

| | CTC on a self-supervised encoder | Encoder-decoder (Whisper) |
|---|---|---|
| Output | One symbol per frame, collapsed | Tokens, one at a time |
| Language model | None built in | The decoder |
| Typical error | Misspelt words that sound right | Fluent words that were not said |
| Starting point | Pretrained on untranscribed audio, then fine-tuned on a small transcribed set | Trained on transcribed audio. Usable without fine-tuning. |
| Long audio | Any length | 30-second windows |

- These are tendencies that follow from the design. Measure both on your own audio.
- A language on a model's list was seen in training. The list says nothing about its error rate.

Notes: Whisper's language list includes Swahili, Amharic, Hausa, Yoruba and Somali. Check the model card for any other language.

---
type: section
title: Part 3: Measure and build
subtitle: Error rates, splits, and the Task 6 design choice

---
title: Worked example: WER by alignment

$$
\mathrm{WER} = \frac{S + D + I}{N_{\mathrm{ref}}}
$$

| Reference | the | lab | opens | | today |
|---|---|---|---|---|---|
| Hypothesis | | lab | opens | on | monday |
| Operation | D | ok | ok | I | S |

- One deletion, one insertion, one substitution over 4 reference words: WER = 3/4 = 0.75
- Other alignments with 3 edits exist. The total is what the edit distance fixes.
- Reference "yes", hypothesis "yes yes yes": 2 insertions over 1 word. WER = 2.0.

Notes: Align on the board first. WER above 1 surprises students: the denominator is the reference length and insertions are unbounded.

---
title: WER alone hides what went wrong

| Reference | Hypothesis | WER | CER |
|---|---|---|---|
| habari ya asubuhi | habari za asubuhi | 0.333 | 0.059 |
| ninakupenda | nakupenda | 1.000 | 0.182 |
| Habari ya asubuhi | habari ya asubuhi | 0.333 | 0.059 |
| ƙofa | kofa | 1.000 | 0.250 |

- Swahili packs subject, tense and object into one word, so one slip costs a whole word. Report CER beside WER.
- Row 3 is a capital letter. Lower-casing both sides is a fair rule.
- Row 4 is Hausa "ƙofa" (door) with a plain k. Mapping ƙ to k would hide a consonant the language distinguishes.
- Write the normalisation rules before scoring. Report raw and normalised numbers.

Notes: All numbers are printed in notebook sections 4 and 6. "Ninakupenda" is "I love you"; "nakupenda" is its common short form. A low WER can still get names, numbers and negation wrong, so count those separately.

---
title: What makes African-language ASR hard

- **Tone.** In Yoruba, ọkọ (husband) and ọkọ̀ (vehicle) differ in tone alone. Yoruba spelling marks tone. Standard Hausa spelling does not, so the audio carries what the text omits.
- **Code-switching.** Speakers mix languages inside a sentence, such as Swahili with English. Whisper takes one language token per 30-second window.
- **Hours.** FLEURS has about 12 hours of transcribed speech per language for 102 languages. Whisper trained on 680,000 hours in total.
- **Who is recorded.** Read speech from a few speakers is the easiest to collect and the least like a question asked aloud in a market.
- Each of these changes what you must report: tone-mark policy, language per utterance, hours and speakers

Notes: FLEURS: Conneau et al. 2022, abstract. MMS's data is readings of religious texts, which is read speech in one register. Keep claims at this level; the course has measured none of these models.

---
title: Lab result: one training voice does not cover new voices
layout: split

- Toy: three "words", each a pair of tones. A "voice" scales the pitch.
- Trained on one voice:
  - 1.00 on that voice
  - 0.36 on new voices, near chance
- Trained on six voices: 0.69 and 0.50. The model is too simple to hold six voices.
- A feature that ignores the voice: 1.00 on both
- Your dev score on synthetic audio is the first blue bar

![](figures/09_shift.png)

Notes: Tutorial 9 section 5, seed 42, 300 clips per bar. The recogniser stores one mean spectrum per word. The voice-invariant feature is the ratio of the two strongest peaks. Pretrained speech encoders that heard many speakers are the learned version of that feature.

---
title: Split by speaker, and record with consent

- Two clips from one speaker share a voice, a microphone and a room
- Put them on both sides of a split and the test measures the first bar of the last slide
- Split by speaker, and by recording session where you can
- Report results by language and recording condition, with the number of clips and speakers behind each
- For any recording you make:
  - ask for consent and say what the audio will be used for
  - record how it may be shared
  - transcribe only. Do not infer identity, emotion or other personal attributes from a voice.

Notes: A voice identifies a person. Treat recordings as personal data, including recordings of classmates made for the exercise.

---
title: Cascade or end-to-end?
subtitle: The design choice in Task 6 subtask 1

| | Cascade: ASR, then visual QA on text | End-to-end: audio and image into one model |
|---|---|---|
| Parts | An ASR model and your subtask 2 system | An audio-language model |
| Debugging | Read the transcript | No intermediate text |
| Errors | An ASR error reaches the answer stage | No transcript to get wrong |
| Keeps | Words only | Words, and how they were said |
| Languages | Any language with an ASR model | Languages the audio model was trained on |

- Unified audio-language models put a speech encoder in front of an LLM, as day 8 did with images. Qwen-Audio (Chu et al., 2023) is one example.
- Start with the cascade. It gives you a transcript to inspect and a WER to report.

Notes: Qwen-Audio uses a Whisper encoder with a language model. Whether any audio-language model handles Kiswahili or Amharic is for the team to measure.

---
title: Errors multiply through a cascade
layout: stack

![](figures/09_cascade.png)

- ASR right on 90% of questions; answers right 60% of the time after a right transcript and 20% after a wrong one: 0.9 × 0.6 + 0.1 × 0.2 = 0.56

Notes: The 0.6 and 0.2 are assumed for the example. At 70% the result is 0.48. Right panel: 0.95 to the power 8 is 0.66, so a 5% word error rate leaves a third of 8-word questions with an error. Not every word error changes the answer; that is what the 0.2 stands for.

---
title: A plan for synthetic training audio and human test audio

- Record 30 to 50 development questions with consenting speakers. This is your only estimate of test conditions.
- Score three systems on those recordings:
  - the text system on the written question (the ceiling)
  - the cascade
  - the end-to-end model, if you have one
- Measure WER and CER on synthetic and on human audio. The gap is the shift.
- Prefer an ASR model pretrained on human speech to one trained on the synthetic audio
- If you fine-tune on synthetic audio, vary speed and add noise, and check on the human set that it helped
- Freeze this plan on day 10 with the rest of your experiment plan

Notes: The first sub-bullet gives the answer rate with a perfect transcript. The difference between it and the cascade is the cost of ASR errors.

---
type: big
title: Your cascade has a transcript to score. Your end-to-end model has none. How do you compare them fairly?

Notes: Compare on the task metric, BERTScore F1 of the final answer, on the same human-recorded questions. WER is a diagnostic for the cascade only. Add the text-question ceiling as a third row. Ask what sample size makes a difference believable: day 10 covers paired tests and intervals.

---
title: Lab: 40 minutes

- Open `course/tutorials/09_speech.ipynb`. NumPy only.
- Sections 1 to 5, together:
  - STFT of a tone; relabel and resample
  - read a CTC output; align a transcript and count S, D, I
  - train on one voice, test on new voices; cascade arithmetic
- Section 6, in pairs: write two functions until the tests pass
  - `ctc_collapse(path, blank)`
  - `cer(ref, hyp)` with a rule for an empty reference
- The audio is synthetic tones. The lab transcribes no speech.

---
title: Readings

### Before the next session (20 minutes)
- Radford et al., 2022. Whisper. Sections 1 and 2, and the limitations.

### For Task 6 teams
- Baevski et al., 2020. wav2vec 2.0.
- Pratap et al., 2023. Scaling Speech Technology to 1,000+ Languages (MMS).
- Conneau et al., 2022. FLEURS.

### Going deeper
- Graves et al., 2006. Connectionist Temporal Classification.
- Babu et al., 2021. XLS-R.
- Chu et al., 2023. Qwen-Audio.

Notes: Links are in the lecture notes.

---
title: Exit ticket

Write three lines and hand them in.

- One number from the lab and what it shows
- One way the tone toy differs from human speech
- For Task 6 teams: cascade or end-to-end first, and why. For others: where a train-test shift could hide in your own task.

Quiz 9 opens the next session: six questions, ten minutes.
