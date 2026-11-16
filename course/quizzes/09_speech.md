# Quiz 9 — Speech and multilingual ASR

[Course home](../../README.md) · [Lecture notes](../lectures/09_speech.md) · Sat at the start of day 10 · 10 minutes · 10 points · closed book

**1. (1 point)** A clip recorded at 48 kHz is loaded by code that assumes 16 kHz, with no resampling. Compared with the original, the model receives audio that is

- A. three times shorter and higher in pitch.
- B. three times longer and lower in pitch.
- C. the same length with the high frequencies removed.
- D. unchanged, because the samples are the same numbers.

**2. (1 point)** With `_` as the blank, what does the CTC path `b b _ a _ a b` collapse to?

- A. bab
- B. bbaab
- C. baab
- D. ba

**3. (1 point)** What does wav2vec 2.0 need for its pretraining stage?

- A. Audio with transcripts.
- B. Audio without transcripts.
- C. Text without audio.
- D. Audio with speaker labels.

**4. (1 point)** A team splits its clips at random and finds a low error rate. Each speaker has clips in both training and test. What does the test measure?

- A. Performance on new speakers.
- B. Performance on new languages.
- C. Performance on speakers the model has already heard.
- D. Nothing, because random splits are invalid for audio.

**5. (3 points)** Reference: "we meet at the market today". Hypothesis: "we met at market to day".

a. Write an alignment and mark each position as correct, S, D or I. (2)
b. Compute the WER. (1)

**6. (3 points)** In a cascade for spoken visual question answering, the ASR stage transcribes 80% of questions correctly. The answer stage is right 70% of the time after a correct transcript and 10% of the time after a wrong one.

a. What share of questions does the cascade answer correctly? (1)
b. You can raise ASR to 90%, or raise the answer stage to 80% after a correct transcript. Which gives the better cascade? Show both numbers. (1)
c. The 80% was measured on Task 6 development audio. Give one reason it may not hold on the test set. (1)
