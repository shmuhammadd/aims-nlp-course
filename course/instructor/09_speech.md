# Lecture 9 — instructor guide

Keep this file private if the cohort can see the repository. Change the numbers in quiz questions 5 and 6 and exercise 1 before reusing them for marks.

## Teaching the session

- Open with the question slide and list the class's answers on the board: accent, noise, speed, hesitation, microphone. Tick them off at the one-voice lab result.
- Draw 12 frames and the word "sannu" and ask three students to assign frames to letters. They will differ. Each assignment is a CTC path; then show why the two n's need a blank.
- Do the WER alignment on the board before showing the table. Then ask for the WER of "yes" against "yes yes yes". Most students say 0.67; it is 2.0, because the denominator is the reference length.
- Common confusion: relabelling the sample rate is taken for resampling. The lab's 3.0 s and 146.7 Hz settle it.
- Common confusion: students expect six training voices to fix the one-voice problem. In the toy it gives 0.69 and 0.50, because the recogniser keeps one mean spectrum per word. The data helped less than the feature did.
- The 0.6 and 0.2 in the cascade example are assumed numbers. Say so when the slide is up.
- The claims about Whisper, wav2vec 2.0, XLS-R, MMS and FLEURS come from the papers' abstracts and, for Whisper's failure modes, its limitations section. The course has run none of these models for this session.
- Yoruba example: ọkọ (husband) and ọkọ̀ (vehicle). If a Yoruba speaker is in the room, ask them to say both.
- Recordings of classmates for exercise 3A are personal data. Get consent in writing and delete the files after marking unless the speaker agrees otherwise.

## Practical questions

**1.** The denominator is the number of reference words and insertions are not bounded by it. Reference "yes", hypothesis "yes yes yes": two insertions over one word gives 2.0.

**2.** The samples are played back at a third of the speed, so the clip is three times as long and every frequency is divided by three. A model trained on 16 kHz speech receives slow, low audio and fails without an error message. Resampling computes new samples at the new rate after filtering out content above the new limit.

**3.** Clips from one speaker share a voice, a microphone and often a room. If they sit on both sides of a split, the test rewards memorising those conditions and overstates performance on new speakers. In the lab, 1.00 against 0.36.

## Notebook

Solutions for the two stub cells are in [solutions/09_speech.py](solutions/09_speech.py). `python scripts/run_notebooks.py --only 09_speech` runs the notebook with those solutions in place of the stubs.

Reference outputs (seed 42): STFT shape (98, 201), dominant 440.0 Hz; relabelled clip 3.0 s at 146.7 Hz, resampled 1.0 s at 440.0 Hz; best CTC path `a _ b _` with probability 0.168 among 81 paths; WER 0.75 with S, D, I = 1, 1, 1; "habari ya asubuhi" against "habari za asubuhi" WER 0.333 with 1 of 17 characters; "ninakupenda" against "nakupenda" WER 1.0 with 2 of 11 characters. One voice: 1.0 and 0.363; six voices: 0.687 and 0.5; peak-ratio feature: 1.0 and 1.0. Cascade: 0.60, 0.56 and 0.48; $0.95^8=0.663$. Test A: 15 paths collapse to "ab" with total probability 0.6041. Test B: CER 0.059 for the two "habari" pairs and 0.250 for "ƙofa" against "kofa".

The question after test A: the best path holds 0.168 of the 0.604, so greedy decoding's own path accounts for 28% of the probability of the string it outputs.

## Exercise sheet

**Exercise 1.** a. `___` → "" (0.4 × 0.5 × 0.8 = 0.16); `__a`, `_a_`, `_aa`, `a__`, `aa_`, `aaa` → "a"; `a_a` → "aa" (0.6 × 0.5 × 0.2 = 0.06). $P(\text{"a"})=1-0.16-0.06=0.78$. Summing the six paths directly gives 0.04 + 0.16 + 0.04 + 0.24 + 0.24 + 0.06 = 0.78. b. D(the), ok(clinic), S(opens→open), ok(at), ok(nine), S(today→to), I(day): $S=2$, $D=1$, $I=1$, WER $=4/6=0.667$. One point each.

**Exercise 2.** a. Count the labels returned by `alignment`. b. Corpus WER is $2/5=0.4$; the mean of utterance WERs is $(2.0+0)/2=1.0$. Corpus WER is the standard report because short utterances do not dominate; accept the mean if the student argues that every utterance matters equally and reports both. c. Raw values: WER 0.333, 1.0, 1.0, 1.0 and CER 1/17 = 0.059, 2/11 = 0.182, 1/4 = 0.250, 2/6 = 0.333. Lower-casing and removing punctuation bring pairs 1 and 4 to zero. Pair 2 should stay an error. A student who maps ƙ to k must defend it; the expected answer rejects that rule. Working code 2, assertions 1, the table with stated rules 1.

**Exercise 3.** A. Full marks need the three properties for both sets, a concrete list of differences, and a comparison table that includes the written-question ceiling. WER and CER are required only if a model was run. Give no credit for numbers that were not measured. B. With seed 42, noise 0.3, a random two-thirds split and voices 1, 4, 7 and 10 held out, a nearest-neighbour recogniser scores 0.95 on the random split and 0.28 on the speaker-disjoint split. Students' values will differ with their draws; the gap should be large. The pilot is a design and estimates nothing about a population: look for speaker counts, sessions, a speaker-disjoint split, consent and stated normalisation.

Marking anchors. Derivation (2): result 1, working 1. Implementation (4): working code 2, test 1, reproducible run 1. Investigation (4): controlled design 1, measured results 1, explanation 1, limitation 1. Give no marks for numbers that do not appear in an output.

## Quiz 9 key

1. B. 2. C. 3. B. 4. C.

5. a. ok(we), S(meet→met), ok(at), D(the), ok(market), S(today→to), I(day). Accept S(today→day) with I(to). 2 points for a valid alignment with four edits, 1 point for one slip. b. $S=2$, $D=1$, $I=1$; WER $=4/6=0.67$.

6. a. $0.8\times0.7+0.2\times0.1=0.58$. b. ASR at 90%: $0.9\times0.7+0.1\times0.1=0.64$. Answer stage at 80%: $0.8\times0.8+0.2\times0.1=0.66$. The answer stage gives more. c. Development audio is synthetic and test audio is human recordings, so the ASR rate was measured on easier or different audio. Accept any answer that names the synthetic-to-human shift, new speakers or recording conditions.
