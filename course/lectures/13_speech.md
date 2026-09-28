# Lecture 13 — Speech, audio representations, and multilingual ASR

[Course home](../../README.md) · Week 3, day 3 · 120 minutes

## Learning outcomes

- Trace a waveform into time-frequency features.
- Distinguish CTC and encoder-decoder speech objectives.
- Calculate WER/CER and design speaker-disjoint evaluation.

## Session plan

| Minutes | Activity |
|---|---|
| 0–10 | Retrieval quiz: recall the previous session; day 1 uses the prerequisite diagnostic |
| 10–35 | Concepts and concrete examples from the first half of these notes |
| 35–60 | Board derivation, worked example, and failure analysis |
| 60–65 | Break |
| 65–105 | Guided [notebook tutorial](../tutorials/13_speech.ipynb) |
| 105–115 | Practical questions in pairs, then discuss evidence |
| 115–120 | Exit ticket: one result, one limitation, one next experiment |

## Teaching notes

### Audio is sampled evidence
A waveform is a sequence of amplitudes sampled at a rate such as 16 kHz. Relabeling the sample rate is not resampling: it changes interpreted time and pitch. Convert channels deliberately and inspect clipping and silence. The short-time Fourier transform applies a window to overlapping segments, then computes frequency content. Mel filterbanks pool frequency bins nonlinearly; log scaling compresses dynamic range. The lab computes a log-magnitude STFT, not a mel spectrogram, so the distinction remains explicit.

### Alignment is latent
Speech frames greatly outnumber output characters or subwords. CTC sums probabilities over paths that collapse to a target sequence, using a blank symbol and a rule that merges consecutive repeated labels before removing blanks. A blank between repeated labels permits repeated output characters. CTC assumes a particular conditional factorization given encoder outputs; it does not itself use an autoregressive output language model.

Encoder-decoder ASR attends from text outputs to audio representations and can use preceding generated tokens. Whisper is a useful example of large-scale supervised speech modelling. wav2vec 2.0 illustrates self-supervised representation learning with masked latent audio prediction and a contrastive task. Data, language exposure, and decoding matter as much as architecture labels.

### Error metrics and normalization
Word error rate is \((S+D+I)/N_{ref}\), where substitutions, deletions, and insertions come from an edit alignment. WER can exceed one when there are many insertions. Character error rate is useful when word boundaries are ambiguous, but script and normalization choices matter. Report raw and normalized variants if normalization changes the practical task; never silently remove meaningful distinctions.

A low WER does not ensure correct names, negation, or numbers. Include critical-token accuracy and qualitative error categories. Break down performance by language, recording condition, and other consented relevant factors. Small-group results need sample counts and uncertainty.

### From ASR to spoken interaction
A cascade runs ASR, text reasoning, and speech synthesis; errors can accumulate and latency adds across stages. Unified audio-language models may preserve prosody or timing that transcripts discard, but need careful modality-specific evaluation. For a course project, transcribing permitted short clips is enough; do not infer emotion, identity, or sensitive attributes from voices.

Split by speaker and session to avoid near-identical acoustic conditions across train and test. Use consented speech with documented access conditions. The mandatory lab uses a synthetic tone only to inspect signal processing and authored transcripts only to inspect metrics. It does not claim to transcribe speech.

## Worked example

Reference “the lab opens today”; hypothesis “lab opens on monday”. One optimal alignment has D=1 (“the”), I=1 (“on”), S=1 (“today”→“monday”), so WER=3/4=0.75. Equivalent optimal edit alignments may distribute operations differently; total distance is stable.

## Guided tutorial

**Question:** Compute an STFT on a synthetic waveform, collapse CTC paths, and implement WER through dynamic programming.

Open the notebook and predict each result before running it. Spend 5 minutes on setup and data inspection, 15 on the mechanism, 10 on the controlled intervention, and 10 on interpretation. The core uses NumPy and authored/synthetic data; the notebook identifies its limits. Save outputs, configuration, and a short interpretation. Do not report an expected trend as a measured result.

**Real-model or research extension:** Run `extensions/hf_audio.py --audio path.wav` on a permitted short clip. It resamples with librosa and runs a pretrained Whisper pipeline; score the output against a human transcript.

See the [extension guide](../extensions/README.md) for commands, hardware assumptions, and validation status.

## Practical questions

1. Why can WER exceed 100%?
2. What is wrong with treating a 48 kHz recording as 16 kHz without resampling?
3. Why should two clips from one speaker stay in the same split?

## After class

Complete the [exercise sheet](../exercises/13_speech.md) in 45–60 minutes. Read the first reference's abstract and method overview before the next session; remaining references are optional depth. For long papers, focus on the objective, one figure/table, and one limitation.

## Readings

- [Whisper](https://arxiv.org/abs/2212.04356)
- [wav2vec 2.0](https://arxiv.org/abs/2006.11477)
- [Connectionist Temporal Classification](https://www.cs.toronto.edu/~graves/icml_2006.pdf)
