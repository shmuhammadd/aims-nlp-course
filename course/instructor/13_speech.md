# Lecture 13 — instructor discussion and marking guide

Teaching support; distribute after the exercise deadline if using this repository privately. These files are public in a public clone, so use modified instances for high-stakes assessment.

## Practical questions

**1. Why can WER exceed 100%?**

Insertions can exceed the reference word count.

**2. What is wrong with treating a 48 kHz recording as 16 kHz without resampling?**

It changes playback duration and frequencies rather than producing the desired samples.

**3. Why should two clips from one speaker stay in the same split?**

Speaker and recording characteristics can leak, overstating generalization to new speakers.

## Exercise guidance

**Exercise 1.** Both collapse to aab; merge consecutive identical symbols first, then remove blanks.

**Exercise 2.** Define empty-reference behavior explicitly; this course uses 0 for both empty and raises for nonempty hypothesis with empty reference.

**Exercise 3.** A pilot is not a population benchmark; report speaker counts, clip durations, recording conditions, and critical-token errors.

## Marking anchors

Derivation (2): correct result (1), reasoning/conventions (1). Implementation (4): functioning method (2), meaningful verification (1), reproducibility (1). Investigation (4): controlled design (1), evidence/results or a fully specified protocol when requested (1), interpretation (1), limitations/provenance (1). Accept justified alternative methods. Do not award marks for invented measurements.
