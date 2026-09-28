# Lecture 14 — instructor discussion and marking guide

Teaching support; distribute after the exercise deadline if using this repository privately. These files are public in a public clone, so use modified instances for high-stakes assessment.

## Practical questions

**1. Why can good single-frame accuracy coexist with poor temporal reasoning?**

Relations across frames and missed events are not tested by isolated images.

**2. How could transcript timestamps corrupt multimodal grounding?**

Misalignment associates words with the wrong visual event or action.

**3. Is forward noising alone a generative diffusion model?**

No; it lacks a trained denoiser and reverse sampling process.

## Exercise guidance

**Exercise 1.** 5880, 11760, and 23520; audio/text/special tokens and preprocessing/encoder compute are not included.

**Exercise 2.** Report offset sensitivity, not just one favorable alignment; reverse timestamps/order consistently for the intended test.

**Exercise 3.** Include section/table references, reported setup, and a proposed replication; do not equate a technical report with independent validation.

## Marking anchors

Derivation (2): correct result (1), reasoning/conventions (1). Implementation (4): functioning method (2), meaningful verification (1), reproducibility (1). Investigation (4): controlled design (1), evidence/results or a fully specified protocol when requested (1), interpretation (1), limitations/provenance (1). Accept justified alternative methods. Do not award marks for invented measurements.
