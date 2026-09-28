# Lecture 11 — instructor discussion and marking guide

Teaching support; distribute after the exercise deadline if using this repository privately. These files are public in a public clone, so use modified instances for high-stakes assessment.

## Practical questions

**1. Why normalize embeddings before cosine retrieval?**

It removes vector magnitude as a direct source of similarity; protect against zero norms.

**2. What happens when two captions in a batch are both valid for one image?**

A diagonal-only loss treats one valid caption as a negative; use appropriate multi-positive labels or deduplicate with care.

**3. Can a CLIP similarity score be read as a calibrated probability?**

Not generally; candidate labels, temperature, and distribution affect the score.

## Exercise guidance

**Exercise 1.** (softmax_rows(S)-I + softmax_rows(Sᵀ)ᵀ-I)/(2N).

**Exercise 2.** Shuffling pair labels changes the task; duplicate captions require relevance sets rather than diagonal-only accuracy.

**Exercise 3.** Keep images fixed and report prompt sensitivity, provenance, and limits; do not select prompts on the final test images.

## Marking anchors

Derivation (2): correct result (1), reasoning/conventions (1). Implementation (4): functioning method (2), meaningful verification (1), reproducibility (1). Investigation (4): controlled design (1), evidence/results or a fully specified protocol when requested (1), interpretation (1), limitations/provenance (1). Accept justified alternative methods. Do not award marks for invented measurements.
