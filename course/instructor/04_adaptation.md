# Lecture 04 — instructor discussion and marking guide

Teaching support; distribute after the exercise deadline if using this repository privately. These files are public in a public clone, so use modified instances for high-stakes assessment.

## Practical questions

**1. Why initialize only one LoRA factor to zero?**

Both-zero factors produce zero gradients in both factors; one random factor enables the other to learn.

**2. Should prompt tokens be removed from the input for response-only SFT?**

No, keep them as context and mask their loss labels.

**3. What must be saved alongside an adapter to reproduce inference?**

Base model ID/revision, tokenizer/template, adapter config and weights, software versions, and generation settings.

## Exercise guidance

**Exercise 1.** With G=dL/dY, dA=s Xᵀ G Bᵀ and dB=s Aᵀ Xᵀ G; use pre-update values for both.

**Exercise 2.** Keep data and base W fixed for each paired comparison. An r beyond useful residual rank need not improve generalization.

**Exercise 3.** Prompt and padding ignored; EOS may be supervised as a real end token. Reject unusable empty/truncated examples explicitly.

## Marking anchors

Derivation (2): correct result (1), reasoning/conventions (1). Implementation (4): functioning method (2), meaningful verification (1), reproducibility (1). Investigation (4): controlled design (1), evidence/results or a fully specified protocol when requested (1), interpretation (1), limitations/provenance (1). Accept justified alternative methods. Do not award marks for invented measurements.
