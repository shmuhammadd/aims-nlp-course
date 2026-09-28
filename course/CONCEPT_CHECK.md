# Individual concepts check

[Course home](../README.md) · 25 minutes · 20 points · 10% of the grade

Closed collaboration; a single handwritten reference sheet is allowed. Explain reasoning. No model calls are needed. Instructors should vary the numbers because the answer guide is public.

1. **Causal modelling (4).** Explain why teacher-forced decoder training can evaluate all positions in parallel without future-token leakage (2). Given next-token probabilities 0.5 and 0.25, compute natural-log NLL and perplexity (2).
2. **Adaptation (4).** A frozen 1024×1024 projection uses LoRA rank 4. Calculate adapter parameters, ignoring bias (2). Explain why setting both LoRA factors to zero prevents the first gradient update (2).
3. **Grounded systems (4).** A RAG system scores 80% with oracle evidence and 40% with retrieved evidence. Give a justified next experiment (2). Explain why citation presence alone is not grounding (2).
4. **Multimodal reasoning (4).** Design an image intervention that tests whether a VLM reads bar heights rather than question wording (2). Explain why unordered frame accuracy cannot establish “before/after” understanding (2).
5. **Evaluation (4).** A system answers 6 of 10 questions and gets 5 answers correct. Compute coverage, selective accuracy, and overall accuracy counting abstentions as wrong (3). Name the split on which an abstention threshold should be chosen (1).
