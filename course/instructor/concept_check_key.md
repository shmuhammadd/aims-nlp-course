# Concepts check — marking key

1. The mask prevents a query from using later positions while teacher forcing supplies known prefixes (2). NLL=−ln(.5)−ln(.25)=ln(8)=2.0794 (1); perplexity=exp(NLL/2)=sqrt(8)=2.8284 (1).
2. 4×(1024+1024)=8192 adapter parameters (2). Each factor's gradient contains the other factor, so both-zero initialization makes both gradients zero (2).
3. Evaluate retrieval ranks/coverage and compare retrieved, oracle, and distractor contexts on the same questions; control the answer model and budget (2). A citation may be irrelevant or fail to entail the claim, so verify support at claim level (2).
4. Swap heights while retaining labels/question and check that the answer follows the changed image; blank/shuffled controls add evidence (2). Before/after requires ordering across frames; a bag of frames can preserve individual content while discarding order (2).
5. Coverage=6/10=.6 (1), selective accuracy=5/6≈.8333 (1), overall accuracy=5/10=.5 (1). Choose threshold on development data (1).

Award partial credit for correct reasoning with minor arithmetic slips. Accept an equivalent controlled experiment in question 3 or 4.
