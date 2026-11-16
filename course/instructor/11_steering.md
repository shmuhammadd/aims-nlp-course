# Lecture 11 — instructor guide

Keep this file private if the cohort can see the repository. Change the numbers in quiz question 5 and exercise 1 before reusing them for marks.

## Teaching the session

- Open with the question slide and take answers for one minute. Most students say "prompt it" or "fine-tune it". Task 7 rules out both.
- Do the two-dimensional example on the board before showing the table. Students who can do it by hand find the notebook easy.
- Ask the class to predict the relevance curve before showing the trade-off figure.
- The forward-hook slide is a sketch. The course has not run it. Send Task 7 teams to the organisers' repository.
- Common confusion: a probe "finding" a feature is taken as proof the model uses it. The layer-2 question near the end is there to break that.

## Practical questions

**1.** Either later layers recompute language from the tokens and overwrite the edit, or the probe found information nothing downstream reads at that layer. Steer at every layer from 2 upward, or patch layer 2 from a Swahili run: if the output moves only when later layers are edited, the first explanation holds.

**2.** The target projection, taken as the mean projection of target-language training activations. It must come from data that is not used for scoring.

**3.** Attention can re-read the source-language tokens at every later layer; normalisation rescales the edited vector; the unembedding may read language from a different direction than the one that was moved.

## Notebook

Solutions for the two stub cells are in [solutions/11_steering.py](solutions/11_steering.py). `python scripts/run_notebooks.py --only 11_steering` runs the notebook with those solutions in place of the stubs.

Reference outputs (seed 42): cosine 0.997; probe 0.95 along $v$ and 0.60 along a random direction; at α = 1 compliance 0.993 and relevance 0.998; at α = 3 relevance 0.352; after ablation language accuracy 0.52 and topic accuracy 0.998; clamp gives compliance 0.975 and relevance 0.995.

## Exercise sheet

**Exercise 1.** $v=(-2,2)$, $\hat v=(-1,1)/\sqrt2$, projection of $h$ is $(-3+1)/\sqrt2=-1.41$. Add with α = 1: $(1,3)$, projection $1.41$. The target mean's projection is $2/\sqrt2=1.41$, so clamping also gives $(1,3)$. Ablate: $(2,2)$, projection 0. One point for $v$, $\hat v$ and the first projection; one for the three results.

**Exercise 2.** Full marks need the three-method table from one run, the α = 0 assertion, and a sentence on which method wins on the harmonic mean and why. Norm scaling keeps the push proportional, so the collapse arrives later than with a fixed vector; accept any result the student's output supports.

**Exercise 3.** With overlap 0 relevance never falls, because the push has no topic component. As overlap grows, the largest safe α shrinks. Good answers for detection in a real model: project the steering vector on directions for other known features, or check whether steering changes the distribution of answer topics on a fixed question set.

Marking anchors. Derivation (2): result 1, working 1. Implementation (4): working code 2, test 1, reproducible run 1. Investigation (4): controlled design 1, measured results 1, explanation 1, limitation 1. Give no marks for numbers that do not appear in an output.

## Quiz 11 key

1. B. 2. C. 3. C. 4. B.

5. a. $v=(-2,2)$; $\hat v=(-1,1)/\sqrt2$; projection $(-1+3)/\sqrt2=1.41$. b. $h'=(1,3)+0.5(-2,2)=(0,4)$. c. $h'=(1,3)-1.41\cdot(-1,1)/\sqrt2=(1,3)-(-1,1)=(2,2)$.

6. Any two of: clamp the projection; scale α by the activation norm; lower α; steer one layer or fewer layers; steer only the first generated tokens; remove the component of $v$ shared with content (1 each, 2 maximum). Choose on development questions, not on the test submission (1).
