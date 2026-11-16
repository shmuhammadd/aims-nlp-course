# Lecture 15 — instructor guide

Keep this file private if the cohort can see the repository. Change the numbers in exercise 1 and in the concept check before reusing them for marks.

## Teaching the session

- Announce the order of talks before class, and which teams present in the 60–100 block. With more than five teams, run the rest as a poster rotation or in the final independent-work period, as PROJECT.md sets out.
- Keep the synthesis to three minutes a week. Ask one student per row for a number they remember from that day's lab.
- Put the five-row timing table on the board and leave it up during the talks. Stop every talk at 6:00.
- Session 15 has no quiz. The concept check replaces it. Use CONCEPT_CHECK.md with the numbers changed; the key is in [concept_check_key.md](concept_check_key.md). Collect the reference sheets with the scripts.
- Read each team's day-14 review before class. The reviewer's "claim that needs evidence" is the best defence question.
- Common confusion: students present every experiment they ran. Ask for one result, one failure, one next experiment.
- Common confusion: a development score is described as "our SemEval result". Ask for the split after every number.
- Common confusion: passing a removal ablation is taken as proof that the input is used. The decoy row in the lab is there for this.

## Marking the defence

The defence carries 5 of the capstone's 100 points ("Presentation and individual understanding" in PROJECT.md). A suggested split:

- **Talk (2).** One result with metric, split and interval; one failure with a count; one next experiment. Deduct one point for a number with no split, or for running past six minutes.
- **Individual answers (2).** Each member answers at least one question alone. Full marks for a correct answer, or for "I do not know" followed by the check that would settle it. No marks for a guess presented as fact.
- **Contributions (1).** What each member says they did matches `contributions.md` and the repository history.

Ask the quietest member first. Ask each member one of: "Which split is that number from?", "What would change your conclusion?", "Explain this line of your config", "What did the ablation hold fixed?". Mark each member separately when the answers differ. A careful negative result can earn full marks. Rank, model size and paper acceptance earn nothing here.

Assess the paper, evaluation and package afterwards under the full rubric in PROJECT.md. If official evaluation has not opened, assess readiness and development evidence under the same rubric.

## Practical questions

**1. What experiment separates modality use from a memorised answer?** A counterfactual swap: keep the question wording and replace the image, audio or document with another item's. A system that uses the input changes its answer. Removal and mismatch of each modality, on the same items, complete the picture.

**2. How does the paper report results if the official evaluation has not opened?** It presents measured development results with metric, split and track labels, and states that official evaluation is pending. It keeps the export command and the later deadlines. It contains no invented test score, receipt or rank.

**3. What makes a negative result useful?** A sound baseline, a controlled comparison on the same items, an interval, enough documentation to rerun it, and a clear statement of the hypothesis it weakens.

## Notebook

Solutions for the two stub cells are in [solutions/15_capstone.py](solutions/15_capstone.py). `python scripts/run_notebooks.py --only 15_capstone` runs the notebook with those solutions in place of the stubs.

Reference outputs (seed 42): case 0 has the policy "Use plot B." and answer 7; full accuracy 1.0 and coverage 1.0; text-only and image-only accuracy 0.0 and coverage 0.0 with selective accuracy `None`; paired interval [1, 1]; `selective_scores` on the toy list gives coverage 0.750, selective accuracy 0.778, overall accuracy 0.583; the decoy changes 12 of 12 answers, with 12 right only with the real chart, 0 right only with the decoy, decoy accuracy 0.0 and 12 of 12 decoy answers citing two evidence IDs.

Questions in the notebook. Section 1: change the chart or the policy of the case and keep the question, which is what the decoy does. Section 2: selective accuracy divides by the number answered, which is zero; reporting 0 would claim the system answered and was wrong. Section 3: the `id` field, together with `condition`. Section 4: check that the cited evidence supports the answer, for example by recomputing the answer from the cited chart.

## Exercise sheet

**Exercise 1.** a. Full: $(6+4)/12=0.833$. Ablated: $(6+1)/12=0.583$. Paired difference: $(4-1)/12=0.25$. McNemar with $b=4$, $c=1$: $p=[C(5,4)+C(5,5)]/2^5=6/32=0.1875$. Twelve items do not show the full system to be better at the 0.05 level. b. Coverage $8/12=0.667$; selective accuracy $6/8=0.75$; overall accuracy $6/12=0.5$. One point for each part.

**Exercise 2.** Reference (seed 42): under the mismatched policy the system answers all 12 cases (coverage 1.0) and 7 are right (accuracy 0.583). Five neighbouring cases share the same letter, and in two more the two bars happen to have the same height, so 5 answers change. With three options a shifted decoy often matches the original by chance. A good answer says the decoy must be checked to differ, or drawn to differ, before the test means anything. Two points for working code, one for the evidence-ID assertion, one for a JSONL file of 36 lines that reproduces.

**Exercise 3.** One point each for: a script that states the comparison, metric, split and interval from a logged run; a failure with a count; three questions with answers, one naming a result that would change the conclusion; a dated next action with a named person in `participation.md`. The full paper is marked separately. A result from the synthetic chart lab is not a shared-task result.

Marking anchors. Derivation (2): result 1, working 1. Implementation (4): working code 2, test 1, reproducible run 1. Investigation (4): controlled design 1, measured results 1, explanation 1, limitation 1. Give no marks for numbers that do not appear in an output or a log.
