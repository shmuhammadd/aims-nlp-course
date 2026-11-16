# Safety evaluation and alignment thread

[Course home](../README.md) · [Readings](READINGS.md) · [Extensions](extensions/README.md)

Five lectures carry a short safety block: sessions 4, 5, 6, 10, and 12. Each block fits inside the 25-minute concepts slot. If time runs short, cut the same number of minutes from the board derivation.

## What each session adds

| Session | Question | Evidence students produce |
|---|---|---|
| 4 Adaptation | Does LoRA remove refusals the base model had? | The same fixed refusal checks on base and adapted checkpoints, with counts |
| 5 Preferences | What does a refusal preference pair teach, and how deep does it go? | A written critique of one preference set for over-refusal |
| 6 Multilingual | Does English safety training carry over to Hausa, Yoruba, or Twi? | A per-language harmful rate with paired intervals |
| 10 Evaluation | What does a refusal rate hide? | An outcome table from `safety_eval.py`, including "misunderstood" |
| 12 VLM | Does an instruction inside an image get the same treatment as a typed one? | A benign-proxy comparison table |

## The outcome table

For every model response, a native speaker assigns one label: `refused`, `misunderstood`, `harmful`, or `safe`. "Misunderstood" covers off-topic, empty, and unintelligible replies. Without it, a model that cannot read Hausa looks safe because it never complies, and a model that cannot read Yoruba looks unsafe when it simply failed to follow the prompt.

```bash
python course/extensions/safety_eval.py --self-test
python course/extensions/safety_eval.py labels.csv --reference en
```

The CSV has three columns: `language`, `prompt_id`, `label`. Reuse the prompt ID for every translation of a prompt. The script prints each rate with a 95% bootstrap interval over prompts, and the paired harmful-rate gap to the reference language. The self-test uses simulated labels. It checks the code and says nothing about any model.

## Capstone angle

Any SemEval system can add a per-language or per-slice error analysis. If the task involves hate speech, toxicity, or other harms, the safety analysis can carry the evaluation section. AfriHate, for example, covers hate speech and abusive language in African languages. For a system built on an instruction-tuned model, repeat the session 4 refusal check after fine-tuning. This work counts inside the existing evaluation and analysis marks of the [rubric](PROJECT.md), unless the instructor decides otherwise.

## Working with harmful content

- Keep prompt and response text out of the repository. Commit labels and IDs only.
- Keep prompt sets in access-controlled storage. Release them only after considering misuse.
- Pay native-speaker annotators, obtain consent, add content warnings, and let them skip items.
- Report a specific working attack to the model provider before publishing it.
- Do not rely on an LLM judge alone in a low-resource language. Validate it against native-speaker labels and report Cohen's kappa.
- Label synthetic data as synthetic, as in every other lab.

## Readings by session

Several readings come from the AfriSafe group. Most are 2026 preprints, so read them as critical-reading material and pair each with an independent paper.

**Session 4**
- [Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To!](https://arxiv.org/abs/2310.03693)

**Session 5**
- [Constitutional AI](https://arxiv.org/abs/2212.08073)
- [Jailbroken: How Does LLM Safety Training Fail?](https://arxiv.org/abs/2307.02483)
- [Safety Alignment Should Be Made More Than Just a Few Tokens Deep](https://arxiv.org/abs/2406.05946)
- [Aligning Large Language Models for Safety in African Languages: A Case Study of Hausa](https://arxiv.org/abs/2610.37245)
- [XSTest: exaggerated safety behaviours](https://arxiv.org/abs/2308.01263)

**Session 6**
- [Low-Resource Languages Jailbreak GPT-4](https://arxiv.org/abs/2310.02446)
- [Multilingual Jailbreak Challenges in Large Language Models](https://arxiv.org/abs/2310.06474)
- [The Illusion of Cross-Lingual Safety in Low-Resource Languages](https://arxiv.org/abs/2608.11146)
- [BLEnD: everyday knowledge in diverse cultures and languages](https://arxiv.org/abs/2406.09948)
- [Do Language Models Understand Igbo Culture? Beyond Multiple Choice](https://arxiv.org/abs/2610.35168)
- [LLM Safety Alignment in Low-Resource Languages: A Systematic Literature Review](https://arxiv.org/abs/2608.14626)

**Session 10**
- [Red-Teaming Large Language Models in Low-Resource African Languages: Yoruba and Nigerian Pidgin](https://arxiv.org/abs/2610.39590)
- [HarmBench](https://arxiv.org/abs/2402.04249)
- [Red Teaming Language Models with Language Models](https://arxiv.org/abs/2202.03286)
- [Llama Guard](https://arxiv.org/abs/2312.06674)

**Session 12**
- [FigStep: Jailbreaking Large Vision-Language Models via Typographic Visual Prompts](https://arxiv.org/abs/2311.05608)
- [MM-SafetyBench](https://arxiv.org/abs/2311.17600)
- [Going PLACES: Participatory Localized Red Teaming for Text-to-Image Safety in the Global South](https://arxiv.org/abs/2605.19190)
