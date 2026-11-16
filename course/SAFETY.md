# Safety evaluation and alignment thread

[Course home](../README.md) · [Readings](READINGS.md) · [Extensions](extensions/README.md)

Six sessions carry a short safety block: 4, 5, 6, 8, 10 and 11. Each block fits inside the 25-minute concepts slot. If time runs short, cut the same number of minutes from the board derivation.

## What each session adds

| Session | Question | Evidence students produce |
|---|---|---|
| 4 Adaptation | Does LoRA remove refusals the base model had? | The same fixed refusal checks on base and adapted checkpoints, with counts |
| 5 Alignment | What does a refusal preference pair teach, and how deep does it go? | A written critique of one preference set for over-refusal |
| 6 Multilingual | Does English safety training carry over to Hausa, Yoruba, or Twi? | A per-language harmful rate with paired intervals |
| 8 Vision-language models | Does an instruction inside an image get the same treatment as a typed one? | A benign-proxy comparison table |
| 10 Benchmarking | What does a refusal rate hide? | An outcome table from `safety_eval.py`, including "misunderstood" |
| 11 Steering | Can the arithmetic that adds a refusal direction also remove one? | A written answer; the course steers language and refusal-on only |

## The outcome table

For every model response, a native speaker assigns one label: `refused`, `misunderstood`, `harmful`, or `safe`. "Misunderstood" covers off-topic, empty, and unintelligible replies. Without it, a model that cannot read Hausa looks safe because it never complies, and a model that cannot read Yoruba looks unsafe when it simply failed to follow the prompt.

```bash
python course/extensions/safety_eval.py --self-test
python course/extensions/safety_eval.py labels.csv --reference en
```

The CSV has three columns: `language`, `prompt_id`, `label`. Reuse the prompt ID for every translation of a prompt. The script prints each rate with a 95% bootstrap interval over prompts, and the paired harmful-rate gap to the reference language. The self-test uses simulated labels. It checks the code and says nothing about any model.

## Capstone angle

Any SemEval system can add a per-language or per-slice error analysis. Task 7 has a bonus track on refusing in the requested language, which is a safety evaluation in its own right. For a system built on an instruction-tuned model, repeat the session 4 refusal check after fine-tuning. This work counts inside the existing evaluation and analysis marks of the [rubric](PROJECT.md), unless the instructor decides otherwise.

## Working with harmful content

- Keep prompt and response text out of the repository. Commit labels and IDs only.
- Keep prompt sets in access-controlled storage. Release them only after considering misuse.
- Pay native-speaker annotators, obtain consent, add content warnings, and let them skip items.
- Report a specific working attack to the model provider before publishing it.
- Do not rely on an LLM judge alone in a low-resource language. Validate it against native-speaker labels and report Cohen's kappa.
- Label synthetic data as synthetic, as in every other lab.

## Foundational readings by session

Read the first paper listed for each session before class. The rest are optional depth.

**Session 4**
- [Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To!](https://arxiv.org/abs/2310.03693)

**Session 5**
- [Training language models to follow instructions with human feedback (InstructGPT)](https://arxiv.org/abs/2203.02155)
- [Training a Helpful and Harmless Assistant with RLHF](https://arxiv.org/abs/2204.05862)
- [Constitutional AI](https://arxiv.org/abs/2212.08073)
- [Open Problems and Fundamental Limitations of RLHF](https://arxiv.org/abs/2307.15217)
- [Jailbroken: How Does LLM Safety Training Fail?](https://arxiv.org/abs/2307.02483)
- [Universal and Transferable Adversarial Attacks on Aligned Language Models](https://arxiv.org/abs/2307.15043)
- [Are aligned neural networks adversarially aligned?](https://arxiv.org/abs/2306.15447)
- [Safety Alignment Should Be Made More Than Just a Few Tokens Deep](https://arxiv.org/abs/2406.05946)
- [XSTest: exaggerated safety behaviours](https://arxiv.org/abs/2308.01263)

**Session 6**
- [Low-Resource Languages Jailbreak GPT-4](https://arxiv.org/abs/2310.02446)
- [Multilingual Jailbreak Challenges in Large Language Models](https://arxiv.org/abs/2310.06474)
- [All Languages Matter: On the Multilingual Safety of Large Language Models](https://arxiv.org/abs/2310.00905)
- [Towards Measuring the Representation of Subjective Global Opinions in Language Models](https://arxiv.org/abs/2306.16388)
- [BLEnD: everyday knowledge in diverse cultures and languages](https://arxiv.org/abs/2406.09948)

**Session 8**
- [Visual Adversarial Examples Jailbreak Aligned Large Language Models](https://arxiv.org/abs/2306.13213)
- [FigStep: Jailbreaking Large Vision-Language Models via Typographic Visual Prompts](https://arxiv.org/abs/2311.05608)
- [MM-SafetyBench](https://arxiv.org/abs/2311.17600)
- [Red-Teaming the Stable Diffusion Safety Filter](https://arxiv.org/abs/2210.04610)

**Session 10**
- [Red Teaming Language Models to Reduce Harms](https://arxiv.org/abs/2209.07858)
- [Red Teaming Language Models with Language Models](https://arxiv.org/abs/2202.03286)
- [HarmBench](https://arxiv.org/abs/2402.04249)
- [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685)
- [Llama Guard](https://arxiv.org/abs/2312.06674)

**Session 11**
- [Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717)
- [Representation Engineering: A Top-Down Approach to AI Transparency](https://arxiv.org/abs/2310.01405)
- [Steering Language Models With Activation Engineering](https://arxiv.org/abs/2308.10248)

## Additional references: African-language safety

These come from the AfriSafe group. Most are 2026 preprints, so read them as critical-reading material after the foundational papers. Each applies a method from the list above to African languages.

**Session 5**
- [Aligning Large Language Models for Safety in African Languages: A Case Study of Hausa](https://arxiv.org/abs/2610.37245)

**Session 6**
- [The Illusion of Cross-Lingual Safety in Low-Resource Languages](https://arxiv.org/abs/2608.11146)
- [Do Language Models Understand Igbo Culture? Beyond Multiple Choice](https://arxiv.org/abs/2610.35168)
- [LLM Safety Alignment in Low-Resource Languages: A Systematic Literature Review](https://arxiv.org/abs/2608.14626)

**Session 8**
- [Going PLACES: Participatory Localized Red Teaming for Text-to-Image Safety in the Global South](https://arxiv.org/abs/2605.19190) (FAccT 2026)

**Session 10**
- [Red-Teaming Large Language Models in Low-Resource African Languages: Yoruba and Nigerian Pidgin](https://arxiv.org/abs/2610.39590)
- [AfriHate: hate speech and abusive language datasets for African languages](https://aclanthology.org/2025.naacl-long.92/) (NAACL 2025)
