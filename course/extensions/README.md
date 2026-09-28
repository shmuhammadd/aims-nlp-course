# Optional experiments with pretrained models

[Course home](../../README.md) · [Setup](../SETUP.md) · [Validation status](../VALIDATION.md)

These extensions connect the offline mechanisms to actual model libraries. They are optional: no paid APIs or cloud accounts are required, and a download failure does not block the core course. Install `requirements-models.txt` in a separate environment. Run commands below from the repository root. Model scripts default to CPU float32, keep generation short, and never upload inputs.

The scripts print software versions, seed, model ID, and resolved revision where available. Use `--revision COMMIT_SHA` for exact reruns and `--output outputs/new-run.json` to save inference records. Existing output files are protected. Record data hashes and prompt versions in a project report. `--device cuda` is available when supported; CUDA performance was not assumed in designing the course.

## Planning guide

These are approximate planning allowances, not measurements on your hardware. First downloads can take longer than the experiment.

| Extension | Model/data | Suggested available memory | Scope |
|---|---|---|---|
| Text | Qwen3-0.6B | 4–8 GB RAM for short CPU inference | One prompt and tokenization audit |
| Tiny decoder | Randomly initialized ~small 2-layer GPT-2 architecture | 2 GB RAM | Offline 100-step training demonstration |
| LoRA | Qwen3-0.6B | 8–16 GB RAM or a suitable GPU | Four authored examples; ten updates |
| Sentiment | Local permitted TSV exports | 2–4 GB RAM for a small subset | Character TF-IDF baseline |
| Image-text retrieval | CLIP ViT-B/32 | 2–4 GB RAM | Three generated images |
| Visual QA | SmolVLM-256M-Instruct | 4–8 GB RAM | One generated chart |
| Speech | Whisper tiny | 2–4 GB RAM | A local clip ≤20 seconds |
| Video | SmolVLM2-256M-Video-Instruct | 4–8 GB RAM, potentially more | A local clip ≤10 seconds; 2–16 frames |

Use development examples to select one configuration before final testing. Do not treat one generated output as a benchmark. Larger language/vision/omni models in the readings are discussion material rather than mandatory downloads.

## Lectures 1–3: tokenization, inference, and actual decoder training

```bash
python course/extensions/hf_text.py --output outputs/text-run.json
python course/extensions/train_tiny_decoder.py --steps 100 --output outputs/tiny-decoder
```

1. Inspect the serialized chat prompt, token counts, and decoder configuration.
2. Compare tokenizer counts on five matched examples from two languages; do not claim quality from counts alone.
3. In the tiny decoder run, follow shifted causal loss, prompt/padding handling, development selection, and the future-token invariance check.
4. Change one training setting and rerun into a different directory. Plot or tabulate the logged train/dev losses. The tiny decoder trains real attention layers but has too little data for useful language generation.

Expected checks: causal prefix logits remain unchanged by suffix edits; a training loss can fall without proving downstream usefulness. The text script measures generation wall time including prefill, excluding model loading. It does not separately measure time to first token or pure decode throughput.

## Lecture 4: actual response-only LoRA SFT

```bash
python course/extensions/hf_lora.py --steps 10 --rank 4 --adapter-dir outputs/adapter-r4 --output outputs/lora-r4.json
```

Inspect the printed supervised token IDs. Prompt and padding labels are ignored while prompt tokens remain available for attention. The model performs the causal shift internally. Only q/v adapters train; the base stays frozen. Train/dev prompts are distinct, but four arithmetic examples are only a pipeline smoke test.

Repeat with rank 8 in a new directory, keeping examples and seed fixed. Compare trainable parameters and development loss. A decrease is not guaranteed in ten updates; report it honestly. For a project, replace the smoke-test data with a permitted, source-disjoint dataset and add final held-out evaluation. The script does not claim a full instruction-tuning benchmark or implement QLoRA.

## Lectures 5, 7–10: preference, retrieval, reasoning, and evaluation

```bash
python course/extensions/hf_text.py --prompt 'Evidence [D1]: The library opens at eight on weekdays. Question: When does the library open? Answer using the evidence and cite its ID.' --output outputs/rag-answer.json
python course/extensions/hf_text.py --prompt 'Compute (7+5)*3. Return only the number.' --output outputs/arithmetic.json
```

For preferences, collect two responses under deliberately different prompt styles and label them with a blinded rubric. For RAG, use your notebook's retrieved evidence and repeat with oracle, shuffled, and missing evidence. For tool use, parse proposed JSON and pass it through the notebook validator; do not execute raw generated code. The supplied text script is greedy; to study sampling, explicitly add sampling settings, a fixed number of candidates, and seed logging.

Evaluate exact task correctness, citation support, format validity, and resource cost separately. Use the notebook-10 metric functions on a fixed set. This pathway reuses one small checkpoint so experiments remain feasible.

## Lecture 6: African-language sentiment on permitted local data

Read the [AfriSenti paper](https://aclanthology.org/2023.emnlp-main.862/) and [official data repository](https://github.com/afrisenti-semeval/afrisent-semeval-2023). Respect the applicable source conditions. Export local UTF-8 TSVs with `id`, `text`, `label`, and `language` columns. Prefix IDs with dataset/language where needed to make them globally unique. Keep original split boundaries and source groups. The script does not download tweets.

```bash
python course/extensions/afrisenti_baseline.py --train path/to/train.tsv --dev path/to/dev.tsv --test path/to/test.tsv --output outputs/sentiment.json
```

The baseline fits character 2–5-gram TF-IDF on training data, selects logistic-regression C on development macro-F1, then evaluates the selected model once on test. It rejects overlapping IDs and exact normalized text across splits; manually audit near duplicates and source dependence as well. Report per-language counts and scores. The three synthetic TSVs in `course/data/` are only a smoke-test fixture for this script:

```bash
python course/extensions/afrisenti_baseline.py --train course/data/sentiment_train.tsv --dev course/data/sentiment_dev.tsv --test course/data/sentiment_test.tsv --output outputs/sentiment-smoke.json
```

## Lectures 11–12: pretrained cross-modal retrieval and visual QA

```bash
python course/extensions/hf_vision.py --mode clip --condition original --output outputs/clip-original.json
python course/extensions/hf_vision.py --mode clip --condition swapped --output outputs/clip-swapped.json
python course/extensions/hf_vision.py --mode vlm --condition original --output outputs/vlm-original.json
python course/extensions/hf_vision.py --mode vlm --condition blank --output outputs/vlm-blank.json
python course/extensions/hf_vision.py --mode vlm --condition swapped --output outputs/vlm-swapped.json
```

CLIP compares three generated colored-square images against three descriptions. Check both retrieval directions; candidate softmax scores are not calibrated probabilities. SmolVLM answers the highest-bar question for an authored chart; the original answer is B, the swapped answer A, and the blank condition should be unknown. Inspect actual generated responses rather than assuming these expected answers occur.

The VLM uses the checkpoint's processor and chat template. Compare it against the explicit geometry baseline from lecture 12. Add style and position changes to avoid giving the specialized baseline an unexamined fixed-layout advantage. Keep model revision, image preprocessing, prompt, and token budget fixed across interventions.

## Lecture 13: pretrained speech recognition

```bash
python course/extensions/hf_audio.py --audio path/to/consented-clip.wav --output outputs/asr.json
```

Use a short consented clip and an independently prepared reference transcript. The script checks duration, averages channels, resamples to 16 kHz, and requests transcription rather than translation. `--language` supplies a supported language hint when appropriate. Compute WER/CER with declared normalization; inspect names, numbers, and negation separately. A synthetic tone is useful for signal processing but is not a speech-recognition benchmark.

## Lecture 14: optional video model

Install the extra dependencies with `python -m pip install -r requirements-video.txt`, then:

```bash
python course/extensions/hf_video.py --video path/to/short-clip.mp4 --frames 8 --output outputs/video.json
```

Use a permitted clip of at most ten seconds. Ask one temporal question with a clear reference answer; repeat with reversed and shuffled frame sequences exported as separate clips. Record the sampling timestamps when extending the script into a benchmark. The script limits requested frames but the actual processor and token budget must still be inspected. It is an extension, not a requirement for the offline lecture-14 lab.

## Model API references

- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B)
- [PEFT 0.17 quick tour](https://huggingface.co/docs/peft/v0.17.0/en/quicktour)
- [CLIP model card](https://huggingface.co/openai/clip-vit-base-patch32)
- [SmolVLM image model card](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct)
- [Transformers 4.57.1 SmolVLM documentation](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/smolvlm)
- [Transformers 4.57.1 Whisper documentation](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/whisper)
- [SmolVLM2 video model card](https://huggingface.co/HuggingFaceTB/SmolVLM2-256M-Video-Instruct)

Model cards and APIs evolve. The course pins an older supported library line for consistency; do not copy a current 5.x example into the 4.57 environment without checking compatibility.
