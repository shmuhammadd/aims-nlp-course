"""Transcribe a permitted local audio clip of at most 20 seconds."""
from pathlib import Path
from common import parser, setup, report


def main():
    p = parser(__doc__, "openai/whisper-tiny")
    p.add_argument("--audio", required=True, type=Path)
    p.add_argument("--language", default=None, help="Optional supported language name/code")
    args = p.parse_args()
    if not args.audio.is_file():
        p.error("Provide an existing permitted local audio file")
    torch = setup(args)
    import librosa
    import soundfile as sf
    import numpy as np
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    info = sf.info(args.audio)
    if info.duration > 20:
        p.error("Use a clip no longer than 20 seconds for this teaching extension")
    wave, original_sr = sf.read(args.audio, dtype="float32", always_2d=True)
    wave = wave.mean(axis=1)
    if len(wave) == 0 or not np.isfinite(wave).all():
        p.error("Audio must be nonempty and finite")
    wave = librosa.resample(wave, orig_sr=original_sr, target_sr=16000)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(args.model, revision=args.revision, dtype=torch.float32,
                                                     trust_remote_code=False).to(args.device).eval()
    processor = AutoProcessor.from_pretrained(args.model, revision=args.revision, trust_remote_code=False)
    pipe = pipeline("automatic-speech-recognition", model=model, tokenizer=processor.tokenizer,
                    feature_extractor=processor.feature_extractor, device=0 if args.device == "cuda" else -1)
    generation = {"task": "transcribe", "max_new_tokens": 128, "do_sample": False}
    if args.language:
        generation["language"] = args.language
    with torch.inference_mode():
        result = pipe({"raw": wave, "sampling_rate": 16000}, generate_kwargs=generation)
    report(args, model, {"audio_file": args.audio.name, "seconds": len(wave)/16000, "original_sample_rate": original_sr,
                        "sample_rate": 16000, "text": result["text"], "language_hint": args.language,
                        "note": "Score against a human reference with the lecture-13 normalization policy"})


if __name__ == "__main__":
    main()
