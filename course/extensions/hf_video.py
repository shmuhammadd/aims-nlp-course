"""Optional short-video VLM inference; requires the extra video dependencies."""
from pathlib import Path
from common import parser, setup, report


def main():
    p = parser(__doc__, "HuggingFaceTB/SmolVLM2-256M-Video-Instruct")
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--frames", type=int, default=8)
    p.add_argument("--prompt", default="Describe the order of the visible events in two sentences.")
    args = p.parse_args()
    if not args.video.is_file() or not 2 <= args.frames <= 16:
        p.error("Provide a local video and 2–16 sampled frames")
    torch = setup(args)
    import av
    with av.open(str(args.video)) as container:
        stream = container.streams.video[0]
        if stream.duration is None:
            p.error("Video duration metadata is required; export a short constant-frame-rate clip")
        duration = float(stream.duration * stream.time_base)
    if duration > 10:
        p.error("Use a clip no longer than 10 seconds")
    from transformers import AutoProcessor, AutoModelForImageTextToText
    processor = AutoProcessor.from_pretrained(args.model, revision=args.revision, trust_remote_code=False)
    model = AutoModelForImageTextToText.from_pretrained(args.model, revision=args.revision, dtype=torch.float32,
                                                       trust_remote_code=False, attn_implementation="eager").to(args.device).eval()
    messages = [{"role": "user", "content": [{"type": "video", "path": str(args.video.resolve())},
                                               {"type": "text", "text": args.prompt}]}]
    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                           return_dict=True, return_tensors="pt", num_frames=args.frames,
                                           video_load_backend="pyav").to(args.device)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    answer = processor.batch_decode(output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    report(args, model, {"video_file": args.video.name, "seconds": duration, "requested_frames": args.frames,
                        "prompt": args.prompt, "answer": answer, "input_shapes": {k: list(v.shape) for k, v in inputs.items()},
                        "note": "Record sampled timestamps when extending this into a temporal benchmark"})


if __name__ == "__main__":
    main()
