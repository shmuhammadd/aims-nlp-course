"""Run pretrained CLIP or SmolVLM on original generated images."""
from common import parser, setup, report


def main():
    p = parser(__doc__, None)
    p.add_argument("--mode", choices=["clip", "vlm"], default="clip")
    p.add_argument("--condition", choices=["original", "blank", "swapped"], default="original")
    args = p.parse_args()
    if args.model is None:
        args.model = "openai/clip-vit-base-patch32" if args.mode == "clip" else "HuggingFaceTB/SmolVLM-256M-Instruct"
    torch = setup(args)
    from PIL import Image, ImageDraw, ImageFont
    if args.mode == "clip":
        from transformers import CLIPModel, CLIPProcessor
        model = CLIPModel.from_pretrained(args.model, revision=args.revision, trust_remote_code=False).to(args.device).eval()
        processor = CLIPProcessor.from_pretrained(args.model, revision=args.revision, trust_remote_code=False)
        images = []
        for color in ["red", "green", "blue"]:
            im = Image.new("RGB", (224, 224), "white")
            ImageDraw.Draw(im).rectangle((48, 48, 176, 176), fill=color)
            images.append(im)
        texts = ["a red square on a white background", "a green square on a white background", "a blue square on a white background"]
        if args.condition == "blank":
            images = [Image.new("RGB", (224, 224), "white") for _ in images]
        elif args.condition == "swapped":
            images = images[::-1]
        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(args.device)
        with torch.inference_mode():
            scores = model(**inputs).logits_per_image.cpu()
        report(args, model, {"condition": args.condition, "labels": texts, "scores": scores.tolist(),
                            "selected_label_indices": scores.argmax(dim=1).tolist(),
                            "data": "course-authored-colored-squares-v1", "note": "Scores are not calibrated probabilities"})
    else:
        from transformers import AutoProcessor, AutoModelForVision2Seq
        processor = AutoProcessor.from_pretrained(args.model, revision=args.revision, trust_remote_code=False)
        model = AutoModelForVision2Seq.from_pretrained(args.model, revision=args.revision, torch_dtype=torch.float32,
                                                       trust_remote_code=False, attn_implementation="eager").to(args.device).eval()
        values = [7, 3, 5] if args.condition == "swapped" else [3, 7, 5]
        im = Image.new("RGB", (384, 320), "white")
        draw = ImageDraw.Draw(im); font = ImageFont.load_default(size=22)
        if args.condition != "blank":
            draw.line((40, 270, 350, 270), fill="black", width=2)
            for i, (name, value) in enumerate(zip("ABC", values)):
                x = 65+100*i
                draw.rectangle((x, 270-value*25, x+50, 270), fill=["red", "green", "blue"][i])
                draw.text((x+15, 280), name, fill="black", font=font)
        question = "Which bar is tallest, A, B, or C? If no bars are visible, say unknown."
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=prompt, images=[im], return_tensors="pt").to(args.device)
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=48, do_sample=False)
        answer = processor.batch_decode(generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        report(args, model, {"condition": args.condition, "question": question, "answer": answer,
                            "expected": "unknown" if args.condition == "blank" else "ABC"[values.index(max(values))],
                            "image_size": list(im.size), "input_shapes": {k: list(v.shape) for k, v in inputs.items()},
                            "data": "course-authored-chart-v1", "warning": "One controlled probe, not a benchmark"})


if __name__ == "__main__":
    main()
