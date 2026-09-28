"""Optional text-model inference and tokenizer audit; downloads on first run."""
from common import parser, setup, report


def main():
    p = parser(__doc__, "Qwen/Qwen3-0.6B")
    p.add_argument("--prompt", default="Explain why a test set should not select a learning rate in two sentences.")
    p.add_argument("--max-new-tokens", type=int, default=96)
    args = p.parse_args()
    if not 1 <= args.max_new_tokens <= 1024:
        p.error("--max-new-tokens must be in [1, 1024]")
    torch = setup(args)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, dtype=torch.float32, trust_remote_code=False).to(args.device).eval()
    messages = [{"role": "user", "content": args.prompt}]
    # Qwen3 permits explicit non-thinking mode; other templates may ignore this keyword.
    rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tok(rendered, return_tensors="pt", add_special_tokens=False, return_token_type_ids=False).to(args.device)
    if inputs.input_ids.shape[1] > 1024:
        p.error("Teaching limit: use at most 1024 prompt tokens")
    if args.device == "cuda":
        torch.cuda.synchronize()
    import time
    start = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tok.eos_token_id)
    if args.device == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    new = generated[0, inputs.input_ids.shape[1]:]
    audit = [{"text": s, "tokens": len(tok.encode(s, add_special_tokens=False)), "utf8_bytes": len(s.encode("utf-8"))}
             for s in ["The class starts today.", "Darasi ya fara yau.", "Darasa linaanza leo."]]
    report(args, model, {"prompt": args.prompt, "rendered_prompt": rendered, "answer": tok.decode(new, skip_special_tokens=True),
                        "input_tokens": inputs.input_ids.shape[1], "output_tokens": len(new),
                        "generation_seconds_including_prefill": seconds, "tokenizer_audit": audit,
                        "config": {k: getattr(model.config, k, None) for k in ["hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads"]}})


if __name__ == "__main__":
    main()
