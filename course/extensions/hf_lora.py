"""Response-only LoRA SFT with an explicit loop and authored teaching data."""
from pathlib import Path
from common import parser, setup, report


def main():
    p = parser(__doc__, "Qwen/Qwen3-0.6B")
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--adapter-dir", type=Path, default=Path("outputs/course-adapter"))
    args = p.parse_args()
    if not 1 <= args.steps <= 200 or not 1 <= args.rank <= 32:
        p.error("Teaching limits: steps 1–200, rank 1–32")
    if args.adapter_dir.exists():
        p.error("Choose a new --adapter-dir to preserve earlier runs")
    torch = setup(args)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, TaskType, get_peft_model
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision, trust_remote_code=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, torch_dtype=torch.float32, trust_remote_code=False).to(args.device)
    model = get_peft_model(base, LoraConfig(task_type=TaskType.CAUSAL_LM, r=args.rank, lora_alpha=2*args.rank,
                                          lora_dropout=0., target_modules=["q_proj", "v_proj"]))
    model.print_trainable_parameters()
    examples = [("Reply with the sum of 2 and 3 only.", "5"), ("Reply with the sum of 4 and 1 only.", "5"),
                ("Reply with the sum of 3 and 3 only.", "6"), ("Reply with the sum of 1 and 2 only.", "3")]
    # Authored smoke-test data, not a claim of mathematical generalization.
    dev = [("Reply with the sum of 2 and 4 only.", "6"), ("Reply with the sum of 4 and 3 only.", "7")]
    def make_batch(rows):
        encoded = []
        for prompt, response in rows:
            prefix = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
            prompt_ids = tok.encode(prefix, add_special_tokens=False)
            answer_ids = tok.encode(response, add_special_tokens=False) + [tok.eos_token_id]
            seq = prompt_ids + answer_ids
            if len(seq) > 256:
                raise ValueError("Example too long; do not silently truncate the supervised response")
            encoded.append((seq, [-100]*len(prompt_ids)+answer_ids))
        length = max(len(x) for x, _ in encoded)
        inputs = torch.tensor([x+[tok.pad_token_id]*(length-len(x)) for x, _ in encoded], device=args.device)
        labels = torch.tensor([y+[-100]*(length-len(y)) for _, y in encoded], device=args.device)
        mask = torch.tensor([[1]*len(x)+[0]*(length-len(x)) for x, _ in encoded], device=args.device)
        assert (labels != -100).any(dim=1).all()
        return {"input_ids": inputs, "labels": labels, "attention_mask": mask}
    train_batch = make_batch(examples); dev_batch = make_batch(dev)
    print("First-row supervised IDs:", train_batch["labels"][0][train_batch["labels"][0] != -100].tolist())
    def evaluate():
        model.eval()
        with torch.no_grad():
            return float(model(**dev_batch).loss)
    before = evaluate()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    history = []
    # Microbatch one example to keep activation memory bounded.
    for step in range(args.steps):
        model.train(); optimizer.zero_grad()
        i = step % len(examples)
        loss = model(**{k: v[i:i+1] for k, v in train_batch.items()}).loss
        loss.backward(); torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
        optimizer.step(); history.append(float(loss.detach()))
        print("step", step, "train loss", history[-1])
    after = evaluate()
    args.adapter_dir.mkdir(parents=True)
    model.save_pretrained(args.adapter_dir); tok.save_pretrained(args.adapter_dir)
    report(args, base, {"dev_loss_before": before, "dev_loss_after": after, "train_losses": history,
                        "rank": args.rank, "steps": args.steps, "adapter_dir": str(args.adapter_dir),
                        "data": "course-authored-arithmetic-v1", "warning": "Pipeline smoke test; no held-out benchmark claim"})


if __name__ == "__main__":
    main()
