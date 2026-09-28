"""Train an actual small causal Transformer from random initialization, offline."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, default=Path("outputs/tiny-decoder"))
    args = p.parse_args()
    if not 1 <= args.steps <= 2000:
        p.error("--steps must be in [1, 2000]")
    if args.output.exists():
        p.error("Choose a new output directory to preserve earlier runs")
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel, set_seed
    set_seed(args.seed)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    train_docs = ["the class starts today", "the lab opens today", "the class studies models", "the lab studies language"]
    dev_docs = ["the class opens today", "the lab starts today"]
    vocab = ["<pad>", "<bos>", "<eos>", "<unk>"] + sorted(set("".join(train_docs)))
    ids = {c: i for i, c in enumerate(vocab)}
    def batch(docs):
        sequences = [[1] + [ids.get(c, 3) for c in s] + [2] for s in docs]
        length = max(map(len, sequences))
        x = torch.tensor([s + [0] * (length-len(s)) for s in sequences])
        mask = (x != 0).long()
        labels = x.clone(); labels[mask == 0] = -100
        return {"input_ids": x, "attention_mask": mask, "labels": labels}
    train, dev = batch(train_docs), batch(dev_docs)
    config = GPT2Config(vocab_size=len(vocab), n_positions=64, n_embd=32, n_layer=2, n_head=2,
                        bos_token_id=1, eos_token_id=2, pad_token_id=0, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0)
    model = GPT2LMHeadModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    history = []; best_loss = float("inf"); best = None
    for step in range(args.steps):
        model.train(); optimizer.zero_grad()
        loss = model(**train).loss; loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if step % 10 == 0 or step == args.steps-1:
            model.eval()
            with torch.no_grad():
                value = model(**dev).loss.item()
            history.append({"step": step, "train_loss": loss.item(), "dev_loss": value})
            print(history[-1])
            if value < best_loss:
                best_loss = value; best = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best); model.eval()
    # Future-token intervention on a real Transformer.
    a = train["input_ids"][:1, :8].clone(); b = a.clone(); b[:, 5:] = 3
    with torch.no_grad():
        la = model(a).logits; lb = model(b).logits
    assert torch.allclose(la[:, :5], lb[:, :5], atol=1e-5)
    args.output.mkdir(parents=True)
    model.save_pretrained(args.output)
    (args.output / "vocab.json").write_text(json.dumps(vocab))
    (args.output / "run.json").write_text(json.dumps({"seed": args.seed, "history": history, "best_dev_loss": best_loss, "parameters": model.num_parameters(), "data": "course-authored-v1", "torch": torch.__version__}, indent=2))
    print("Saved selected checkpoint to", args.output, "(no final test claim)")


if __name__ == "__main__":
    main()
