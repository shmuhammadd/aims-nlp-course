"""Offline integration checks for optional text/LoRA CLIs with a tiny random Qwen3.

Requires requirements-models.txt. This tests API plumbing and real optimization,
not pretrained checkpoint quality or remote download availability.
"""
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM, set_seed
    set_seed(42)
    with tempfile.TemporaryDirectory(prefix="aims-model-check-") as td:
        base = Path(td) / "random-qwen3"
        words = ["<pad>", "<unk>", "<bos>", "<eos>", "user", "assistant", "Reply", "with", "the", "sum", "of", "and", "only", ".", "1", "2", "3", "4", "5", "6", "7", "Explain", "why", "a", "test", "set", "should", "not", "select", "learning", "rate", "in", "two", "sentences"]
        tokenizer = Tokenizer(WordLevel({w: i for i, w in enumerate(words)}, unk_token="<unk>"))
        tokenizer.pre_tokenizer = Whitespace()
        tok = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="<unk>", pad_token="<pad>", bos_token="<bos>", eos_token="<eos>")
        tok.chat_template = "{% for message in messages %}{{ message['role'] + ' ' + message['content'] + ' ' }}{% endfor %}{% if add_generation_prompt %}{{ 'assistant ' }}{% endif %}"
        tok.save_pretrained(base)
        config = Qwen3Config(vocab_size=len(words), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                             num_attention_heads=2, num_key_value_heads=2, head_dim=16, max_position_embeddings=512,
                             bos_token_id=2, eos_token_id=3, pad_token_id=0, tie_word_embeddings=True)
        model = Qwen3ForCausalLM(config)
        model.save_pretrained(base)
        commands = [
            ["hf_text.py", "--model", str(base), "--max-new-tokens", "4", "--output", str(Path(td)/"text.json")],
            ["hf_lora.py", "--model", str(base), "--steps", "2", "--rank", "2", "--adapter-dir", str(Path(td)/"adapter"), "--output", str(Path(td)/"lora.json")],
        ]
        for name, *args in commands:
            result = subprocess.run([sys.executable, str(ROOT/"course/extensions"/name), *args], text=True, capture_output=True, timeout=180)
            if result.returncode:
                print(result.stdout); print(result.stderr, file=sys.stderr)
                raise SystemExit(result.returncode)
            print("PASS", name, "with a local randomly initialized model")
        record = json.loads((Path(td)/"lora.json").read_text())
        assert len(record["result"]["train_losses"]) == 2
        assert all(math.isfinite(x) for x in record["result"]["train_losses"])
        from safetensors.torch import load_file
        weights = load_file(str(Path(td)/"adapter/adapter_model.safetensors"))
        assert any(torch.count_nonzero(v).item() > 0 for k,v in weights.items() if "lora_B" in k)
        print("PASS: LoRA B weights changed from their zero initialization")
        # Import APIs used by the image/audio scripts without downloading weights.
        from transformers import CLIPModel, CLIPProcessor, AutoModelForVision2Seq, AutoModelForSpeechSeq2Seq, AutoProcessor
        assert all([CLIPModel, CLIPProcessor, AutoModelForVision2Seq, AutoModelForSpeechSeq2Seq, AutoProcessor])
        print("PASS: required image/audio APIs import in the pinned environment")


if __name__ == "__main__":
    main()
