import argparse
import glob
import math
import os

import torch
import torch.nn as nn
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import common

DEFAULT_MODEL = "Qwen/Qwen3-4B"


def load_model(name=DEFAULT_MODEL, dtype="float16", device="cuda"):
    """Load a causal LM and its tokenizer for evaluation (and for compression: no grads, no KV cache)."""
    if name.endswith(".pt"):
        return load_compressed_model(name, device)
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=getattr(torch, dtype), device_map=device)
    model.eval()
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tokenizer


def load_compressed_model(path, device="cuda"):
    model = torch.load(path, map_location=device, weights_only=False, mmap=True)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model.name_or_path)
    return model, tokenizer


def wikitext2_test_text():
    try:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    except (PermissionError, OSError, ConnectionError, ValueError):
        cache = os.environ.get("HF_DATASETS_CACHE", os.path.expanduser("~/.cache/huggingface/datasets"))
        ds = Dataset.from_file(sorted(glob.glob(os.path.join(cache, "wikitext/wikitext-2-raw-v1/*/*/wikitext-test.arrow")))[0])
    return "\n\n".join(ds["text"])


@torch.no_grad()
def eval_ppl_wikitext2(model, tokenizer, seqlen=2048):
    device = next(model.parameters()).device
    ids = tokenizer(wikitext2_test_text(), return_tensors="pt").input_ids
    loss_fct = nn.CrossEntropyLoss(reduction="sum")
    nll, n_tok = 0.0, 0
    for i in range(ids.shape[1] // seqlen):
        batch = ids[:, i * seqlen : (i + 1) * seqlen].to(device)
        logits = model(batch).logits.float()
        nll += loss_fct(logits[:, :-1].reshape(-1, logits.shape[-1]), batch[:, 1:].reshape(-1)).item()
        n_tok += seqlen - 1
    return math.exp(nll / n_tok)


def compression_rate(model):
    """0 for a model that was not compressed in this process; otherwise needs
    common.record_base(model) to have been called before compressing."""
    if not hasattr(model, "_base_linear_numel"):
        return 0.0
    return common.compressed_ratio(model=model)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="WikiText-2 PPL of a (base) model")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    args = ap.parse_args()
    model, tok = load_model(args.model, args.dtype)
    print(f"ppl: {eval_ppl_wikitext2(model, tok):.4f}")
    