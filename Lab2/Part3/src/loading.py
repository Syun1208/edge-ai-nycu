from __future__ import annotations

import logging

import torch
import torch.nn as nn


WIKITEXT_DATASET = "wikitext"
WIKITEXT_CONFIG = "wikitext-2-raw-v1"
TEXT_SEPARATOR = "\n\n"
SEQUENCE_LENGTH = 2048

logger = logging.getLogger(__name__)


def load_model(
    *,
    name: str,
    dtype: torch.dtype,
) -> tuple[nn.Module, object]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"📥 loading {name} ({dtype})")
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name,
        dtype=dtype,
        device_map="cuda",
    )
    model.eval()
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, tokenizer


def wikitext_token_ids(
    *,
    tokenizer: object,
    split: str,
) -> torch.Tensor:
    from datasets import load_dataset

    dataset = load_dataset(
        WIKITEXT_DATASET,
        WIKITEXT_CONFIG,
        split=split,
    )
    return tokenizer(
        TEXT_SEPARATOR.join(dataset["text"]),
        return_tensors="pt",
    ).input_ids
