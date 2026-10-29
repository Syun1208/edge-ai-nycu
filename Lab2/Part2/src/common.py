from __future__ import annotations

import random

import torch
import torch.nn as nn
from datasets import load_dataset

FP_BITS = 16
CALIBRATION_DATASET = "Salesforce/wikitext"
CALIBRATION_CONFIG = "wikitext-2-raw-v1"
CALIBRATION_SPLIT = "train"
TEXT_SEPARATOR = "\n\n"


def target_linears(*, model: nn.Module) -> dict[str, nn.Linear]:
    return {name: module for name, module in model.model.layers.named_modules() if isinstance(module, nn.Linear)}


def record_base(model: nn.Module) -> None:
    model._base_linear_numel = sum(module.weight.numel() for module in target_linears(model=model).values())


def set_storage_bits(
    *,
    module: nn.Linear,
    bits: float,
) -> None:
    module._storage_bits = float(bits)


def compressed_ratio(*, model: nn.Module) -> float:
    base_bits = model._base_linear_numel * FP_BITS
    used_bits = sum(
        getattr(module, "_storage_bits", module.weight.numel() * FP_BITS)
        for module in target_linears(model=model).values()
    )
    return 1.0 - used_bits / base_bits


def quant_bits_per_weight(
    *,
    wbits: int,
    group_size: int,
) -> float:
    return wbits + (FP_BITS + wbits) / group_size


def get_calib_data(
    *,
    tokenizer,
    nsamples: int = 128,
    seqlen: int = 2048,
    seed: int = 0,
) -> torch.Tensor:
    dataset = load_dataset(
        CALIBRATION_DATASET,
        CALIBRATION_CONFIG,
        split=CALIBRATION_SPLIT,
    )
    token_ids = tokenizer(
        TEXT_SEPARATOR.join(dataset["text"]),
        return_tensors="pt",
    ).input_ids

    generator = random.Random(seed)
    starts = [generator.randint(0, token_ids.shape[1] - seqlen - 1) for _ in range(nsamples)]
    return torch.cat(
        [token_ids[:, start : start + seqlen] for start in starts],
        dim=0,
    )
