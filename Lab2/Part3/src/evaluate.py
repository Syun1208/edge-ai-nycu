from __future__ import annotations

import logging
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn

from src.cuda_quantizer import build_tile_quantizer
from src.formats import NVFP4_BLOCK
from src.loading import SEQUENCE_LENGTH, load_model, wikitext_token_ids
from src.model_quantizer import (
    FormatCounter,
    MxOperandQuantizer,
    Nvfp4OperandQuantizer,
    OperandQuantizer,
    quantize_model,
)
from src.progress import EVALUATE, progress
from src.thresholds import OPERANDS, PROJECTIONS, ThresholdTable
from src.tiles import ALL_FP4, TileQuantizer, TorchTileQuantizer


METHODS = ("fp16", "mxfp4", "nvfp4", "adaptquant")

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    method: str
    model: str
    perplexity: float
    weight_bits: float
    activation_bits: float
    weight_formats: Mapping[str, float] = field(default_factory=dict)
    activation_formats: Mapping[str, float] = field(default_factory=dict)
    seconds_quantize: float = 0.0


@torch.no_grad()
def evaluate_perplexity(
    *,
    model: nn.Module,
    token_ids: torch.Tensor,
    sequence_length: int = SEQUENCE_LENGTH,
) -> float:
    device = next(model.parameters()).device
    loss_function = nn.CrossEntropyLoss(reduction="sum")
    windows = token_ids.shape[1] // sequence_length

    total_nll, total_tokens = 0.0, 0
    with progress(
        stage=EVALUATE,
        total=windows,
    ) as bar:
        for index in range(windows):
            batch = token_ids[:, index * sequence_length : (index + 1) * sequence_length].to(device)
            logits = model(batch).logits.float()
            total_nll += loss_function(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                batch[:, 1:].reshape(-1),
            ).item()
            total_tokens += sequence_length - 1
            bar.set_postfix_str(f"ppl {math.exp(total_nll / total_tokens):.4f}")
            bar.update(1)
    return math.exp(total_nll / total_tokens)


def build_operand_quantizer(
    *,
    method: str,
    tile_quantizer: TileQuantizer,
    thresholds_path: Path,
    counters: Mapping[str, FormatCounter],
) -> OperandQuantizer | None:
    if method == "fp16":
        return None
    if method == "nvfp4":
        return Nvfp4OperandQuantizer()
    if method == "mxfp4":
        thresholds = ThresholdTable.uniform(pair=ALL_FP4)
    else:
        thresholds = ThresholdTable.load(path=thresholds_path)
        logger.info(f"🔖 thresholds from {thresholds_path} (derived on {thresholds.reference_model})")
        for operand in OPERANDS:
            for projection in PROJECTIONS:
                pair = thresholds.pair(
                    operand=operand,
                    projection=projection,
                )
                logger.debug(f"   {operand:<11} {projection:<9} t1={pair.lower} t2={pair.upper}")
    return MxOperandQuantizer(
        tile_quantizer=tile_quantizer,
        thresholds=thresholds,
        counters=counters,
    )


def nominal_bits(*, method: str) -> float:
    return {"fp16": 16.0, "nvfp4": 4.0 + 8 / NVFP4_BLOCK}.get(method, 0.0)


def run_evaluation(
    *,
    model_name: str,
    method: str,
    thresholds_path: Path,
    backend: str,
) -> EvaluationResult:
    model, tokenizer = load_model(
        name=model_name,
        dtype=torch.float16,
    )
    counters = {operand: FormatCounter() for operand in OPERANDS}
    tile_quantizer = TorchTileQuantizer() if method in ("fp16", "nvfp4") else build_tile_quantizer(backend=backend)
    quantizer = build_operand_quantizer(
        method=method,
        tile_quantizer=tile_quantizer,
        thresholds_path=thresholds_path,
        counters=counters,
    )

    started = time.time()
    if quantizer is not None:
        quantize_model(
            model=model,
            quantizer=quantizer,
        )
    seconds_quantize = time.time() - started
    weight_bits = counters["weights"].effective_bits() or nominal_bits(method=method)
    weight_formats = counters["weights"].shares()
    logger.info(f"💾 weight effective bits {weight_bits:.3f} {weight_formats}")

    token_ids = wikitext_token_ids(
        tokenizer=tokenizer,
        split="test",
    )
    perplexity = evaluate_perplexity(
        model=model,
        token_ids=token_ids,
    )
    return EvaluationResult(
        method=method,
        model=model_name,
        perplexity=perplexity,
        weight_bits=weight_bits,
        activation_bits=counters["activations"].effective_bits() or nominal_bits(method=method),
        weight_formats=weight_formats,
        activation_formats=counters["activations"].shares(),
        seconds_quantize=seconds_quantize,
    )
