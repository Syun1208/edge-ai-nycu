from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn as nn

from src.formats import FORMATS, nvfp4_fake_quantize
from src.progress import QUANTIZE, progress
from src.thresholds import PROJECTIONS, ThresholdTable
from src.tiles import TileQuantizer, pad_rows

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class FormatCounter:
    counts: torch.Tensor | None = None

    def add(self, *, formats: torch.Tensor) -> None:
        batch = torch.bincount(
            formats.flatten().long(),
            minlength=len(FORMATS),
        )
        self.counts = batch if self.counts is None else self.counts + batch

    def effective_bits(self) -> float:
        if self.counts is None:
            return 0.0
        counts = self.counts.double().cpu()
        bits = torch.tensor([element.effective_bits for element in FORMATS], dtype=torch.float64)
        return float((counts * bits).sum() / counts.sum())

    def shares(self) -> dict[str, float]:
        if self.counts is None:
            return {}
        counts = self.counts.double().cpu()
        return {element.name: float(count / counts.sum()) for element, count in zip(FORMATS, counts)}


class OperandQuantizer(Protocol):
    def quantize(
        self,
        *,
        matrix: torch.Tensor,
        operand: str,
        projection: str,
    ) -> torch.Tensor: ...


class MxOperandQuantizer:
    def __init__(
        self,
        *,
        tile_quantizer: TileQuantizer,
        thresholds: ThresholdTable,
        counters: Mapping[str, FormatCounter],
    ) -> None:
        self._tile_quantizer = tile_quantizer
        self._thresholds = thresholds
        self._counters = counters

    def quantize(
        self,
        *,
        matrix: torch.Tensor,
        operand: str,
        projection: str,
    ) -> torch.Tensor:
        padded, rows = pad_rows(matrix=matrix)
        quantized, formats = self._tile_quantizer.quantize(
            matrix=padded.contiguous(),
            pair=self._thresholds.pair(
                operand=operand,
                projection=projection,
            ),
        )
        self._counters[operand].add(formats=formats)
        return quantized[:rows]


class Nvfp4OperandQuantizer:
    def quantize(
        self,
        *,
        matrix: torch.Tensor,
        operand: str,
        projection: str,
    ) -> torch.Tensor:
        return nvfp4_fake_quantize(matrix=matrix)


class ActivationHook:
    def __init__(
        self,
        *,
        quantizer: OperandQuantizer,
        projection: str,
    ) -> None:
        self._quantizer = quantizer
        self._projection = projection

    def __call__(
        self,
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        hidden = inputs[0]
        flat = hidden.reshape(-1, hidden.shape[-1])
        quantized = self._quantizer.quantize(
            matrix=flat,
            operand="activations",
            projection=self._projection,
        )
        return (quantized.reshape(hidden.shape), *inputs[1:])


def projection_of(*, name: str) -> str:
    return name.rsplit(".", 1)[-1]


def target_linears(*, model: nn.Module) -> list[tuple[str, nn.Linear]]:
    return [
        (name, module)
        for name, module in model.model.layers.named_modules()
        if isinstance(module, nn.Linear) and projection_of(name=name) in PROJECTIONS
    ]


@torch.no_grad()
def quantize_model(
    *,
    model: nn.Module,
    quantizer: OperandQuantizer,
) -> list[torch.utils.hooks.RemovableHandle]:
    linears = target_linears(model=model)
    handles = []
    with progress(
        stage=QUANTIZE,
        total=len(linears),
    ) as bar:
        for name, linear in linears:
            projection = projection_of(name=name)
            linear.weight.data = quantizer.quantize(
                matrix=linear.weight.data,
                operand="weights",
                projection=projection,
            )
            handles.append(
                linear.register_forward_pre_hook(
                    ActivationHook(
                        quantizer=quantizer,
                        projection=projection,
                    )
                )
            )
            bar.update(1)
    logger.info(f"🧩 quantized {len(linears)} Linear layers (weights offline, activations on every forward)")
    return handles
