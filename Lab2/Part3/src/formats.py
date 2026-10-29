from __future__ import annotations

import math
from dataclasses import dataclass

import torch


TILE_ROWS = 16
TILE_COLS = 32
TILE_SIZE = TILE_ROWS * TILE_COLS
NVFP4_BLOCK = 16
SCALE_BITS = 8


@dataclass(frozen=True, slots=True)
class ElementFormat:
    name: str
    payload_bits: int
    mantissa_bits: int
    min_exponent: int
    max_exponent: int
    max_value: float

    @property
    def effective_bits(self) -> float:
        return self.payload_bits + SCALE_BITS / TILE_COLS


FP4 = ElementFormat(
    name="MXFP4",
    payload_bits=4,
    mantissa_bits=1,
    min_exponent=0,
    max_exponent=2,
    max_value=6.0,
)


FP6 = ElementFormat(
    name="MXFP6",
    payload_bits=6,
    mantissa_bits=3,
    min_exponent=0,
    max_exponent=2,
    max_value=7.5,
)


FP8 = ElementFormat(
    name="MXFP8",
    payload_bits=8,
    mantissa_bits=3,
    min_exponent=-6,
    max_exponent=8,
    max_value=448.0,
)


FORMATS = (FP4, FP6, FP8)
CAP = math.inf


def floor_log2(values: torch.Tensor) -> torch.Tensor:
    return torch.frexp(values).exponent - 1


def quantize_element(
    *,
    values: torch.Tensor,
    element: ElementFormat,
) -> torch.Tensor:
    magnitude = values.abs()
    exponent = floor_log2(magnitude).clamp(min=element.min_exponent)
    step = torch.ldexp(
        torch.ones_like(magnitude),
        exponent - element.mantissa_bits,
    )
    rounded = torch.minimum(
        torch.round(magnitude / step) * step,
        torch.tensor(element.max_value, device=values.device),
    )
    return torch.copysign(
        rounded,
        values,
    )


def mx_quantize_blocks(
    *,
    blocks: torch.Tensor,
    block_max: torch.Tensor,
    element: ElementFormat,
) -> torch.Tensor:
    exponent = (floor_log2(block_max) - element.max_exponent).clamp(
        min=-127,
        max=127,
    )
    codes = quantize_element(
        values=torch.ldexp(blocks, -exponent),
        element=element,
    )
    return torch.where(
        block_max > 0,
        torch.ldexp(codes, exponent),
        torch.zeros_like(blocks),
    )


def nvfp4_fake_quantize(*, matrix: torch.Tensor) -> torch.Tensor:
    rows, cols = matrix.shape
    values = matrix.float()
    global_max = values.abs().amax()
    if float(global_max) == 0.0:
        return torch.zeros_like(matrix)
    global_scale = global_max / (FP8.max_value * FP4.max_value)
    blocks = values.reshape(rows, cols // NVFP4_BLOCK, NVFP4_BLOCK)
    block_scale = quantize_element(
        values=blocks.abs().amax(
            dim=-1,
            keepdim=True,
        )
        / FP4.max_value
        / global_scale,
        element=FP8,
    )
    scale = block_scale * global_scale
    codes = quantize_element(
        values=blocks / scale.clamp(min=torch.finfo(torch.float32).tiny),
        element=FP4,
    )
    return torch.where(
        scale > 0,
        codes * scale,
        torch.zeros_like(blocks),
    ).reshape(rows, cols).to(matrix.dtype)
