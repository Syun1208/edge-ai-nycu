from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import torch

from src.formats import CAP, FORMATS, FP4, FP6, FP8, TILE_COLS, TILE_ROWS, TILE_SIZE, mx_quantize_blocks


@dataclass(frozen=True, slots=True)
class ThresholdPair:
    lower: float
    upper: float

    def to_json(self) -> list[float | None]:
        return [None if math.isinf(self.lower) else self.lower, None if math.isinf(self.upper) else self.upper]

    @classmethod
    def from_json(cls, values: Sequence[float | None]) -> ThresholdPair:
        lower, upper = (CAP if value is None else float(value) for value in values)
        return cls(
            lower=lower,
            upper=upper,
        )


ALL_FP4 = ThresholdPair(
    lower=CAP,
    upper=CAP,
)


ALL_FP6 = ThresholdPair(
    lower=-CAP,
    upper=CAP,
)


ALL_FP8 = ThresholdPair(
    lower=-CAP,
    upper=-CAP,
)


FORCED = {FP4.name: ALL_FP4, FP6.name: ALL_FP6, FP8.name: ALL_FP8}


class TileQuantizer(Protocol):
    def quantize(
        self,
        *,
        matrix: torch.Tensor,
        pair: ThresholdPair,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


def to_tiles(*, matrix: torch.Tensor) -> torch.Tensor:
    rows, cols = matrix.shape
    return matrix.float().reshape(rows // TILE_ROWS, TILE_ROWS, cols // TILE_COLS, TILE_COLS).permute(0, 2, 1, 3)


def from_tiles(
    *,
    tiles: torch.Tensor,
    rows: int,
    cols: int,
) -> torch.Tensor:
    return tiles.permute(0, 2, 1, 3).reshape(rows, cols)


def tile_statistic(*, tiles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    tile_max = tiles.abs().amax(dim=(-1, -2))
    tile_rms = torch.sqrt(tiles.square().sum(dim=(-1, -2)) / TILE_SIZE)
    statistic = torch.log2(tile_max / tile_rms.clamp(min=torch.finfo(torch.float32).tiny))
    return statistic, tile_max


def select_formats(
    *,
    statistic: torch.Tensor,
    tile_max: torch.Tensor,
    pair: ThresholdPair,
) -> torch.Tensor:
    formats = torch.where(
        statistic < pair.lower,
        0,
        torch.where(
            statistic < pair.upper,
            1,
            2,
        ),
    )
    return torch.where(
        tile_max > 0,
        formats,
        0,
    ).to(torch.int8)


class TorchTileQuantizer:
    def quantize(
        self,
        *,
        matrix: torch.Tensor,
        pair: ThresholdPair,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rows, cols = matrix.shape
        tiles = to_tiles(matrix=matrix)
        statistic, tile_max = tile_statistic(tiles=tiles)
        formats = select_formats(
            statistic=statistic,
            tile_max=tile_max,
            pair=pair,
        )
        block_max = tiles.abs().amax(
            dim=-1,
            keepdim=True,
        )

        quantized = torch.zeros_like(tiles)
        for index, element in enumerate(FORMATS):
            selected = formats == index
            if not bool(selected.any()):
                continue
            candidate = mx_quantize_blocks(
                blocks=tiles,
                block_max=block_max,
                element=element,
            )
            quantized = torch.where(
                selected[..., None, None],
                candidate,
                quantized,
            )
        return (
            from_tiles(
                tiles=quantized,
                rows=rows,
                cols=cols,
            ).to(matrix.dtype),
            formats,
        )


def pad_rows(*, matrix: torch.Tensor) -> tuple[torch.Tensor, int]:
    rows = matrix.shape[0]
    padding = (-rows) % TILE_ROWS
    if padding == 0:
        return matrix, rows
    return torch.nn.functional.pad(matrix, (0, 0, 0, padding)), rows
