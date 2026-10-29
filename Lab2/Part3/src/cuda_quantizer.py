from __future__ import annotations

import logging
import time
from pathlib import Path

import torch

from src.tiles import ThresholdPair, TileQuantizer, TorchTileQuantizer

CUDA_KERNEL = Path(__file__).resolve().with_name("mx_tile_fake_quant.cu")


EXTENSION_NAME = "mx_tile_fake_quant"
SELF_TEST_SHAPE = (256, 1024)
SELF_TEST_TOLERANCE = 1e-3

logger = logging.getLogger(__name__)


class CudaTileQuantizer:
    def __init__(self, *, extension: object) -> None:
        self._extension = extension

    def quantize(
        self,
        *,
        matrix: torch.Tensor,
        pair: ThresholdPair,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output, formats = self._extension.mx_tile_fake_quant(
            matrix,
            pair.lower,
            pair.upper,
        )
        return output, formats


def compile_cuda_extension() -> object:
    from torch.utils.cpp_extension import load

    return load(
        name=EXTENSION_NAME,
        sources=[str(CUDA_KERNEL)],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )


def agreement(
    *,
    candidate: TileQuantizer,
    reference: TileQuantizer,
) -> float:
    generator = torch.Generator(device="cuda").manual_seed(0)
    matrix = (
        torch.randn(
            SELF_TEST_SHAPE,
            generator=generator,
            device="cuda",
        )
        * torch.rand(
            SELF_TEST_SHAPE,
            generator=generator,
            device="cuda",
        ).pow(4)
        * 8
    ).half()
    pair = ThresholdPair(
        lower=2.0,
        upper=3.0,
    )
    expected, expected_formats = reference.quantize(
        matrix=matrix,
        pair=pair,
    )
    actual, actual_formats = candidate.quantize(
        matrix=matrix,
        pair=pair,
    )
    value_mismatch = (expected != actual).float().mean().item()
    format_mismatch = (expected_formats != actual_formats).float().mean().item()
    return max(value_mismatch, format_mismatch)


def build_tile_quantizer(*, backend: str) -> TileQuantizer:
    reference = TorchTileQuantizer()
    if backend == "torch":
        logger.info("🧮 tile quantizer: PyTorch")
        return reference
    try:
        started = time.time()
        candidate = CudaTileQuantizer(extension=compile_cuda_extension())
        mismatch = agreement(
            candidate=candidate,
            reference=reference,
        )
    except Exception:
        logger.warning("🚧 CUDA extension unavailable, falling back to the PyTorch quantizer", exc_info=True)
        return reference
    if mismatch > SELF_TEST_TOLERANCE:
        logger.warning(f"🚧 CUDA kernel disagrees with PyTorch on {mismatch:.2%} of values, using PyTorch")
        return reference
    logger.info(f"🧮 tile quantizer: CUDA kernel (built in {time.time() - started:.0f}s, mismatch {mismatch:.4%})")
    return candidate
