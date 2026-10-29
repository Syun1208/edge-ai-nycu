from __future__ import annotations

import math

import torch
import torch.nn as nn

from common import FP_BITS, quant_bits_per_weight, set_storage_bits, target_linears
from gptq import PERCENT_DAMP, round_to_nearest, run_sequential

DEFAULT_RATE = 0.50
SVD_RATIO = 0.20
RANK_MULTIPLE = 128
FACTOR_BITS = 8
FACTOR_GROUP_SIZE = 128
MAX_DAMP_TRIES = 6


def svd_rank(
    *,
    out_features: int,
    in_features: int,
    svd_ratio: float = SVD_RATIO,
) -> int:
    budget = (1 - svd_ratio) * out_features * in_features / (out_features + in_features)
    return max(RANK_MULTIPLE, math.floor(budget / RANK_MULTIPLE) * RANK_MULTIPLE)


def svd_plan(
    *,
    model: nn.Module,
    rate: float,
) -> tuple[dict[str, int], set[str], float, float]:
    linears = target_linears(model=model)
    base_bits = sum(linear.weight.numel() for linear in linears.values()) * FP_BITS
    factor_cost = quant_bits_per_weight(
        wbits=FACTOR_BITS,
        group_size=FACTOR_GROUP_SIZE,
    )

    ranks = {
        name: svd_rank(
            out_features=linear.out_features,
            in_features=linear.in_features,
        )
        for name, linear in linears.items()
    }
    factor_numel = {name: ranks[name] * (linear.out_features + linear.in_features) for name, linear in linears.items()}
    svd_ratio = 1 - sum(factor_numel.values()) * FP_BITS / base_bits

    used_bits = sum(factor_numel.values()) * FP_BITS
    quantized: set[str] = set()
    for name in linears:
        if 1 - used_bits / base_bits >= rate:
            break
        quantized.add(name)
        used_bits -= factor_numel[name] * (FP_BITS - factor_cost)
    return ranks, quantized, svd_ratio, 1 - used_bits / base_bits


@torch.no_grad()
def whitened_factors(
    *,
    weight: torch.Tensor,
    hessian: torch.Tensor,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    weight = weight.float()
    hessian = hessian.clone()
    dead = torch.diag(hessian) == 0
    hessian[dead, dead] = 1
    damp = PERCENT_DAMP * torch.mean(torch.diag(hessian))
    identity = torch.eye(
        hessian.shape[0],
        device=hessian.device,
    )

    for _ in range(MAX_DAMP_TRIES):
        cholesky, info = torch.linalg.cholesky_ex(hessian + damp * identity)
        if info.item() == 0:
            break
        damp *= 10
    left_vectors, singular_values, right_vectors = torch.linalg.svd(
        weight @ cholesky,
        full_matrices=False,
    )

    left = left_vectors[:, :rank] * singular_values[:rank]
    right = torch.linalg.solve_triangular(
        cholesky,
        right_vectors[:rank],
        upper=False,
        left=False,
    )
    return left, right


def factorized_linear(
    *,
    linear: nn.Linear,
    left: torch.Tensor,
    right: torch.Tensor,
) -> nn.Sequential:
    rank = left.shape[1]
    down = nn.Linear(
        linear.in_features,
        rank,
        bias=False,
        device=linear.weight.device,
        dtype=linear.weight.dtype,
    )
    up = nn.Linear(
        rank,
        linear.out_features,
        bias=linear.bias is not None,
        device=linear.weight.device,
        dtype=linear.weight.dtype,
    )
    down.weight.data = right.to(linear.weight.dtype)
    up.weight.data = left.to(linear.weight.dtype)
    if linear.bias is not None:
        up.bias.data = linear.bias.data.clone()
    return nn.Sequential(down, up)


def replace_submodule(
    *,
    block: nn.Module,
    name: str,
    module: nn.Module,
) -> None:
    parent_name, _, child_name = name.rpartition(".")
    setattr(block.get_submodule(parent_name), child_name, module)


def compress(
    model: nn.Module,
    tokenizer,
    rate: float = DEFAULT_RATE,
    wbits: int | None = None,
) -> nn.Module:
    ranks, quantized, svd_ratio, planned_ratio = svd_plan(
        model=model,
        rate=rate,
    )
    print(
        f"[case3] SVD ratio {svd_ratio:.2%}, {len(quantized)}/{len(ranks)} layers with {FACTOR_BITS}-bit factors, "
        f"planned ratio {planned_ratio:.2%}",
        flush=True,
    )
    factor_cost = quant_bits_per_weight(
        wbits=FACTOR_BITS,
        group_size=FACTOR_GROUP_SIZE,
    )

    def factorize_step(
        *,
        block_index: int,
        block: nn.Module,
        names: list[str],
        hessians: dict[str, torch.Tensor],
    ) -> None:
        for name in names:
            full_name = f"{block_index}.{name}"
            linear = block.get_submodule(name)
            left, right = whitened_factors(
                weight=linear.weight.data,
                hessian=hessians[name],
                rank=ranks[full_name],
            )
            factorized = factorized_linear(
                linear=linear,
                left=left,
                right=right,
            )

            bits_per_weight = factor_cost if full_name in quantized else FP_BITS
            for factor in factorized:
                if full_name in quantized:
                    factor.weight.data = round_to_nearest(
                        weight=factor.weight.data,
                        wbits=FACTOR_BITS,
                        group_size=FACTOR_GROUP_SIZE,
                    )
                set_storage_bits(
                    module=factor,
                    bits=factor.weight.numel() * bits_per_weight,
                )
            replace_submodule(
                block=block,
                name=name,
                module=factorized,
            )

    run_sequential(
        model=model,
        tokenizer=tokenizer,
        handle_step=factorize_step,
        tag="case3",
    )
    return model
