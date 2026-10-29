from __future__ import annotations

import math

import torch
import torch.nn as nn

from common import FP_BITS, quant_bits_per_weight, set_storage_bits, target_linears
from gptq import PERCENT_DAMP, quantize_linear, run_sequential

DEFAULT_RATE = 0.85
DEFAULT_WBITS = 3
GROUP_SIZE = 128
CHANNEL_MULTIPLE = 128
FFN_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def pruned_channels_per_block(
    *,
    model: nn.Module,
    rate: float,
    wbits: int,
) -> tuple[int, float, float]:
    linears = target_linears(model=model)
    base_numel = sum(linear.weight.numel() for linear in linears.values())
    cost = quant_bits_per_weight(
        wbits=wbits,
        group_size=GROUP_SIZE,
    )
    keep_fraction = (1 - rate) * FP_BITS / cost

    mlp = model.model.layers[0].mlp
    hidden_size = mlp.down_proj.out_features
    block_count = len(model.model.layers)
    weights_per_channel = len(FFN_PROJECTIONS) * hidden_size
    removed_numel = (1 - keep_fraction) * base_numel
    channels = math.ceil(removed_numel / (block_count * weights_per_channel) / CHANNEL_MULTIPLE) * CHANNEL_MULTIPLE

    pruning_rate = block_count * channels * weights_per_channel / base_numel
    planned_ratio = 1 - (1 - pruning_rate) * cost / FP_BITS
    return channels, pruning_rate, planned_ratio


@torch.no_grad()
def select_channels(
    *,
    down_weight: torch.Tensor,
    hessian: torch.Tensor,
    pruned: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    importance = (down_weight.float() ** 2).sum(0) * torch.diag(hessian)
    order = torch.argsort(importance)
    removed, kept = order[:pruned], order[pruned:]
    return torch.sort(kept).values, torch.sort(removed).values


@torch.no_grad()
def compensated_down_weight(
    *,
    down_weight: torch.Tensor,
    hessian: torch.Tensor,
    kept: torch.Tensor,
    removed: torch.Tensor,
) -> torch.Tensor:
    weight = down_weight.float()
    kept_hessian = hessian[kept][:, kept]
    damp = PERCENT_DAMP * torch.mean(torch.diag(kept_hessian))
    kept_hessian = kept_hessian + damp * torch.eye(
        kept.numel(),
        device=hessian.device,
    )
    cross = weight[:, removed] @ hessian[removed][:, kept]
    correction = torch.linalg.solve(kept_hessian, cross.t()).t()
    return weight[:, kept] + correction


def sliced_linear(
    *,
    linear: nn.Linear,
    weight: torch.Tensor,
) -> nn.Linear:
    out_features, in_features = weight.shape
    sliced = nn.Linear(
        in_features,
        out_features,
        bias=False,
        device=linear.weight.device,
        dtype=linear.weight.dtype,
    )
    sliced.weight.data = weight.to(linear.weight.dtype).contiguous()
    return sliced


def compress(
    model: nn.Module,
    tokenizer,
    rate: float = DEFAULT_RATE,
    wbits: int = DEFAULT_WBITS,
) -> nn.Module:
    pruned, pruning_rate, planned_ratio = pruned_channels_per_block(
        model=model,
        rate=rate,
        wbits=wbits,
    )
    intermediate = model.model.layers[0].mlp.down_proj.in_features
    print(
        f"[case4] prune {pruned}/{intermediate} FFN channels per block, pruning rate {pruning_rate:.2%}, "
        f"{wbits}-bit group {GROUP_SIZE}, planned ratio {planned_ratio:.2%}",
        flush=True,
    )
    cost = quant_bits_per_weight(
        wbits=wbits,
        group_size=GROUP_SIZE,
    )

    def prune_and_quantize_step(
        *,
        block_index: int,
        block: nn.Module,
        names: list[str],
        hessians: dict[str, torch.Tensor],
    ) -> None:
        if names != ["mlp.down_proj"]:
            for name in names:
                quantize_linear(
                    linear=block.get_submodule(name),
                    hessian=hessians[name],
                    wbits=wbits,
                    group_size=GROUP_SIZE,
                )
            return

        mlp = block.mlp
        hessian = hessians["mlp.down_proj"]
        kept, removed = select_channels(
            down_weight=mlp.down_proj.weight.data,
            hessian=hessian,
            pruned=pruned,
        )
        down_weight = compensated_down_weight(
            down_weight=mlp.down_proj.weight.data,
            hessian=hessian,
            kept=kept,
            removed=removed,
        )

        for projection in ("gate_proj", "up_proj"):
            linear = getattr(mlp, projection)
            sliced = sliced_linear(
                linear=linear,
                weight=linear.weight.data[kept],
            )
            set_storage_bits(
                module=sliced,
                bits=sliced.weight.numel() * cost,
            )
            setattr(mlp, projection, sliced)

        mlp.down_proj = sliced_linear(
            linear=mlp.down_proj,
            weight=down_weight,
        )
        quantize_linear(
            linear=mlp.down_proj,
            hessian=hessian[kept][:, kept].clone(),
            wbits=wbits,
            group_size=GROUP_SIZE,
        )

    run_sequential(
        model=model,
        tokenizer=tokenizer,
        handle_step=prune_and_quantize_step,
        tag="case4",
    )
    return model
