from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn as nn

from common import get_calib_data, quant_bits_per_weight, set_storage_bits, target_linears

SEQUENTIAL = [
    ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
    ["self_attn.o_proj"],
    ["mlp.gate_proj", "mlp.up_proj"],
    ["mlp.down_proj"],
]
CLIP_RATIOS = (1.0, 0.95, 0.9, 0.85, 0.8)
MIN_SCALE = 1e-8
PERCENT_DAMP = 0.01
BLOCK_SIZE = 128
FP_BITS = 16
LOW_LINEAR_MODULES = ("mlp.up_proj", "mlp.gate_proj")


def group_params(
    *,
    weight: torch.Tensor,
    wbits: int,
    clip_ratios: tuple[float, ...] = CLIP_RATIOS,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_level = 2**wbits - 1
    weight_min = weight.min(1, keepdim=True)[0].clamp(max=0)
    weight_max = weight.max(1, keepdim=True)[0].clamp(min=0)

    best_error = torch.full_like(weight_min, float("inf"))
    best_scale = torch.ones_like(weight_min)
    best_zero = torch.zeros_like(weight_min)
    for ratio in clip_ratios:
        scale = ((weight_max - weight_min) * ratio / max_level).clamp(min=MIN_SCALE)
        zero = torch.round(-weight_min * ratio / scale)
        reconstructed = fake_quant(
            weight=weight,
            scale=scale,
            zero=zero,
            max_level=max_level,
        )
        error = ((reconstructed - weight) ** 2).sum(1, keepdim=True)
        better = error < best_error
        best_error = torch.where(better, error, best_error)
        best_scale = torch.where(better, scale, best_scale)
        best_zero = torch.where(better, zero, best_zero)
    return best_scale, best_zero


def fake_quant(
    *,
    weight: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    max_level: int,
) -> torch.Tensor:
    levels = torch.clamp(
        torch.round(weight / scale) + zero,
        min=0,
        max=max_level,
    )
    return scale * (levels - zero)


@torch.no_grad()
def gptq_quantize(
    *,
    weight: torch.Tensor,
    hessian: torch.Tensor,
    wbits: int,
    group_size: int,
    blocksize: int = BLOCK_SIZE,
    percdamp: float = PERCENT_DAMP,
    act_order: bool = True,
) -> torch.Tensor:
    weight = weight.float().clone()
    columns = weight.shape[1]
    max_level = 2**wbits - 1

    dead = torch.diag(hessian) == 0
    hessian[dead, dead] = 1
    weight[:, dead] = 0

    scales, zeros = [], []
    for group in range(columns // group_size):
        scale, zero = group_params(
            weight=weight[:, group * group_size : (group + 1) * group_size],
            wbits=wbits,
        )
        scales.append(scale)
        zeros.append(zero)

    if act_order:
        permutation = torch.argsort(
            torch.diag(hessian),
            descending=True,
        )
    else:
        permutation = torch.arange(
            columns,
            device=weight.device,
        )
    weight = weight[:, permutation]
    hessian = hessian[permutation][:, permutation]

    hessian += percdamp * torch.mean(torch.diag(hessian)) * torch.eye(
        columns,
        device=weight.device,
    )
    hessian_inverse = torch.linalg.cholesky(
        torch.cholesky_inverse(torch.linalg.cholesky(hessian)),
        upper=True,
    )

    quantized = torch.zeros_like(weight)
    for block_start in range(0, columns, blocksize):
        block_end = min(block_start + blocksize, columns)
        block_weight = weight[:, block_start:block_end].clone()
        block_inverse = hessian_inverse[block_start:block_end, block_start:block_end]
        block_quantized = torch.zeros_like(block_weight)
        block_error = torch.zeros_like(block_weight)

        for column in range(block_end - block_start):
            column_weight = block_weight[:, column]
            diagonal = block_inverse[column, column]
            group = permutation[block_start + column].item() // group_size
            column_quantized = fake_quant(
                weight=column_weight.unsqueeze(1),
                scale=scales[group],
                zero=zeros[group],
                max_level=max_level,
            ).flatten()

            block_quantized[:, column] = column_quantized
            error = (column_weight - column_quantized) / diagonal
            block_weight[:, column:] -= error.unsqueeze(1) @ block_inverse[column, column:].unsqueeze(0)
            block_error[:, column] = error

        quantized[:, block_start:block_end] = block_quantized
        weight[:, block_end:] -= block_error @ hessian_inverse[block_start:block_end, block_end:]

    return quantized[:, torch.argsort(permutation)]


def block_out(output: torch.Tensor | tuple[torch.Tensor, ...]) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


class HessianHook:
    def __init__(self, *, layer: nn.Linear) -> None:
        in_features = layer.weight.shape[1]
        self.hessian = torch.zeros(
            in_features,
            in_features,
            device=layer.weight.device,
            dtype=torch.float32,
        )
        self.samples = 0

    def __call__(
        self,
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        activations = inputs[0].reshape(-1, inputs[0].shape[-1]).float()
        rows = activations.shape[0]
        self.hessian *= self.samples / (self.samples + rows)
        self.samples += rows
        activations = math.sqrt(2 / self.samples) * activations
        self.hessian += activations.t() @ activations


class Catcher(nn.Module):
    def __init__(
        self,
        *,
        module: nn.Module,
        inputs: list[torch.Tensor],
        cache: dict[str, object],
    ) -> None:
        super().__init__()
        self.module = module
        self.inputs = inputs
        self.cache = cache
        self.attention_type = getattr(module, "attention_type", None)

    def forward(self, *args: torch.Tensor, **kwargs: object) -> None:
        self.inputs.append(args[0] if args else kwargs.pop("hidden_states"))
        kwargs.pop("hidden_states", None)
        self.cache.update(kwargs)
        raise ValueError


@torch.no_grad()
def capture_block_inputs(
    *,
    model: nn.Module,
    calibration: torch.Tensor,
) -> tuple[list[torch.Tensor], dict[str, object]]:
    device = next(model.parameters()).device
    layers = model.model.layers
    inputs: list[torch.Tensor] = []
    cache: dict[str, object] = {}

    layers[0] = Catcher(
        module=layers[0],
        inputs=inputs,
        cache=cache,
    )
    for index in range(calibration.shape[0]):
        try:
            model(calibration[index : index + 1].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module
    return inputs, cache


@torch.no_grad()
def round_to_nearest(
    *,
    weight: torch.Tensor,
    wbits: int,
    group_size: int,
) -> torch.Tensor:
    rows, columns = weight.shape
    groups = weight.float().reshape(-1, group_size)
    scale, zero = group_params(
        weight=groups,
        wbits=wbits,
    )
    return fake_quant(
        weight=groups,
        scale=scale,
        zero=zero,
        max_level=2**wbits - 1,
    ).reshape(rows, columns).to(weight.dtype)


@torch.no_grad()
def quantize_linear(
    *,
    linear: nn.Linear,
    hessian: torch.Tensor,
    wbits: int,
    group_size: int,
    act_order: bool = True,
) -> None:
    linear.weight.data = gptq_quantize(
        weight=linear.weight.data,
        hessian=hessian,
        wbits=wbits,
        group_size=group_size,
        act_order=act_order,
    ).to(linear.weight.dtype)
    set_storage_bits(
        module=linear,
        bits=linear.weight.numel()
        * quant_bits_per_weight(
            wbits=wbits,
            group_size=group_size,
        ),
    )


@torch.no_grad()
def run_sequential(
    *,
    model: nn.Module,
    tokenizer,
    handle_step: Callable[..., None],
    tag: str,
    nsamples: int = 128,
    seqlen: int = 2048,
) -> None:
    calibration = get_calib_data(
        tokenizer=tokenizer,
        nsamples=nsamples,
        seqlen=seqlen,
    )
    inputs, cache = capture_block_inputs(
        model=model,
        calibration=calibration,
    )

    for block_index, block in enumerate(model.model.layers):
        for names in SEQUENTIAL:
            submodules = dict(block.named_modules())
            hooks = {name: HessianHook(layer=submodules[name]) for name in names}
            handles = [submodules[name].register_forward_hook(hooks[name]) for name in names]
            for hidden_states in inputs:
                block(hidden_states, **cache)
            for handle in handles:
                handle.remove()

            handle_step(
                block_index=block_index,
                block=block,
                names=names,
                hessians={name: hook.hessian for name, hook in hooks.items()},
            )
            del hooks

        inputs = [block_out(block(hidden_states, **cache)) for hidden_states in inputs]
        print(f"[{tag}] block {block_index} done", flush=True)
        torch.cuda.empty_cache()


def run_gptq(
    *,
    model: nn.Module,
    tokenizer,
    bits_of: dict[str, int | None],
    group_size: int = 128,
    nsamples: int = 128,
    seqlen: int = 2048,
    act_order: bool = True,
) -> None:
    def quantize_step(
        *,
        block_index: int,
        block: nn.Module,
        names: list[str],
        hessians: dict[str, torch.Tensor],
    ) -> None:
        submodules = dict(block.named_modules())
        for name in names:
            wbits = bits_of.get(f"{block_index}.{name}")
            if wbits is None:
                continue
            quantize_linear(
                linear=submodules[name],
                hessian=hessians[name],
                wbits=wbits,
                group_size=group_size,
                act_order=act_order,
            )

    run_sequential(
        model=model,
        tokenizer=tokenizer,
        handle_step=quantize_step,
        tag="gptq",
        nsamples=nsamples,
        seqlen=seqlen,
    )


def mixed_precision_plan(
    *,
    model: nn.Module,
    target_ratio: float,
    hi_bits: int,
    lo_bits: int,
    group_size: int,
    lo_modules: tuple[str, ...] = LOW_LINEAR_MODULES,
) -> tuple[dict[str, int], float]:
    linears = target_linears(model=model)
    base_bits = sum(module.weight.numel() for module in linears.values()) * FP_BITS
    high_cost = quant_bits_per_weight(
        wbits=hi_bits,
        group_size=group_size,
    )
    low_cost = quant_bits_per_weight(
        wbits=lo_bits,
        group_size=group_size,
    )
    plan = {name: hi_bits for name in linears}
    used_bits = sum(module.weight.numel() * high_cost for module in linears.values())

    block_count = len(model.model.layers)
    middle = (block_count - 1) / 2
    order = sorted(range(block_count), key=lambda block: abs(block - middle))
    for block in order:
        if 1 - used_bits / base_bits > target_ratio:
            break
        for module_name in lo_modules:
            name = f"{block}.{module_name}"
            plan[name] = lo_bits
            used_bits -= linears[name].weight.numel() * (high_cost - low_cost)
    return plan, 1 - used_bits / base_bits
