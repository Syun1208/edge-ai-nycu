from __future__ import annotations

import torch.nn as nn

from common import quant_bits_per_weight, target_linears
from gptq import run_gptq

GROUP_SIZE = 512
DEFAULT_RATE = 0.81
DEFAULT_WBITS = 3
FP_BITS = 16


def compress(
    model: nn.Module,
    tokenizer,
    rate: float = DEFAULT_RATE,
    wbits: int = DEFAULT_WBITS,
) -> nn.Module:
    planned_ratio = 1 - quant_bits_per_weight(
        wbits=wbits,
        group_size=GROUP_SIZE,
    ) / FP_BITS
    if planned_ratio < rate:
        raise ValueError(f"{wbits}-bit with group {GROUP_SIZE} gives {planned_ratio:.2%}, below {rate:.2%}")
    print(
        f"[case2] all layers at {wbits}-bit, group {GROUP_SIZE}, planned ratio {planned_ratio:.2%}",
        flush=True,
    )

    run_gptq(
        model=model,
        tokenizer=tokenizer,
        bits_of={name: wbits for name in target_linears(model=model)},
        group_size=GROUP_SIZE,
    )
    return model
