from __future__ import annotations

import torch.nn as nn

from gptq import mixed_precision_plan, run_gptq

GROUP_SIZE = 128
DEFAULT_RATE = 0.755
DEFAULT_WBITS = 4


def compress(
    model: nn.Module,
    tokenizer,
    rate: float = DEFAULT_RATE,
    wbits: int = DEFAULT_WBITS,
) -> nn.Module:
    low_bits = wbits - 1
    plan, planned_ratio = mixed_precision_plan(
        model=model,
        target_ratio=rate,
        hi_bits=wbits,
        lo_bits=low_bits,
        group_size=GROUP_SIZE,
    )
    low_count = sum(bits == low_bits for bits in plan.values())
    print(
        f"[case1] {low_count} layers at {low_bits}-bit, planned ratio {planned_ratio:.2%}",
        flush=True,
    )

    run_gptq(
        model=model,
        tokenizer=tokenizer,
        bits_of=plan,
        group_size=GROUP_SIZE,
    )
    return model
