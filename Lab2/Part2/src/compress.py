from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn as nn

SRC = Path(__file__).resolve().parent
CASES = [1, 2, 3, 4]

sys.path.insert(0, str(SRC))

import common
from evaluate import DEFAULT_MODEL, load_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        type=int,
        required=True,
        choices=CASES,
    )
    parser.add_argument(
        "--rate",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--wbits",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16"],
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    return parser


def compress_case(
    *,
    case: int,
    rate: float,
    wbits: int | None,
    model_name: str,
    dtype: str,
    seed: int,
) -> nn.Module:
    torch.manual_seed(seed)
    model, tokenizer = load_model(model_name, dtype)
    common.record_base(model)
    options = {"rate": rate} if wbits is None else {"rate": rate, "wbits": wbits}
    importlib.import_module(f"case{case}").compress(
        model=model,
        tokenizer=tokenizer,
        **options,
    )
    return model


def save_checkpoint(
    *,
    model: nn.Module,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, path)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    model = compress_case(
        case=arguments.case,
        rate=arguments.rate,
        wbits=arguments.wbits,
        model_name=arguments.model,
        dtype=arguments.dtype,
        seed=arguments.seed,
    )
    save_checkpoint(
        model=model,
        path=arguments.output,
    )
    print(f"[case{arguments.case}] saved to {arguments.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
