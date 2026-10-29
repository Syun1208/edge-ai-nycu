from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable, Sequence
from pathlib import Path

import torch

from src.derive import run_derivation
from src.evaluate import METHODS, run_evaluation
from src.progress import setup_logging


ROOT = Path(__file__).resolve().parent
DEFAULT_TARGET_MODEL = "Qwen/Qwen3-4B"
DEFAULT_REFERENCE_MODEL = "meta-llama/Llama-3.1-8B"
DEFAULT_THRESHOLDS = ROOT / "thresholds.json"

logger = logging.getLogger(__name__)


def run_evaluate_command(arguments: argparse.Namespace) -> int:
    result = run_evaluation(
        model_name=arguments.model,
        method=arguments.method,
        thresholds_path=arguments.thresholds,
        backend=arguments.backend,
    )
    print(f"method: {result.method}")
    print(f"model: {result.model}")
    print(f"weight_bits: {result.weight_bits:.3f}")
    print(f"activation_bits: {result.activation_bits:.3f}")
    print(f"ppl: {result.perplexity:.4f}")
    if arguments.output:
        arguments.output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        arguments.output.write_text(
            json.dumps(
                {
                    "method": result.method,
                    "model": result.model,
                    "ppl": result.perplexity,
                    "weight_bits": result.weight_bits,
                    "activation_bits": result.activation_bits,
                    "weight_formats": dict(result.weight_formats),
                    "activation_formats": dict(result.activation_formats),
                    "seconds_quantize": result.seconds_quantize,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return 0


def run_derive_command(arguments: argparse.Namespace) -> int:
    run_derivation(
        reference_model=arguments.reference_model,
        output=arguments.output,
        backend=arguments.backend,
        target_bits=arguments.target_bits,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AdaptQuant: per-tile MX format selection (fake quantization).")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="show debug logs",
    )
    subcommands = parser.add_subparsers(
        dest="command",
        required=True,
    )

    evaluate = subcommands.add_parser(
        "evaluate",
        help="quantize the target model and report WikiText-2 PPL",
    )
    evaluate.add_argument(
        "--model",
        default=DEFAULT_TARGET_MODEL,
        help="target model (never used to derive thresholds)",
    )
    evaluate.add_argument(
        "--method",
        default="adaptquant",
        choices=METHODS,
        help="fp16 baseline, uniform MXFP4, uniform NVFP4, or AdaptQuant",
    )
    evaluate.add_argument(
        "--thresholds",
        type=Path,
        default=DEFAULT_THRESHOLDS,
        help="threshold table produced by the derive command",
    )
    evaluate.add_argument(
        "--backend",
        default="cuda",
        choices=["cuda", "torch"],
        help="tile quantizer: CUDA kernel (falls back to PyTorch) or PyTorch only",
    )
    evaluate.add_argument(
        "--output",
        type=Path,
        default=None,
        help="optional JSON file for the result",
    )
    evaluate.set_defaults(handler=run_evaluate_command)

    derive = subcommands.add_parser(
        "derive",
        help="derive the per-projection thresholds once on a reference model",
    )
    derive.add_argument(
        "--reference-model",
        default=DEFAULT_REFERENCE_MODEL,
        help="reference model used once, offline",
    )
    derive.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_THRESHOLDS,
        help="where to write the threshold table",
    )
    derive.add_argument(
        "--backend",
        default="cuda",
        choices=["cuda", "torch"],
        help="tile quantizer used while measuring distortion",
    )
    derive.add_argument(
        "--target-bits",
        type=float,
        default=None,
        help="optional mean effective bits b* of both operands (Algorithm 1, lines 15 and 16)",
    )
    derive.set_defaults(handler=run_derive_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    setup_logging(verbose=arguments.verbose)
    if not torch.cuda.is_available():
        logger.error("❌ a CUDA GPU is required")
        return 1
    logger.info(f"🚀 {arguments.command} on {torch.cuda.get_device_name()}")
    handler: Callable[[argparse.Namespace], int] = arguments.handler
    return handler(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
