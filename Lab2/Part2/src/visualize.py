from __future__ import annotations

import argparse
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

PART2 = Path(__file__).resolve().parent.parent
LOG_DIR = PART2 / "logs"
OUTPUT = PART2 / "report" / "figures" / "tradeoff"
CASES = [1, 2, 3, 4]
PPL_LIMITS = {1: 15.0, 2: 20.0, 3: 50.0, 4: 50.0}
SURFACE = "#ffffff"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
BASELINE = "#8a8984"
FIGURE_SIZE = (6.3, 2.6)
DPI = 400


@dataclass(frozen=True)
class Family:
    label: str
    color: str
    marker: str


QUANTIZATION = Family("Quantization (GPTQ)", "#2a78d6", "o")
SVD = Family("SVD + 8-bit factors", "#eb6834", "s")
PRUNING = Family("Pruning + GPTQ", "#1baf7a", "D")
FAMILY_OF = {1: QUANTIZATION, 2: QUANTIZATION, 3: SVD, 4: PRUNING}


@dataclass(frozen=True)
class EvalResult:
    ppl: float
    ratio: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=LOG_DIR,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT,
    )
    return parser


def read_eval(path: Path) -> EvalResult:
    text = path.read_text()
    ppl = float(re.search(r"^ppl: ([\d.]+)", text, re.M).group(1))
    ratio = float(re.search(r"^comp_rate: ([\d.]+)%", text, re.M).group(1))
    return EvalResult(ppl=ppl, ratio=ratio)


def read_results(log_dir: Path) -> tuple[EvalResult, dict[int, EvalResult]]:
    base = read_eval(log_dir / "eval_base.log")
    return base, {case: read_eval(log_dir / f"eval_case{case}.log") for case in CASES}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.edgecolor": TEXT_SECONDARY,
            "axes.labelcolor": TEXT_PRIMARY,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
        }
    )


def draw_baseline(*, axis: Axes, ppl: float, label_x: float) -> None:
    axis.axhline(ppl, color=BASELINE, linestyle=(0, (4, 3)), linewidth=0.9, zorder=1)
    axis.text(
        label_x,
        ppl,
        f"fp16 {ppl:.2f}",
        color=TEXT_SECONDARY,
        fontsize=7.5,
        ha="left",
        va="bottom",
    )


def draw_tradeoff(*, axis: Axes, base: EvalResult, results: dict[int, EvalResult]) -> None:
    axis.grid(color=GRID, linewidth=0.5, zorder=0)
    draw_baseline(axis=axis, ppl=base.ppl, label_x=2)
    offsets = {1: (-9, 5), 2: (-9, 0), 3: (9, 0), 4: (-9, 0)}
    alignments = {1: "right", 2: "right", 3: "left", 4: "right"}
    for case, result in results.items():
        family = FAMILY_OF[case]
        axis.scatter(
            result.ratio,
            result.ppl,
            s=46,
            marker=family.marker,
            color=family.color,
            edgecolor=SURFACE,
            linewidth=1.2,
            zorder=3,
        )
        axis.annotate(
            f"Case {case} ({result.ratio:.2f}%, {result.ppl:.2f})",
            (result.ratio, result.ppl),
            textcoords="offset points",
            xytext=offsets[case],
            ha=alignments[case],
            va="center",
            fontsize=7.5,
            color=TEXT_PRIMARY,
            linespacing=1.1,
        )
    axis.set_xlim(0, 92)
    axis.set_ylim(12, 25)
    axis.set_xlabel("Compression Ratio (%)")
    axis.set_ylabel("WikiText-2 PPL")
    axis.set_title("(a) PPL vs. Compression Ratio", loc="left")


def draw_margins(*, axis: Axes, results: dict[int, EvalResult]) -> None:
    axis.grid(axis="y", color=GRID, linewidth=0.5, zorder=0)
    positions = list(range(len(results)))
    for position, (case, result) in zip(positions, results.items()):
        share = result.ppl / PPL_LIMITS[case] * 100
        axis.bar(
            position,
            share,
            width=0.58,
            color=FAMILY_OF[case].color,
            edgecolor=SURFACE,
            linewidth=1,
            zorder=2,
        )
    axis.axhline(100, color=TEXT_PRIMARY, linestyle=(0, (4, 3)), linewidth=0.9, zorder=1)
    axis.text(
        len(results) - 0.45,
        101.5,
        "limit",
        ha="right",
        va="bottom",
        fontsize=7.5,
        color=TEXT_SECONDARY,
    )
    axis.set_xticks(
        positions,
        [f"Case {case}\n{result.ppl:.2f} / {PPL_LIMITS[case]:.0f}" for case, result in results.items()],
    )
    axis.tick_params(axis="x", length=0)
    axis.set_xlim(-0.55, len(results) - 0.45)
    axis.set_ylim(0, 112)
    axis.set_ylabel("PPL / PPL limit (%)")
    axis.set_title("(b) PPL relative to the case limit", loc="left")


def family_legend() -> list[Line2D]:
    return [
        Line2D(
            [],
            [],
            marker=family.marker,
            linestyle="none",
            markersize=6,
            markerfacecolor=family.color,
            markeredgecolor=SURFACE,
            label=family.label,
        )
        for family in (QUANTIZATION, SVD, PRUNING)
    ]


def plot_results(*, base: EvalResult, results: dict[int, EvalResult], output: Path) -> list[Path]:
    apply_style()
    figure, (left, right) = plt.subplots(
        1,
        2,
        figsize=FIGURE_SIZE,
        gridspec_kw={"width_ratios": [1.25, 1], "wspace": 0.3},
    )
    draw_tradeoff(axis=left, base=base, results=results)
    draw_margins(axis=right, results=results)
    figure.legend(
        handles=family_legend(),
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
        handletextpad=0.4,
        columnspacing=1.4,
    )
    figure.subplots_adjust(left=0.08, right=0.97, bottom=0.2, top=0.8)
    output.parent.mkdir(parents=True, exist_ok=True)
    paths = [output.with_suffix(".png"), output.with_suffix(".pdf")]
    for path in paths:
        figure.savefig(path, dpi=DPI)
    plt.close(figure)
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    base, results = read_results(arguments.log_dir)
    for path in plot_results(base=base, results=results, output=arguments.output):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
