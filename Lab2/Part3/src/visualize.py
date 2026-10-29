from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

PART3 = Path(__file__).resolve().parent.parent
RESULTS = PART3 / "report" / "results"
THRESHOLDS = PART3 / "thresholds.json"
OUTPUT = PART3 / "report" / "figures" / "reproduction"
METHODS = ("fp16", "mxfp4", "nvfp4", "adaptquant")
METHOD_LABELS = {"fp16": "FP16 / BF16", "mxfp4": "MXFP4", "nvfp4": "NVFP4", "adaptquant": "AdaptQuant"}
PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
SURFACE = "#ffffff"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
FIGURE_SIZE = (6.3, 2.5)
DODGE = 0.14
DPI = 400


@dataclass(frozen=True, slots=True)
class Series:
    label: str
    color: str
    marker: str
    filled: bool


PAPER = Series(
    label="Paper (Table 5)",
    color="#8a8984",
    marker="o",
    filled=False,
)
OURS = Series(
    label="This reproduction",
    color="#2a78d6",
    marker="o",
    filled=True,
)
WEIGHTS = Series(
    label="Weights",
    color="#1baf7a",
    marker="s",
    filled=True,
)
ACTIVATIONS = Series(
    label="Activations",
    color="#eb6834",
    marker="D",
    filled=True,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=RESULTS,
    )
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=THRESHOLDS,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT,
    )
    return parser


def read_ppl(*, results: Path) -> tuple[dict[str, float], dict[str, float]]:
    reference = json.loads((results / "reference.adaptquant.json").read_text(encoding="utf-8"))
    ours = {method: json.loads((results / f"{method}.json").read_text(encoding="utf-8"))["ppl"] for method in METHODS}
    return {method: reference["ppl"][method] for method in METHODS}, ours


def read_lower_thresholds(*, path: Path) -> dict[str, dict[str, float]]:
    table = json.loads(path.read_text(encoding="utf-8"))
    return {operand: {projection: table[operand][projection][0] for projection in PROJECTIONS} for operand in ("weights", "activations")}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
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
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
        }
    )


def draw_points(
    *,
    axis: Axes,
    series: Series,
    x_values: Sequence[float],
    y_values: Sequence[float],
) -> None:
    axis.scatter(
        x_values,
        y_values,
        s=40,
        marker=series.marker,
        facecolor=series.color if series.filled else SURFACE,
        edgecolor=series.color if not series.filled else SURFACE,
        linewidth=1.3,
        label=series.label,
        zorder=3,
    )


def draw_perplexity(
    *,
    axis: Axes,
    paper: Mapping[str, float],
    ours: Mapping[str, float],
) -> None:
    rows = list(range(len(METHODS)))[::-1]
    axis.grid(axis="x", color=GRID, linewidth=0.5, zorder=0)
    for row, method in zip(rows, METHODS):
        axis.plot(
            [paper[method], ours[method]],
            [row, row],
            color=GRID,
            linewidth=2,
            zorder=1,
        )
        axis.text(
            max(paper[method], ours[method]) + 0.25,
            row,
            f"{ours[method]:.2f} vs {paper[method]:.2f}",
            va="center",
            ha="left",
            fontsize=7.5,
            color=TEXT_PRIMARY,
        )
    draw_points(
        axis=axis,
        series=PAPER,
        x_values=[paper[method] for method in METHODS],
        y_values=rows,
    )
    draw_points(
        axis=axis,
        series=OURS,
        x_values=[ours[method] for method in METHODS],
        y_values=rows,
    )
    axis.set_yticks(rows, [METHOD_LABELS[method] for method in METHODS])
    axis.tick_params(axis="y", length=0)
    axis.set_xlim(13, 21.5)
    axis.set_ylim(-0.6, len(METHODS) - 0.4)
    axis.set_xlabel("WikiText-2 PPL (Qwen3-4B)")
    axis.set_title("(a) Reproduced PPL, W + A quantized", loc="left")
    axis.legend(
        loc="lower right",
        frameon=False,
        handletextpad=0.3,
    )


def draw_thresholds(
    *,
    axis: Axes,
    thresholds: Mapping[str, Mapping[str, float]],
) -> None:
    rows = list(range(len(PROJECTIONS)))[::-1]
    axis.grid(axis="x", color=GRID, linewidth=0.5, zorder=0)
    for series, operand, offset in ((WEIGHTS, "weights", DODGE), (ACTIVATIONS, "activations", -DODGE)):
        draw_points(
            axis=axis,
            series=series,
            x_values=[thresholds[operand][projection] for projection in PROJECTIONS],
            y_values=[row + offset for row in rows],
        )
    axis.set_yticks(rows, list(PROJECTIONS))
    axis.tick_params(axis="y", length=0)
    axis.set_xlim(1.6, 3.05)
    axis.set_xlabel(r"Derived threshold $t_1$")
    axis.set_title(r"(b) Thresholds from Llama-3.1-8B", loc="left")
    axis.legend(
        loc="upper right",
        frameon=False,
        handletextpad=0.3,
    )


def plot_reproduction(
    *,
    paper: Mapping[str, float],
    ours: Mapping[str, float],
    thresholds: Mapping[str, Mapping[str, float]],
    output: Path,
) -> list[Path]:
    apply_style()
    figure, (left, right) = plt.subplots(
        1,
        2,
        figsize=FIGURE_SIZE,
        gridspec_kw={"width_ratios": [1.3, 1], "wspace": 0.45},
    )
    draw_perplexity(
        axis=left,
        paper=paper,
        ours=ours,
    )
    draw_thresholds(
        axis=right,
        thresholds=thresholds,
    )
    figure.subplots_adjust(left=0.13, right=0.98, bottom=0.18, top=0.88)
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    paths = [output.with_suffix(".png"), output.with_suffix(".pdf")]
    for path in paths:
        figure.savefig(path, dpi=DPI)
    plt.close(figure)
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    paper, ours = read_ppl(results=arguments.results)
    for path in plot_reproduction(
        paper=paper,
        ours=ours,
        thresholds=read_lower_thresholds(path=arguments.thresholds),
        output=arguments.output,
    ):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
