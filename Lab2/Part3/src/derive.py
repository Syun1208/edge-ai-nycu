from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn

from src.cuda_quantizer import build_tile_quantizer
from src.formats import CAP, FORMATS, FP4, FP6, FP8, TILE_COLS, nvfp4_fake_quantize
from src.loading import SEQUENCE_LENGTH, load_model, wikitext_token_ids
from src.model_quantizer import projection_of, target_linears
from src.progress import MEASURE, progress
from src.thresholds import OPERANDS, PROJECTIONS, ThresholdTable
from src.tiles import FORCED, ThresholdPair, TileQuantizer, pad_rows, tile_statistic, to_tiles


GRID_STEP = 0.02
GRID_POINTS = 226
SLACK_SEARCH_MAX_DB = 60.0
SLACK_SEARCH_STEPS = 48

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class GridHistogram:
    device: torch.device
    counts: torch.Tensor = field(init=False)
    errors: torch.Tensor = field(init=False)
    reference: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        self.counts = torch.zeros(
            GRID_POINTS,
            dtype=torch.float64,
            device=self.device,
        )
        self.errors = torch.zeros(
            len(FORMATS),
            GRID_POINTS,
            dtype=torch.float64,
            device=self.device,
        )
        self.reference = torch.zeros(
            GRID_POINTS,
            dtype=torch.float64,
            device=self.device,
        )

    def add(
        self,
        *,
        statistic: torch.Tensor,
        normalized_errors: torch.Tensor,
        normalized_reference: torch.Tensor,
    ) -> None:
        positions = torch.clamp(
            torch.floor(statistic / GRID_STEP + 1e-6).long(),
            min=0,
            max=GRID_POINTS - 1,
        )
        self.counts.index_add_(0, positions, torch.ones_like(statistic, dtype=torch.float64))
        for index in range(len(FORMATS)):
            self.errors[index].index_add_(0, positions, normalized_errors[index].double())
        self.reference.index_add_(0, positions, normalized_reference.double())


class TileDistortion:
    def __init__(self, *, tile_quantizer: TileQuantizer) -> None:
        self._tile_quantizer = tile_quantizer

    def measure(
        self,
        *,
        matrix: torch.Tensor,
        channel_weights: torch.Tensor,
        histogram: GridHistogram,
    ) -> None:
        tiles = to_tiles(matrix=matrix)
        statistic, tile_max = tile_statistic(tiles=tiles)
        weights = (channel_weights / channel_weights.mean()).float().reshape(1, -1, 1, TILE_COLS)
        signal = (weights * tiles.square()).sum(dim=(-1, -2))
        alive = (tile_max > 0) & (signal > 0)

        errors = []
        for element in FORMATS:
            quantized, _ = self._tile_quantizer.quantize(
                matrix=matrix,
                pair=FORCED[element.name],
            )
            errors.append(
                self._weighted_error(
                    tiles=tiles,
                    quantized=quantized,
                    weights=weights,
                )
            )
        reference = self._weighted_error(
            tiles=tiles,
            quantized=nvfp4_fake_quantize(matrix=matrix),
            weights=weights,
        )
        histogram.add(
            statistic=statistic[alive],
            normalized_errors=torch.stack([error[alive] / signal[alive] for error in errors]),
            normalized_reference=reference[alive] / signal[alive],
        )

    @staticmethod
    def _weighted_error(
        *,
        tiles: torch.Tensor,
        quantized: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        return (weights * (tiles - to_tiles(matrix=quantized)).square()).sum(dim=(-1, -2))


class ActivationCollector:
    def __init__(
        self,
        *,
        name: str,
        linear: nn.Linear,
        distortion: TileDistortion,
        histogram: GridHistogram,
        second_moments: dict[str, torch.Tensor],
    ) -> None:
        self._name = name
        self._linear = linear
        self._distortion = distortion
        self._histogram = histogram
        self._second_moments = second_moments

    def __call__(
        self,
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
    ) -> None:
        flat = inputs[0].reshape(-1, inputs[0].shape[-1])
        padded, _ = pad_rows(matrix=flat)
        self._second_moments[self._name] = flat.float().square().mean(dim=0)
        self._distortion.measure(
            matrix=padded.contiguous(),
            channel_weights=self._linear.weight.float().square().sum(dim=0),
            histogram=self._histogram,
        )


@dataclass(frozen=True, slots=True)
class Solution:
    pair: ThresholdPair
    bits: float
    tiles: float
    margin_db: float
    feasible: bool


class ThresholdSolver:
    def __init__(self, *, histograms: Mapping[str, Mapping[str, GridHistogram]]) -> None:
        self._histograms = histograms

    def solve(
        self,
        *,
        operand: str,
        slack_db: float,
    ) -> dict[str, Solution]:
        return {
            projection: self._solve_projection(
                layers=[
                    histogram
                    for name, histogram in self._histograms[operand].items()
                    if projection_of(name=name) == projection
                ],
                slack_db=slack_db,
            )
            for projection in PROJECTIONS
        }

    @staticmethod
    def spent_bits(*, solutions: Mapping[str, Solution]) -> float:
        tiles = sum(solution.tiles for solution in solutions.values())
        return sum(solution.bits * solution.tiles for solution in solutions.values()) / tiles

    def _solve_projection(
        self,
        *,
        layers: Sequence[GridHistogram],
        slack_db: float,
    ) -> Solution:
        counts = self._prefix(values=torch.stack([layer.counts for layer in layers]).sum(dim=0))
        errors = [
            self._prefix(values=torch.stack([layer.errors[index] for layer in layers]))
            for index in range(len(FORMATS))
        ]
        reference = torch.stack([layer.reference.sum() for layer in layers])
        layer_tiles = torch.stack([layer.counts.sum() for layer in layers])

        cut = torch.arange(
            GRID_POINTS + 1,
            device=counts.device,
        )
        lower, upper = torch.meshgrid(cut, cut, indexing="ij")
        last = GRID_POINTS
        ordered = upper >= lower
        fp4_tiles = counts[lower]
        fp6_tiles = counts[upper] - counts[lower]
        fp8_tiles = counts[last] - counts[upper]
        bits = (
            FP4.effective_bits * fp4_tiles + FP6.effective_bits * fp6_tiles + FP8.effective_bits * fp8_tiles
        ) / counts[last]
        error = (
            errors[0][:, lower]
            + errors[1][:, upper]
            - errors[1][:, lower]
            + errors[2][:, last, None, None]
            - errors[2][:, upper]
        )
        budget = reference[:, None, None] * 10 ** (slack_db / 10)
        feasible = ordered & (error <= budget).all(dim=0)

        if bool(feasible.any()):
            cost = torch.where(
                feasible,
                bits,
                torch.inf,
            )
            best = cost.min()
            ties = feasible & (cost <= best + 1e-12)
            choice = torch.where(
                ties,
                fp4_tiles * (GRID_POINTS + 2) + upper,
                -torch.inf,
            ).flatten().argmax()
        else:
            choice = torch.where(
                ordered,
                error.sum(dim=0),
                torch.inf,
            ).flatten().argmin()
        row, column = divmod(int(choice), GRID_POINTS + 1)

        layer_sqnr = -10 * torch.log10(error[:, row, column] / layer_tiles)
        reference_sqnr = -10 * torch.log10(reference / layer_tiles)
        return Solution(
            pair=ThresholdPair(
                lower=self._grid_value(index=row),
                upper=self._grid_value(index=column),
            ),
            bits=float(bits[row, column]),
            tiles=float(counts[last]),
            margin_db=float((layer_sqnr - reference_sqnr).min()),
            feasible=bool(feasible.any()),
        )

    @staticmethod
    def _prefix(*, values: torch.Tensor) -> torch.Tensor:
        zero = torch.zeros_like(values[..., :1])
        return torch.cat([zero, values.cumsum(dim=-1)], dim=-1)

    @staticmethod
    def _grid_value(*, index: int) -> float:
        return CAP if index >= GRID_POINTS else round(index * GRID_STEP, 2)


def solve_weights_for_bits(
    *,
    solver: ThresholdSolver,
    target_bits: float,
) -> tuple[dict[str, Solution], float]:
    low, high = 0.0, SLACK_SEARCH_MAX_DB
    for _ in range(SLACK_SEARCH_STEPS):
        middle = (low + high) / 2
        spent = ThresholdSolver.spent_bits(
            solutions=solver.solve(
                operand="weights",
                slack_db=middle,
            )
        )
        if spent > target_bits:
            low = middle
        else:
            high = middle
    return (
        solver.solve(
            operand="weights",
            slack_db=high,
        ),
        high,
    )


@torch.no_grad()
def run_derivation(
    *,
    reference_model: str,
    output: Path,
    backend: str,
    target_bits: float | None,
) -> ThresholdTable:
    model, tokenizer = load_model(
        name=reference_model,
        dtype=torch.bfloat16,
    )
    device = next(model.parameters()).device
    distortion = TileDistortion(tile_quantizer=build_tile_quantizer(backend=backend))
    linears = target_linears(model=model)
    histograms = {operand: {name: GridHistogram(device=device) for name, _ in linears} for operand in OPERANDS}
    second_moments: dict[str, torch.Tensor] = {}

    sequence = wikitext_token_ids(
        tokenizer=tokenizer,
        split="train",
    )[:, :SEQUENCE_LENGTH].to(device)
    logger.info(f"📝 reference corpus: one WikiText-2 train sequence of {sequence.shape[1]} tokens")
    handles = [
        linear.register_forward_pre_hook(
            ActivationCollector(
                name=name,
                linear=linear,
                distortion=distortion,
                histogram=histograms["activations"][name],
                second_moments=second_moments,
            )
        )
        for name, linear in linears
    ]
    model(sequence)
    for handle in handles:
        handle.remove()

    with progress(
        stage=MEASURE,
        total=len(linears),
    ) as bar:
        for name, linear in linears:
            distortion.measure(
                matrix=linear.weight.data,
                channel_weights=second_moments[name],
                histogram=histograms["weights"][name],
            )
            bar.update(1)

    solver = ThresholdSolver(histograms=histograms)
    solutions = {
        "activations": solver.solve(
            operand="activations",
            slack_db=0.0,
        ),
        "weights": solver.solve(
            operand="weights",
            slack_db=0.0,
        ),
    }
    if target_bits is not None:
        activation_bits = ThresholdSolver.spent_bits(solutions=solutions["activations"])
        solutions["weights"], slack = solve_weights_for_bits(
            solver=solver,
            target_bits=2 * target_bits - activation_bits,
        )
        logger.info(f"🎯 target {target_bits:.3f} bits: weight slack {slack:.3f} dB")

    for operand in OPERANDS:
        logger.info(f"📊 {operand}: {ThresholdSolver.spent_bits(solutions=solutions[operand]):.3f} effective bits")
        for projection, solution in solutions[operand].items():
            logger.info(
                f"   {projection:<9} t1={solution.pair.lower:<5} t2={solution.pair.upper:<5} "
                f"bits={solution.bits:.3f} margin={solution.margin_db:+.3f} dB feasible={solution.feasible}"
            )

    table = ThresholdTable(
        pairs={operand: {projection: solution.pair for projection, solution in solutions[operand].items()} for operand in OPERANDS},
        reference_model=reference_model,
    )
    table.save(path=output)
    logger.info(f"💾 thresholds written to {output}")
    return table
