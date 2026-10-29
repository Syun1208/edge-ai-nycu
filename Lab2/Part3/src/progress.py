from __future__ import annotations

import logging
import sys
from dataclasses import dataclass

from tqdm import tqdm


COUNT_BAR = (
    "{desc} ▕{bar}▏ {percentage:3.0f}% • {n_fmt}/{total_fmt} • {rate_fmt}"
    " • ⏳ {elapsed}<{remaining}{postfix}"
)


@dataclass(frozen=True, slots=True)
class Stage:
    emoji: str
    title: str
    colour: str


LOAD = Stage(
    emoji="📥",
    title="Load",
    colour="cyan",
)


QUANTIZE = Stage(
    emoji="🧩",
    title="Quantize weights",
    colour="green",
)


MEASURE = Stage(
    emoji="🔎",
    title="Measure tiles",
    colour="magenta",
)


EVALUATE = Stage(
    emoji="📊",
    title="WikiText-2 PPL",
    colour="blue",
)


class TqdmLoggingHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        tqdm.write(
            self.format(record),
            file=sys.stderr,
        )


def setup_logging(*, verbose: bool) -> None:
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    root.handlers[:] = [handler]


def progress(
    *,
    stage: Stage,
    total: int,
) -> tqdm:
    return tqdm(
        total=total,
        desc=f"{stage.emoji} {stage.title}",
        colour=stage.colour,
        bar_format=COUNT_BAR,
        dynamic_ncols=True,
        file=sys.stderr,
    )
