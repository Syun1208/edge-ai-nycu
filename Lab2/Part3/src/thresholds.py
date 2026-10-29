from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from src.tiles import ThresholdPair


PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
OPERANDS = ("weights", "activations")


@dataclass(frozen=True, slots=True)
class ThresholdTable:
    pairs: Mapping[str, Mapping[str, ThresholdPair]]
    reference_model: str

    def pair(
        self,
        *,
        operand: str,
        projection: str,
    ) -> ThresholdPair:
        return self.pairs[operand][projection]

    def save(self, *, path: Path) -> None:
        payload = {
            "reference_model": self.reference_model,
            **{
                operand: {projection: pair.to_json() for projection, pair in table.items()}
                for operand, table in self.pairs.items()
            },
        }
        path.write_text(
            json.dumps(
                payload,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, *, path: Path) -> ThresholdTable:
        payload = json.loads(path.read_text(encoding="utf-8"))
        missing = [
            f"{operand}.{projection}"
            for operand in OPERANDS
            for projection in PROJECTIONS
            if projection not in payload.get(operand, {})
        ]
        if missing:
            raise ValueError(f"{path} misses thresholds for {missing}")
        return cls(
            pairs={
                operand: {projection: ThresholdPair.from_json(payload[operand][projection]) for projection in PROJECTIONS}
                for operand in OPERANDS
            },
            reference_model=payload["reference_model"],
        )

    @classmethod
    def uniform(cls, *, pair: ThresholdPair) -> ThresholdTable:
        return cls(
            pairs={operand: {projection: pair for projection in PROJECTIONS} for operand in OPERANDS},
            reference_model="none",
        )
