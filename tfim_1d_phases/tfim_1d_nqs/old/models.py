from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any


class Phase(str, Enum):
    FERRO = "FERRO"
    ANTIFERRO = "ANTIFERRO"
    PARA = "PARA"


@dataclass
class PointHistory:
    iters: list[int]
    energy: list[float]
    e_var: list[float]
    m2: list[float]
    n2: list[float]


@dataclass
class TrainingResult:
    J: float
    h: float
    e_exact_finite: float
    e_exact_thermo: float
    history: PointHistory

    @property
    def e_final(self) -> float:
        return self.history.energy[-1]

    @property
    def m2_final(self) -> float:
        return self.history.m2[-1]

    @property
    def n2_final(self) -> float:
        return self.history.n2[-1]

    @property
    def rel_error_pct(self) -> float:
        denom = abs(self.e_exact_finite)
        return 0.0 if denom == 0 else abs(self.e_final - self.e_exact_finite) / denom * 100.0

    @property
    def phase(self) -> Phase:
        if self.m2_final > 0.1 and self.n2_final < 0.1:
            return Phase.FERRO
        if self.n2_final > 0.1 and self.m2_final < 0.1:
            return Phase.ANTIFERRO
        return Phase.PARA

    def to_dict(self) -> dict[str, Any]:
        return {
            "J": self.J,
            "h": self.h,
            "e_exact_finite": self.e_exact_finite,
            "e_exact_thermo": self.e_exact_thermo,
            "history": asdict(self.history),
            "e_final": self.e_final,
            "m2_final": self.m2_final,
            "n2_final": self.n2_final,
            "rel_error_pct": self.rel_error_pct,
            "phase": self.phase.value,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "TrainingResult":
        history = PointHistory(**data["history"])
        return TrainingResult(
            J=data["J"],
            h=data["h"],
            e_exact_finite=data["e_exact_finite"],
            e_exact_thermo=data["e_exact_thermo"],
            history=history,
        )


@dataclass
class ExperimentDataset:
    metadata: dict[str, Any]
    results: list[TrainingResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata,
            "results": [r.to_dict() for r in self.results],
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "ExperimentDataset":
        return ExperimentDataset(
            metadata=data["metadata"],
            results=[TrainingResult.from_dict(r) for r in data["results"]],
        )