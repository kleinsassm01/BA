from __future__ import annotations

from dataclasses import dataclass, field, asdict
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
    # Extensions for autocorrelation diagnostics
    tau_corr: list[float] = field(default_factory=list)
    m4: list[float] = field(default_factory=list)


@dataclass
class AutocorrAnalysis:
    lags: list[int]
    acf: list[float]
    tau_int: float
    tau_int_window: int
    n_samples: int


@dataclass
class TrainingResult:
    J: float
    h: float
    e_exact_finite: float
    e_exact_thermo: float
    history: PointHistory
    N: int = 0
    autocorr: AutocorrAnalysis | None = None

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
    def m4_final(self) -> float:
        if not self.history.m4:
            return float("nan")
        return self.history.m4[-1]

    @property
    def binder_U4(self) -> float:
        # Binder cumulant U4 = 1 - <m^4> / (3 <m^2>^2).
        m2 = self.m2_final
        m4 = self.m4_final
        if m2 <= 0 or not (m4 == m4):
            return float("nan")
        return 1.0 - m4 / (3.0 * m2 * m2)

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
        out: dict[str, Any] = {
            "J": self.J, "h": self.h, "N": self.N,
            "e_exact_finite": self.e_exact_finite,
            "e_exact_thermo": self.e_exact_thermo,
            "history": asdict(self.history),
            "e_final": self.e_final,
            "m2_final": self.m2_final,
            "n2_final": self.n2_final,
            "m4_final": self.m4_final,
            "binder_U4": self.binder_U4,
            "rel_error_pct": self.rel_error_pct,
            "phase": self.phase.value,
        }
        if self.autocorr is not None:
            out["autocorr"] = asdict(self.autocorr)
        return out

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "TrainingResult":
        hist_data = dict(data["history"])
        hist_data.setdefault("tau_corr", [])
        hist_data.setdefault("m4", [])
        history = PointHistory(**hist_data)

        autocorr = None
        if "autocorr" in data and data["autocorr"] is not None:
            autocorr = AutocorrAnalysis(**data["autocorr"])

        return TrainingResult(
            J=data["J"],
            h=data["h"],
            N=int(data.get("N", 0)),
            e_exact_finite=data["e_exact_finite"],
            e_exact_thermo=data["e_exact_thermo"],
            history=history,
            autocorr=autocorr,
        )


@dataclass
class ExperimentDataset:
    metadata: dict[str, Any]
    results: list[TrainingResult]

    def results_for_N(self, N: int) -> list[TrainingResult]:
        return [r for r in self.results if r.N == N]

    def N_values(self) -> list[int]:
        return sorted({r.N for r in self.results})

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata,
            "results": [r.to_dict() for r in self.results],
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "ExperimentDataset":
        ds = ExperimentDataset(
            metadata=data["metadata"],
            results=[TrainingResult.from_dict(r) for r in data["results"]],
        )
        if ds.results and ds.results[0].N == 0:
            N_meta = data["metadata"].get("model_config", {}).get("N", 0)
            for r in ds.results:
                r.N = N_meta
        return ds
