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
    # --- extensions for autocorrelation diagnostics -------------------------
    # tau_corr = NetKet's per-step integrated autocorrelation estimate of the
    # energy samples, reported at every log step during training.
    tau_corr: list[float] = field(default_factory=list)
    # m4 = <m^4> recorded during training; needed for the Binder cumulant U4.
    m4: list[float] = field(default_factory=list)


@dataclass
class AutocorrAnalysis:
    """
    Post-training autocorrelation analysis of the energy samples at the
    optimized variational parameters. Computed once per (N, J) point via a
    long, dedicated MCMC chain (see autocorr.py).
    """
    lags: list[int]           # lag index t = 0, 1, ..., t_max
    acf: list[float]          # normalized autocorrelation function C(t)/C(0)
    tau_int: float            # integrated autocorrelation time (Sokal window)
    tau_int_window: int       # window size W chosen by the Sokal criterion
    n_samples: int            # MC samples the ACF was built from


@dataclass
class TrainingResult:
    J: float
    h: float
    e_exact_finite: float
    e_exact_thermo: float
    history: PointHistory
    # Extensions --------------------------------------------------------------
    # N is stored per-result so results can be pooled across multiple system
    # sizes in one dataset. Defaults to 0 for backward-compat with old JSON;
    # the loader patches this from metadata when a single-N file is read.
    N: int = 0
    # Optional because legacy datasets (from the original app) don't have it.
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
        """
        Binder cumulant U4 = 1 - <m^4> / (3 <m^2>^2).

        Canonical diagnostic for a second-order phase transition: curves
        computed at different N cross at the critical point, making the
        crossing a size-independent estimator of J_c.
        """
        m2 = self.m2_final
        m4 = self.m4_final
        if m2 <= 0 or not (m4 == m4):  # NaN guard
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
            "J": self.J,
            "h": self.h,
            "N": self.N,
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
        # Backfill N on legacy results if the JSON stored only a single-N run.
        if ds.results and ds.results[0].N == 0:
            N_meta = data["metadata"].get("model_config", {}).get("N", 0)
            for r in ds.results:
                r.N = N_meta
        return ds
