from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from .activations import collect_activations
from .config import Config
from .figures import make_figure1, make_figure3, make_figure4, make_figure5
from .metrics import local_shell_probe, logpsi_probe, within_net_cka
from .models import GraphNQS
from .physics import build_hamiltonian, exact_ground_state, logpsi_exact
from .probes import shared_probe
from .training import train_or_load_depth


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DepthRecord:
    raw: dict[str, Any]
    model: Any
    params: Any
    multiplier: int
    blocks: int
    seed: int
    energy: float
    residual: bool

    @staticmethod
    def from_raw(raw_record: dict[str, Any]) -> DepthRecord:
        return DepthRecord(
            raw=raw_record,
            model=raw_record["model"],
            params=raw_record["params"],
            multiplier=int(raw_record["mult"]),
            blocks=int(raw_record["blocks"]),
            seed=int(raw_record["seed"]),
            energy=float(raw_record["energy"]),
            residual=bool(raw_record["residual"]),
        )


@dataclass(frozen=True)
class ProblemContext:
    key: str
    problem: Any
    hamiltonian: Any
    hilbert_space: Any
    exact_energy: Any
    exact_wavefunction: Any
    depth_records: list[DepthRecord]

    @property
    def raw_depth_records(self) -> list[dict[str, Any]]:
        return [record.raw for record in self.depth_records]


@dataclass(frozen=True)
class ArchitectureResult:
    activations: Any
    layer_names: list[str]
    energy: float
    seed: int

    def to_figure_dict(self) -> dict[str, Any]:
        return {"acts": self.activations, "layers": self.layer_names, "energy": self.energy, "seed": self.seed}


@dataclass(frozen=True)
class DepthAnalysisResult:
    multiplier: int
    blocks: int
    seed: int
    energy_per_site: float
    cka: Any
    cka_layer_names: list[str]
    probe: Any
    probe_layer_names: list[str]
    logpsi_r2: Any
    logpsi_target: str

    def to_figure_dict(self) -> dict[str, Any]:
        return {
            "mult": self.multiplier,
            "blocks": self.blocks,
            "seed": self.seed,
            "e_per_site": self.energy_per_site,
            "cka": self.cka,
            "cka_layers": self.cka_layer_names,
            "probe": self.probe,
            "probe_layers": self.probe_layer_names,
            "logpsi_r2": self.logpsi_r2,
            "logpsi_target": self.logpsi_target,
        }


class AnalysisRunner:
    def __init__(self, config: Config, *, force: bool = False, plot_only: bool = False):
        self.config = config
        self.force = force
        self.plot_only = plot_only
        self.random_generator = np.random.default_rng(config.seed)

    def run(self) -> list[Any]:
        Path(self.config.out_dir).mkdir(parents=True, exist_ok=True)
        figure_paths: list[Any] = []

        main_context = self._load_main_problem_context()
        main_probe_samples = self._make_shared_probe(main_context)

        self._append_if_present(figure_paths, self._build_figure1(main_context, main_probe_samples))
        self._append_if_present(figure_paths, self._build_figure3_if_enabled(main_context, main_probe_samples))
        self._append_if_present(figure_paths, self._build_figure5(main_context, main_probe_samples, filename="figure5-critical_2d_tfim.png"))
        figure_paths.extend(self._build_simple_ising_figures_if_enabled())

        self._log_saved_figures(figure_paths)
        return figure_paths

    def _load_main_problem_context(self) -> ProblemContext:
        logger.info("[1] Main problem training/loading")
        context = self._train_depth_sweep(self.config.figures.figure3.problem, residual=True)

        if not context.depth_records:
            raise RuntimeError("no main-problem records found")

        return context

    def _build_simple_ising_figures_if_enabled(self) -> list[Any]:
        figure_config = self.config.figures.figure4
        if not self._is_enabled(figure_config):
            return []

        logger.info("[3] Simple-Ising baseline training/loading")
        simple_context = self._train_depth_sweep(figure_config.problem, residual=True)

        if not simple_context.depth_records:
            return []

        simple_probe_samples = self._make_shared_probe(simple_context)

        logger.info("[4] Simple-Ising depth analysis")
        simple_analysis_results = self._analyze_depth_records(simple_context, simple_probe_samples, figure_config)
        figure_paths = [
            make_figure4(
                self._as_figure_dicts(simple_analysis_results),
                simple_context.problem.shape,
                self.config.out_dir,
                filename="figure3-simple_2d_ising.png",
            )
        ]

        self._append_if_present(
            figure_paths,
            self._build_figure5(simple_context, simple_probe_samples, filename="figure5-critical_2d_ising.png"),
        )
        return figure_paths

    def _build_figure1(self, context: ProblemContext, probe_samples: Any):
        figure_config = self.config.figures.figure1
        if not self._is_enabled(figure_config):
            return None

        seeds = self._depth_seeds(figure_config.depth_multiplier, min_count=figure_config.seeds)
        raw_records = train_or_load_depth(
            self.config,
            context.problem,
            context.hamiltonian,
            context.hilbert_space,
            mult=figure_config.depth_multiplier,
            residual=True,
            seeds=seeds,
            force=self.force,
            train_missing=not self.plot_only,
        )

        if len(raw_records) < 2:
            logger.warning("figure1 needs two cached/trained seeds; skipped")
            return None

        comparison_records = raw_records[:2]
        comparison_activations = [collect_activations(record["model"], record["params"], probe_samples) for record in comparison_records]
        return make_figure1(comparison_records, comparison_activations, probe_samples, figure_config.layer, self.config.out_dir)

    def _build_figure3_if_enabled(self, context: ProblemContext, probe_samples: Any):
        figure_config = self.config.figures.figure3
        if not self._is_enabled(figure_config):
            return None

        logger.info("[2] Critical 2D TFIM depth analysis")
        analysis_results = self._analyze_depth_records(context, probe_samples, figure_config)

        return make_figure3(
            self._as_figure_dicts(analysis_results),
            context.problem.shape,
            self.config.out_dir,
            min_distance=figure_config.long_range_min_distance,
            filename="figure3-critical_2d_tfim.png",
            title="Critical 2D TFIM",
        )

    def _build_figure5(self, context: ProblemContext, probe_samples: Any, *, filename: str):
        figure_config = self.config.figures.figure5
        if not self._is_enabled(figure_config):
            return None

        residual_records_by_multiplier = self._index_residual_records_by_multiplier(context.depth_records)
        architecture_results = self._collect_architecture_results(figure_config, context, probe_samples, residual_records_by_multiplier)
        available_pairs = self._available_architecture_pairs(figure_config, architecture_results)

        if not available_pairs:
            return None

        return make_figure5(
            {label: result.to_figure_dict() for label, result in architecture_results.items()},
            available_pairs,
            self.config.out_dir,
            filename=filename,
            title=context.problem.name.replace("_", " "),
        )

    def _train_depth_sweep(self, problem_key: str, *, residual: bool = True) -> ProblemContext:
        problem = self.config.physics[problem_key]
        hamiltonian, hilbert_space = build_hamiltonian(problem)
        exact_energy, exact_wavefunction = self._exact_solution_if_available(problem, hamiltonian, hilbert_space)

        depth_records: list[DepthRecord] = []
        for depth_multiplier in self.config.analysis.depth_multipliers:
            seeds = self._depth_seeds(depth_multiplier)
            logger.info("%s: %sx depth, residual=%s, seeds=%s", problem_key, depth_multiplier, residual, len(seeds))

            raw_records = train_or_load_depth(
                self.config,
                problem,
                hamiltonian,
                hilbert_space,
                mult=depth_multiplier,
                residual=residual,
                seeds=seeds,
                force=self.force,
                train_missing=not self.plot_only,
            )
            if raw_records:
                depth_records.append(DepthRecord.from_raw(raw_records[0]))

        return ProblemContext(
            key=problem_key,
            problem=problem,
            hamiltonian=hamiltonian,
            hilbert_space=hilbert_space,
            exact_energy=exact_energy,
            exact_wavefunction=exact_wavefunction,
            depth_records=depth_records,
        )

    def _analyze_depth_records(self, context: ProblemContext, probe_samples: Any, figure_config: Any) -> list[DepthAnalysisResult]:
        distance_metric = self._message_distance_metric()
        logpsi_targets, target_source = self._logpsi_regression_target(context, probe_samples)

        analysis_results: list[DepthAnalysisResult] = []
        for depth_record in context.depth_records:
            activations = collect_activations(depth_record.model, depth_record.params, probe_samples)
            cka_layer_names = self._activation_layer_names(depth_record.model, figure_config.cka_kind)
            probe_layer_names = self._activation_layer_names(depth_record.model, figure_config.probe_kind)

            local_probe_results = local_shell_probe(
                activations,
                probe_layer_names,
                probe_samples,
                context.problem.shape,
                self.random_generator,
                rows_max=self.config.analysis.probe_rows,
                train_frac=self.config.analysis.probe_train_frac,
                val_frac=self.config.analysis.probe_val_frac,
                threshold=self.config.analysis.probe_r2_threshold,
                max_distance=self.config.analysis.max_probe_distance,
                metric=distance_metric,
            )

            analysis_results.append(
                self._depth_analysis_result(
                    depth_record,
                    context.problem,
                    activations,
                    cka_layer_names,
                    probe_layer_names,
                    local_probe_results,
                    logpsi_targets,
                    target_source,
                )
            )

        return analysis_results

    def _depth_analysis_result(
        self,
        depth_record: DepthRecord,
        problem: Any,
        activations: Any,
        cka_layer_names: list[str],
        probe_layer_names: list[str],
        local_probe_results: Any,
        logpsi_targets: Any,
        target_source: str,
    ) -> DepthAnalysisResult:
        return DepthAnalysisResult(
            multiplier=depth_record.multiplier,
            blocks=depth_record.blocks,
            seed=depth_record.seed,
            energy_per_site=depth_record.energy / problem.n_spins,
            cka=within_net_cka(activations, cka_layer_names),
            cka_layer_names=cka_layer_names,
            probe=local_probe_results,
            probe_layer_names=probe_layer_names,
            logpsi_r2=logpsi_probe(
                activations,
                probe_layer_names,
                logpsi_targets,
                self.random_generator,
                train_frac=self.config.analysis.probe_train_frac,
                val_frac=self.config.analysis.probe_val_frac,
            ),
            logpsi_target=target_source,
        )

    def _collect_architecture_results(
        self,
        figure_config: Any,
        context: ProblemContext,
        probe_samples: Any,
        residual_records_by_multiplier: dict[int, DepthRecord],
    ) -> dict[str, ArchitectureResult]:
        architecture_results: dict[str, ArchitectureResult] = {}

        for architecture_spec in figure_config.archs:
            architecture_record = self._load_architecture_record(architecture_spec, figure_config, context, residual_records_by_multiplier)
            if architecture_record is None:
                continue

            architecture_label = str(architecture_spec["label"])
            activations = collect_activations(architecture_record.model, architecture_record.params, probe_samples)
            layer_names = self._activation_layer_names(
                architecture_record.model,
                figure_config.activation_kind,
                drop_first_layers=figure_config.drop_first_layers,
            )
            architecture_results[architecture_label] = ArchitectureResult(
                activations=activations,
                layer_names=layer_names,
                energy=architecture_record.energy,
                seed=architecture_record.seed,
            )

        return architecture_results

    def _load_architecture_record(
        self,
        architecture_spec: dict[str, Any],
        figure_config: Any,
        context: ProblemContext,
        residual_records_by_multiplier: dict[int, DepthRecord],
    ) -> DepthRecord | None:
        uses_residual_connections = self._uses_residual_connections(architecture_spec)
        depth_multiplier = self._architecture_depth_multiplier(architecture_spec)

        if uses_residual_connections and depth_multiplier in residual_records_by_multiplier:
            return residual_records_by_multiplier[depth_multiplier]

        seeds = self._architecture_seeds(depth_multiplier, uses_residual_connections=uses_residual_connections, min_count=figure_config.seeds_per_arch)
        raw_records = train_or_load_depth(
            self.config,
            context.problem,
            context.hamiltonian,
            context.hilbert_space,
            mult=depth_multiplier,
            residual=uses_residual_connections,
            seeds=seeds,
            force=self.force,
            train_missing=not self.plot_only,
        )

        return DepthRecord.from_raw(raw_records[0]) if raw_records else None

    @staticmethod
    def _logpsi_regression_target(context: ProblemContext, probe_samples: Any) -> tuple[Any, str]:
        if context.exact_wavefunction is not None:
            return logpsi_exact(context.hilbert_space, context.exact_wavefunction, probe_samples), "exact"

        best_record = min(context.depth_records, key=lambda record: record.energy)
        predicted_logpsi = best_record.model.apply({"params": best_record.params}, jnp.asarray(probe_samples))
        return np.asarray(predicted_logpsi, dtype=float), "reference"

    def _make_shared_probe(self, context: ProblemContext):
        return shared_probe(
            context.hilbert_space,
            context.exact_wavefunction,
            context.raw_depth_records,
            self.config.analysis.probe_samples,
            self.random_generator,
        )

    def _exact_solution_if_available(self, problem: Any, hamiltonian: Any, hilbert_space: Any):
        if problem.n_spins > self.config.analysis.max_exact_spins:
            return None, None

        return exact_ground_state(hamiltonian, hilbert_space)

    def _depth_seeds(self, depth_multiplier: int, *, min_count: int = 1, offset: int = 0) -> list[int]:
        seed_count = max(self.config.analysis.seeds_per_depth, min_count)
        depth_seed_offset = 1000 * int(depth_multiplier)
        return [self.config.seed + offset + depth_seed_offset + seed_index for seed_index in range(seed_count)]

    def _architecture_seeds(self, depth_multiplier: int, *, uses_residual_connections: bool, min_count: int) -> list[int]:
        architecture_seed_offset = 70000 + (0 if uses_residual_connections else 20000)
        return self._depth_seeds(depth_multiplier, min_count=min_count, offset=architecture_seed_offset)

    @staticmethod
    def _activation_layer_names(model: GraphNQS, activation_kind: str, *, drop_first_layers: int = 0) -> list[str]:
        layer_names = model.layer_names(activation_kind)
        return layer_names[int(drop_first_layers) :] if activation_kind not in {"embed", "input"} and drop_first_layers > 0 else layer_names

    def _message_distance_metric(self) -> str:
        return "chebyshev" if self.config.model.use_diag_messages else "manhattan"

    @staticmethod
    def _index_residual_records_by_multiplier(depth_records: list[DepthRecord]) -> dict[int, DepthRecord]:
        return {record.multiplier: record for record in depth_records if record.residual}

    @staticmethod
    def _available_architecture_pairs(figure_config: Any, architecture_results: dict[str, ArchitectureResult]) -> list[tuple[str, str]]:
        available_pairs: list[tuple[str, str]] = []

        for pair in figure_config.pairs:
            if len(pair) < 2:
                logger.warning("skipping malformed architecture pair: %r", pair)
                continue

            left_label, right_label = str(pair[0]), str(pair[1])
            if left_label in architecture_results and right_label in architecture_results:
                available_pairs.append((left_label, right_label))

        return available_pairs

    @staticmethod
    def _is_enabled(config_section: Any) -> bool:
        return bool(config_section.enabled)

    @staticmethod
    def _uses_residual_connections(architecture_spec: dict[str, Any]) -> bool:
        return bool(architecture_spec["residual"])

    @staticmethod
    def _architecture_depth_multiplier(architecture_spec: dict[str, Any]) -> int:
        return int(architecture_spec["multiplier"])

    @staticmethod
    def _append_if_present(paths: list[Any], path: Any) -> None:
        if path:
            paths.append(path)

    @staticmethod
    def _as_figure_dicts(results: list[DepthAnalysisResult]) -> list[dict[str, Any]]:
        return [result.to_figure_dict() for result in results]

    @staticmethod
    def _log_saved_figures(figure_paths: list[Any]) -> None:
        logger.info("saved figures:")
        for path in figure_paths:
            logger.info("  %s", path)


def run(config: Config, *, force: bool = False, plot_only: bool = False):
    return AnalysisRunner(config, force=force, plot_only=plot_only).run()