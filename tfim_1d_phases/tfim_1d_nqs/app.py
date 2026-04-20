from dataclasses import replace

from .config import (
    AutocorrConfig,
    CriticalZoomConfig,
    ModelConfig,
    MultiNConfig,
    OutputConfig,
    ScanConfig,
    TrainingConfig,
)
from .logging_utils import setup_logging
from .models import ExperimentDataset, TrainingResult
from .persistence import DatasetRepository
from .plotting import ResultPlotter
from .reporting import ConsoleReporter
from .trainer import TFIMTrainer


class TFIMExperimentApp:

    def __init__(
        self,
        model_cfg: ModelConfig,
        train_cfg: TrainingConfig,
        scan_cfg: ScanConfig,
        output_cfg: OutputConfig,
        autocorr_cfg: AutocorrConfig | None = None,
    ) -> None:
        output_cfg.ensure_dirs()
        self.logger = setup_logging(output_cfg.output_dir / "run.log")

        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.scan_cfg = scan_cfg
        self.output_cfg = output_cfg
        self.autocorr_cfg = autocorr_cfg or AutocorrConfig(enabled=False)

        self.trainer = TFIMTrainer(model_cfg, train_cfg, self.logger, self.autocorr_cfg)
        self.plotter = ResultPlotter()
        self.reporter = ConsoleReporter(self.logger)
        self.repo = DatasetRepository()

    def run_training_and_save(self) -> ExperimentDataset:
        J_values = self.scan_cfg.J_values
        self.reporter.log_run_config(
            self.model_cfg, self.train_cfg,
            len(J_values), float(J_values.min()), float(J_values.max()),
        )

        results = self.trainer.scan(J_values)

        dataset = ExperimentDataset(
            metadata=_build_metadata(
                self.model_cfg, self.train_cfg, self.autocorr_cfg, J_values,
                N_values=[self.model_cfg.N],
            ),
            results=results,
        )
        self.repo.save(dataset, self.output_cfg.output_dir / self.output_cfg.dataset_file)
        self.reporter.log_summary(results)
        return dataset

    def generate_plots_from_dataset(self, dataset: ExperimentDataset) -> None:
        N = dataset.metadata["model_config"]["N"]
        results = dataset.results

        self.plotter.plot_phase_diagram(
            results, N, self.output_cfg.output_dir / self.output_cfg.phase_plot
        )
        self.plotter.plot_training_convergence(
            results, N, self.output_cfg.output_dir / self.output_cfg.convergence_plot
        )
        self.plotter.plot_training_histories(
            results, self.output_cfg.output_dir / self.output_cfg.histories_plot
        )

    def load_and_plot(self) -> ExperimentDataset:
        dataset = self.repo.load(self.output_cfg.output_dir / self.output_cfg.dataset_file)
        self.generate_plots_from_dataset(dataset)
        return dataset


class TFIMMultiNApp:

    def __init__(
        self,
        base_model_cfg: ModelConfig,
        train_cfg: TrainingConfig,
        scan_cfg: ScanConfig,
        multi_N_cfg: MultiNConfig,
        zoom_cfg: CriticalZoomConfig,
        autocorr_cfg: AutocorrConfig,
        output_cfg: OutputConfig,
        include_coarse_scan: bool = True,
        include_zoom_scan: bool = True,
    ) -> None:
        output_cfg.ensure_dirs()
        self.logger = setup_logging(output_cfg.output_dir / "run.log")

        self.base_model_cfg = base_model_cfg
        self.train_cfg = train_cfg
        self.scan_cfg = scan_cfg
        self.multi_N_cfg = multi_N_cfg
        self.zoom_cfg = zoom_cfg
        self.autocorr_cfg = autocorr_cfg
        self.output_cfg = output_cfg
        self.include_coarse_scan = include_coarse_scan
        self.include_zoom_scan = include_zoom_scan

        self.plotter = ResultPlotter()
        self.reporter = ConsoleReporter(self.logger)
        self.repo = DatasetRepository()

    def _merged_J_values(self):
        import numpy as np
        pieces = []
        if self.include_coarse_scan:
            pieces.append(self.scan_cfg.J_values)
        if self.include_zoom_scan:
            pieces.append(self.zoom_cfg.combined_window())
        if not pieces:
            raise ValueError("At least one of coarse or zoom scan must be enabled.")
        return np.unique(np.concatenate(pieces))

    def run_training_and_save(self) -> ExperimentDataset:
        J_values = self._merged_J_values()
        all_results: list[TrainingResult] = []

        self.logger.info(
            "=== Multi-N sweep: N in %s | h=%.3f | %d J-points ===",
            list(self.multi_N_cfg.N_values), self.base_model_cfg.h, len(J_values),
        )

        for N in self.multi_N_cfg.N_values:
            self.logger.info("--- Starting scan for N=%d ---", N)
            model_cfg = replace(self.base_model_cfg, N=N)
            trainer = TFIMTrainer(
                model_cfg, self.train_cfg, self.logger, self.autocorr_cfg
            )
            self.reporter.log_run_config(
                model_cfg, self.train_cfg,
                len(J_values), float(J_values.min()), float(J_values.max()),
            )
            results_N = trainer.scan(J_values)
            all_results.extend(results_N)
            self._checkpoint(all_results, J_values)

        dataset = ExperimentDataset(
            metadata=_build_metadata(
                self.base_model_cfg, self.train_cfg, self.autocorr_cfg, J_values,
                N_values=list(self.multi_N_cfg.N_values),
                zoom_cfg=self.zoom_cfg,
            ),
            results=all_results,
        )
        self.repo.save(dataset, self.output_cfg.output_dir / self.output_cfg.dataset_file)
        self.reporter.log_summary(all_results)
        return dataset

    def _checkpoint(self, results: list[TrainingResult], J_values) -> None:
        dataset = ExperimentDataset(
            metadata=_build_metadata(
                self.base_model_cfg, self.train_cfg, self.autocorr_cfg, J_values,
                N_values=list(self.multi_N_cfg.N_values),
                zoom_cfg=self.zoom_cfg,
            ),
            results=results,
        )
        self.repo.save(
            dataset,
            self.output_cfg.output_dir / (self.output_cfg.dataset_file + ".partial"),
        )

    def generate_plots_from_dataset(self, dataset: ExperimentDataset) -> None:
        od = self.output_cfg.output_dir

        ref_N = dataset.N_values()[0]
        ref_results = dataset.results_for_N(ref_N)
        self.plotter.plot_phase_diagram(ref_results, ref_N, od / self.output_cfg.phase_plot)
        self.plotter.plot_training_convergence(ref_results, ref_N, od / self.output_cfg.convergence_plot)
        self.plotter.plot_training_histories(ref_results, od / self.output_cfg.histories_plot)

        self.plotter.plot_multi_N_overlay(dataset, od / self.output_cfg.overlay_plot)
        self.plotter.plot_critical_zoom(
            dataset, self.zoom_cfg, od / self.output_cfg.critical_zoom_plot
        )
        self.plotter.plot_binder_cumulant(
            dataset, self.zoom_cfg, od / self.output_cfg.binder_plot
        )
        self.plotter.plot_tau_corr_vs_step(dataset, od / self.output_cfg.tau_step_plot)
        self.plotter.plot_tau_int_vs_J(dataset, od / self.output_cfg.tau_vs_J_plot)
        self.plotter.plot_energy_variance(
        dataset, od / self.output_cfg.energy_variance_plot
    )
        self.plotter.plot_binder_crossings(
            dataset, self.zoom_cfg, od / self.output_cfg.binder_crossings_plot
        )
        self.plotter.plot_fss_order_parameter(
            dataset, od / self.output_cfg.fss_order_parameter_plot
        )
        self.plotter.plot_curvature_peak_scaling(
            dataset, self.zoom_cfg, od / self.output_cfg.curvature_peak_scaling_plot
        )

    def load_and_plot(self) -> ExperimentDataset:
        dataset = self.repo.load(self.output_cfg.output_dir / self.output_cfg.dataset_file)
        self.generate_plots_from_dataset(dataset)
        return dataset


def _build_metadata(
    model_cfg: ModelConfig,
    train_cfg: TrainingConfig,
    autocorr_cfg: AutocorrConfig,
    J_values,
    N_values: list[int],
    zoom_cfg: CriticalZoomConfig | None = None,
) -> dict:
    meta = {
        "model_config": {
            "N": model_cfg.N,
            "h": model_cfg.h,
            "N_values": N_values,
        },
        "training_config": {
            "n_iter": train_cfg.n_iter,
            "alpha": train_cfg.alpha,
            "n_samples": train_cfg.n_samples,
            "lr": train_cfg.lr,
            "n_chains": train_cfg.n_chains,
            "n_discard_per_chain": train_cfg.n_discard_per_chain,
            "sr_diag_shift": train_cfg.sr_diag_shift,
            "log_every": train_cfg.log_every,
        },
        "autocorr_config": {
            "enabled": autocorr_cfg.enabled,
            "n_samples": autocorr_cfg.n_samples,
            "n_chains": autocorr_cfg.n_chains,
            "n_discard": autocorr_cfg.n_discard,
            "max_lag": autocorr_cfg.max_lag,
            "sokal_c": autocorr_cfg.sokal_c,
        },
        "scan_config": {
            "J_values": [float(x) for x in J_values],
        },
    }
    if zoom_cfg is not None:
        meta["zoom_config"] = {
            "J_center_ferro": zoom_cfg.J_center_ferro,
            "J_center_antiferro": zoom_cfg.J_center_antiferro,
            "zoom_halfwidth": zoom_cfg.zoom_halfwidth,
            "n_points_per_side": zoom_cfg.n_points_per_side,
        }
    return meta
