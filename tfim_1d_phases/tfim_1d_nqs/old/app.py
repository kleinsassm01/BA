from .config import ModelConfig, OutputConfig, ScanConfig, TrainingConfig
from .logging_utils import setup_logging
from .models import ExperimentDataset
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
    ) -> None:
        output_cfg.ensure_dirs()
        self.logger = setup_logging(output_cfg.output_dir / "run.log")

        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.scan_cfg = scan_cfg
        self.output_cfg = output_cfg

        self.trainer = TFIMTrainer(model_cfg, train_cfg, self.logger)
        self.plotter = ResultPlotter()
        self.reporter = ConsoleReporter(self.logger)
        self.repo = DatasetRepository()

    def run_training_and_save(self) -> ExperimentDataset:
        J_values = self.scan_cfg.J_values
        self.reporter.log_run_config(
            self.model_cfg, self.train_cfg, len(J_values), float(J_values.min()), float(J_values.max())
        )

        results = self.trainer.scan(J_values)

        dataset = ExperimentDataset(
            metadata={
                "model_config": {
                    "N": self.model_cfg.N,
                    "h": self.model_cfg.h,
                },
                "training_config": {
                    "n_iter": self.train_cfg.n_iter,
                    "alpha": self.train_cfg.alpha,
                    "n_samples": self.train_cfg.n_samples,
                    "lr": self.train_cfg.lr,
                    "n_chains": self.train_cfg.n_chains,
                    "n_discard_per_chain": self.train_cfg.n_discard_per_chain,
                    "sr_diag_shift": self.train_cfg.sr_diag_shift,
                    "log_every": self.train_cfg.log_every,
                },
                "scan_config": {
                    "J_values": [float(x) for x in J_values],
                },
            },
            results=results,
        )

        self.repo.save(dataset, self.output_cfg.output_dir / self.output_cfg.dataset_file)
        self.reporter.log_summary(results)
        return dataset

    def generate_plots_from_dataset(self, dataset: ExperimentDataset) -> None:
        N = dataset.metadata["model_config"]["N"]
        results = dataset.results

        self.plotter.plot_phase_diagram(results, N, self.output_cfg.output_dir / self.output_cfg.phase_plot)
        self.plotter.plot_training_convergence(results, N, self.output_cfg.output_dir / self.output_cfg.convergence_plot)
        self.plotter.plot_training_histories(results, self.output_cfg.output_dir / self.output_cfg.histories_plot)

    def load_and_plot(self) -> ExperimentDataset:
        dataset = self.repo.load(self.output_cfg.output_dir / self.output_cfg.dataset_file)
        self.generate_plots_from_dataset(dataset)
        return dataset