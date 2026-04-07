import logging

from .config import ModelConfig, TrainingConfig
from .models import TrainingResult


class ConsoleReporter:
    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger

    def log_run_config(self, model_cfg: ModelConfig, train_cfg: TrainingConfig, j_count: int, j_min: float, j_max: float) -> None:
        self.logger.info("NQS for 1D Transverse Field Ising Model")
        self.logger.info("System: N=%d, h=%.3f", model_cfg.N, model_cfg.h)
        self.logger.info("Scan: %d J values from %.3f to %.3f", j_count, j_min, j_max)
        self.logger.info(
            "Training: n_iter=%d alpha=%d n_samples=%d lr=%.5f",
            train_cfg.n_iter, train_cfg.alpha, train_cfg.n_samples, train_cfg.lr
        )

    def log_summary(self, results: list[TrainingResult]) -> None:
        for r in results:
            self.logger.info(
                "J=% .3f | E/N=%.6f | exact=%.6f | err=%.3f%% | m2=%.4f | n2=%.4f | phase=%s",
                r.J, r.e_final, r.e_exact_finite, r.rel_error_pct, r.m2_final, r.n2_final, r.phase.value
            )