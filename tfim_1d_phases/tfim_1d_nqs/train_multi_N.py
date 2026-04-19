from .app import TFIMMultiNApp
from .config import (
    AutocorrConfig,
    CriticalZoomConfig,
    ModelConfig,
    MultiNConfig,
    OutputConfig,
    ScanConfig,
    TrainingConfig,
)


def main() -> None:
    """
    Multi-N TFIM sweep entry point.

    Trains NQS for every N in MultiNConfig.N_values on both the coarse
    J-scan and a fine J-scan around the quantum critical points at J = +/- h.
    Records per-step tau_corr during training and runs a dedicated
    post-training MC chain to estimate the integrated autocorrelation time
    tau_int at each (N, J). Renders the full plot suite.
    """
    base_model_cfg = ModelConfig(h=1.0)
    train_cfg = TrainingConfig(n_iter=200, alpha=4, n_samples=2048, lr=0.01)
    scan_cfg = ScanConfig.default()
    multi_N_cfg = MultiNConfig(N_values=(32, 4))
    zoom_cfg = CriticalZoomConfig()
    autocorr_cfg = AutocorrConfig(enabled=True)
    output_cfg = OutputConfig(output_dir="./outputs_multiN")

    app = TFIMMultiNApp(
        base_model_cfg=base_model_cfg,
        train_cfg=train_cfg,
        scan_cfg=scan_cfg,
        multi_N_cfg=multi_N_cfg,
        zoom_cfg=zoom_cfg,
        autocorr_cfg=autocorr_cfg,
        output_cfg=output_cfg,
        include_coarse_scan=True,
        include_zoom_scan=True,
    )

    dataset = app.run_training_and_save()
    app.generate_plots_from_dataset(dataset)


if __name__ == "__main__":
    main()
