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
    base_model_cfg = ModelConfig(h=1.0)
    train_cfg = TrainingConfig(
        n_iter=300, alpha=4, n_samples=2048, lr=0.01,
        n_chains=16, n_discard_per_chain=100,
        sr_diag_shift=0.01, log_every=5,
    )
    scan_cfg = ScanConfig.default()
    multi_N_cfg = MultiNConfig(N_values=(4, 8, 16, 32, 64))
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
