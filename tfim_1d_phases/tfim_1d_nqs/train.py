from .app import TFIMExperimentApp
from .config import AutocorrConfig, ModelConfig, OutputConfig, ScanConfig, TrainingConfig


def main() -> None:
    model_cfg = ModelConfig(N=10, h=1.0)
    train_cfg = TrainingConfig(n_iter=300, alpha=4, n_samples=512, lr=0.01)
    scan_cfg = ScanConfig.default()
    output_cfg = OutputConfig(output_dir="./outputs")

    autocorr_cfg = AutocorrConfig(enabled=False)

    app = TFIMExperimentApp(model_cfg, train_cfg, scan_cfg, output_cfg, autocorr_cfg)
    dataset = app.run_training_and_save()
    app.generate_plots_from_dataset(dataset)


if __name__ == "__main__":
    main()
