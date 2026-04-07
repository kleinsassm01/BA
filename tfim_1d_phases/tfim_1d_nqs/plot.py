from .app import TFIMExperimentApp
from .config import ModelConfig, OutputConfig, ScanConfig, TrainingConfig


def main() -> None:
    model_cfg = ModelConfig()
    train_cfg = TrainingConfig()
    scan_cfg = ScanConfig.default()
    output_cfg = OutputConfig(output_dir="./outputs")

    app = TFIMExperimentApp(model_cfg, train_cfg, scan_cfg, output_cfg)
    app.load_and_plot()


if __name__ == "__main__":
    main()