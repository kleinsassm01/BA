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
    app = TFIMMultiNApp(
        base_model_cfg=ModelConfig(h=1.0),
        train_cfg=TrainingConfig(),
        scan_cfg=ScanConfig.default(),
        multi_N_cfg=MultiNConfig(),
        zoom_cfg=CriticalZoomConfig(),
        autocorr_cfg=AutocorrConfig(enabled=True),
        output_cfg=OutputConfig(output_dir="./outputs_multiN"),
    )
    app.load_and_plot()


if __name__ == "__main__":
    main()
