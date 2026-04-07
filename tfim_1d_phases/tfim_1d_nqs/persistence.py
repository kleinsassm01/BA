import json
from pathlib import Path

from .models import ExperimentDataset


class DatasetRepository:
    @staticmethod
    def save(dataset: ExperimentDataset, file_path: Path) -> None:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(dataset.to_dict(), f, indent=2)

    @staticmethod
    def load(file_path: Path) -> ExperimentDataset:
        with open(file_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return ExperimentDataset.from_dict(payload)