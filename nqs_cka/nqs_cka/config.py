from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class ModelConfig:
    features: int = 16
    blocks: int = 2
    residual: bool = True
    use_prenorm: bool = True
    layernorm_epsilon: float = 1.0e-5
    layerscale_init: float | None = None
    learned_layerscale: bool = True
    gated_updates: bool = True
    mlp_hidden_mult: int = 2
    final_mlp_hidden_mult: int = 2
    activation: str = "gelu"
    use_diag_messages: bool = False
    include_self_in_message: bool = True
    symmetrize_spin_flip_output: bool = True
    dtype: str = "float32"

    def with_depth(
        self,
        blocks: int,
        *,
        residual: bool | None = None,
        retry: int = 0,
        training: TrainingConfig | None = None,
    ) -> ModelConfig:
        resolved_residual: bool = self.residual if residual is None else bool(residual)
        return replace(
            self,
            blocks=int(blocks),
            residual=resolved_residual,
            layerscale_init=self._layerscale_for_depth(blocks, retry=retry, training=training),
        )

    def _layerscale_for_depth(self, blocks: int, *, retry: int = 0, training: TrainingConfig | None = None) -> float | None:
        layerscale = self.layerscale_init
        if not retry or training is None:
            return layerscale

        if layerscale is None:
            layerscale = 1.0 / max(1.0, float(blocks)) ** 0.5

        return layerscale * training.retry_layerscale_decay**retry


@dataclass(frozen=True)
class TrainingConfig:
    n_iter: int = 120
    n_samples: int = 768
    n_chains: int = 16
    chunk_size: int = 128
    learning_rate: float = 0.02
    min_learning_rate: float = 0.004
    diag_shift: float = 0.04
    depth_adaptive: bool = True
    lr_depth_power: float = 0.5
    diag_shift_depth_power: float = 0.5
    retries: int = 1
    retry_lr_decay: float = 0.5
    retry_diag_mult: float = 2.0
    retry_layerscale_decay: float = 0.7

    def for_depth(self, depth_multiplier: int, *, retry: int = 0) -> TrainingConfig:
        learning_rate, diag_shift = self._depth_scaled_values(depth_multiplier)
        learning_rate, diag_shift = self._retry_scaled_values(learning_rate, diag_shift, retry)
        return replace(self, learning_rate=learning_rate, diag_shift=diag_shift)

    def _depth_scaled_values(self, depth_multiplier: int) -> tuple[float, float]:
        if not self.depth_adaptive:
            return self.learning_rate, self.diag_shift

        learning_rate = max(self.min_learning_rate, self.learning_rate / float(depth_multiplier) ** self.lr_depth_power)
        diag_shift = self.diag_shift * float(depth_multiplier) ** self.diag_shift_depth_power
        return learning_rate, diag_shift

    def _retry_scaled_values(self, learning_rate: float, diag_shift: float, retry: int) -> tuple[float, float]:
        if not retry:
            return learning_rate, diag_shift

        return learning_rate * self.retry_lr_decay**retry, diag_shift * self.retry_diag_mult**retry


@dataclass(frozen=True)
class PhysicsConfig:
    name: str = "critical_2d_tfim"
    dimension: int = 2
    length: int = 8
    J: float = 1.0
    h: float = 3.044
    J2: float = 0.0

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(self.length) for _ in range(int(self.dimension)))

    @property
    def n_spins(self) -> int:
        n_spins = 1
        for axis_size in self.shape:
            n_spins *= int(axis_size)
        return n_spins


@dataclass(frozen=True)
class AnalysisConfig:
    depth_multipliers: tuple[int, ...] = (1, 2, 4, 8)
    blocks_per_unit: int = 2
    seeds_per_depth: int = 1
    probe_samples: int = 2048
    probe_rows: int = 40000
    probe_train_frac: float = 0.6
    probe_val_frac: float = 0.2
    probe_r2_threshold: float = 0.35
    max_probe_distance: int | None = None
    max_exact_spins: int = 20


@dataclass(frozen=True)
class FigureConfig:
    enabled: bool = True

    def is_enabled(self) -> bool:
        return bool(self.enabled)


@dataclass(frozen=True)
class Figure1Config(FigureConfig):
    depth_multiplier: int = 1
    layer: str = "post1"
    seeds: int = 2


@dataclass(frozen=True)
class Figure3Config(FigureConfig):
    problem: str = "main"
    cka_kind: str = "delta"
    probe_kind: str = "post"
    probe_target: str = "local_shell"
    long_range_min_distance: int = 3


@dataclass(frozen=True)
class Figure4Config(FigureConfig):
    problem: str = "simple"
    cka_kind: str = "post_with_embed"
    probe_kind: str = "post_with_embed"
    probe_target: str = "logpsi"


@dataclass(frozen=True)
class ArchitectureSpec:
    label: str
    residual: bool
    multiplier: int

    @staticmethod
    def from_raw(raw_spec: Mapping[str, Any]) -> ArchitectureSpec:
        return ArchitectureSpec(
            label=str(raw_spec["label"]),
            residual=bool(raw_spec["residual"]),
            multiplier=int(raw_spec["multiplier"]),
        )

    def to_figure_dict(self) -> dict[str, Any]:
        return {"label": self.label, "residual": self.residual, "multiplier": self.multiplier}


@dataclass(frozen=True)
class Figure5Config(FigureConfig):
    problem: str = "main"
    activation_kind: str = "delta"
    drop_first_layers: int = 1
    seeds_per_arch: int = 1
    archs: tuple[ArchitectureSpec, ...] = field(default_factory=tuple)
    pairs: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    @property
    def raw_archs(self) -> tuple[dict[str, Any], ...]:
        return tuple(architecture.to_figure_dict() for architecture in self.archs)


@dataclass(frozen=True)
class FiguresConfig:
    figure1: Figure1Config = field(default_factory=Figure1Config)
    figure3: Figure3Config = field(default_factory=Figure3Config)
    figure4: Figure4Config = field(default_factory=Figure4Config)
    figure5: Figure5Config = field(default_factory=Figure5Config)


@dataclass(frozen=True)
class Config:
    seed: int = 0
    out_dir: str = "results"
    cache_dir: str = "cache"
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    physics: dict[str, PhysicsConfig] = field(default_factory=dict)
    figures: FiguresConfig = field(default_factory=FiguresConfig)

    @property
    def cache_path(self) -> Path:
        return Path(self.out_dir) / self.cache_dir


@dataclass(frozen=True)
class RawConfig:
    data: dict[str, Any]

    @staticmethod
    def from_yaml(path: str | Path) -> RawConfig:
        return RawConfig(data=_clean_none(yaml.safe_load(Path(path).read_text()) or {}))

    def section(self, name: str) -> dict[str, Any]:
        section_data = self.data.get(name, {})
        return dict(section_data) if isinstance(section_data, dict) else {}

    def value(self, name: str, default: Any) -> Any:
        return self.data.get(name, default)


class ConfigFactory:
    def __init__(self, raw_config: RawConfig):
        self.raw_config = raw_config

    def build(self) -> Config:
        return Config(
            seed=int(self.raw_config.value("seed", 0)),
            out_dir=str(self.raw_config.value("out_dir", "results")),
            cache_dir=str(self.raw_config.value("cache_dir", "cache")),
            model=self._model_config(),
            training=self._training_config(),
            analysis=self._analysis_config(),
            physics=self._physics_configs(),
            figures=self._figures_config(),
        )

    def _model_config(self) -> ModelConfig:
        return ModelConfig(**self.raw_config.section("model"))

    def _training_config(self) -> TrainingConfig:
        return TrainingConfig(**self.raw_config.section("training"))

    def _analysis_config(self) -> AnalysisConfig:
        analysis_data = self.raw_config.section("analysis")
        if "depth_multipliers" in analysis_data:
            analysis_data["depth_multipliers"] = tuple(int(multiplier) for multiplier in analysis_data["depth_multipliers"])
        return AnalysisConfig(**analysis_data)

    def _physics_configs(self) -> dict[str, PhysicsConfig]:
        physics_data = self.raw_config.section("physics")
        physics_configs = {name: PhysicsConfig(**config_data) for name, config_data in physics_data.items()}
        return physics_configs or {"main": PhysicsConfig()}

    def _figures_config(self) -> FiguresConfig:
        figures_data = self.raw_config.section("figures")
        return FiguresConfig(
            figure1=Figure1Config(**dict(figures_data.get("figure1", {}))),
            figure3=Figure3Config(**dict(figures_data.get("figure3", {}))),
            figure4=Figure4Config(**dict(figures_data.get("figure4", {}))),
            figure5=self._figure5_config(figures_data),
        )

    @staticmethod
    def _figure5_config(figures_data: Mapping[str, Any]) -> Figure5Config:
        figure5_data = dict(figures_data.get("figure5", {}))
        figure5_data["pairs"] = _architecture_pairs(figure5_data.get("pairs", ()))
        figure5_data["archs"] = _architecture_specs(figure5_data.get("archs", ()))
        return Figure5Config(**figure5_data)


def load_config(path: str | Path) -> Config:
    return ConfigFactory(RawConfig.from_yaml(path)).build()


def model_with_blocks(
    base: ModelConfig,
    blocks: int,
    residual: bool | None = None,
    retry: int = 0,
    train: TrainingConfig | None = None,
) -> ModelConfig:
    return base.with_depth(blocks, residual=residual, retry=retry, training=train)


def training_for_depth(train: TrainingConfig, mult: int, retry: int = 0) -> TrainingConfig:
    return train.for_depth(mult, retry=retry)


def _clean_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _clean_none(child_value) for key, child_value in value.items()}

    if isinstance(value, list):
        return [_clean_none(child_value) for child_value in value]

    return None if value == "null" else value


def _architecture_pairs(raw_pairs: Any) -> tuple[tuple[str, str], ...]:
    return tuple((str(pair[0]), str(pair[1])) for pair in raw_pairs)


def _architecture_specs(raw_architectures: Any) -> tuple[ArchitectureSpec, ...]:
    return tuple(ArchitectureSpec.from_raw(raw_architecture) for raw_architecture in raw_architectures)