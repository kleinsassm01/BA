from __future__ import annotations

from .models import BiRNNNQS, CNNNQS, GNNNQS


def make_configs():
    return {
        "CNN-3layer-k3": CNNNQS(
            channels=(10, 10, 10),
            kernels=(3, 3, 3),
            init_std=0.02,
        ),
        "CNN-5layer-k3": CNNNQS(
            channels=(7, 7, 7, 7, 7),
            kernels=(3, 3, 3, 3, 3),
            init_std=0.02,
        ),
        "BiRNN-1layer": BiRNNNQS(
            n_layers=1,
            hidden_dim=17,
            init_std=0.02,
        ),
        "BiRNN-2layer": BiRNNNQS(
            n_layers=2,
            hidden_dim=9,
            init_std=0.02,
        ),
        "GNN-3layer-local": GNNNQS(
            n_layers=3,
            channels=7,
            activation="gelu",
            init_std=0.02,
        ),
    }


def model_family(model) -> str:
    if isinstance(model, CNNNQS):
        return "CNN"
    if isinstance(model, BiRNNNQS):
        return "BiRNN"
    if isinstance(model, GNNNQS):
        return "GNN"


def select_train_names(train_arg: str | None, train_only: list[str] | None, configs: dict) -> list[str]:
    if train_only is not None:
        selected = train_only if train_only else list(configs.keys())
        bad = [x for x in selected if x not in configs]
        if bad:
            raise ValueError(f"Unknown model names in --train_only: {bad}")
        return selected

    if train_arg is None or train_arg == "all":
        return list(configs.keys())

    if train_arg in configs:
        return [train_arg]

    key = train_arg.lower()

    if key == "cnn":
        return [k for k, v in configs.items() if isinstance(v, CNNNQS)]
    if key in ("rnn", "birnn"):
        return [k for k, v in configs.items() if isinstance(v, BiRNNNQS)]
    if key == "gnn":
        return [k for k, v in configs.items() if isinstance(v, GNNNQS)]
