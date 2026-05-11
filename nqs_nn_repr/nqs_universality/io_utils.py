from __future__ import annotations

import json
import numpy as np

from .config import ExperimentConfig
from .features import flatten_activations_for_cka


def save_all_data(
    cfg: ExperimentConfig,
    metadata,
    samples_physical,
    samples_uniform,
    samples_tangent,
    activations_physical,
    activations_uniform,
    full_tangents,
    layer_tangents,
    saliency_all,
    local_all,
    multidist,
    correlations,
):
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    np.save(cfg.out_dir / "samples_physical.npy", samples_physical.astype(np.float32))
    np.save(cfg.out_dir / "samples_uniform.npy", samples_uniform.astype(np.float32))
    np.save(cfg.out_dir / "samples_tangent.npy", samples_tangent.astype(np.float32))

    act_arrays = {}
    act_keys = {"physical": {}, "uniform": {}}
    ctr = 0

    for skey, acts in [("physical", activations_physical), ("uniform", activations_uniform)]:
        for mn in metadata["model_order"]:
            act_keys[skey][mn] = {}
            for ln, arr in acts[mn].items():
                k = f"act_{ctr:05d}"
                act_arrays[k] = arr.astype(np.float32)
                act_keys[skey][mn][ln] = k
                ctr += 1

    np.savez_compressed(cfg.out_dir / "activations.npz", **act_arrays)

    tan_arrays = {}
    tan_keys = {"full": {}, "layer": {}}
    ctr = 0

    for mn in metadata["model_order"]:
        k = f"tan_{ctr:05d}"
        tan_arrays[k] = full_tangents[mn].astype(np.float32)
        tan_keys["full"][mn] = k
        ctr += 1

        tan_keys["layer"][mn] = {}
        for ln, arr in layer_tangents[mn].items():
            k = f"tan_{ctr:05d}"
            tan_arrays[k] = arr.astype(np.float32)
            tan_keys["layer"][mn][ln] = k
            ctr += 1

    np.savez_compressed(cfg.out_dir / "tangent_features.npz", **tan_arrays)

    sal_arrays = {}
    sal_keys = {}
    for mn in metadata["model_order"]:
        k = f"sal_{mn}"
        sal_arrays[k] = saliency_all[mn].astype(np.float32)
        sal_keys[mn] = k
    np.savez_compressed(cfg.out_dir / "saliency.npz", **sal_arrays)

    metadata["activation_keys"] = act_keys
    metadata["tangent_feature_keys"] = tan_keys
    metadata["saliency_keys"] = sal_keys

    with open(cfg.out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    with open(cfg.out_dir / "local_decoding.json", "w") as f:
        json.dump(local_all, f, indent=2)
    with open(cfg.out_dir / "multidist_decoding.json", "w") as f:
        json.dump(multidist, f, indent=2)
    with open(cfg.out_dir / "correlations.json", "w") as f:
        json.dump({n: c.tolist() for n, c in correlations.items()}, f, indent=2)

    print(f"\nAll data saved -> {cfg.out_dir.resolve()}")


def load_all_data(cfg: ExperimentConfig):
    with open(cfg.out_dir / "metadata.json") as f:
        metadata = json.load(f)
    with open(cfg.out_dir / "local_decoding.json") as f:
        local_all = json.load(f)
    with open(cfg.out_dir / "multidist_decoding.json") as f:
        multidist = json.load(f)
    with open(cfg.out_dir / "correlations.json") as f:
        correlations = {n: np.array(v) for n, v in json.load(f).items()}

    act_npz = np.load(cfg.out_dir / "activations.npz")
    tan_npz = np.load(cfg.out_dir / "tangent_features.npz")
    sal_npz = np.load(cfg.out_dir / "saliency.npz")

    activations_physical = {}
    activations_uniform = {}
    act_cka = {}

    for mn in metadata["model_order"]:
        activations_physical[mn] = {}
        activations_uniform[mn] = {}

        for ln in metadata["models"][mn]["activation_layer_order"]:
            k_phys = metadata["activation_keys"]["physical"][mn][ln]
            k_unif = metadata["activation_keys"]["uniform"][mn][ln]
            activations_physical[mn][ln] = np.asarray(act_npz[k_phys])
            activations_uniform[mn][ln] = np.asarray(act_npz[k_unif])

        act_cka[mn] = flatten_activations_for_cka(activations_physical[mn])

    full_tangents = {}
    layer_tangents = {}
    for mn in metadata["model_order"]:
        full_tangents[mn] = np.asarray(
            tan_npz[metadata["tangent_feature_keys"]["full"][mn]]
        )
        layer_tangents[mn] = {}
        for ln in metadata["models"][mn]["tangent_layer_order"]:
            k = metadata["tangent_feature_keys"]["layer"][mn][ln]
            layer_tangents[mn][ln] = np.asarray(tan_npz[k])

    saliency_all = {}
    for mn in metadata["model_order"]:
        k = metadata["saliency_keys"][mn]
        saliency_all[mn] = np.asarray(sal_npz[k])

    return (
        metadata,
        act_cka,
        activations_physical,
        activations_uniform,
        full_tangents,
        layer_tangents,
        saliency_all,
        local_all,
        multidist,
        correlations,
    )
