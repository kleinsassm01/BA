from __future__ import annotations

import json

import jax
import jax.numpy as jnp

from .config import ExperimentConfig
from .features import (
    compute_input_saliency,
    extract_activations,
    extract_tangent_features,
    flatten_activations_for_cka,
)
from .hamiltonian import make_tfim
from .io_utils import load_all_data, save_all_data
from .metrics import (
    compute_local_term_decoding,
    compute_multidistance_decoding,
    compute_zz_correlation,
)
from .model_registry import make_configs
from .plotting import make_all_plots
from .sampling import get_shared_samples, load_existing_probe_samples, make_uniform_spin_samples
from .training import count_params, model_metadata, train_nqs


def base_metadata(cfg: ExperimentConfig, model_order: list[str]):
    return {
        "experiment": "NQS representation universality",
        "hamiltonian": {
            "name": "1D TFIM",
            "N": cfg.N,
            "J": cfg.J,
            "h": cfg.h,
            "pbc": cfg.pbc,
        },
        "training": {
            "n_samples": cfg.n_samples_train,
            "n_iter": cfg.n_iter,
            "lr": cfg.lr,
            "optimizer": "Adam cosine decay",
            "use_sr": cfg.use_sr,
        },
        "metrics": {
            "activation_samples": cfg.n_activation_samples,
            "uniform_probe_samples": cfg.n_uniform_probe_samples,
            "tangent_samples": cfg.n_tangent_samples,
            "saliency_samples": cfg.n_saliency_samples,
            "ridge_alpha": cfg.ridge_alpha,
            "probe_train_frac": cfg.probe_train_frac,
            "max_decode_distance": cfg.max_decode_distance,
            "max_corr_distance": cfg.max_corr_distance,
        },
        "model_order": model_order,
        "models": {},
    }


def _load_existing_metadata(cfg: ExperimentConfig):
    metadata_path = cfg.out_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            return json.load(f)
    return None


def _load_existing_analysis_for_incremental(cfg: ExperimentConfig, all_names: list[str], train_names: list[str]):
    existing = {
        "act_phys": {},
        "act_unif": {},
        "tangents": {},
        "layer_tangents": {},
        "saliency": {},
        "correlations": {},
    }

    print("\nLoading existing saved data for non-retrained models ...")
    (
        _old_meta,
        _old_act_cka,
        old_act_phys,
        old_act_unif,
        old_tangents,
        old_layer_tangents,
        old_saliency,
        _old_local,
        _old_multidist,
        old_corr,
    ) = load_all_data(cfg)

    for mn in all_names:
        if mn in train_names:
            continue
        if mn in old_act_phys:
            existing["act_phys"][mn] = old_act_phys[mn]
        if mn in old_act_unif:
            existing["act_unif"][mn] = old_act_unif[mn]
        if mn in old_tangents:
            existing["tangents"][mn] = old_tangents[mn]
        if mn in old_layer_tangents:
            existing["layer_tangents"][mn] = old_layer_tangents[mn]
        if mn in old_saliency:
            existing["saliency"][mn] = old_saliency[mn]
        if mn in old_corr:
            existing["correlations"][mn] = old_corr[mn]

    print(f"  Loaded saved data for: {list(existing['act_phys'].keys())}")
    return existing


def _assert_incremental_has_existing_data(existing, all_names: list[str], train_names: list[str]):
    missing_old = [mn for mn in all_names if mn not in train_names and mn not in existing["act_phys"]]
    if missing_old:
        raise RuntimeError(
            "Incremental training requested, but saved data is missing for: "
            f"{missing_old}. Train all models once first, or point --out_dir to an output "
            "directory that already contains the non-retrained models."
        )


def _assert_complete_outputs(metadata, act_phys, act_unif, full_tangents, layer_tangents, saliency_all, correlations):
    for mn in metadata["model_order"]:
        if mn not in metadata["models"]:
            raise RuntimeError(f"Missing metadata for {mn}")
        if mn not in act_phys:
            raise RuntimeError(f"Missing physical activations for {mn}")
        if mn not in act_unif:
            raise RuntimeError(f"Missing uniform activations for {mn}")
        if mn not in full_tangents:
            raise RuntimeError(f"Missing tangent features for {mn}")
        if mn not in layer_tangents:
            raise RuntimeError(f"Missing layer tangent features for {mn}")
        if mn not in saliency_all:
            raise RuntimeError(f"Missing saliency for {mn}")
        if mn not in correlations:
            raise RuntimeError(f"Missing correlations for {mn}")


def run_experiment(cfg: ExperimentConfig, train_names: list[str], make_plots: bool = False):
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    hi, _graph, H = make_tfim(N=cfg.N, J=cfg.J, h=cfg.h, pbc=cfg.pbc)
    all_configs = make_configs()
    all_names = list(all_configs.keys())
    incremental = set(train_names) != set(all_names)

    print(f"\nActive models: {all_names}")
    print(f"Training models: {train_names}")

    metadata = base_metadata(cfg, all_names)
    existing_metadata = _load_existing_metadata(cfg) if incremental else None
    if existing_metadata:
        for mn in all_names:
            if mn not in train_names and mn in existing_metadata.get("models", {}):
                metadata["models"][mn] = existing_metadata["models"][mn]

    print("\nParameter counts:")
    for name, model in all_configs.items():
        dummy = jnp.ones((1, cfg.N))
        variables = model.init(jax.random.PRNGKey(0), dummy)
        n_params = count_params(variables["params"])
        marker = " *" if name in train_names else ""
        print(f"  {name:20s}  {n_params:>6d}{marker}")

    results = {}
    for seed, (name, model) in enumerate(all_configs.items()):
        if name not in train_names:
            continue
        print(f"\n{'=' * 40}")
        print(f"Training {name}")
        print(f"{'=' * 40}")

        results[name] = train_nqs(
            hi,
            H,
            model,
            seed=seed,
            n_samples=cfg.n_samples_train,
            n_iter=cfg.n_iter,
            lr=cfg.lr,
            use_sr=cfg.use_sr,
        )
        energy = results[name]["energy"]
        print(f"{name} final energy: {energy}")
        metadata["models"][name] = model_metadata(
            model,
            results[name]["vstate"].parameters,
            energy,
        )

    existing = {
        "act_phys": {},
        "act_unif": {},
        "tangents": {},
        "layer_tangents": {},
        "saliency": {},
        "correlations": {},
    }
    if incremental:
        existing = _load_existing_analysis_for_incremental(cfg, all_names, train_names)
        _assert_incremental_has_existing_data(existing, all_names, train_names)

    print("\nPreparing probe samples ...")
    samples_phys_old, samples_unif_old, samples_tang_old = load_existing_probe_samples(cfg)

    if incremental and samples_phys_old is not None:
        samples_phys = samples_phys_old[:cfg.n_activation_samples].astype("float32")
        print("  Reusing existing physical probe samples.")
    else:
        ref_name = cfg.ref_name if cfg.ref_name in results else train_names[0]
        samples_phys = get_shared_samples(results[ref_name]["vstate"], cfg.n_activation_samples)

    if incremental and samples_unif_old is not None:
        samples_unif = samples_unif_old[:cfg.n_uniform_probe_samples].astype("float32")
        print("  Reusing existing uniform probe samples.")
    else:
        samples_unif = make_uniform_spin_samples(cfg.N, cfg.n_uniform_probe_samples, seed=54321)

    if incremental and samples_tang_old is not None:
        samples_tang = samples_tang_old[:cfg.n_tangent_samples].astype("float32")
        print("  Reusing existing tangent probe samples.")
    else:
        samples_tang = samples_phys[:cfg.n_tangent_samples]

    samples_sal = samples_phys[:cfg.n_saliency_samples]

    print("\nExtracting activations on physical samples ...")
    act_phys = dict(existing["act_phys"])
    act_cka = {}
    for name, r in results.items():
        print(f"  {name}")
        act_phys[name] = extract_activations(
            r["model"],
            r["vstate"].parameters,
            samples_phys,
            cfg.activation_batch_size,
        )

    print("\nExtracting activations on uniform samples ...")
    act_unif = dict(existing["act_unif"])
    for name, r in results.items():
        print(f"  {name}")
        act_unif[name] = extract_activations(
            r["model"],
            r["vstate"].parameters,
            samples_unif,
            cfg.activation_batch_size,
        )

    for name in all_names:
        act_cka[name] = flatten_activations_for_cka(act_phys[name])

    print("\nLocal Hamiltonian-term decoding ...")
    local_phys = compute_local_term_decoding(act_phys, samples_phys, metadata, cfg)
    local_unif = compute_local_term_decoding(act_unif, samples_unif, metadata, cfg)
    local_all = {"physical": local_phys, "uniform": local_unif}

    print("\nMulti-distance z_i z_{i+d} decoding ...")
    md_phys = compute_multidistance_decoding(act_phys, samples_phys, metadata, cfg)
    md_unif = compute_multidistance_decoding(act_unif, samples_unif, metadata, cfg)
    multidist = {"physical": md_phys, "uniform": md_unif}

    print("\nExtracting tangent features ...")
    full_tangents = dict(existing["tangents"])
    layer_tangents = dict(existing["layer_tangents"])
    for name, r in results.items():
        print(f"\n  {name}")
        fJ, lJ = extract_tangent_features(
            r["model"],
            r["vstate"].parameters,
            samples_tang,
            cfg.jacobian_batch_size,
        )
        print(f"    full tangent: {fJ.shape}")
        full_tangents[name] = fJ
        layer_tangents[name] = lJ

    print("\nComputing input saliency ...")
    saliency_all = dict(existing["saliency"])
    for name, r in results.items():
        print(f"  {name}")
        saliency_all[name] = compute_input_saliency(
            r["model"],
            r["vstate"].parameters,
            samples_sal,
            cfg.saliency_batch_size,
        )

    print("\nComputing correlation functions ...")
    correlations = dict(existing["correlations"])
    for name, r in results.items():
        s = get_shared_samples(r["vstate"], cfg.n_activation_samples)
        correlations[name] = compute_zz_correlation(s, cfg.max_corr_distance, cfg.pbc)
        print(
            f"  {name}  "
            f"C(0)={correlations[name][0]:.4f}  "
            f"C(1)={correlations[name][1]:.4f}"
        )

    _assert_complete_outputs(metadata, act_phys, act_unif, full_tangents, layer_tangents, saliency_all, correlations)

    save_all_data(
        cfg,
        metadata,
        samples_phys,
        samples_unif,
        samples_tang,
        act_phys,
        act_unif,
        full_tangents,
        layer_tangents,
        saliency_all,
        local_all,
        multidist,
        correlations,
    )

    if make_plots:
        make_all_plots(
            cfg,
            metadata,
            act_cka,
            act_phys,
            full_tangents,
            layer_tangents,
            saliency_all,
            local_all,
            multidist,
            correlations,
        )
