from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

from .models import apply_with_activations


def collect_activations(model: Any, params: Any, samples: Any) -> dict[str, np.ndarray]:
    """
    Run the model on a batch of spin samples and collect intermediate activations.

    What happens:
    - `samples` is converted to a JAX array so it can be passed through the model.
    - `apply_with_activations(...)` evaluates the model and returns named intermediate tensors.
    - Each activation tensor is converted back to a NumPy array with `float` dtype.
    - The result is a dictionary keyed by activation/layer name.

    Example:
        samples.shape == (2048, 64)

        activations = collect_activations(model, params, samples)

        # Possible output structure:
        {
            "embed": np.ndarray shape (2048, 64, features),
            "block0/post": np.ndarray shape (2048, 64, features),
            "block1/post": np.ndarray shape (2048, 64, features),
            ...
        }

    This helper gives downstream analyses, such as CKA or probing, a consistent
    NumPy-based activation dictionary independent of the model's internal JAX format.
    """
    jax_samples = jnp.asarray(samples)
    activation_tensors = apply_with_activations(model, params, jax_samples)
    return {
        layer_name: np.asarray(layer_activations, dtype=float)
        for layer_name, layer_activations in activation_tensors.items()
    }
