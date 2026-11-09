"""Strict loading for official model_state/net.module.* and simplified checkpoints."""

import random
from collections import OrderedDict

import numpy as np
import torch


def read_checkpoint(path):
    # Full training checkpoints contain RNG/optimizer state. Load trusted files only.
    return torch.load(path, map_location="cpu", weights_only=False)


def model_state(checkpoint):
    state = checkpoint
    for key in ("model_state", "model", "state_dict"):
        if isinstance(checkpoint, dict) and key in checkpoint:
            state = checkpoint[key]
            break
    if not isinstance(state, dict):
        raise TypeError("Checkpoint does not contain a model state dictionary")
    result = OrderedDict()
    for key, value in state.items():
        while key.startswith(("module.", "net.")):
            key = key.split(".", 1)[1]
        if key in result:
            raise ValueError(f"Duplicate parameter after prefix conversion: {key}")
        result[key] = value
    return result


def load_weights(model, path):
    checkpoint = read_checkpoint(path)
    model.load_state_dict(model_state(checkpoint), strict=True)
    return checkpoint


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
