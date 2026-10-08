"""Restricted loading for LMNDM PyTorch checkpoints."""

from __future__ import annotations

from pathlib import Path

import torch


def load_checkpoint(path, *, map_location="cpu"):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return torch.load(path, map_location=map_location, weights_only=True)


def strip_module_prefix(state_dict):
    return {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }
