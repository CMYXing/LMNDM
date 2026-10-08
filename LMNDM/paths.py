"""Paths to resources in a cloned LMNDM repository."""

from __future__ import annotations

import os
from pathlib import Path


def repository_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_model_root() -> Path:
    configured = os.environ.get("LMNDM_MODEL_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return repository_root() / "models"
