"""Runtime device and thread helpers."""

from __future__ import annotations

from functools import lru_cache
import os

import torch


@lru_cache(maxsize=1)
def mps_supports_3d() -> bool:
    """Return whether the local MPS backend supports the required 3-D ops."""
    if not torch.backends.mps.is_available():
        return False
    try:
        device = torch.device("mps")
        inputs = torch.zeros((1, 1, 4, 4, 4), device=device)
        conv = torch.nn.Conv3d(1, 2, kernel_size=3, padding=1).to(device)
        transpose = torch.nn.ConvTranspose3d(2, 1, kernel_size=2, stride=2).to(device)
        transpose(conv(inputs))
        torch.mps.synchronize()
        return True
    except (NotImplementedError, RuntimeError):
        return False


def resolve_device(device="auto") -> torch.device:
    """Resolve ``auto``, CPU, CUDA, or MPS."""
    if isinstance(device, torch.device):
        return device
    if device is None:
        device = "auto"

    value = str(device).strip().lower()
    if value == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if mps_supports_3d():
            return torch.device("mps")
        return torch.device("cpu")
    if value == "cpu":
        return torch.device("cpu")
    if value == "mps":
        if not mps_supports_3d():
            raise RuntimeError(
                "the installed PyTorch/MPS backend does not support the required "
                "3-D convolution operators; use --device cpu"
            )
        return torch.device("mps")
    if value == "cuda":
        value = "cuda:0"
    if value.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        gpu_id = int(value.split(":", 1)[1])
        if gpu_id >= torch.cuda.device_count():
            raise ValueError(
                f"requested CUDA device {gpu_id}, but only "
                f"{torch.cuda.device_count()} device(s) are available"
            )
        return torch.device(value)
    raise ValueError("device must be auto, cpu, mps, cuda, or cuda:<index>")


def configure_threads(num_threads: int | None = None) -> int:
    """Limit CPU threads for predictable laptop power and memory usage."""
    if num_threads is None:
        num_threads = min(4, os.cpu_count() or 1)
    num_threads = max(int(num_threads), 1)
    torch.set_num_threads(num_threads)
    try:
        import SimpleITK as sitk

        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(num_threads)
    except ImportError:
        pass
    return num_threads
