"""Compute device for the networks. `auto` = Apple GPU (MPS) when PyTorch can see it,
else CPU. The command sandbox hides Metal, so runs meant for the GPU must be launched
outside it; a run started inside it silently falls back to CPU (measured 2026-09-21:
training step 977 ms on CPU vs 38 ms on MPS for the 128-filter network)."""
import torch


def resolve_device(name: str = "auto") -> torch.device:
    name = (name or "auto").lower()
    if name == "auto":
        return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("device mps requested but MPS is not available (inside the sandbox?)")
    return torch.device(name)
