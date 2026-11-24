"""Resolve UI and CLI device choices before any Kitchen operation."""

import re


def normalize_device(value):
    value = str(value or "auto").strip().lower()
    return {"automatic": "auto", "gpu (cuda)": "cuda", "gpu": "cuda"}.get(value, value)


def resolve_device(value):
    value = normalize_device(value)
    if value == "auto":
        import torch
        # Use the same CUDA target as the explicit GPU option. Do not pass the
        # UI's sentinel string to Torch, Kitchen, or tensor.to().
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value != "cpu" and not re.fullmatch(r"cuda(?::\d+)?", value):
        raise ValueError("Device must be Automatic, CPU, GPU (CUDA), or cuda:N")
    return value
