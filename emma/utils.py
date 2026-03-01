from __future__ import annotations
import os
import random
import numpy as np
import torch
import torch.nn.functional as F


def get_device(prefer: str | None = None) -> torch.device:
    """Deterministic device selector.
    - prefer == 'cpu' -> CPU; prefer == 'cuda' -> CUDA if available; prefer == 'mps' -> MPS if available.
    - If prefer is None, pick CUDA > MPS > CPU.
    """
    if prefer is not None:
        p = prefer.lower()
        if p == "cpu":
            return torch.device("cpu")
        if p == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if p == "mps" and getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
            return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    os.environ["PYTHONHASHSEED"] = str(seed)


def info_nce_loss(v_pred: torch.Tensor,
                  v_tgt: torch.Tensor,
                  temperature: float = 0.1) -> torch.Tensor:
    """In-batch InfoNCE with the diagonal as positives.

    Args:
        v_pred: (B, D) predicted vectors (grad-enabled)
        v_tgt:  (B, D) target vectors
        temperature: softmax temperature; defaults to 0.1.

    Returns:
        Scalar loss encouraging v_pred[i] to align with v_tgt[i] vs other targets in the batch.
    """
    if v_pred.ndim != 2 or v_tgt.ndim != 2:
        raise ValueError("info_nce_loss expects (B, D) tensors")
    if v_pred.size(0) != v_tgt.size(0) or v_pred.size(1) != v_tgt.size(1):
        raise ValueError("v_pred and v_tgt must have matching shape")

    v_pred = F.normalize(v_pred, dim=-1)
    v_tgt = F.normalize(v_tgt, dim=-1)
    logits = (v_pred @ v_tgt.t()) / max(temperature, 1e-6)
    labels = torch.arange(v_pred.size(0), device=v_pred.device)
    return F.cross_entropy(logits, labels)
