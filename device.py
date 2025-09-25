from __future__ import annotations
import torch

def get_device(prefer: str | None = None) -> torch.device:
    """
    Deterministic device selector.
    - If prefer == 'cpu' -> always return CPU.
    - If prefer == 'cuda' and CUDA available -> CUDA.
    - If prefer == 'mps' and MPS available -> MPS.
    - If prefer is None -> pick CUDA > MPS > CPU.
    """
    if prefer is not None:
        p = prefer.lower()
        if p == "cpu":
            return torch.device("cpu")
        if p == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if p == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        # If user requested something unavailable, fall through to auto selection

    # Auto selection: prefer cuda, then mps, then cpu
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
