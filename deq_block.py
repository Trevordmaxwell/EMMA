from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm

try:
    import torchdeq
    _has_deq_attr = any(
        hasattr(torchdeq, name)
        for name in ("deq", "get_deq", "core", "DEQ", "DEQBase")
    )
    HAS_TORCHDEQ = _has_deq_attr
except Exception:
    torchdeq = None
    HAS_TORCHDEQ = False

def _coerce_to_tensor(maybe):
    """Return a torch.Tensor from various possible return types from DEQ call."""
    # If it's already a tensor, good.
    if isinstance(maybe, torch.Tensor):
        return maybe
    # If it's a list/tuple, pick the first tensor-like element
    if isinstance(maybe, (list, tuple)):
        for item in maybe:
            if isinstance(item, torch.Tensor):
                return item
        # if none are tensors, try converting the first element
        try:
            return torch.as_tensor(maybe[0])
        except Exception:
            pass
    # Try to convert general array-like / numeric to tensor
    try:
        return torch.as_tensor(maybe)
    except Exception:
        raise TypeError("Could not coerce DEQ return value to torch.Tensor")

class ResidualUpdate(nn.Module):
    def __init__(self, dim_z: int, dim_x: int, dim_v: int, hidden: int = 128, use_spectral_norm: bool = True):
        super().__init__()
        lin1 = nn.Linear(dim_z + dim_x + dim_v, hidden)
        lin2 = nn.Linear(hidden, dim_z)
        nn.init.xavier_uniform_(lin1.weight)
        nn.init.xavier_uniform_(lin2.weight)
        if use_spectral_norm:
            self.lin1 = spectral_norm(lin1)
            self.lin2 = spectral_norm(lin2)
        else:
            self.lin1 = lin1
            self.lin2 = lin2

    def forward(self, z: torch.Tensor, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([z, x, v], dim=-1)
        h = torch.tanh(self.lin1(inp))
        dz = self.lin2(h)
        return z + 0.5 * dz

class FixedPointBlock(nn.Module):
    def __init__(self, updater: ResidualUpdate, max_iter: int = 20, tol: float = 1e-4, relax: float = 0.5):
        super().__init__()
        self.updater = updater
        self.max_iter = max_iter
        self.tol = tol
        self.relax = relax

    def forward(self, z0: torch.Tensor, x: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, int]:
        if HAS_TORCHDEQ and torchdeq is not None:
            def f(z):
                return self.updater(z, x, v)

            deq_callable = None
            try:
                if hasattr(torchdeq, "deq"):
                    deq_callable = getattr(torchdeq, "deq")
                elif hasattr(torchdeq, "get_deq"):
                    candidate = getattr(torchdeq, "get_deq")
                    if callable(candidate):
                        try:
                            deq_callable = candidate()
                        except TypeError:
                            deq_callable = candidate
                elif hasattr(torchdeq, "core") and hasattr(torchdeq.core, "deq"):
                    deq_callable = getattr(torchdeq.core, "deq")
                elif hasattr(torchdeq, "DEQ"):
                    deq_callable = getattr(torchdeq, "DEQ")
            except Exception:
                deq_callable = None

            if callable(deq_callable):
                try:
                    # Try common signature first
                    out = deq_callable(f, z0, max_iter=self.max_iter, tol=self.tol, method="anderson")
                    # many DEQ APIs return (z_star, info) or sometimes just z_star or list-like
                    if isinstance(out, tuple) and len(out) == 2:
                        z_star, info = out
                    else:
                        # Could be z_star or [z_star, info]; try to coerce sensibly
                        if isinstance(out, (list, tuple)) and len(out) >= 1:
                            z_star = out[0]
                            info = out[1] if len(out) > 1 else None
                        else:
                            z_star = out
                            info = None
                    # coerce to tensor if needed
                    z_star = _coerce_to_tensor(z_star)
                    iters = int(getattr(info, "nstep", self.max_iter)) if info is not None else self.max_iter
                    return z_star, iters
                except TypeError:
                    # Try a simpler call pattern if signature differs
                    try:
                        out = deq_callable(f, z0)
                        if isinstance(out, tuple) and len(out) == 2:
                            z_star, info = out
                        else:
                            if isinstance(out, (list, tuple)) and len(out) >= 1:
                                z_star = out[0]
                                info = out[1] if len(out) > 1 else None
                            else:
                                z_star = out
                                info = None
                        z_star = _coerce_to_tensor(z_star)
                        iters = int(getattr(info, "nstep", self.max_iter)) if info is not None else self.max_iter
                        return z_star, iters
                    except Exception:
                        # fall through to iterative fallback
                        pass
                except Exception:
                    # Any other runtime error — fall back to iterative solver.
                    pass

        # Fallback iterative solver (Picard with relaxation)
        z = z0
        iters = 0
        last_res = None
        with torch.enable_grad():
            for k in range(self.max_iter):
                z_next = self.updater(z, x, v)
                z_new = (1.0 - self.relax) * z + self.relax * z_next
                last_res = torch.max(torch.abs(z_new - z))
                if last_res.item() < self.tol:
                    z = z_new
                    iters = k + 1
                    break
                z = z_new
                iters = k + 1
        try:
            self._last_residual = float(last_res.item()) if last_res is not None else float('nan')
        except Exception:
            self._last_residual = float('nan')
        return z, iters
