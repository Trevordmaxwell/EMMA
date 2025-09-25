from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

def _normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)

def hrr_bind(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    A = torch.fft.rfft(a, dim=-1)
    B = torch.fft.rfft(b, dim=-1)
    C = A * B
    c = torch.fft.irfft(C, n=a.shape[-1], dim=-1)
    return c

def hrr_unbind(c: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    A = torch.fft.rfft(a, dim=-1)
    C = torch.fft.rfft(c, dim=-1)
    B_hat = C * torch.conj(A) / (A.abs()**2 + 1e-8)
    b_hat = torch.fft.irfft(B_hat, n=a.shape[-1], dim=-1)
    return b_hat

class VSAMemory(nn.Module):
    """Kanerva-style localized associative memory with HRR binding."""
    def __init__(self, dim: int, n_slots: int = 256, k_top: int = 16, decay: float = 0.995, device: torch.device | None = None):
        super().__init__()
        self.dim = dim
        self.n_slots = n_slots
        self.k_top = k_top
        self.decay = decay
        self.device = device if device is not None else torch.device('cpu')
        with torch.no_grad():
            addresses = torch.randn(n_slots, dim, device=self.device)
            addresses = _normalize(addresses)
        self.register_buffer('addresses', addresses)
        self.register_buffer('memory', torch.zeros(n_slots, dim, device=self.device))

    def reset(self) -> None:
        self.memory.zero_()

    def _select_indices(self, key: torch.Tensor) -> torch.Tensor:
        if key.dim() == 1:
            sims = F.cosine_similarity(self.addresses, key.unsqueeze(0), dim=-1)
            idx = torch.topk(sims, k=self.k_top, dim=-1).indices
        else:
            sims = F.cosine_similarity(self.addresses.unsqueeze(0), key.unsqueeze(1), dim=-1)
            idx = torch.topk(sims, k=self.k_top, dim=-1).indices
        return idx

    def write(self, key: torch.Tensor, value: torch.Tensor, strength: float = 1.0) -> None:
        """Write bound(key, value) to top-k addresses. key/value: (D,) or (B,D)."""
        assert key.shape == value.shape
        key_n = _normalize(key)
        val_n = _normalize(value)
        bound = hrr_bind(key_n, val_n)
        idx = self._select_indices(key_n.detach())
        with torch.no_grad():
            self.memory.mul_(self.decay)
            if bound.dim() == 1:
                self.memory.index_add_(0, idx, bound.detach().unsqueeze(0).expand(self.k_top, -1) * strength)
            else:
                B = bound.shape[0]
                for b in range(B):
                    self.memory.index_add_(0, idx[b], bound[b].detach().unsqueeze(0).expand(self.k_top, -1) * strength)

    def read(self, key: torch.Tensor) -> torch.Tensor:
        """Read by summing selected address contents and unbinding with key."""
        key_n = _normalize(key)
        idx = self._select_indices(key_n)
        if key_n.dim() == 1:
            content = self.memory.index_select(0, idx).sum(dim=0)
            return _normalize(hrr_unbind(content, key_n))
        else:
            outs = []
            for b in range(key_n.shape[0]):
                content = self.memory.index_select(0, idx[b]).sum(dim=0)
                outs.append(_normalize(hrr_unbind(content, key_n[b])))
            return torch.stack(outs, dim=0)
