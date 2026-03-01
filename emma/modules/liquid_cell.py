from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class LiquidCell(nn.Module):
    """A tiny 'liquid' continuous-time RNN-like cell with learned time-constant.
    h_{t+1} = h_t + dt * ( -h_t / tau(x,h) + tanh(W_in x + W_rec h + b) )
    tau(x,h) = softplus( W_tau [x;h] + b_tau ) + tau_min
    """
    def __init__(self, dim_in: int, dim_hid: int, dt: float = 1.0, tau_min: float = 0.1):
        super().__init__()
        self.W_in = nn.Linear(dim_in, dim_hid)
        self.W_rec = nn.Linear(dim_hid, dim_hid, bias=False)
        self.b = nn.Parameter(torch.zeros(dim_hid))
        self.W_tau = nn.Linear(dim_in + dim_hid, dim_hid)
        self.dt = dt
        self.tau_min = tau_min
        # Ensure tau_min is not too small relative to dt to avoid unstable updates
        if self.tau_min < self.dt * 1.01:
            self.tau_min = float(self.dt * 1.01)
        nn.init.xavier_uniform_(self.W_in.weight)
        nn.init.xavier_uniform_(self.W_rec.weight)
        nn.init.xavier_uniform_(self.W_tau.weight)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        pre = torch.tanh(self.W_in(x) + self.W_rec(h) + self.b)
        tau = F.softplus(self.W_tau(torch.cat([x, h], dim=-1))) + self.tau_min
        # safety: ensure tau is at least slightly larger than dt (avoid division blowups)
        tau = torch.clamp(tau, min=self.dt * 1.01)
        # compute update in a numerically clearer way
        dh = -h / (tau + 1e-8) + pre
        h_next = h + self.dt * dh
        # sanitize improbable values (final safety - keeps training alive while we debug)
        h_next = torch.nan_to_num(h_next, nan=0.0, posinf=1e6, neginf=-1e6)
        h_next = h_next.clamp(-1e3, 1e3)
        return h_next
