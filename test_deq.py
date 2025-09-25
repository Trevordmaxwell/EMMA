from __future__ import annotations
import torch
from src.modules.deq_block import ResidualUpdate, FixedPointBlock

def main():
    torch.manual_seed(0)
    B, Z, X, V = 4, 32, 16, 64
    updater = ResidualUpdate(dim_z=Z, dim_x=X, dim_v=V, hidden=64)
    deq = FixedPointBlock(updater, max_iter=10, tol=1e-5, relax=0.5)

    z0 = torch.zeros(B, Z)
    x = torch.randn(B, X)
    v = torch.randn(B, V)

    z_star, iters = deq(z0, x, v)
    print("Converged in steps:", iters)
    # simple loss to test backward
    loss = z_star.pow(2).mean()
    loss.backward()
    print("Backward ok; mean grad abs:", float(sum(p.grad.abs().mean() for p in updater.parameters()) / sum(1 for p in updater.parameters())))

if __name__ == '__main__':
    main()
