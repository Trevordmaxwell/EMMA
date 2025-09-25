from __future__ import annotations
import torch
from src.modules.vsa_memory import VSAMemory, _normalize, hrr_bind, hrr_unbind

def main():
    D = 512
    mem = VSAMemory(dim=D, n_slots=128, k_top=8, decay=0.999)
    K = 20
    keys = _normalize(torch.randn(K, D))
    vals = _normalize(torch.randn(K, D))

    # write pairs
    for i in range(K):
        mem.write(keys[i], vals[i])

    # read back and measure cosine
    recovered = torch.stack([mem.read(keys[i]) for i in range(K)], dim=0)
    cos = torch.nn.functional.cosine_similarity(recovered, vals, dim=-1)
    print("Mean cosine after write/read:", float(cos.mean()))
    print("Min / Max:", float(cos.min()), float(cos.max()))

if __name__ == '__main__':
    main()
