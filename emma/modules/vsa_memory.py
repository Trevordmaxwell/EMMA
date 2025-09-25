from __future__ import annotations
import math
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
    """Kanerva-style associative memory with bucket-localized writes/reads."""

    def __init__(
        self,
        dim: int,
        n_slots: int = 256,
        k_top: int = 16,
        decay: float = 0.995,
        bucket_count: int = 32,
        write_strength: float = 1.0,
        bucket_temp: float = 1.0,
        device: torch.device | None = None,
        write_norm_clip: float | None = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.n_slots = n_slots
        self.k_top = max(1, k_top)
        self.decay = float(decay)
        self.bucket_count = max(1, min(int(bucket_count), n_slots))
        self.write_strength = float(write_strength) if write_strength is not None else 1.0
        self.bucket_temp = float(bucket_temp) if bucket_temp else 1.0
        self.device = device if device is not None else torch.device('cpu')
        self.write_norm_clip = None if write_norm_clip is None else float(write_norm_clip)

        with torch.no_grad():
            addresses = torch.randn(n_slots, dim, device=self.device)
            addresses = _normalize(addresses)
        self.register_buffer('addresses', addresses)
        self.register_buffer('memory', torch.zeros(n_slots, dim, device=self.device))

        with torch.no_grad():
            bucket_proj = torch.randn(self.bucket_count, dim, device=self.device)
            bucket_proj = _normalize(bucket_proj)
        self.register_buffer('bucket_projections', bucket_proj)

        bucket_ids = torch.arange(n_slots, device=self.device, dtype=torch.long) % self.bucket_count
        self.register_buffer('bucket_ids', bucket_ids)
        self._bucket_slot_index: list[torch.Tensor] = []
        for b in range(self.bucket_count):
            idx = torch.nonzero(bucket_ids == b, as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                idx = torch.tensor([b % n_slots], device=self.device, dtype=torch.long)
            self._bucket_slot_index.append(idx)

        self._write_trace: list[dict] = []
        self._read_trace: list[dict] = []
        self._init_collision_trackers()

    def _init_collision_trackers(self) -> None:
        self._bucket_slot_usage: list[set[int]] = [set() for _ in range(self.bucket_count)]
        self._bucket_collision_counts: list[int] = [0 for _ in range(self.bucket_count)]
        self._bucket_write_counts: list[int] = [0 for _ in range(self.bucket_count)]
        self._collision_total: int = 0
        self._write_total: int = 0

    def reset(self) -> None:
        self.memory.zero_()
        self._write_trace = []
        self._read_trace = []
        self._init_collision_trackers()

    def _hash_bucket(self, key: torch.Tensor) -> torch.Tensor:
        key_n = _normalize(key)
        scores = torch.matmul(key_n, self.bucket_projections.t())
        return scores.argmax(dim=-1)

    def _bucket_topk(self, key_vec: torch.Tensor, bucket_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        bucket_slots = self._bucket_slot_index[bucket_id]
        addr_subset = self.addresses.index_select(0, bucket_slots)
        sims = F.cosine_similarity(addr_subset, key_vec.unsqueeze(0), dim=-1)
        k = min(self.k_top, sims.numel())
        if k <= 0:
            k = 1
        top_vals, top_idx = torch.topk(sims, k=k, dim=-1)
        selected = bucket_slots.index_select(0, top_idx)
        return selected, top_vals

    def write(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        strength: float = 1.0,
        seq_indices: torch.Tensor | None = None,
    ) -> None:
        """Write bound(key, value) within hashed buckets."""
        assert key.shape == value.shape
        key = key.to(self.device)
        value = value.to(self.device)
        if key.dim() == 1:
            key = key.unsqueeze(0)
            value = value.unsqueeze(0)
        B = key.size(0)
        if seq_indices is None:
            seq_idx = torch.arange(B, device=self.device, dtype=torch.long)
        else:
            seq_idx = seq_indices.to(self.device).view(-1)
            if seq_idx.numel() != B:
                raise ValueError('seq_indices must match batch size in write()')

        with torch.no_grad():
            key_n = _normalize(key)
            val_n = _normalize(value)
            bound = hrr_bind(key_n, val_n)
            scale = float(strength) * self.write_strength
            if scale == 0.0:
                scale = self.write_strength
            for i in range(B):
                bucket_id = int(self._hash_bucket(key_n[i:i+1]).item())
                bucket_slots = self._bucket_slot_index[bucket_id]
                current = self.memory.index_select(0, bucket_slots)
                current = current * self.decay
                self.memory.index_copy_(0, bucket_slots, current)
                idx, _ = self._bucket_topk(key_n[i], bucket_id)
                add_vec = bound[i] * scale
                if self.write_norm_clip is not None and math.isfinite(self.write_norm_clip):
                    norm = torch.linalg.norm(add_vec, ord=2)
                    if float(norm) > self.write_norm_clip + 1e-8:
                        add_vec = add_vec * (self.write_norm_clip / (float(norm) + 1e-8))
                self.memory.index_add_(0, idx, add_vec.unsqueeze(0).expand(idx.size(0), -1))
                indices = idx.tolist()
                usage = self._bucket_slot_usage[bucket_id]
                collisions = sum(1 for slot in indices if slot in usage)
                self._bucket_collision_counts[bucket_id] += collisions
                self._bucket_write_counts[bucket_id] += len(indices)
                self._collision_total += collisions
                self._write_total += len(indices)
                usage.update(indices)
                if idx.numel() > 0:
                    updated = self.memory.index_select(0, idx)
                    updated = _normalize(updated)
                    self.memory.index_copy_(0, idx, updated)
                self._write_trace.append({
                    'bucket': bucket_id,
                    'indices': idx.clone(),
                    'seq': int(seq_idx[i].item()),
                })

    def read(
        self,
        key: torch.Tensor,
        return_info: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]] | torch.Tensor:
        """Read by summing bucket-local address contents and unbinding with key."""
        key = key.to(self.device)
        single = False
        if key.dim() == 1:
            key = key.unsqueeze(0)
            single = True

        key_n = _normalize(key)
        outs = []
        buckets: list[int] = []
        entropies: list[torch.Tensor] = []
        indices_trace: list[torch.Tensor] = []
        for i in range(key_n.size(0)):
            bucket_id = int(self._hash_bucket(key_n[i:i+1]).item())
            idx, sims = self._bucket_topk(key_n[i], bucket_id)
            content = self.memory.index_select(0, idx).sum(dim=0)
            value = _normalize(hrr_unbind(content, key_n[i]))
            outs.append(value)
            buckets.append(bucket_id)
            if sims.numel() > 0:
                temp = self.bucket_temp if self.bucket_temp > 0 else 1.0
                probs = torch.softmax(sims / temp, dim=-1)
                entropy = -(probs * torch.log(probs.clamp_min(1e-9))).sum()
            else:
                entropy = torch.tensor(0.0, device=self.device)
            entropies.append(entropy)
            indices_trace.append(idx.clone())
            self._read_trace.append({
                'bucket': bucket_id,
                'entropy': float(entropy.item()),
                'indices': idx.clone(),
            })

        outs_tensor = torch.stack(outs, dim=0)
        if single and not return_info:
            outs_tensor = outs_tensor.squeeze(0)

        if return_info:
            bucket_tensor = torch.tensor(buckets, device=self.device, dtype=torch.long)
            entropy_tensor = torch.stack(entropies, dim=0)
            return outs_tensor, bucket_tensor, entropy_tensor, indices_trace
        return outs_tensor

    def address_sims(self, key: torch.Tensor) -> torch.Tensor:
        """Return cosine similarities to all addresses for diagnostics."""
        key_n = _normalize(key)
        if key_n.dim() == 1:
            sims = F.cosine_similarity(self.addresses, key_n.unsqueeze(0), dim=-1)
        else:
            sims = F.cosine_similarity(self.addresses.unsqueeze(0), key_n.unsqueeze(1), dim=-1)
        return sims

    def collision_rates(self) -> tuple[float, list[float]]:
        """Return overall and per-bucket collision rates for the current write trace."""
        if self._write_total <= 0:
            return 0.0, [0.0 for _ in range(self.bucket_count)]
        per_bucket = []
        for collisions, writes in zip(self._bucket_collision_counts, self._bucket_write_counts):
            if writes <= 0:
                per_bucket.append(0.0)
            else:
                per_bucket.append(collisions / float(writes))
        overall = self._collision_total / float(self._write_total)
        return overall, per_bucket

    def bucket_load_variance(self) -> float:
        if self._write_total <= 0:
            return 0.0
        counts = torch.tensor(self._bucket_write_counts, dtype=torch.float32, device=self.device)
        total = counts.sum()
        if total <= 0:
            return 0.0
        probs = counts / total
        var = torch.var(probs, unbiased=False)
        return float(var.item())
