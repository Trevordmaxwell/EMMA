from __future__ import annotations
import math
import torch
from typing import Optional
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm

from emma.modules.vsa_memory import VSAMemory, _normalize, hrr_bind
from emma.modules.deq_block import ResidualUpdate, FixedPointBlock
from emma.modules.liquid_cell import LiquidCell
from emma.utils import info_nce_loss

class EMMA(nn.Module):
    """
    EMMA: Equilibrium + VSA Memory + (Liquid) backbone.

    - Non-diff external memory (writes under no_grad).
    - Write-step supervision uses TRUE value ids (value_ids).
    - Aux prediction head at query.
    - Learnable logit scale for CE calibration.
    - Optional memory-into-DEQ at query (non-diff read).
    - Reports write_cos (mean cosine at writes) and num_write_steps per forward.
    """
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        hid_dim: int,
        mem_dim: int,
        num_values: int,
        n_slots: int = 256,
        k_top: int = 16,
        oracle_write: bool = False,
        deq_max_iter: int = 15,
        mem_into_deq: bool = False,
        mem_scale: float = 1.0,
        warm_start_epochs: int = 0,
        bucket_count: int = 32,
        write_decay: float = 0.995,
        write_strength: float = 1.0,
        bucket_temp: float = 1.0,
        max_writes_per_seq: int | None = None,
        use_spectral_norm: bool = True,
        spectral_norm_keys: bool = False,
        value_dropout: float = 0.0,
        cleanup_blend_raw: float | None = None,
        cleanup_blend_clean: float | None = None,
        write_norm_clip: float | None = None,
        memory_device: str | torch.device | None = 'cpu',
    ):
        super().__init__()
        self.oracle_write = oracle_write
        self.mem_into_deq = mem_into_deq
        self.mem_scale = mem_scale
        self.warm_start_epochs = warm_start_epochs
        self.max_writes_per_seq = max_writes_per_seq
        # Runtime knobs set by trainer
        self.oracle_mix_alpha: float = 0.0
        self.logit_scale_max: Optional[float] = None
        self.memory_device = torch.device(memory_device) if memory_device is not None else torch.device('cpu')

        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.key_embed = nn.Embedding(num_values, mem_dim)
        self.value_embed = nn.Embedding(num_values, mem_dim)
        nn.init.normal_(self.value_embed.weight, std=0.1)
        nn.init.normal_(self.key_embed.weight, std=0.1)

        self.updater = ResidualUpdate(
            dim_z=hid_dim,
            dim_x=emb_dim,
            dim_v=mem_dim,
            hidden=max(128, hid_dim*2),
            use_spectral_norm=use_spectral_norm,
        )
        self.deq = FixedPointBlock(self.updater, max_iter=deq_max_iter, tol=1e-4, relax=0.5)

        self.liquid = LiquidCell(dim_in=hid_dim, dim_hid=hid_dim)
        self.h0 = nn.Parameter(torch.zeros(hid_dim))

        self.memory = VSAMemory(
            dim=mem_dim,
            n_slots=n_slots,
            k_top=k_top,
            decay=write_decay,
            bucket_count=bucket_count,
            write_strength=write_strength,
            bucket_temp=bucket_temp,
            device=self.memory_device,
            write_norm_clip=write_norm_clip,
        )
        self.z_to_value = nn.Linear(hid_dim*2, mem_dim)
        # Classification helper: derive a memory key from hidden state
        self.cls_key = nn.Linear(hid_dim, mem_dim)
        if spectral_norm_keys:
            self.z_to_value = spectral_norm(self.z_to_value)
            self.cls_key = spectral_norm(self.cls_key)

        self._logit_scale_raw = nn.Parameter(torch.tensor(2.0))  # softplus(2) ~ 2.13
        self.read_cleanup_topk = 1
        self.enable_read_cleanup = True
        self.read_sharpen_topk: int | None = None
        self.read_sharpen_temp: float | None = None
        self.read_sharpen_eval_only = False
        self.read_sharpen_mask_value: float | None = None
        self.read_sharpen_mask_margin: float = 5.0
        self.cleanup_softmax = False
        self.cleanup_temp = 1.0
        self.value_dropout_p = max(0.0, float(value_dropout)) if value_dropout is not None else 0.0
        self.cleanup_blend_raw = cleanup_blend_raw
        self.cleanup_blend_clean = cleanup_blend_clean
        self.post_gate_write_cooldown_batches: int = 0
        self._write_cooldown_batches_remaining: int = 0
        self._write_cooldown_step_counter: int = 0

    @staticmethod
    def _apply_read_sharpen(
        logits: torch.Tensor,
        topk: int | None,
        temperature: float | None,
        mask_value: float | None,
        mask_margin: float | None,
        training: bool,
        eval_only: bool,
    ) -> torch.Tensor:
        if logits is None:
            return logits
        out = logits
        if temperature is not None:
            try:
                temp = float(temperature)
            except Exception:
                temp = None
            if temp is not None and temp > 0.0 and not math.isclose(temp, 1.0):
                out = out / temp
        if eval_only and training:
            return logits
        if topk is not None:
            try:
                k = int(topk)
            except Exception:
                k = None
            if k is not None and 0 < k < out.size(-1):
                mask = torch.ones_like(out, dtype=torch.bool)
                top_idx = torch.topk(out, k=k, dim=-1).indices
                mask.scatter_(dim=-1, index=top_idx, value=False)
                mask_fill = mask_value
                if mask_fill is None:
                    margin = mask_margin if mask_margin is not None else 5.0
                    floor = (out.min(dim=-1, keepdim=True)[0] - margin).expand_as(out)
                    out = torch.where(mask, floor, out)
                else:
                    if not math.isfinite(mask_fill):
                        mask_fill = -1e9
                    out = out.masked_fill(mask, mask_fill)
        return out

    @property
    def logit_scale(self) -> torch.Tensor:
        s = F.softplus(self._logit_scale_raw) + 1e-3
        max_s = getattr(self, 'logit_scale_max', None)
        if max_s is not None:
            try:
                s = torch.clamp(s, max=float(max_s))
            except Exception:
                pass
        return s

    def forward(
        self,
        tokens: torch.Tensor,      # (B, L)
        key_ids: torch.Tensor,     # (B,)
        write_pos: torch.Tensor,   # (B,)
        query_pos: torch.Tensor,   # (B,)
        value_ids: Optional[torch.Tensor] = None,  # (B,) TRUE value id (label)
        current_epoch: int = 0,
        disable_write: bool = False,
        shuffle_read: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        B, L = tokens.shape
        device = tokens.device
        use_oracle = self.oracle_write

        self.memory.reset()
        self.memory.to(self.memory_device)
        h = self.h0.unsqueeze(0).expand(B, -1)
        z = torch.zeros(B, h.size(1), device=device)

        x = self.embed(tokens)
        key_vecs = _normalize(self.key_embed(key_ids))
        V_proto = _normalize(self.value_embed.weight)
        scale = self.logit_scale

        max_writes_limit = getattr(self, 'max_writes_per_seq', None)
        if max_writes_limit is not None and max_writes_limit > 0:
            writes_remaining = torch.full((B,), int(max_writes_limit), device=device, dtype=torch.long)
        else:
            writes_remaining = None
        writes_executed = torch.zeros(B, device=device, dtype=torch.float32)
        writes_skipped = torch.zeros((), device=device)

        bucket_entropy_sum = torch.zeros((), device=device)
        bucket_entropy_count = torch.zeros((), device=device)
        bucket_usage = torch.zeros(getattr(self.memory, 'bucket_count', 1), device=device, dtype=torch.float32)

        logits_main = None
        logits_pred_full = None
        aux_write_loss = torch.zeros((), device=device)
        aux_nce_loss = torch.zeros((), device=device)
        fp_iters_total = 0

        # NEW: direct write-cos + read-cos accounting
        write_cos_sum = torch.zeros((), device=device)
        write_step_count = torch.zeros((), device=device)
        read_cos_sum = torch.zeros((), device=device)
        read_cos_raw_sum = torch.zeros((), device=device)
        read_cleanup_cos_sum = torch.zeros((), device=device)
        read_step_count = torch.zeros((), device=device)
        read_gap_sum = torch.zeros((), device=device)

        write_samples: list[dict[str, float | int]] = []
        read_samples: list[dict[str, float | int]] = []

        for t in range(L):
            x_t = x[:, t, :]

            if self.mem_into_deq and (t == query_pos).any():
                mask_q = (t == query_pos)
                v_read_local = torch.zeros(B, self.memory.dim, device=device)
                mem_keys = _normalize(key_vecs[mask_q]).detach().to(self.memory_device)
                mem_inject = self.memory.read(mem_keys).to(device)
                if self.enable_read_cleanup:
                    mem_inject, _, _ = self._cleanup_read(mem_inject, V_proto)
                mem_inject = F.normalize(mem_inject, dim=-1)
                v_read_local[mask_q] = mem_inject
                v_t = v_read_local * self.mem_scale
            else:
                v_t = torch.zeros(B, self.memory.dim, device=device)

            z, nstep = self.deq(z, x_t, v_t)
            fp_iters_total += nstep

            h = self.liquid(z, h)

            zh = torch.cat([z, h], dim=-1)
            v_pred_raw = self.z_to_value(zh)
            if self.value_dropout_p > 0.0:
                v_pred_raw = F.dropout(v_pred_raw, p=self.value_dropout_p, training=self.training)
            v_pred = _normalize(v_pred_raw)

            # WRITE
            if (t == write_pos).any():
                mask_w = (t == write_pos)
                allowed_mask = mask_w
                cooldown_active = getattr(self, '_write_cooldown_batches_remaining', 0) > 0
                if writes_remaining is not None:
                    allowed_mask = mask_w & (writes_remaining > 0)
                    writes_skipped = writes_skipped + (mask_w.sum() - allowed_mask.sum()).to(device)
                if cooldown_active:
                    writes_skipped = writes_skipped + allowed_mask.sum().to(device)
                    allowed_mask = allowed_mask & torch.zeros_like(allowed_mask)
                if not allowed_mask.any():
                    continue
                seq_idx = allowed_mask.nonzero(as_tuple=False).squeeze(-1)
                # true value at write
                if value_ids is not None:
                    v_true = _normalize(self.value_embed(value_ids[allowed_mask]))
                else:
                    v_true = _normalize(self.value_embed(key_ids[allowed_mask]))
                if use_oracle:
                    v_write = v_true
                    write_cos_sum = write_cos_sum + seq_idx.numel()
                    write_step_count = write_step_count + seq_idx.numel()
                    seq_cpu = seq_idx.detach().cpu()
                    for seq_val in seq_cpu.tolist():
                        write_samples.append({
                            'seq': int(seq_val),
                            'time': int(t),
                            'cos': 1.0,
                        })
                else:
                    # Optional oracle/predicted mix controlled by trainer via self.oracle_mix_alpha
                    alpha = float(getattr(self, 'oracle_mix_alpha', 0.0) or 0.0)
                    if alpha > 0.0:
                        v_write = _normalize(alpha * v_true + (1.0 - alpha) * v_pred[allowed_mask])
                    else:
                        v_write = v_pred[allowed_mask]
                    # Aux losses computed against predicted vector vs true (unchanged)
                    cos = torch.cosine_similarity(v_pred[allowed_mask], v_true, dim=-1)
                    aux_write_loss = aux_write_loss + (1.0 - cos).mean()
                    write_cos_sum = write_cos_sum + cos.sum()
                    write_step_count = write_step_count + cos.numel()
                    seq_cpu = seq_idx.detach().cpu()
                    cos_cpu = cos.detach().cpu()
                    for seq_val, cos_val in zip(seq_cpu.tolist(), cos_cpu.tolist()):
                        write_samples.append({
                            'seq': int(seq_val),
                            'time': int(t),
                            'cos': float(cos_val),
                        })
                    # InfoNCE write loss using HRR-bound targets as positives
                    if value_ids is not None:
                        key_w = _normalize(key_vecs[allowed_mask])
                        v_true_w = _normalize(v_true)
                        v_pred_w = _normalize(v_pred[allowed_mask])
                        v_true_bound = hrr_bind(key_w, v_true_w)
                        v_pred_bound = hrr_bind(key_w, v_pred_w)
                        temp = float(getattr(self, 'nce_temperature', 0.1) or 0.1)
                        aux_nce = info_nce_loss(v_pred_bound, v_true_bound, temperature=temp)
                        aux_nce_loss = aux_nce_loss + aux_nce
                # external memory write (optional disable during eval)
                if not disable_write:
                    key_w = _normalize(key_vecs[allowed_mask]).detach().to(self.memory_device)
                    self.memory.write(
                        key_w,
                        v_write.detach().to(self.memory_device),
                        seq_indices=seq_idx.detach().to(self.memory_device),
                    )
                if writes_remaining is not None:
                    writes_remaining[seq_idx] = torch.clamp(writes_remaining[seq_idx] - 1, min=0)
                writes_executed[seq_idx] += 1.0

            # QUERY / CLASSIFY
            if (t == query_pos).any():
                mask_q = (t == query_pos)
                seq_idx = mask_q.nonzero(as_tuple=False).squeeze(-1)
                # Optionally shuffle keys before read (eval-time causal test)
                if shuffle_read:
                    kv = key_vecs[mask_q]
                    if kv.shape[0] > 1:
                        perm = torch.randperm(kv.shape[0], device=kv.device)
                        kv = kv[perm]
                    v_mem_raw, bucket_ids, bucket_entropies, _ = self.memory.read(
                        _normalize(kv).detach().to(self.memory_device),
                        return_info=True,
                    )
                else:
                    v_mem_raw, bucket_ids, bucket_entropies, _ = self.memory.read(
                        _normalize(key_vecs[mask_q]).detach().to(self.memory_device),
                        return_info=True,
                    )

                bucket_ids_cpu = bucket_ids.detach().to('cpu')
                bucket_entropies_cpu = bucket_entropies.detach().to('cpu')

                v_mem_raw = v_mem_raw.to(device)
                bucket_entropies_device = bucket_entropies.to(device)
                bucket_entropy_sum = bucket_entropy_sum + bucket_entropies_device.sum()
                bucket_entropy_count = bucket_entropy_count + bucket_entropies.numel()
                bucket_ids_device = bucket_ids.to(device)
                if bucket_ids_device.numel() > 0 and bucket_usage.numel() >= bucket_ids_device.numel():
                    bucket_usage.index_add_(
                        0,
                        bucket_ids_device,
                        torch.ones(bucket_ids_device.size(0), device=device, dtype=torch.float32),
                    )
                addr_ent = bucket_entropies_device.mean() if bucket_entropies_device.numel() > 0 else torch.tensor(float('nan'), device=device)

                if self.enable_read_cleanup:
                    v_mem_clean, v_mem_norm, cleanup_sim = self._cleanup_read(v_mem_raw, V_proto)
                else:
                    v_mem_norm = F.normalize(v_mem_raw, dim=-1)
                    v_mem_clean = v_mem_norm
                    cleanup_sim = torch.ones(v_mem_norm.shape[0], device=v_mem_norm.device)

                if self.cleanup_blend_raw is not None and self.cleanup_blend_clean is not None:
                    alpha_raw = float(self.cleanup_blend_raw)
                    alpha_clean = float(self.cleanup_blend_clean)
                    blend = alpha_raw + alpha_clean
                    if blend == 0.0:
                        blend = 1.0
                    v_mem_for_logits = F.normalize(alpha_raw * v_mem_norm + alpha_clean * v_mem_clean, dim=-1)
                else:
                    v_mem_for_logits = v_mem_clean

                logits_q_mem  = scale * (v_mem_for_logits @ V_proto.t())
                logits_q_mem = self._apply_read_sharpen(
                    logits_q_mem,
                    getattr(self, 'read_sharpen_topk', None),
                    getattr(self, 'read_sharpen_temp', None),
                    getattr(self, 'read_sharpen_mask_value', None),
                    getattr(self, 'read_sharpen_mask_margin', None),
                    self.training,
                    getattr(self, 'read_sharpen_eval_only', False),
                )
                logits_q_pred = scale * (F.normalize(v_pred[mask_q], dim=-1) @ V_proto.t())

                if logits_main is None:
                    C = V_proto.size(0)
                    logits_main = torch.zeros(B, C, device=device)
                    logits_pred_full = torch.zeros(B, C, device=device)
                logits_main[mask_q] = logits_q_mem
                logits_pred_full[mask_q] = logits_q_pred

                # read alignment vs true value at query
                if value_ids is not None:
                    v_true_q = _normalize(self.value_embed(value_ids[mask_q]))
                    v_mem_norm = F.normalize(v_mem_norm, dim=-1)
                    rc_raw = torch.sum(v_mem_norm * v_true_q, dim=-1)
                    rc_clean = torch.sum(v_mem_clean * v_true_q, dim=-1)
                    read_cos_sum = read_cos_sum + rc_clean.mean()
                    read_cos_raw_sum = read_cos_raw_sum + rc_raw.mean()
                    read_cleanup_cos_sum = read_cleanup_cos_sum + cleanup_sim.mean()
                    read_gap_sum = read_gap_sum + (cleanup_sim - rc_raw).mean()
                    read_step_count = read_step_count + 1.0

                    seq_cpu = seq_idx.detach().cpu()
                    rc_clean_cpu = rc_clean.detach().cpu()
                    rc_raw_cpu = rc_raw.detach().cpu()
                    cleanup_cpu = cleanup_sim.detach().cpu()
                    bucket_ids_list = bucket_ids_cpu
                    bucket_entropy_list = bucket_entropies_cpu
                    for i, seq_val in enumerate(seq_cpu.tolist()):
                        record = {
                            'seq': int(seq_val),
                            'time': int(t),
                            'cos': float(rc_clean_cpu[i].item()),
                            'cos_raw': float(rc_raw_cpu[i].item()),
                            'cleanup': float(cleanup_cpu[i].item()),
                            'bucket': int(bucket_ids_list[i].item()) if bucket_ids_list.numel() > i else None,
                            'entropy': float(bucket_entropy_list[i].item()) if bucket_entropy_list.numel() > i else None,
                        }
                        read_samples.append(record)

                # addressing entropy already captured via bucket_entropies

        # Safe averages
        denom = torch.clamp(write_step_count, min=1.0)
        aux_nce_loss = aux_nce_loss / denom
        avg_write_cos = (write_cos_sum / torch.clamp_min(write_step_count, 1.0)).detach()
        avg_read_cos = (read_cos_sum / torch.clamp_min(read_step_count, 1.0)).detach()
        avg_read_cos_raw = (read_cos_raw_sum / torch.clamp_min(read_step_count, 1.0)).detach()
        avg_read_cleanup = (read_cleanup_cos_sum / torch.clamp_min(read_step_count, 1.0)).detach()
        avg_read_gap = (read_gap_sum / torch.clamp_min(read_step_count, 1.0)).detach()
        avg_bucket_entropy = (bucket_entropy_sum / torch.clamp_min(bucket_entropy_count, 1.0)).detach()
        writes_per_seq_mean = writes_executed.mean().detach()
        writes_per_seq_max = writes_executed.max().detach()
        write_bucket_entropy = torch.tensor(float('nan'), device=device)
        bucket_coverage = 0
        write_trace = getattr(self.memory, '_write_trace', None)
        if isinstance(write_trace, list) and write_trace:
            counts: dict[int, int] = {}
            for entry in write_trace:
                bucket = int(entry.get('bucket', -1))
                counts[bucket] = counts.get(bucket, 0) + 1
            total_writes = float(sum(counts.values()))
            if total_writes > 0:
                probs = torch.tensor([c / total_writes for c in counts.values()], device=device)
                write_bucket_entropy = -(probs * torch.log(probs.clamp_min(1e-9))).sum()
                bucket_coverage = len(counts)
        read_bucket_active = int((bucket_usage > 0).sum().item()) if bucket_usage.numel() > 0 else 0
        collision_overall, collision_per_bucket = self.memory.collision_rates()
        load_variance = float(self.memory.bucket_load_variance())
        if collision_per_bucket:
            collision_mean = float(sum(collision_per_bucket) / len(collision_per_bucket))
            collision_max = float(max(collision_per_bucket))
            collision_min = float(min(collision_per_bucket))
        else:
            collision_mean = 0.0
            collision_max = 0.0
            collision_min = 0.0
        metrics = {
            "avg_fp_iters": fp_iters_total / max(1, L),
            "aux_logits": logits_pred_full,
            "aux_loss": aux_write_loss,
            "aux_nce_loss": aux_nce_loss,
            "write_cos": avg_write_cos,              # mean cosine at write steps this forward
            "read_cos": avg_read_cos,                # mean cosine at query reads this forward
            "read_cos_raw": avg_read_cos_raw,
            "read_cleanup_cos": avg_read_cleanup,
            "read_cleanup_gap": avg_read_gap,
            "num_write_steps": int(write_step_count.item()),
            "addr_entropy": avg_bucket_entropy,
            "bucket_entropy": avg_bucket_entropy,
            "write_bucket_entropy": write_bucket_entropy.detach(),
            "bucket_usage_active": bucket_coverage,
            "read_bucket_active": read_bucket_active,
            "writes_per_seq_mean": writes_per_seq_mean,
            "writes_per_seq_max": writes_per_seq_max,
            "writes_skipped": writes_skipped.detach(),
            "residual_norm": (float(getattr(self.deq, "_last_residual", float('nan'))) if hasattr(self, "deq") else float('nan')),
            "bucket_collision_rate": float(collision_overall),
            "bucket_collision_rate_mean": collision_mean,
            "bucket_collision_rate_max": collision_max,
            "bucket_collision_rate_min": collision_min,
            "bucket_load_variance": load_variance,
        }
        if write_samples:
            metrics['write_samples'] = write_samples
        if read_samples:
            metrics['read_samples'] = read_samples
        return logits_main, metrics


    def forward_classify(
        self,
        tokens: torch.Tensor,      # (B, L)
        labels: Optional[torch.Tensor] = None,  # (B,) class id 0..C-1
        current_epoch: int = 0,
    ) -> tuple[torch.Tensor, dict]:
        """Sequence classification mode (e.g., ListOps-lite).
        Writes a class vector per step (oracle/pred mix per schedule), reads at EOS.
        """
        B, L = tokens.shape
        device = tokens.device
        self.memory.reset()
        self.memory.to(self.memory_device)
        h = self.h0.unsqueeze(0).expand(B, -1)
        z = torch.zeros(B, h.size(1), device=device)

        x = self.embed(tokens)
        V_proto = _normalize(self.value_embed.weight)  # (C, D)
        scale = self.logit_scale

        aux_write_loss = torch.zeros((), device=device)
        aux_nce_loss = torch.zeros((), device=device)
        fp_iters_total = 0
        write_cos_sum = torch.zeros((), device=device)
        write_step_count = torch.zeros((), device=device)

        for t in range(L):
            x_t = x[:, t, :]
            v_t = torch.zeros(B, self.memory.dim, device=device)
            z, nstep = self.deq(z, x_t, v_t)
            fp_iters_total += nstep
            h = self.liquid(z, h)
            zh = torch.cat([z, h], dim=-1)
            v_pred_raw = self.z_to_value(zh)
            if self.value_dropout_p > 0.0:
                v_pred_raw = F.dropout(v_pred_raw, p=self.value_dropout_p, training=self.training)
            v_pred = _normalize(v_pred_raw)

            # WRITE: per-step write using mixed oracle/pred schedule if labels available
            if labels is not None:
                v_true = _normalize(self.value_embed(labels))  # (B, D)
                alpha = float(getattr(self, 'oracle_mix_alpha', 0.0) or 0.0)
                if getattr(self, 'oracle_write', False):
                    alpha = 1.0
                if alpha > 0.0:
                    v_write = _normalize(alpha * v_true + (1.0 - alpha) * v_pred)
                else:
                    v_write = v_pred
                # aux losses / cos tracking
                cos = torch.cosine_similarity(v_pred, v_true, dim=-1)
                aux_write_loss = aux_write_loss + (1.0 - cos).mean()
                write_cos_sum = write_cos_sum + cos.mean()
                write_step_count = write_step_count + 1.0
            else:
                v_write = v_pred

            # write with key from hidden state at step t (external memory)
            k_t = _normalize(self.cls_key(h.detach()))
            seq_idx = torch.arange(B, device=device, dtype=torch.long)
            self.memory.write(
                k_t.detach().to(self.memory_device),
                v_write.detach().to(self.memory_device),
                seq_indices=seq_idx.detach().to(self.memory_device),
            )

        # READ at EOS using key from final hidden
        k_read = _normalize(self.cls_key(h.detach()))
        v_mem_raw = self.memory.read(k_read.detach().to(self.memory_device)).to(device)
        if self.enable_read_cleanup:
            v_mem_clean, v_mem_norm, cleanup_sim = self._cleanup_read(v_mem_raw, V_proto)
        else:
            v_mem_norm = F.normalize(v_mem_raw, dim=-1)
            v_mem_clean = v_mem_norm
            cleanup_sim = torch.ones(v_mem_clean.shape[0], device=v_mem_clean.device)
        logits = scale * (v_mem_clean @ V_proto.t())  # (B,C)
        logits = self._apply_read_sharpen(
            logits,
            getattr(self, 'read_sharpen_topk', None),
            getattr(self, 'read_sharpen_temp', None),
            getattr(self, 'read_sharpen_mask_value', None),
            getattr(self, 'read_sharpen_mask_margin', None),
            self.training,
            getattr(self, 'read_sharpen_eval_only', False),
        )

        # addressing entropy at EOS
        try:
            sims = self.memory.address_sims(k_read.detach().to(self.memory_device))
            p = torch.softmax(sims/1.0, dim=-1)
            ent = -(p * (p.clamp(min=1e-9).log())).sum(dim=-1)
            addr_ent = ent.mean().to(device)
        except Exception:
            addr_ent = torch.tensor(float('nan'), device=device)

        # Metrics aggregation
        avg_write_cos = (write_cos_sum / torch.clamp_min(write_step_count, 1.0)).detach()
        metrics = {
            "avg_fp_iters": fp_iters_total / max(1, L),
            "aux_logits": None,
            "aux_loss": aux_write_loss,
            "aux_nce_loss": aux_nce_loss,
            "write_cos": avg_write_cos,
            "read_cos": torch.tensor(float('nan'), device=device),
            "read_cos_raw": torch.tensor(float('nan'), device=device),
            "read_cleanup_cos": cleanup_sim.mean().detach() if cleanup_sim.numel() > 0 else torch.tensor(float('nan'), device=device),
            "num_write_steps": int(write_step_count.item()),
            "addr_entropy": (addr_ent if "addr_ent" in locals() else torch.tensor(float('nan'), device=device)),
            "residual_norm": (float(getattr(self.deq, "_last_residual", float('nan'))) if hasattr(self, "deq") else float('nan')),
        }
        return logits, metrics

    def _cleanup_read(self, v_raw: torch.Tensor, proto: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project raw memory reads onto nearest prototype for stability."""
        v_norm = F.normalize(v_raw, dim=-1)
        sims = torch.matmul(v_norm, proto.t())
        if sims.dim() == 1:
            sims = sims.unsqueeze(0)
            v_norm = v_norm.unsqueeze(0)
        topk = max(1, int(getattr(self, 'read_cleanup_topk', 1)))
        temp = float(getattr(self, 'cleanup_temp', 1.0) or 1.0)
        softmax_mode = bool(getattr(self, 'cleanup_softmax', False))
        if topk == 1 and not softmax_mode:
            best = sims.argmax(dim=-1)
            snapped = proto.index_select(0, best)
        else:
            vals, idx = sims.topk(topk, dim=-1)
            proto_selected = proto.index_select(0, idx.reshape(-1))
            proto_selected = proto_selected.view(v_norm.size(0), topk, -1)
            if softmax_mode:
                weights = vals / max(temp, 1e-6)
                weights = F.softmax(weights, dim=-1)
                snapped = torch.sum(proto_selected * weights.unsqueeze(-1), dim=1)
                snapped = F.normalize(snapped, dim=-1)
            else:
                snapped = F.normalize(proto_selected.mean(dim=1), dim=-1)
        cleanup_cos = torch.sum(v_norm * snapped, dim=-1)
        return snapped, v_norm, cleanup_cos
