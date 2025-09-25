from __future__ import annotations
import torch
from typing import Optional
import torch.nn as nn
import torch.nn.functional as F

from src.modules.vsa_memory import VSAMemory, _normalize
from src.modules.deq_block import ResidualUpdate, FixedPointBlock
from src.modules.liquid_cell import LiquidCell

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
    ):
        super().__init__()
        self.oracle_write = oracle_write
        self.mem_into_deq = mem_into_deq
        self.mem_scale = mem_scale
        self.warm_start_epochs = warm_start_epochs
        # Runtime knobs set by trainer
        self.oracle_mix_alpha: float = 0.0
        self.logit_scale_max: Optional[float] = None

        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.key_embed = nn.Embedding(num_values, mem_dim)
        self.value_embed = nn.Embedding(num_values, mem_dim)
        nn.init.normal_(self.value_embed.weight, std=0.1)
        nn.init.normal_(self.key_embed.weight, std=0.1)

        self.updater = ResidualUpdate(dim_z=hid_dim, dim_x=emb_dim, dim_v=mem_dim, hidden=max(128, hid_dim*2))
        self.deq = FixedPointBlock(self.updater, max_iter=deq_max_iter, tol=1e-4, relax=0.5)

        self.liquid = LiquidCell(dim_in=hid_dim, dim_hid=hid_dim)
        self.h0 = nn.Parameter(torch.zeros(hid_dim))

        self.memory = VSAMemory(dim=mem_dim, n_slots=n_slots, k_top=k_top, decay=0.997)
        self.z_to_value = nn.Linear(hid_dim*2, mem_dim)

        self._logit_scale_raw = nn.Parameter(torch.tensor(2.0))  # softplus(2) ~ 2.13

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
        self.memory = self.memory.to('cpu')
        h = self.h0.unsqueeze(0).expand(B, -1)
        z = torch.zeros(B, h.size(1), device=device)

        x = self.embed(tokens)
        key_vecs = _normalize(self.key_embed(key_ids))
        V_proto = _normalize(self.value_embed.weight)
        scale = self.logit_scale

        logits_main = None
        logits_pred_full = None
        aux_write_loss = torch.zeros((), device=device)
        aux_nce_loss = torch.zeros((), device=device)
        fp_iters_total = 0

        # NEW: direct write-cos + read-cos accounting
        write_cos_sum = torch.zeros((), device=device)
        write_step_count = torch.zeros((), device=device)
        read_cos_sum = torch.zeros((), device=device)
        read_step_count = torch.zeros((), device=device)

        for t in range(L):
            x_t = x[:, t, :]

            if self.mem_into_deq and (t == query_pos).any():
                mask_q = (t == query_pos)
                v_read_local = torch.zeros(B, self.memory.dim, device=device)
                v_read_local[mask_q] = self.memory.read(key_vecs[mask_q].detach().to('cpu')).to(device)
                v_t = v_read_local * self.mem_scale
            else:
                v_t = torch.zeros(B, self.memory.dim, device=device)

            z, nstep = self.deq(z, x_t, v_t)
            fp_iters_total += nstep

            h = self.liquid(z, h)

            zh = torch.cat([z, h], dim=-1)
            v_pred = _normalize(self.z_to_value(zh))

            # WRITE
            if (t == write_pos).any():
                mask_w = (t == write_pos)
                # true value at write
                if value_ids is not None:
                    v_true = _normalize(self.value_embed(value_ids[mask_w]))
                else:
                    v_true = _normalize(self.value_embed(key_ids[mask_w]))
                if use_oracle:
                    v_write = v_true
                else:
                    # Optional oracle/predicted mix controlled by trainer via self.oracle_mix_alpha
                    alpha = float(getattr(self, 'oracle_mix_alpha', 0.0) or 0.0)
                    if alpha > 0.0:
                        v_write = _normalize(alpha * v_true + (1.0 - alpha) * v_pred[mask_w])
                    else:
                        v_write = v_pred[mask_w]
                    # Aux losses computed against predicted vector vs true (unchanged)
                    cos = torch.cosine_similarity(v_pred[mask_w], v_true, dim=-1)
                    aux_write_loss = aux_write_loss + (1.0 - cos).mean()
                    write_cos_sum = write_cos_sum + cos.mean()
                    write_step_count = write_step_count + 1.0
                    # InfoNCE write loss (optional CE over prototypes)
                    logits_nce = self.logit_scale * (F.normalize(v_pred[mask_w], dim=-1) @ V_proto.t())
                    if value_ids is not None:
                        aux_nce_loss = aux_nce_loss + F.cross_entropy(logits_nce, value_ids[mask_w])
                # external memory write on CPU (optional disable during eval)
                if not disable_write:
                    self.memory.write(key_vecs[mask_w].detach().to('cpu'), v_write.detach().to('cpu'))

            # QUERY / CLASSIFY
            if (t == query_pos).any():
                mask_q = (t == query_pos)
                # Optionally shuffle keys before read (eval-time causal test)
                if shuffle_read:
                    kv = key_vecs[mask_q]
                    if kv.shape[0] > 1:
                        perm = torch.randperm(kv.shape[0], device=kv.device)
                        kv = kv[perm]
                    v_mem = self.memory.read(kv.detach().to('cpu')).to(device)
                else:
                    v_mem = self.memory.read(key_vecs[mask_q].detach().to('cpu')).to(device)
                logits_q_mem  = scale * (F.normalize(v_mem, dim=-1) @ V_proto.t())
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
                    rc = torch.cosine_similarity(F.normalize(v_mem, dim=-1), v_true_q, dim=-1)
                    read_cos_sum = read_cos_sum + rc.mean()
                    read_step_count = read_step_count + 1.0

        # Safe averages
        avg_write_cos = (write_cos_sum / torch.clamp_min(write_step_count, 1.0)).detach()
        avg_read_cos = (read_cos_sum / torch.clamp_min(read_step_count, 1.0)).detach()
        metrics = {
            "avg_fp_iters": fp_iters_total / max(1, L),
            "aux_logits": logits_pred_full,
            "aux_loss": aux_write_loss,
            "aux_nce_loss": aux_nce_loss,
            "write_cos": avg_write_cos,              # mean cosine at write steps this forward
            "read_cos": avg_read_cos,                # mean cosine at query reads this forward
            "num_write_steps": int(write_step_count.item()),
        }
        return logits_main, metrics
