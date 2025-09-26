
from __future__ import annotations
import argparse, yaml, time, sys, os, math, csv
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from rich import print as rprint

from emma.utils import get_device
from emma.utils import set_seed
from emma.data import make_dataloaders, make_dataloaders_listops_lite
from emma.schedules import NCEScheduler
# GRU baseline removed in the minimal package; only EMMA is exposed.
from emma.model import EMMA

def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=-1)
    return (preds == targets).float().mean().item()


class PerSampleLogger:
    """Write per-sequence telemetry to a CSV sidecar for early batches."""

    def __init__(self, path: str | Path, limit: int):
        self.path = Path(path)
        self.limit = max(0, int(limit or 0))
        self._fh = None
        self._writer = None
        self._logged = 0

    def _ensure_writer(self) -> None:
        if self._writer is None and self.limit > 0:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.path.open('w', encoding='utf-8', newline='')
            self._writer = csv.writer(self._fh)
            self._writer.writerow([
                'split', 'epoch', 'batch', 'time', 'event', 'seq_idx',
                'target', 'cos', 'cos_raw', 'cleanup', 'bucket', 'entropy',
            ])

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self._logged)

    def log(
        self,
        *,
        split: str,
        epoch: int,
        batch: int,
        targets: torch.Tensor | None,
        write_samples: list[dict] | None,
        read_samples: list[dict] | None,
    ) -> None:
        if self.limit <= 0 or self.remaining <= 0:
            return
        self._ensure_writer()
        if self._writer is None:
            return
        targets_list = None
        if targets is not None:
            try:
                targets_list = targets.tolist()
            except Exception:
                targets_list = None

        def _lookup_target(seq_idx: int):
            if targets_list is None:
                return ''
            if 0 <= seq_idx < len(targets_list):
                return targets_list[seq_idx]
            return ''

        def _write_row(event: str, record: dict):
            if self.remaining <= 0:
                return
            seq_idx = int(record.get('seq', -1))
            target_val = _lookup_target(seq_idx)
            row = [
                split,
                int(epoch),
                int(batch),
                int(record.get('time', -1)),
                event,
                seq_idx,
                target_val,
                record.get('cos', float('nan')),
                record.get('cos_raw', float('nan')),
                record.get('cleanup', float('nan')),
                record.get('bucket', ''),
                record.get('entropy', float('nan')),
            ]
            self._writer.writerow(row)
            self._logged += 1

        for rec in write_samples or []:
            if self.remaining <= 0:
                break
            _write_row('write', rec)
        for rec in read_samples or []:
            if self.remaining <= 0:
                break
            _write_row('read', rec)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
            self._writer = None

def train_epoch(model, loader, device, optimizer, criterion, model_type,
                lam_pred=0.0, lam_write=0.0, lam_nce=0.0, lam_jac=0.0, logprint=None,
                epoch_metrics: dict | None = None, epoch_index: int | None = None,
                sample_logger: PerSampleLogger | None = None,
                load_balance_coeff: float = 0.0):
    import math, torch
    from torch.nn.utils import clip_grad_norm_

    model.train()
    total_loss = 0.0
    total_acc  = 0.0
    n = 0

    fp_sum = 0.0; fp_n = 0
    wcos_sum = 0.0; wcos_n = 0
    rcos_sum = 0.0; rcos_n = 0
    rcos_raw_sum = 0.0; rcos_raw_n = 0
    cleanup_sum = 0.0; cleanup_n = 0
    bucket_ent_sum = 0.0; bucket_ent_n = 0
    writes_mean_sum = 0.0; writes_mean_n = 0
    write_bucket_ent_sum = 0.0; write_bucket_ent_n = 0
    ent_sum = 0.0; ent_n = 0
    res_sum = 0.0; res_n = 0
    nce_sum = 0.0; nce_count = 0
    wcos_sq_sum = 0.0
    rcos_sq_sum = 0.0
    wcos_min = float('inf'); wcos_max = float('-inf')
    rcos_min = float('inf'); rcos_max = float('-inf')
    bucket_usage_sum = 0.0; bucket_usage_n = 0
    bucket_usage_min = float('inf'); bucket_usage_max = float('-inf')
    read_bucket_sum = 0.0; read_bucket_n = 0
    read_bucket_min = float('inf'); read_bucket_max = float('-inf')
    read_gap_sum = 0.0; read_gap_n = 0
    collision_rate_sum = 0.0; collision_rate_n = 0
    collision_mean_sum = 0.0; collision_mean_n = 0
    collision_max_sum = 0.0; collision_max_n = 0
    load_variance_sum = 0.0; load_variance_n = 0

    for batch_idx, batch in enumerate(loader):
        tokens    = batch['tokens'].to(device)
        key_id    = batch['key_id'].to(device)
        write_pos = batch['write_pos'].to(device)
        query_pos = batch['query_pos'].to(device)
        target    = batch['target'].to(device)

        optimizer.zero_grad(set_to_none=True)

        if model_type != 'gru':
            logits, metrics = model(tokens, key_id, write_pos, query_pos, value_ids=target)
        else:
            logits, metrics = model(tokens)

        # Main CE loss (memory read)
        ce_main = criterion(logits, target)
        loss = ce_main

        # Auxiliary prediction CE (predicted head), if available
        if model_type != 'gru' and lam_pred > 0.0 and isinstance(metrics, dict):
            aux_logits = metrics.get('aux_logits', None)
            if isinstance(aux_logits, torch.Tensor) and aux_logits.shape == logits.shape:
                ce_pred = criterion(aux_logits, target)
                loss = loss + lam_pred * ce_pred

        # ✅ Differentiable write-alignment loss from model (normalize by #write steps)
        if model_type != 'gru' and lam_write > 0.0 and isinstance(metrics, dict):
            aux = metrics.get('aux_loss', None)
            if isinstance(aux, torch.Tensor):
                steps = metrics.get('num_write_steps', 0)
                if isinstance(steps, int) and steps > 0:
                    aux = aux / float(steps)
                loss = loss + lam_write * aux
            else:
                # Fallback: non-diff scalar; last resort only
                wc = metrics.get('write_cos', None)
                if wc is not None:
                    try:
                        loss = loss + lam_write * (1.0 - float(wc))
                    except Exception:
                        pass

        # Optional InfoNCE write loss (tensor preferred; off by default)
        if model_type != 'gru' and lam_nce > 0.0 and isinstance(metrics, dict):
            nce = metrics.get('aux_nce_loss', None)
            if isinstance(nce, torch.Tensor):
                loss = loss + lam_nce * nce
                try:
                    nce_sum += float(nce.detach().cpu())
                    nce_count += 1
                except Exception:
                    pass
            elif nce is not None:
                try:
                    loss = loss + lam_nce * float(nce)
                except Exception:
                    pass

        # Optional Jacobian/residual penalty
        if lam_jac > 0.0 and isinstance(metrics, dict):
            try:
                resid = metrics.get('residual_norm', None)
                if resid is not None:
                    loss = loss + lam_jac * float(resid)
            except Exception:
                pass

        write_samples = None
        read_samples = None
        if isinstance(metrics, dict):
            write_samples = metrics.pop('write_samples', None)
            read_samples = metrics.pop('read_samples', None)
        if sample_logger is not None and sample_logger.remaining > 0 and (write_samples or read_samples):
            target_cpu = target.detach().cpu()
            sample_logger.log(
                split='train',
                epoch=epoch_index if epoch_index is not None else 0,
                batch=batch_idx,
                targets=target_cpu,
                write_samples=write_samples,
                read_samples=read_samples,
            )

        # Backprop & step
        loss.backward()
        try:
            clip_grad_norm_(model.parameters(), max_norm=1.0)
        except Exception:
            pass
        optimizer.step()

        # Stats
        B = target.shape[0]
        total_loss += float(loss.item()) * B
        total_acc  += (logits.argmax(-1) == target).float().sum().item()
        n += B

        if isinstance(metrics, dict):
            try:
                fpi = metrics.get('avg_fp_iters', None)
                if fpi is not None: fp_sum += float(fpi); fp_n += 1
            except Exception: pass
            try:
                wc = metrics.get('write_cos', None)
                if wc is not None:
                    wc = float(wc)
                    if not math.isnan(wc):
                        wcos_sum += wc; wcos_n += 1
                        wcos_sq_sum += wc * wc
                        wcos_min = min(wcos_min, wc)
                        wcos_max = max(wcos_max, wc)
            except Exception: pass
            try:
                rc = metrics.get('read_cos', None)
                if rc is not None:
                    rc = float(rc)
                    if not math.isnan(rc):
                        rcos_sum += rc; rcos_n += 1
                        rcos_sq_sum += rc * rc
                        rcos_min = min(rcos_min, rc)
                        rcos_max = max(rcos_max, rc)
            except Exception: pass
            try:
                rc_raw = metrics.get('read_cos_raw', None)
                if rc_raw is not None:
                    rc_raw = float(rc_raw)
                    if not math.isnan(rc_raw): rcos_raw_sum += rc_raw; rcos_raw_n += 1
            except Exception: pass
            try:
                rc_clean = metrics.get('read_cleanup_cos', None)
                if rc_clean is not None:
                    rc_clean = float(rc_clean)
                    if not math.isnan(rc_clean): cleanup_sum += rc_clean; cleanup_n += 1
            except Exception: pass
            try:
                gap = metrics.get('read_cleanup_gap', None)
                if gap is not None:
                    gap = float(gap)
                    if not math.isnan(gap): read_gap_sum += gap; read_gap_n += 1
            except Exception: pass
            try:
                be = metrics.get('bucket_entropy', None)
                if be is not None:
                    be = float(be)
                    if not math.isnan(be): bucket_ent_sum += be; bucket_ent_n += 1
            except Exception: pass
            try:
                wp = metrics.get('writes_per_seq_mean', None)
                if wp is not None:
                    wp = float(wp)
                    if not math.isnan(wp): writes_mean_sum += wp; writes_mean_n += 1
            except Exception: pass
            try:
                wbe = metrics.get('write_bucket_entropy', None)
                if wbe is not None:
                    wbe = float(wbe)
                    if not math.isnan(wbe): write_bucket_ent_sum += wbe; write_bucket_ent_n += 1
            except Exception: pass
            try:
                ae = metrics.get('addr_entropy', None)
                if ae is not None:
                    ae = float(ae)
                    if not math.isnan(ae): ent_sum += ae; ent_n += 1
            except Exception: pass
            try:
                rn = metrics.get('residual_norm', None)
                if rn is not None:
                    rn = float(rn)
                    if not math.isnan(rn): res_sum += rn; res_n += 1
            except Exception: pass
            try:
                coll = metrics.get('bucket_collision_rate', None)
                if coll is not None:
                    coll = float(coll)
                    if not math.isnan(coll): collision_rate_sum += coll; collision_rate_n += 1
            except Exception: pass
            try:
                coll_mean = metrics.get('bucket_collision_rate_mean', None)
                if coll_mean is not None:
                    coll_mean = float(coll_mean)
                    if not math.isnan(coll_mean): collision_mean_sum += coll_mean; collision_mean_n += 1
            except Exception: pass
            try:
                coll_max = metrics.get('bucket_collision_rate_max', None)
                if coll_max is not None:
                    coll_max = float(coll_max)
                    if not math.isnan(coll_max): collision_max_sum += coll_max; collision_max_n += 1
            except Exception: pass
            try:
                load_var = metrics.get('bucket_load_variance', None)
                if load_var is not None:
                    load_var = float(load_var)
                    if not math.isnan(load_var):
                        load_variance_sum += load_var
                        load_variance_n += 1
                        if load_balance_coeff > 0.0:
                            loss = loss + load_balance_coeff * load_var
            except Exception: pass
            try:
                bucket_active = metrics.get('bucket_usage_active', None)
                if bucket_active is not None:
                    bucket_active = float(bucket_active)
                    if not math.isnan(bucket_active):
                        bucket_usage_sum += bucket_active
                        bucket_usage_n += 1
                        bucket_usage_min = min(bucket_usage_min, bucket_active)
                        bucket_usage_max = max(bucket_usage_max, bucket_active)
            except Exception:
                pass
            try:
                read_bucket_active = metrics.get('read_bucket_active', None)
                if read_bucket_active is not None:
                    read_bucket_active = float(read_bucket_active)
                    if not math.isnan(read_bucket_active):
                        read_bucket_sum += read_bucket_active
                        read_bucket_n += 1
                        read_bucket_min = min(read_bucket_min, read_bucket_active)
                        read_bucket_max = max(read_bucket_max, read_bucket_active)
            except Exception:
                pass

        cooldown_batches = getattr(model, '_write_cooldown_batches_remaining', None)
        if isinstance(cooldown_batches, int) and cooldown_batches > 0:
            model._write_cooldown_batches_remaining = max(0, cooldown_batches - 1)

    avg_fp = fp_sum / max(1, fp_n)
    avg_w  = wcos_sum / max(1, wcos_n)
    avg_r  = rcos_sum / max(1, rcos_n)
    avg_r_raw = rcos_raw_sum / max(1, rcos_raw_n)
    avg_r_cleanup = cleanup_sum / max(1, cleanup_n)
    avg_bucket_ent = bucket_ent_sum / max(1, bucket_ent_n)
    avg_writes_mean = writes_mean_sum / max(1, writes_mean_n)
    avg_write_bucket_ent = write_bucket_ent_sum / max(1, write_bucket_ent_n)
    avg_read_gap = read_gap_sum / max(1, read_gap_n) if read_gap_n else float('nan')
    avg_collision_rate = collision_rate_sum / max(1, collision_rate_n) if collision_rate_n else float('nan')
    avg_collision_mean = collision_mean_sum / max(1, collision_mean_n) if collision_mean_n else float('nan')
    avg_collision_max = collision_max_sum / max(1, collision_max_n) if collision_max_n else float('nan')
    avg_load_variance = load_variance_sum / max(1, load_variance_n) if load_variance_n else float('nan')

    try:
        from rich import print as rprint
        avg_ent = ent_sum / max(1, ent_n)
        avg_res = res_sum / max(1, res_n)
        if rcos_raw_n:
            rprint(
                f" avg_write_cos={avg_w:.4f} avg_read_cos={avg_r:.4f} avg_read_raw={avg_r_raw:.4f} "
                f"avg_cleanup_cos={avg_r_cleanup:.4f} avg_bucket_ent={avg_bucket_ent:.4f} writes_mean={avg_writes_mean:.2f} "
                f"write_bucket_ent={avg_write_bucket_ent:.4f} avg_fp_iters={avg_fp:.3f} avg_addr_ent={avg_ent:.3f} avg_resid={avg_res:.6f} "
                f"read_gap={avg_read_gap:.4f} coll_rate={avg_collision_rate:.4f} load_var={avg_load_variance:.6f}"
            )
        else:
            rprint(
                f" avg_write_cos={avg_w:.4f} avg_read_cos={avg_r:.4f} avg_bucket_ent={avg_bucket_ent:.4f} "
                f"writes_mean={avg_writes_mean:.2f} write_bucket_ent={avg_write_bucket_ent:.4f} "
                f"avg_fp_iters={avg_fp:.3f} avg_addr_ent={avg_ent:.3f} avg_resid={avg_res:.6f} "
                f"read_gap={avg_read_gap:.4f} coll_rate={avg_collision_rate:.4f} load_var={avg_load_variance:.6f}"
            )
    except Exception:
        pass

    if epoch_metrics is not None:
        def _stat_or_nan(value, count):
            return value if count > 0 else float('nan')

        write_std = float('nan')
        if wcos_n > 0:
            mean = avg_w
            var = max(0.0, (wcos_sq_sum / max(1, wcos_n)) - (mean * mean))
            write_std = math.sqrt(var)
        read_std = float('nan')
        if rcos_n > 0:
            mean = avg_r
            var = max(0.0, (rcos_sq_sum / max(1, rcos_n)) - (mean * mean))
            read_std = math.sqrt(var)

        bucket_mean = bucket_usage_sum / max(1, bucket_usage_n) if bucket_usage_n else float('nan')
        read_bucket_mean = read_bucket_sum / max(1, read_bucket_n) if read_bucket_n else float('nan')

        epoch_metrics.update({
            'train_write_cos_min': _stat_or_nan(wcos_min, wcos_n),
            'train_write_cos_max': _stat_or_nan(wcos_max, wcos_n),
            'train_write_cos_std': write_std,
            'train_read_cos_min': _stat_or_nan(rcos_min, rcos_n),
            'train_read_cos_max': _stat_or_nan(rcos_max, rcos_n),
            'train_read_cos_std': read_std,
            'train_bucket_usage_mean': _stat_or_nan(bucket_mean, bucket_usage_n),
            'train_bucket_usage_min': _stat_or_nan(bucket_usage_min, bucket_usage_n),
            'train_bucket_usage_max': _stat_or_nan(bucket_usage_max, bucket_usage_n),
            'train_read_bucket_mean': _stat_or_nan(read_bucket_mean, read_bucket_n),
            'train_read_bucket_min': _stat_or_nan(read_bucket_min, read_bucket_n),
            'train_read_bucket_max': _stat_or_nan(read_bucket_max, read_bucket_n),
            'train_read_gap_mean': avg_read_gap,
            'train_bucket_collision_rate': avg_collision_rate,
            'train_bucket_collision_mean': avg_collision_mean,
            'train_bucket_collision_max': avg_collision_max,
            'train_bucket_load_variance': avg_load_variance,
        })

    if callable(logprint):
        try:
            lam_val = float(lam_nce)
        except Exception:
            lam_val = 0.0
        avg_nce = (nce_sum / max(1, nce_count)) if nce_count else 0.0
        logprint(f"METRIC lambda_nce_in_train={lam_val:.6f}")
        logprint(f"METRIC aux_nce_loss_epoch_mean={avg_nce:.6f}")
        try:
            logprint(f"METRIC train_read_gap_mean={avg_read_gap:.6f}")
        except Exception:
            pass
        try:
            logprint(f"METRIC train_bucket_collision_rate={avg_collision_rate:.6f}")
        except Exception:
            pass
        try:
            logprint(f"METRIC train_bucket_collision_mean={avg_collision_mean:.6f}")
        except Exception:
            pass
        try:
            logprint(f"METRIC train_bucket_collision_max={avg_collision_max:.6f}")
        except Exception:
            pass
        try:
            logprint(f"METRIC train_bucket_load_variance={avg_load_variance:.6f}")
        except Exception:
            pass

    return total_loss / max(1, n), total_acc / max(1, n), avg_fp, avg_w, avg_r

def eval_epoch(model, loader, device, criterion, model_type: str,
               sample_logger: PerSampleLogger | None = None,
               epoch_index: int | None = None):
    # reduce DEQ work during eval for speed
    saved_deq_iter = getattr(getattr(model, 'deq', None), 'max_iter', None)
    try:
        if saved_deq_iter is not None and saved_deq_iter > 5:
            model.deq.max_iter = 5
    except Exception:
        pass

    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n = 0

    import math
    read_cos_vals: list[float] = []
    read_cos_raw_vals: list[float] = []
    read_cleanup_vals: list[float] = []
    write_cos_vals: list[float] = []
    addr_entropy_vals: list[float] = []
    residual_vals: list[float] = []
    writes_mean_vals: list[float] = []
    write_bucket_entropy_vals: list[float] = []
    topk_hits = 0
    topk_total = 0

    for batch_idx, batch in enumerate(tqdm(loader, desc=f"eval/{model_type}", leave=False)):
        tokens = batch['tokens'].to(device)
        key_id = batch['key_id'].to(device)
        write_pos = batch['write_pos'].to(device)
        query_pos = batch['query_pos'].to(device)
        target = batch['target'].to(device)

        if model_type == 'gru':
            logits = model(tokens, query_pos)
            metrics = {}
        else:
            disable_write = bool(getattr(model, '_eval_disable_write', False))
            shuffle_read = bool(getattr(model, '_eval_shuffle_read', False))
            logits, metrics = model(
                tokens,
                key_id,
                write_pos,
                query_pos,
                value_ids=target,
                current_epoch=0,
                disable_write=disable_write,
                shuffle_read=shuffle_read,
            )
        loss = criterion(logits, target)

        write_samples = None
        read_samples = None
        if isinstance(metrics, dict) and metrics:
            rc = metrics.get('read_cos')
            rc_raw = metrics.get('read_cos_raw')
            rc_cleanup = metrics.get('read_cleanup_cos')
            wc = metrics.get('write_cos')
            ae = metrics.get('addr_entropy')
            rn = metrics.get('residual_norm')
            wp = metrics.get('writes_per_seq_mean')
            wbe = metrics.get('write_bucket_entropy')
            try:
                if rc is not None and not math.isnan(float(rc)):
                    read_cos_vals.append(float(rc))
            except Exception:
                pass
            try:
                if rc_raw is not None and not math.isnan(float(rc_raw)):
                    read_cos_raw_vals.append(float(rc_raw))
            except Exception:
                pass
            try:
                if rc_cleanup is not None and not math.isnan(float(rc_cleanup)):
                    read_cleanup_vals.append(float(rc_cleanup))
            except Exception:
                pass
            try:
                if wc is not None and not math.isnan(float(wc)):
                    write_cos_vals.append(float(wc))
            except Exception:
                pass
            try:
                if ae is not None and not math.isnan(float(ae)):
                    addr_entropy_vals.append(float(ae))
            except Exception:
                pass
            try:
                if rn is not None and not math.isnan(float(rn)):
                    residual_vals.append(float(rn))
            except Exception:
                pass
            try:
                if wp is not None and not math.isnan(float(wp)):
                    writes_mean_vals.append(float(wp))
            except Exception:
                pass
            try:
                if wbe is not None and not math.isnan(float(wbe)):
                    write_bucket_entropy_vals.append(float(wbe))
            except Exception:
                pass

            write_samples = metrics.pop('write_samples', None)
            read_samples = metrics.pop('read_samples', None)

        if model_type != 'gru':
            try:
                k_top = int(getattr(getattr(model, 'memory', object()), 'k_top', 16))
                if k_top > 0:
                    topk = torch.topk(logits, k=min(k_top, logits.size(-1)), dim=-1).indices
                    hits = (topk == target.unsqueeze(-1)).any(dim=-1).float().sum().item()
                    topk_hits += int(hits)
                    topk_total += int(target.numel())
            except Exception:
                pass

        bs = tokens.size(0)
        total_loss += loss.item() * bs
        total_acc += (torch.argmax(logits, dim=-1) == target).float().sum().item()
        n += bs

        if sample_logger is not None and sample_logger.remaining > 0 and (write_samples or read_samples):
            sample_logger.log(
                split='val',
                epoch=epoch_index if epoch_index is not None else 0,
                batch=batch_idx,
                targets=target.detach().cpu(),
                write_samples=write_samples,
                read_samples=read_samples,
            )

    def _mean(values: list[float]) -> float:
        return float(sum(values) / len(values)) if values else float('nan')

    read_cos_last = _mean(read_cos_vals)
    read_cos_raw_last = _mean(read_cos_raw_vals)
    read_cleanup_last = _mean(read_cleanup_vals)
    writes_mean_last = _mean(writes_mean_vals)
    write_bucket_entropy_last = _mean(write_bucket_entropy_vals)
    write_cos_last = _mean(write_cos_vals)
    addr_entropy_last = _mean(addr_entropy_vals)
    residual_last = _mean(residual_vals)
    topk_hit_rate = float(topk_hits) / float(max(1, topk_total)) if topk_total > 0 else float('nan')

    return (
        total_loss / max(1, n),
        total_acc / max(1, n),
        read_cos_last,
        write_cos_last,
        topk_hit_rate,
        addr_entropy_last,
        residual_last,
        read_cos_raw_last,
        read_cleanup_last,
        writes_mean_last,
        write_bucket_entropy_last,
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/needle_tiny.yaml')
    parser.add_argument('--model', type=str, default='emma_liquid', choices=['emma_liquid'])
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--logfile', type=str, default=None, help='Optional path to also write logs (tee).')
    parser.add_argument('--save-config', type=str, default=None, help='Optional path to write the resolved config YAML.')
    parser.add_argument('--checkpoint', type=str, default=None, help='Optional path to save best model checkpoint.')
    parser.add_argument('--metrics-out', type=str, default=None, help='Optional path to write run-level metrics (JSON recommended).')
    parser.add_argument('--per-epoch-out', type=str, default=None, help='Optional CSV to append per-epoch eval metrics.')
    parser.add_argument('--sample-log', type=str, default=None, help='Optional base path for per-sample telemetry CSV (suffix _train/_val).')
    parser.add_argument('--sample-log-limit', type=int, default=128, help='Maximum rows to record per split for sample telemetry.')
    parser.add_argument('--eval-no-write', action='store_true', help='Disable memory writes during eval (causality test).')
    parser.add_argument('--eval-shuffle-read', action='store_true', help='Shuffle keys before memory read during eval (causality test).')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # simple tee logger (non-intrusive)
    log_fh = None
    def logprint(msg: str):
        rprint(msg)
        if log_fh is not None:
            print(msg, file=log_fh, flush=True)
    if args.logfile is not None:
        os.makedirs(os.path.dirname(args.logfile), exist_ok=True)
        log_fh = open(args.logfile, 'a', encoding='utf-8')

    # Optionally persist the resolved config for reproducibility
    if args.save_config:
        try:
            os.makedirs(os.path.dirname(args.save_config), exist_ok=True)
            with open(args.save_config, 'w') as _cf:
                yaml.safe_dump(cfg, _cf, sort_keys=False)
        except Exception as e:
            rprint(f"[yellow]warn: failed to save config to {args.save_config}: {e}[/yellow]")
    logprint(f"METRIC trainer_file={__file__}")

    set_seed(cfg.get('seed', 42))
    device = get_device(args.device)

    train_sample_logger = None
    val_sample_logger = None
    sample_limit = max(0, int(args.sample_log_limit or 0))
    if args.sample_log and sample_limit > 0:
        base_path = Path(args.sample_log)
        if base_path.suffix:
            stem = base_path.stem
            suffix = base_path.suffix
        else:
            stem = base_path.name
            suffix = '.csv'
        parent = base_path.parent if str(base_path.parent) else Path('.')
        train_path = parent / f"{stem}_train{suffix}"
        val_path = parent / f"{stem}_val{suffix}"
        train_sample_logger = PerSampleLogger(train_path, sample_limit)
        val_sample_logger = PerSampleLogger(val_path, sample_limit)

    task = cfg.get('task', 'needle')
    if task == 'listops_lite':
        class_count = 10
        train_loader, val_loader = make_dataloaders_listops_lite(
            length=cfg['data']['length'],
            train_size=cfg['data']['train_size'],
            val_size=cfg['data']['val_size'],
            batch_size=cfg['train']['batch_size'],
            num_workers=cfg['train'].get('num_workers', 0),
            seed=cfg.get('seed', 42),
        )
    else:
        class_count = cfg['data']['num_values']
        train_loader, val_loader = make_dataloaders(
            n_pairs=cfg['data'].get('n_pairs', 1),
            decouple_kv=cfg['data'].get('decouple_kv', True),
            num_values=class_count,
            vocab_extra=cfg['data']['vocab_extra'],
            length=cfg['data']['length'],
            train_size=cfg['data']['train_size'],
            val_size=cfg['data']['val_size'],
            batch_size=cfg['train']['batch_size'],
            num_workers=cfg['train'].get('num_workers', 0),
            seed=cfg.get('seed', 42),
        )
    vocab_size = next(iter(train_loader))['vocab_size']

    model_type = args.model
    if model_type != 'emma_liquid':
        raise ValueError(f"Unsupported model '{model_type}'. Minimal package only exposes 'emma_liquid'.")
    emma_cfg = cfg.get('emma', {})
    cfg_emma = cfg.get('emma', {})
    memory_cfg = cfg.get('memory', {})
    default_max_writes = int(cfg.get('data', {}).get('n_pairs', 1) or 1) + 1
    max_writes_per_seq = cfg_emma.get('max_writes_per_seq', default_max_writes)
    if max_writes_per_seq is not None:
        try:
            max_writes_per_seq = int(max_writes_per_seq)
            if max_writes_per_seq <= 0:
                max_writes_per_seq = None
        except Exception:
            max_writes_per_seq = None

    spectral_norm_enable = emma_cfg.get('use_spectral_norm', None)
    if spectral_norm_enable is None:
        spectral_norm_enable = (str(device) != 'mps')
    else:
        spectral_norm_enable = bool(spectral_norm_enable)

    value_dropout = float(emma_cfg.get('value_dropout', 0.0) or 0.0)
    cleanup_blend_raw = None
    cleanup_blend_clean = None
    read_cfg = cfg.get('read', {}) or {}
    cleanup_blend_cfg = read_cfg.get('cleanup_blend') if isinstance(read_cfg, dict) else None
    if isinstance(cleanup_blend_cfg, dict):
        try:
            cleanup_blend_raw = float(cleanup_blend_cfg.get('raw', None))
        except Exception:
            cleanup_blend_raw = None
        try:
            cleanup_blend_clean = float(cleanup_blend_cfg.get('cleanup', None))
        except Exception:
            cleanup_blend_clean = None
        if cleanup_blend_raw is None or cleanup_blend_clean is None:
            cleanup_blend_raw = cleanup_blend_clean = None
    write_norm_clip = memory_cfg.get('write_norm_clip', None)

    memory_device_cfg = memory_cfg.get('device', 'cpu')
    if isinstance(memory_device_cfg, str) and memory_device_cfg.lower() == 'auto':
        memory_device = device
    else:
        memory_device = memory_device_cfg

    model = EMMA(
        vocab_size=vocab_size,
        warm_start_epochs=cfg_emma.get('warm_start_epochs', 0),
        emb_dim=cfg['model']['emb_dim'],
        hid_dim=cfg['model']['hid_dim'],
        mem_dim=cfg['model']['mem_dim'],
        num_values=class_count,
        n_slots=memory_cfg['n_slots'],
        k_top=memory_cfg['k_top'],
        oracle_write=emma_cfg.get('oracle_write', False),
        deq_max_iter=emma_cfg.get('deq_max_iter', 15),
        mem_into_deq=emma_cfg.get('mem_into_deq', False),
        bucket_count=memory_cfg.get('bucket_count', 32),
        write_decay=memory_cfg.get('write_decay', 0.995),
        write_strength=memory_cfg.get('write_strength', 1.0),
        bucket_temp=memory_cfg.get('bucket_temp', 1.0),
        max_writes_per_seq=max_writes_per_seq,
        use_spectral_norm=spectral_norm_enable,
        spectral_norm_keys=emma_cfg.get('spectral_norm_keys', False),
        value_dropout=value_dropout,
        cleanup_blend_raw=cleanup_blend_raw,
        cleanup_blend_clean=cleanup_blend_clean,
        write_norm_clip=write_norm_clip,
        memory_device=memory_device,
    )
    model.to(device)

    load_balance_coeff = float(memory_cfg.get('load_balance_coeff', 0.0) or 0.0)
    write_strength_schedule_cfg = memory_cfg.get('write_strength_schedule', {}) or {}
    write_strength_mult = float(write_strength_schedule_cfg.get('mult_after_open', 1.0) or 1.0)
    write_strength_steps = int(write_strength_schedule_cfg.get('steps', 0) or 0)
    try:
        model._write_strength_base = float(getattr(model.memory, 'write_strength', 1.0))
    except Exception:
        model._write_strength_base = 1.0
    model._write_strength_delta = 0.0
    model._write_strength_steps_remaining = 0
    model._write_strength_schedule_mult = write_strength_mult
    model._write_strength_schedule_steps = write_strength_steps

    try:
        topk_val = read_cfg.get('sharpen_topk', None)
        if topk_val is not None:
            model.read_sharpen_topk = int(topk_val)
    except Exception:
        pass
    try:
        temp_val = read_cfg.get('sharpen_temp', None)
        if temp_val is not None:
            model.read_sharpen_temp = float(temp_val)
    except Exception:
        pass
    mode = read_cfg.get('cleanup_mode')
    if isinstance(mode, str):
        mode_lower = mode.lower()
        if mode_lower in {'softmax', 'hopfield'}:
            model.cleanup_softmax = True
        elif mode_lower in {'nearest', 'none'}:
            model.cleanup_softmax = False
    temp_cleanup = read_cfg.get('cleanup_temp')
    if temp_cleanup is not None:
        try:
            model.cleanup_temp = float(temp_cleanup)
        except Exception:
            pass
    mask_value = read_cfg.get('sharpen_mask_value')
    if mask_value is not None:
        try:
            model.read_sharpen_mask_value = float(mask_value)
        except Exception:
            pass
    mask_margin = read_cfg.get('sharpen_mask_margin')
    if mask_margin is not None:
        try:
            model.read_sharpen_mask_margin = float(mask_margin)
        except Exception:
            pass
    eval_only = read_cfg.get('sharpen_eval_only')
    if eval_only is not None:
        model.read_sharpen_eval_only = bool(eval_only)

    try:
        temp = float(emma_cfg.get('nce_temperature', 0.1) or 0.1)
        setattr(model, 'nce_temperature', temp)
    except Exception:
        setattr(model, 'nce_temperature', 0.1)
    logprint(
        f"METRIC nce_temperature={float(getattr(model, 'nce_temperature', float('nan'))):.6f}"
    )

    try:
        cooldown_batches_cfg = int(cfg_emma.get('post_gate_write_cooldown_batches', 0) or 0)
    except Exception:
        cooldown_batches_cfg = 0
    cooldown_batches_cfg = max(0, cooldown_batches_cfg)
    model.post_gate_write_cooldown_batches = cooldown_batches_cfg
    model._write_cooldown_batches_remaining = 0

    # Optional logit temperature clamp limit
    try:
        if model_type != 'gru' and hasattr(model, 'logit_scale_max'):
            model.logit_scale_max = cfg.get('emma', {}).get('logit_scale_max', model.logit_scale_max)
    except Exception:
        pass

    criterion = nn.CrossEntropyLoss()
    # Optional knobs (no core module edits)
    distill_cfg = cfg.get('distill', {})
    distill_enable = bool(distill_cfg.get('enable', False))
    distill_weight = float(distill_cfg.get('weight', 0.0) or 0.0)
    distill_teacher_deq = int(distill_cfg.get('teacher_deq', 0) or 0)
    distill_temp = float(distill_cfg.get('temp', 1.0) or 1.0)
    reg_cfg = cfg.get('reg', {})
    addr_ent_w = float(reg_cfg.get('addr_entropy_weight', 0.0) or 0.0)
    optimizer = (
        optim.SGD(model.parameters(), lr=cfg['train']['lr'], momentum=0.9, weight_decay=cfg['train'].get('wd', 1e-2))
        if (str(device) == 'mps' and model_type != 'gru')
        else optim.AdamW(model.parameters(), lr=cfg['train']['lr'], weight_decay=cfg['train'].get('wd', 1e-2))
    )
    train_cfg = cfg.get('train', {}) or {}
    base_param_group_lrs = [group['lr'] for group in optimizer.param_groups]
    lr_warm_restart_factor = float(train_cfg.get('lr_warm_restart_factor', 1.0) or 1.0)
    lr_warm_restart_epochs = int(train_cfg.get('lr_warm_restart_epochs', 0) or 0)
    if lr_warm_restart_epochs < 0:
        lr_warm_restart_epochs = 0
    lr_restart_remaining = 0

    lam_pred = cfg.get('loss', {}).get('lambda_pred_ce', 0.0)
    lam_write = cfg.get('loss', {}).get('lambda_write_cos', 0.0)
    lam_jac = float(cfg.get('loss', {}).get('lambda_jacobian', 0.0) or 0.0)
    loss_cfg = cfg.get('loss', {}) or {}
    base_lam_nce = loss_cfg.get('lambda_write_nce')
    if base_lam_nce is None:
        base_lam_nce = loss_cfg.get('lambda_nce', 0.0)
    base_lam_nce = float(base_lam_nce or 0.0)
    warm_start_epochs = cfg.get('emma', {}).get('warm_start_epochs', 0)

    lam_nce_schedule_cfg = loss_cfg.get('info_nce_lambda') or loss_cfg.get('lambda_nce_schedule')
    lam_nce_schedule = None
    if isinstance(lam_nce_schedule_cfg, dict):
        lam_start = float(lam_nce_schedule_cfg.get('start', 0.0) or 0.0)
        lam_end = float(lam_nce_schedule_cfg.get('end', lam_start))
        lam_max = float(lam_nce_schedule_cfg.get('max', lam_end))
        lam_epochs = max(1, int(lam_nce_schedule_cfg.get('epochs', 1) or 1))
        lam_alpha = float(lam_nce_schedule_cfg.get('ema_alpha', 0.0) or 0.0)
        lam_nce_schedule = {
            'start': lam_start,
            'end': lam_end,
            'max': lam_max,
            'epochs': lam_epochs,
            'ema_alpha': lam_alpha,
            'value': lam_start,
        }

    best_val = 0.0
    best_val_postwarm = 0.0
    warm = cfg.get('emma', {}).get('warm_start_epochs', 0)
    best_epoch = -1
    final_val_loss = None
    avg_fp_iters_running = []
    t0 = time.perf_counter()
    # InfoNCE gating configuration
    nce_ema_beta = float(emma_cfg.get('nce_ema_beta', 0.9) or 0.9)
    nce_read_ema_beta = float(emma_cfg.get('nce_read_ema_beta', 0.8) or 0.8)
    drop_threshold = float(emma_cfg.get('nce_post_gate_drop_threshold', 0.01) or 0.01)
    if drop_threshold < 0.0:
        drop_threshold = 0.0
    drop_patience = int(emma_cfg.get('nce_post_gate_drop_patience', 2) or 2)
    if drop_patience < 1:
        drop_patience = 1
    nce_ema = 0.0
    read_ema = 0.0
    nce_scheduler = NCEScheduler(emma_cfg)
    nce_gate_reason = None if nce_scheduler.config.enable else 'disabled'
    lam_nce_current = 0.0
    last_used_lambda = 0.0
    last_avg_w = 0.0
    base_oracle_write = bool(getattr(model, 'oracle_write', False))
    val_acc_history: list[float] = []
    residual_history: list[float] = []
    pre_gate_best_acc = float('nan')
    post_gate_best_acc = float('nan')
    delta_post_vs_pre = float('nan')
    gate_open_epoch_metric = None
    post_gate_drop_epochs = 0
    had_sustained_drop = False
    early_stop_epoch = None

    trip_cfg = emma_cfg.get('tripwire', {})
    trip_read_threshold = float(trip_cfg.get('read_cos_min', 0.5) or 0.5)
    trip_write_threshold = float(trip_cfg.get('write_cos_min', 0.01) or 0.01)
    trip_acc_target = trip_cfg.get('val_acc_min', None)
    if trip_acc_target is not None:
        try:
            trip_acc_target = float(trip_acc_target)
        except Exception:
            trip_acc_target = None
    trip_residual_tolerance = float(trip_cfg.get('residual_tolerance', 5e-4) or 5e-4)

    logging_cfg = cfg.get('logging', {}) or {}
    enable_extended_epoch_metrics = bool(logging_cfg.get('per_epoch_extended', False))

    # metrics CSV init (Phase 1 demo)
    metrics_csv = None
    metrics_out_path = args.metrics_out
    per_epoch_out_path = args.per_epoch_out
    if per_epoch_out_path:
        metrics_csv = per_epoch_out_path
    elif metrics_out_path and str(metrics_out_path).lower().endswith('.csv'):
        metrics_csv = metrics_out_path
    csv_rows: list[list[str]] = []
    epoch_header_extra: list[str] = []
    if enable_extended_epoch_metrics:
        epoch_header_extra = [
            'train_write_cos_min',
            'train_write_cos_max',
            'train_write_cos_std',
            'train_read_cos_min',
            'train_read_cos_max',
            'train_read_cos_std',
            'train_bucket_usage_mean',
            'train_bucket_usage_min',
            'train_bucket_usage_max',
            'train_read_bucket_mean',
            'train_read_bucket_min',
            'train_read_bucket_max',
            'train_read_gap_mean',
            'train_bucket_collision_rate',
            'train_bucket_collision_mean',
            'train_bucket_collision_max',
            'train_bucket_load_variance',
        ]
    total_epochs = int(train_cfg.get('epochs', cfg['train']['epochs']))
    gate_open_epoch_count = 0

    for epoch in range(cfg['train']['epochs']):
        opened_this_epoch = False
        plateau_ready = False
        read_ready = False
        epoch_ready = False
        stop_training = False

        if getattr(model, '_write_strength_steps_remaining', 0) > 0:
            try:
                current_strength = float(getattr(model.memory, 'write_strength', model._write_strength_base))
            except Exception:
                current_strength = model._write_strength_base
            new_strength = current_strength + float(getattr(model, '_write_strength_delta', 0.0))
            if new_strength <= 0.0:
                new_strength = model._write_strength_base
            try:
                model.memory.write_strength = new_strength
            except Exception:
                pass
            model._write_strength_steps_remaining = max(0, getattr(model, '_write_strength_steps_remaining', 0) - 1)

        decision = nce_scheduler.on_epoch_start(epoch, val_acc_history, read_ema)
        lam_nce_current = decision.lambda_value
        plateau_ready = decision.plateau_ready
        read_ready = decision.read_ready
        epoch_ready = decision.epoch_ready
        nce_gate_state = decision.gate_state
        nce_open_steps = decision.open_steps
        if nce_gate_state:
            gate_open_epoch_count += 1
        if lam_nce_schedule:
            progress = min(1.0, (epoch + 1) / lam_nce_schedule['epochs'])
            target = lam_nce_schedule['start'] + (lam_nce_schedule['end'] - lam_nce_schedule['start']) * progress
            alpha = lam_nce_schedule['ema_alpha']
            prev_value = lam_nce_schedule['value']
            if alpha > 0.0:
                schedule_value = alpha * prev_value + (1.0 - alpha) * target
            else:
                schedule_value = target
            schedule_value = min(schedule_value, lam_nce_schedule['max'])
            lam_nce_schedule['value'] = schedule_value
        else:
            schedule_value = 0.0
        if decision.opened_this_epoch:
            opened_this_epoch = True
            if decision.gate_reason:
                nce_gate_reason = decision.gate_reason

        if decision.lambda_transition and gate_open_epoch_metric is None:
            gate_open_epoch_metric = epoch + 1
            if val_acc_history:
                pre_gate_best_acc = max(val_acc_history)
            else:
                pre_gate_best_acc = float('nan')
        if decision.lambda_transition:
            try:
                model._write_cooldown_batches_remaining = int(getattr(model, 'post_gate_write_cooldown_batches', 0) or 0)
            except Exception:
                model._write_cooldown_batches_remaining = 0
            if getattr(model, '_write_strength_schedule_steps', 0) > 0 and getattr(model, '_write_strength_schedule_mult', 1.0) != 1.0:
                steps = max(1, int(getattr(model, '_write_strength_schedule_steps', 0)))
                target_strength = model._write_strength_base * float(getattr(model, '_write_strength_schedule_mult', 1.0))
                try:
                    current_strength = float(getattr(model.memory, 'write_strength', model._write_strength_base))
                except Exception:
                    current_strength = model._write_strength_base
                delta = (target_strength - current_strength) / float(steps)
                model._write_strength_delta = delta
                model._write_strength_steps_remaining = steps
            if lr_warm_restart_factor < 1.0 and lr_warm_restart_epochs > 0 and lr_restart_remaining <= 0:
                lr_restart_remaining = lr_warm_restart_epochs
                for idx, group in enumerate(optimizer.param_groups):
                    base_lr = base_param_group_lrs[idx] if idx < len(base_param_group_lrs) else group['lr']
                    group['lr'] = base_lr * lr_warm_restart_factor
                logprint(
                    f"[lr-restart] epoch={epoch+1} lr={optimizer.param_groups[0]['lr']:.2e} "
                    f"factor={lr_warm_restart_factor:.3f} duration={lr_warm_restart_epochs}"
                )

        logprint(f"METRIC gate_state_before={int(nce_gate_state)}")
        logprint(f"METRIC gate_plateau_ready={int(bool(plateau_ready))}")
        logprint(f"METRIC gate_read_ready={int(bool(read_ready))}")
        logprint(f"METRIC gate_epoch_ready={int(bool(epoch_ready))}")
        logprint(f"METRIC nce_open_steps={int(nce_open_steps)}")
        logprint(f"METRIC lambda_nce_current_epoch={lam_nce_current:.6f}")
        if lam_nce_schedule:
            logprint(f"METRIC lambda_nce_schedule={schedule_value:.6f}")
        if opened_this_epoch:
            rprint(
                f"[nce] gate_open epoch={epoch+1} reason={nce_gate_reason} "
                f"lambda={lam_nce_current:.3f} plateau={int(plateau_ready)} read_ready={int(read_ready)} read_ema={read_ema:.3f}"
            )
        elif nce_gate_state and lam_nce_current > 0.0:
            rprint(
                f"[nce] epoch={epoch+1} lambda_nce={lam_nce_current:.3f} open_steps={nce_open_steps} read_ema={read_ema:.3f}"
            )
        # Warm-start schedule: teacher-force writes for first N epochs (0-indexed: epochs [0..warm-1])
        if model_type != 'gru' and hasattr(model, 'oracle_write'):
            warm = cfg.get('emma', {}).get('warm_start_epochs', 0)
            warm_active = epoch < warm
            gate_forced = nce_scheduler.oracle_hold_remaining > 0
            use_oracle = warm_active or gate_forced or base_oracle_write
            model.oracle_write = use_oracle
            if warm_active:
                rprint(f"[warm-start] epoch={epoch+1} using oracle_write=True")
            elif gate_forced and not warm_active:
                rprint(
                    f"[nce] oracle_write held open epoch={epoch+1} hold_remaining={nce_scheduler.oracle_hold_remaining}"
                )
        # Oracle/pred mix alpha schedule
        if model_type != 'gru' and hasattr(model, 'oracle_mix_alpha'):
            emma_cfg = cfg.get('emma', {})
            warm = int(emma_cfg.get('warm_start_epochs', 0) or 0)
            ramp = int(emma_cfg.get('oracle_mix_ramp_epochs', 0) or 0)
            mix_min = float(emma_cfg.get('oracle_mix_min', 0.0) or 0.0)
            post_gate_epochs = int(emma_cfg.get('oracle_mix_post_gate_epochs', 0) or 0)
            if post_gate_epochs < 0:
                post_gate_epochs = 0
            post_gate_min = emma_cfg.get('oracle_mix_post_gate_min', None)
            if post_gate_min is None:
                post_gate_min = mix_min
            try:
                post_gate_min = float(post_gate_min)
            except Exception:
                post_gate_min = mix_min
            post_gate_min = max(mix_min, max(0.0, min(1.0, post_gate_min)))
            post_gate_start = float(emma_cfg.get('oracle_mix_post_gate_start', 1.0) or 1.0)
            post_gate_start = max(post_gate_min, min(1.0, post_gate_start))
            sched = emma_cfg.get('oracle_mix_schedule', None)
            if isinstance(sched, list) and len(sched) > 0:
                # Epoch-indexed schedule (0-based); clamp to last value if beyond length
                try:
                    alpha = float(sched[min(epoch, len(sched)-1)])
                except Exception:
                    alpha = 0.0
            else:
                if epoch < warm:
                    alpha = 1.0
                else:
                    if ramp and ramp > 0:
                        # Decay from 1 -> mix_min over `ramp` epochs after warm-start
                        k = 1.0 - (epoch - warm + 1) / float(ramp)
                        alpha = max(0.0, min(1.0, k))
                    else:
                        alpha = 0.0
                # Clamp to a minimum mixing floor
                alpha = max(mix_min, alpha)
            if nce_gate_state and post_gate_epochs > 0:
                steps_open = max(0, nce_open_steps - 1)
                progress = min(steps_open, post_gate_epochs)
                if post_gate_epochs == 0:
                    gate_alpha = post_gate_min
                elif post_gate_epochs == 1:
                    gate_alpha = post_gate_min
                else:
                    frac = progress / float(post_gate_epochs)
                    gate_alpha = post_gate_start * (1.0 - frac) + post_gate_min * frac
                alpha = max(alpha, max(post_gate_min, min(post_gate_start, gate_alpha)))
            model.oracle_mix_alpha = float(alpha)
        # Mem-injection scheduling: ramp mem_scale up as oracle mix decays
        if model_type != 'gru' and getattr(model, 'mem_into_deq', False):
            try:
                alpha = float(getattr(model, 'oracle_mix_alpha', 0.0) or 0.0)  # 1 early, ->0 later
                target = float(cfg.get('emma', {}).get('mem_scale', 1.0))
                mmin   = float(cfg.get('emma', {}).get('mem_scale_min', 0.0))
                rng = max(1e-6, 1.0 - float(cfg.get('emma', {}).get('oracle_mix_min', 0.0) or 0.0))
                s = (1.0 - alpha) / rng
                s = max(0.0, min(1.0, s))
            except Exception:
                s = None
            try:
                if s is None:
                    pass
                else:
                    # s in [0,1]: 0 when alpha≈1 (early), 1 when alpha≈mix_min (late)
                    model.mem_scale = mmin + s * (target - mmin)
            except Exception:
                pass
        # Apply LR step right after warm-start if configured
        if epoch == warm and cfg.get('train', {}).get('lr_after_warm_factor', 1.0) < 1.0:
            factor = float(cfg['train']['lr_after_warm_factor'])
            for g in optimizer.param_groups:
                g['lr'] = g['lr'] * factor
            rprint(f"[lr-step] epoch={epoch+1} new_lr={optimizer.param_groups[0]['lr']:.2e}")
            base_param_group_lrs = [group['lr'] for group in optimizer.param_groups]
        # Log current temperature clamp value once per epoch
        try:
            if model_type != 'gru' and hasattr(model, 'logit_scale'):
                ls = getattr(model, 'logit_scale')
                ls_val = float(ls.detach().cpu().item()) if hasattr(ls, 'detach') else float(ls)
                rprint(f"[temp] epoch={epoch+1} logit_scale={ls_val:.3f}")
        except Exception:
            pass
        lam_nce_used = max(float(lam_nce_current), float(schedule_value))

        epoch_train_metrics = {} if enable_extended_epoch_metrics else None
        if task == 'listops_lite':
            # Classification mode
            def _train_epoch_classify():
                model.train()
                tot_loss=tot_acc=0.0; n=0; fp_sum=0.0; fp_n=0; wcos_sum=0.0; wcos_n=0
                for batch in train_loader:
                    tokens = batch['tokens'].to(device)
                    target = batch['target'].to(device)
                    optimizer.zero_grad(set_to_none=True)
                    logits, metrics = model.forward_classify(tokens, labels=target)
                    loss = criterion(logits, target)
                    # Optional addressing entropy regularizer
                    try:
                        if addr_ent_w > 0.0:
                            ae = metrics.get('addr_entropy', None)
                            if ae is not None:
                                loss = loss + addr_ent_w * float(ae)
                    except Exception:
                        pass
                    # Optional self-distillation from deeper DEQ teacher
                    if distill_enable and distill_weight > 0.0 and distill_teacher_deq > 0:
                        saved_k = getattr(getattr(model, 'deq', None), 'max_iter', None)
                        try:
                            if saved_k is not None:
                                model.deq.max_iter = distill_teacher_deq
                            with torch.no_grad():
                                t_logits, _ = model.forward_classify(tokens, labels=target)
                        finally:
                            try:
                                if saved_k is not None:
                                    model.deq.max_iter = saved_k
                            except Exception:
                                pass
                        t = distill_temp if distill_temp > 0 else 1.0
                        log_p = torch.log_softmax(logits / t, dim=-1)
                        q = torch.softmax(t_logits / t, dim=-1)
                        kld = torch.sum(q * (torch.log(q.clamp_min(1e-9)) - log_p), dim=-1).mean()
                        loss = loss + distill_weight * kld
                    try:
                        if lam_jac > 0.0:
                            resid = metrics.get('residual_norm', None)
                            if resid is not None:
                                loss = loss + lam_jac * float(resid)
                    except Exception:
                        pass
                    loss.backward(); optimizer.step()
                    B = target.size(0); tot_loss += float(loss.item())*B; tot_acc += (logits.argmax(-1)==target).float().sum().item(); n+=B
                    fpi = metrics.get('avg_fp_iters', None)
                    if fpi==fpi: fp_sum += float(fpi); fp_n += 1
                    wc = metrics.get('write_cos', None)
                    try:
                        if wc is not None and wc==wc: wcos_sum += float(wc); wcos_n += 1
                    except Exception: pass
                avg_fp = fp_sum/max(1,fp_n); avg_w = wcos_sum/max(1,wcos_n)
                return tot_loss/max(1,n), tot_acc/max(1,n), avg_fp, avg_w, float('nan')
            tr_loss, tr_acc, avg_fp_epoch, avg_w_epoch, avg_r_epoch = _train_epoch_classify()
            if epoch_train_metrics is not None:
                epoch_train_metrics.update({
                    'train_write_cos_min': float('nan'),
                    'train_write_cos_max': float('nan'),
                    'train_write_cos_std': float('nan'),
                    'train_read_cos_min': float('nan'),
                    'train_read_cos_max': float('nan'),
                    'train_read_cos_std': float('nan'),
                    'train_bucket_usage_mean': float('nan'),
                    'train_bucket_usage_min': float('nan'),
                    'train_bucket_usage_max': float('nan'),
                    'train_read_bucket_mean': float('nan'),
                    'train_read_bucket_min': float('nan'),
                    'train_read_bucket_max': float('nan'),
                    'train_read_gap_mean': float('nan'),
                    'train_bucket_collision_rate': float('nan'),
                    'train_bucket_collision_mean': float('nan'),
                    'train_bucket_collision_max': float('nan'),
                    'train_bucket_load_variance': float('nan'),
                })
        else:
            tr_loss, tr_acc, avg_fp_epoch, avg_w_epoch, avg_r_epoch = train_epoch(
                model,
                train_loader,
                device,
                optimizer,
                criterion,
                model_type,
                lam_pred=lam_pred,
                lam_write=lam_write,
                lam_nce=lam_nce_used,
                lam_jac=lam_jac,
                logprint=logprint,
                epoch_metrics=epoch_train_metrics,
                epoch_index=epoch,
                sample_logger=train_sample_logger,
                load_balance_coeff=load_balance_coeff,
            )

        if lam_nce_used > 0.0:
            rprint(f"[nce-used] epoch={epoch+1} lambda_nce={lam_nce_used:.3f}")

        last_used_lambda = lam_nce_used

        # Eval-time toggles for causality tests
        if model_type != 'gru':
            try:
                model._eval_disable_write = bool(args.eval_no_write)
                model._eval_shuffle_read = bool(args.eval_shuffle_read)
            except Exception:
                pass
        
        if task == 'listops_lite':
            def _eval_epoch_classify():
                model.eval(); tot_loss=tot_acc=0.0; n=0; read_cos_last=float('nan'); write_cos_last=float('nan'); topk=float('nan')
                with torch.no_grad():
                    for batch in val_loader:
                        tokens = batch['tokens'].to(device)
                        target = batch['target'].to(device)
                        logits, metrics = model.forward_classify(tokens, labels=target)
                        loss = criterion(logits, target)
                    # Optional addressing entropy regularizer
                    try:
                        if addr_ent_w > 0.0:
                            ae = metrics.get('addr_entropy', None)
                            if ae is not None:
                                loss = loss + addr_ent_w * float(ae)
                    except Exception:
                        pass
                    # Optional self-distillation from deeper DEQ teacher
                    if distill_enable and distill_weight > 0.0 and distill_teacher_deq > 0:
                        saved_k = getattr(getattr(model, 'deq', None), 'max_iter', None)
                        try:
                            if saved_k is not None:
                                model.deq.max_iter = distill_teacher_deq
                            with torch.no_grad():
                                t_logits, _ = model.forward_classify(tokens, labels=target)
                        finally:
                            try:
                                if saved_k is not None:
                                    model.deq.max_iter = saved_k
                            except Exception:
                                pass
                        t = distill_temp if distill_temp > 0 else 1.0
                        log_p = torch.log_softmax(logits / t, dim=-1)
                        q = torch.softmax(t_logits / t, dim=-1)
                        kld = torch.sum(q * (torch.log(q.clamp_min(1e-9)) - log_p), dim=-1).mean()
                        loss = loss + distill_weight * kld
                        B = target.size(0); tot_loss += float(loss.item())*B; tot_acc += (logits.argmax(-1)==target).float().sum().item(); n+=B
                        try:
                            wc = metrics.get('write_cos', None)
                            if wc==wc: write_cos_last = float(wc)
                        except Exception: pass
                return (
                    tot_loss / max(1, n),
                    tot_acc / max(1, n),
                    read_cos_last,
                    write_cos_last,
                    topk,
                    float('nan'),
                    float('nan'),
                )
            va_out = _eval_epoch_classify()
        else:
            va_out = eval_epoch(
                model,
                val_loader,
                device,
                criterion,
                model_type,
                sample_logger=val_sample_logger,
                epoch_index=epoch,
            )

        va_addr_last = float('nan')
        va_resid_last = float('nan')
        va_read_raw_last = float('nan')
        va_cleanup_last = float('nan')
        va_writes_mean_last = float('nan')
        va_write_bucket_entropy_last = float('nan')
        if isinstance(va_out, (tuple, list)):
            if len(va_out) >= 11:
                (
                    va_loss,
                    va_acc,
                    va_read_last,
                    va_write_last,
                    va_topk,
                    va_addr_last,
                    va_resid_last,
                    va_read_raw_last,
                    va_cleanup_last,
                    va_writes_mean_last,
                    va_write_bucket_entropy_last,
                ) = va_out[:11]
            elif len(va_out) >= 9:
                (
                    va_loss,
                    va_acc,
                    va_read_last,
                    va_write_last,
                    va_topk,
                    va_addr_last,
                    va_resid_last,
                    va_read_raw_last,
                    va_cleanup_last,
                ) = va_out[:9]
            elif len(va_out) >= 7:
                va_loss, va_acc, va_read_last, va_write_last, va_topk, va_addr_last, va_resid_last = va_out[:7]
            elif len(va_out) == 5:
                va_loss, va_acc, va_read_last, va_write_last, va_topk = va_out
            else:
                raise ValueError(f"Unexpected eval_epoch() output length: {len(va_out)}")
        elif isinstance(va_out, dict):
            va_loss = float(va_out.get('val_loss', float('nan')))
            va_acc = float(va_out.get('val_acc', float('nan')))
            va_read_last = float(va_out.get('val_read_last', float('nan')))
            va_write_last = float(va_out.get('val_write_last', float('nan')))
            va_topk = float(va_out.get('val_topk', float('nan')))
            va_addr_last = float(va_out.get('addr_entropy_last', float('nan')))
            va_resid_last = float(va_out.get('residual_norm_last', float('nan')))
            va_read_raw_last = float(va_out.get('read_cos_raw_last', float('nan')))
            va_cleanup_last = float(va_out.get('read_cleanup_last', float('nan')))
            va_writes_mean_last = float(va_out.get('writes_per_seq_mean_last', float('nan')))
            va_write_bucket_entropy_last = float(va_out.get('write_bucket_entropy_last', float('nan')))
        else:
            raise TypeError(f"Unexpected eval_epoch() return type: {type(va_out)}")
        final_val_loss = va_loss
        try:
            if (va_cleanup_last == va_cleanup_last) and (va_read_raw_last == va_read_raw_last):
                va_read_gap_last = float(va_cleanup_last - va_read_raw_last)
            else:
                va_read_gap_last = float('nan')
        except Exception:
            va_read_gap_last = float('nan')
        try:
            last_avg_w = float(avg_w_epoch)
            if not (avg_w_epoch != avg_w_epoch):
                nce_ema = nce_ema_beta * nce_ema + (1.0 - nce_ema_beta) * float(avg_w_epoch)
            else:
                nce_ema = nce_ema_beta * nce_ema
        except Exception:
            pass
        try:
            if not (avg_r_epoch != avg_r_epoch):
                read_ema = nce_read_ema_beta * read_ema + (1.0 - nce_read_ema_beta) * float(avg_r_epoch)
            else:
                read_ema = nce_read_ema_beta * read_ema
        except Exception:
            pass

        if lam_nce_used > 0.0 and (va_acc == va_acc):
            if math.isnan(post_gate_best_acc):
                post_gate_best_acc = va_acc
            else:
                post_gate_best_acc = max(post_gate_best_acc, va_acc)
        if not math.isnan(pre_gate_best_acc) and not math.isnan(post_gate_best_acc):
            delta_post_vs_pre = post_gate_best_acc - pre_gate_best_acc
        else:
            delta_post_vs_pre = float('nan')

        if lam_nce_used > 0.0 and (va_acc == va_acc) and not math.isnan(pre_gate_best_acc):
            drop = pre_gate_best_acc - va_acc
            if drop > drop_threshold:
                post_gate_drop_epochs += 1
            else:
                post_gate_drop_epochs = 0
            if post_gate_drop_epochs >= drop_patience and not had_sustained_drop:
                had_sustained_drop = True
                if early_stop_epoch is None:
                    early_stop_epoch = epoch + 1
                rprint(
                    f"[early-stop] epoch={epoch+1} drop={drop:.3f} thresh={drop_threshold:.3f} "
                    f"patience={post_gate_drop_epochs}/{drop_patience}"
                )
                stop_training = True
        elif lam_nce_used <= 0.0:
            post_gate_drop_epochs = 0
        logprint(f"METRIC post_gate_drop_epochs={post_gate_drop_epochs}")

        # Note: gate decisions for the next epoch will use the updated EMAs computed below

        try:
            deq_last_res = float(getattr(getattr(model, 'deq', object()), '_last_residual', float('nan')))
            if deq_last_res == deq_last_res:
                logprint(f"METRIC deq_last_residual={deq_last_res:.6f}")
        except Exception:
            pass

        logprint(f"METRIC gate_state_after={int(nce_gate_state)}")

        if not (avg_fp_epoch != avg_fp_epoch):  # not NaN
            avg_fp_iters_running.append(avg_fp_epoch)
        logprint(f"[bold green]Epoch {epoch+1}/{cfg['train']['epochs']}[/bold green]  train_loss={tr_loss:.4f} acc={tr_acc:.3f}  val_loss={va_loss:.4f} acc={va_acc:.3f}")
        try:
            if va_read_raw_last == va_read_raw_last:
                logprint(f"METRIC val_read_cos_raw_last={va_read_raw_last:.6f}")
        except Exception:
            pass
        try:
            if va_cleanup_last == va_cleanup_last:
                logprint(f"METRIC val_read_cleanup_last={va_cleanup_last:.6f}")
        except Exception:
            pass
        try:
            if va_writes_mean_last == va_writes_mean_last:
                logprint(f"METRIC val_writes_per_seq_mean={va_writes_mean_last:.6f}")
        except Exception:
            pass
        try:
            if va_write_bucket_entropy_last == va_write_bucket_entropy_last:
                logprint(f"METRIC val_write_bucket_entropy={va_write_bucket_entropy_last:.6f}")
        except Exception:
            pass
        if va_resid_last == va_resid_last:
            residual_history.append(float(va_resid_last))
        else:
            residual_history.append(float('nan'))
        reads_ready_trip = (not math.isnan(read_ema)) and (read_ema >= trip_read_threshold)
        writes_ready_trip = (not math.isnan(nce_ema)) and (nce_ema >= trip_write_threshold)
        acc_ready_trip = True
        if trip_acc_target is not None and va_acc == va_acc:
            acc_ready_trip = (va_acc >= trip_acc_target)
        residual_ok_trip = True
        if len(residual_history) >= 2 and trip_residual_tolerance >= 0:
            prev_res = residual_history[-2]
            curr_res = residual_history[-1]
            if prev_res == prev_res and curr_res == curr_res:
                residual_ok_trip = (curr_res <= prev_res + trip_residual_tolerance)
        tripwire_ready = int(reads_ready_trip and writes_ready_trip and residual_ok_trip and acc_ready_trip)
        logprint(f"METRIC tripwire_reads_ready={int(reads_ready_trip)}")
        logprint(f"METRIC tripwire_writes_ready={int(writes_ready_trip)}")
        logprint(f"METRIC tripwire_acc_ready={int(acc_ready_trip)}")
        logprint(f"METRIC tripwire_residual_ok={int(residual_ok_trip)}")
        logprint(f"METRIC tripwire_ready={tripwire_ready}")
        if not math.isnan(va_acc):
            val_acc_history.append(float(va_acc))
        nce_scheduler.decrement_oracle_hold()
        pre_gate_str = 'nan' if math.isnan(pre_gate_best_acc) else f"{pre_gate_best_acc:.6f}"
        post_gate_str = 'nan' if math.isnan(post_gate_best_acc) else f"{post_gate_best_acc:.6f}"
        delta_str = 'nan' if math.isnan(delta_post_vs_pre) else f"{delta_post_vs_pre:.6f}"
        gate_epoch_str = str(gate_open_epoch_metric) if gate_open_epoch_metric is not None else 'nan'

        # Accumulate per-epoch metrics for optional CSV dump
        if metrics_csv:
            row = [
                epoch + 1,
                f"{tr_loss:.6f}",
                f"{tr_acc:.6f}",
                f"{va_loss:.6f}",
                f"{va_acc:.6f}",
                (f"{avg_fp_epoch:.6f}" if (avg_fp_epoch==avg_fp_epoch) else 'nan'),
                (f"{va_read_last:.6f}" if (va_read_last==va_read_last) else 'nan'),
                (f"{va_read_raw_last:.6f}" if (va_read_raw_last==va_read_raw_last) else 'nan'),
                (f"{va_cleanup_last:.6f}" if (va_cleanup_last==va_cleanup_last) else 'nan'),
                (f"{va_read_gap_last:.6f}" if (va_read_gap_last==va_read_gap_last) else 'nan'),
                (f"{va_writes_mean_last:.6f}" if (va_writes_mean_last==va_writes_mean_last) else 'nan'),
                (f"{va_write_bucket_entropy_last:.6f}" if (va_write_bucket_entropy_last==va_write_bucket_entropy_last) else 'nan'),
                (f"{va_write_last:.6f}" if (va_write_last==va_write_last) else 'nan'),
                (f"{va_topk:.6f}" if (va_topk==va_topk) else 'nan'),
                (f"{va_addr_last:.6f}" if (va_addr_last==va_addr_last) else 'nan'),
                (f"{va_resid_last:.6f}" if (va_resid_last==va_resid_last) else 'nan'),
                f"{lam_nce_used:.6f}",
                f"{nce_ema:.6f}",
                f"{read_ema:.6f}",
                pre_gate_str,
                post_gate_str,
                delta_str,
                gate_epoch_str,
            ]
            if enable_extended_epoch_metrics:
                metrics_source = epoch_train_metrics or {}
                for key in epoch_header_extra:
                    val = metrics_source.get(key, float('nan'))
                    if isinstance(val, str):
                        row.append(val)
                    else:
                        try:
                            if val != val:
                                row.append('nan')
                            else:
                                row.append(f"{float(val):.6f}")
                        except Exception:
                            row.append('nan')
            csv_rows.append(row)
        if va_acc > best_val:
            best_val = va_acc
            best_epoch = epoch + 1
            if args.checkpoint:
                os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': best_epoch,
                    'best_val_acc': best_val,
                    'config': cfg,
                    'model_type': model_type,
                }, args.checkpoint)
                logprint(f"[checkpoint] saved best to {args.checkpoint}")
        # Track post-warm best
        warm = cfg.get('emma', {}).get('warm_start_epochs', 0)
        if epoch >= warm and va_acc > best_val_postwarm:
            best_val_postwarm = va_acc
        if lr_restart_remaining > 0:
            lr_restart_remaining -= 1
            if lr_restart_remaining == 0:
                for idx, group in enumerate(optimizer.param_groups):
                    base_lr = base_param_group_lrs[idx] if idx < len(base_param_group_lrs) else group['lr']
                    group['lr'] = base_lr
                base_param_group_lrs = [group['lr'] for group in optimizer.param_groups]
                logprint(f"[lr-restart] restore_lr={optimizer.param_groups[0]['lr']:.2e}")
        if stop_training:
            break
    if not math.isnan(pre_gate_best_acc) and not math.isnan(post_gate_best_acc):
        delta_post_vs_pre = post_gate_best_acc - pre_gate_best_acc
    else:
        delta_post_vs_pre = float('nan')
    if not math.isnan(delta_post_vs_pre):
        gate_success = abs(delta_post_vs_pre) <= 0.005 and not had_sustained_drop
    else:
        gate_success = None
    if total_epochs > 0:
        gate_open_fraction = gate_open_epoch_count / float(total_epochs)
    else:
        gate_open_fraction = float('nan')
    wall = time.perf_counter() - t0
    logprint(f"[bold cyan]Best val accuracy: {best_val:.3f}[/bold cyan]")
    # Emit parse-friendly metrics
    avg_fp_overall = (sum(avg_fp_iters_running) / len(avg_fp_iters_running)) if avg_fp_iters_running else float('nan')
    logprint(f"METRIC best_val_acc={best_val:.6f}")
    try:
        logprint(f"METRIC best_val_acc_postwarm={best_val_postwarm:.6f}")
    except Exception:
        pass
    if final_val_loss is not None:
        logprint(f"METRIC final_val_loss={final_val_loss:.6f}")
    if not (avg_fp_overall != avg_fp_overall):
        logprint(f"METRIC avg_fp_iters={avg_fp_overall:.6f}")
    gate_epoch_metric = nce_scheduler.gate_epoch_metric
    if gate_epoch_metric is not None:
        logprint(f"METRIC nce_gate_epoch={int(gate_epoch_metric)}")
    else:
        logprint("METRIC nce_gate_epoch=nan")
    logprint(f"METRIC nce_lambda_last={last_used_lambda:.6f}")
    logprint(f"METRIC nce_ema_write_cos={nce_ema:.6f}")
    logprint(f"METRIC nce_read_ema_cos={read_ema:.6f}")
    if nce_gate_reason is not None:
        logprint(f"METRIC nce_gate_reason={nce_gate_reason}")
    if gate_open_epoch_metric is not None:
        logprint(f"METRIC gate_open_epoch={int(gate_open_epoch_metric)}")
    else:
        logprint("METRIC gate_open_epoch=nan")
    if not math.isnan(pre_gate_best_acc):
        logprint(f"METRIC pre_gate_best_acc={pre_gate_best_acc:.6f}")
    else:
        logprint("METRIC pre_gate_best_acc=nan")
    if not math.isnan(post_gate_best_acc):
        logprint(f"METRIC post_gate_best_acc={post_gate_best_acc:.6f}")
    else:
        logprint("METRIC post_gate_best_acc=nan")
    if not math.isnan(delta_post_vs_pre):
        logprint(f"METRIC delta_post_vs_pre={delta_post_vs_pre:.6f}")
    else:
        logprint("METRIC delta_post_vs_pre=nan")
    if gate_success is not None:
        logprint(f"METRIC gate_success={int(gate_success)}")
    else:
        logprint("METRIC gate_success=nan")
    if early_stop_epoch is not None:
        logprint(f"METRIC early_stop_epoch={int(early_stop_epoch)}")
    else:
        logprint("METRIC early_stop_epoch=nan")
    logprint(f"METRIC walltime_sec={wall:.3f}")
    if gate_open_fraction == gate_open_fraction:
        logprint(f"METRIC gate_open_fraction={gate_open_fraction:.6f}")
    else:
        logprint("METRIC gate_open_fraction=nan")
    if metrics_csv and csv_rows:
        import csv
        os.makedirs(os.path.dirname(metrics_csv), exist_ok=True)
        with open(metrics_csv, 'w', newline='') as f:
            w = csv.writer(f)
            header = [
                'epoch',
                'train_loss',
                'train_acc',
                'val_loss',
                'val_acc',
                'avg_fp_iters',
                'read_cos_last',
                'read_cos_raw_last',
                'read_cleanup_last',
                'val_read_gap_last',
                'writes_per_seq_mean_last',
                'write_bucket_entropy_last',
                'write_cos_last',
                'topk_hit_rate',
                'addr_entropy_last',
                'residual_norm_last',
                'lambda_nce',
                'nce_ema_write_cos',
                'nce_read_ema_cos',
                'pre_gate_best_acc',
                'post_gate_best_acc',
                'delta_post_vs_pre',
                'gate_open_epoch',
            ]
            if epoch_header_extra:
                header.extend(epoch_header_extra)
            w.writerow(header)
            w.writerows(csv_rows)

    summary = {
        'best_val_acc': float(best_val) if best_val == best_val else None,
        'best_val_acc_postwarm': float(best_val_postwarm) if best_val_postwarm == best_val_postwarm else None,
        'final_val_loss': float(final_val_loss) if final_val_loss == final_val_loss else None,
        'avg_fp_iters': float(avg_fp_overall) if avg_fp_overall == avg_fp_overall else None,
        'nce_gate_epoch': int(gate_epoch_metric) if gate_epoch_metric is not None else None,
        'nce_lambda_last': float(last_used_lambda),
        'nce_ema_write_cos': float(nce_ema) if nce_ema == nce_ema else None,
        'nce_read_ema_cos': float(read_ema) if read_ema == read_ema else None,
        'nce_gate_reason': nce_gate_reason,
        'gate_open_epoch': int(gate_open_epoch_metric) if gate_open_epoch_metric is not None else None,
        'pre_gate_best_acc': float(pre_gate_best_acc) if not math.isnan(pre_gate_best_acc) else None,
        'post_gate_best_acc': float(post_gate_best_acc) if not math.isnan(post_gate_best_acc) else None,
        'delta_post_vs_pre': float(delta_post_vs_pre) if not math.isnan(delta_post_vs_pre) else None,
        'gate_success': bool(gate_success) if gate_success is not None else None,
        'early_stop_epoch': int(early_stop_epoch) if early_stop_epoch is not None else None,
        'walltime_sec': float(wall),
        'gate_open_fraction': float(gate_open_fraction) if gate_open_fraction == gate_open_fraction else None,
    }
    if metrics_out_path and str(metrics_out_path).strip():
        summary_path = Path(metrics_out_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import json
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
        except Exception as exc:
            rprint(f"[warn] failed to write metrics-out JSON to {summary_path}: {exc}")
    if train_sample_logger is not None:
        train_sample_logger.close()
    if val_sample_logger is not None:
        val_sample_logger.close()
    if log_fh is not None:
        log_fh.close()

if __name__ == '__main__':
    main()
