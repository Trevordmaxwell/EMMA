
from __future__ import annotations
import argparse, yaml, time, sys, os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from rich import print as rprint

from src.utils.device import get_device
from src.utils.seeds import set_seed
from src.data.needle import make_dataloaders
from src.models.baselines import GRUClassifier
from src.models.emma import EMMA

def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=-1)
    return (preds == targets).float().mean().item()

def train_epoch(model, loader, device, optimizer, criterion, model_type,
                lam_pred=0.0, lam_write=0.0, lam_nce=0.0):
    import math, torch
    from torch.nn.utils import clip_grad_norm_

    model.train()
    total_loss = 0.0
    total_acc  = 0.0
    n = 0

    fp_sum = 0.0; fp_n = 0
    wcos_sum = 0.0; wcos_n = 0
    rcos_sum = 0.0; rcos_n = 0

    for batch in loader:
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
            elif nce is not None:
                try:
                    loss = loss + lam_nce * float(nce)
                except Exception:
                    pass

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
                    if not math.isnan(wc): wcos_sum += wc; wcos_n += 1
            except Exception: pass
            try:
                rc = metrics.get('read_cos', None)
                if rc is not None:
                    rc = float(rc)
                    if not math.isnan(rc): rcos_sum += rc; rcos_n += 1
            except Exception: pass

    avg_fp = fp_sum / max(1, fp_n)
    avg_w  = wcos_sum / max(1, wcos_n)
    avg_r  = rcos_sum / max(1, rcos_n)

    try:
        from rich import print as rprint
        rprint(f" avg_write_cos={avg_w:.4f} avg_read_cos={avg_r:.4f} avg_fp_iters={avg_fp:.3f}")
    except Exception:
        pass

    return total_loss / max(1, n), total_acc / max(1, n), avg_fp, avg_w, avg_r

def eval_epoch(model, loader, device, criterion, model_type: str):
    # reduce DEQ work during eval for speed
    saved_deq_iter = getattr(getattr(model, 'deq', None), 'max_iter', None)
    try:
        if saved_deq_iter is not None and saved_deq_iter > 5:
            model.deq.max_iter = 5
    except Exception:
        pass
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    # Epoch-end aggregations
    import math
    read_cos_vals = []
    write_cos_vals = []
    topk_hits = 0
    topk_total = 0
    for batch in tqdm(loader, desc=f"eval/{model_type}", leave=False):
        tokens = batch['tokens'].to(device)
        key_id = batch['key_id'].to(device)
        write_pos = batch['write_pos'].to(device)
        query_pos = batch['query_pos'].to(device)
        target = batch['target'].to(device)
        if model_type == 'gru':
            logits = model(tokens, query_pos)
            loss = criterion(logits, target)
        else:
            # Pass through any eval control flags set on model (set in main)
            disable_write = bool(getattr(model, '_eval_disable_write', False))
            shuffle_read  = bool(getattr(model, '_eval_shuffle_read', False))
            logits, metrics = model(tokens, key_id, write_pos, query_pos, value_ids=target, current_epoch=0,
                                    disable_write=disable_write, shuffle_read=shuffle_read)
            loss = criterion(logits, target)
            # Aggregate epoch-end metrics, if present
            if isinstance(metrics, dict):
                rc = metrics.get('read_cos', None)
                wc = metrics.get('write_cos', None)
                try:
                    if rc is not None and not math.isnan(float(rc)):
                        read_cos_vals.append(float(rc))
                except Exception: pass
                try:
                    if wc is not None and not math.isnan(float(wc)):
                        write_cos_vals.append(float(wc))
                except Exception: pass
            # Top-k hit rate (using memory k_top if available)
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
        total_acc  += (torch.argmax(logits, dim=-1) == target).float().sum().item()
        n += bs
    # Safe means
    import numpy as _np
    read_cos_last = float(_np.mean(read_cos_vals)) if read_cos_vals else float('nan')
    write_cos_last = float(_np.mean(write_cos_vals)) if write_cos_vals else float('nan')
    topk_hit_rate = (float(topk_hits) / float(max(1, topk_total))) if topk_total > 0 else float('nan')
    return total_loss / max(1, n), total_acc / max(1, n), read_cos_last, write_cos_last, topk_hit_rate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/needle_tiny.yaml')
    parser.add_argument('--model', type=str, default='gru', choices=['gru', 'emma_gru', 'emma_liquid'])
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--logfile', type=str, default=None, help='Optional path to also write logs (tee).')
    parser.add_argument('--save-config', type=str, default=None, help='Optional path to write the resolved config YAML.')
    parser.add_argument('--checkpoint', type=str, default=None, help='Optional path to save best model checkpoint.')
    parser.add_argument('--metrics-out', type=str, default=None, help='Optional CSV to append per-epoch eval metrics for Phase 1 demo.')
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

    set_seed(cfg.get('seed', 42))
    device = get_device(args.device)

    train_loader, val_loader = make_dataloaders(
        n_pairs=cfg['data'].get('n_pairs', 1),
        decouple_kv=cfg['data'].get('decouple_kv', True),
        num_values=cfg['data']['num_values'],
        vocab_extra=cfg['data']['vocab_extra'],
        length=cfg['data']['length'],
        train_size=cfg['data']['train_size'],
        val_size=cfg['data']['val_size'],
        batch_size=cfg['train']['batch_size'],
        num_workers=cfg['train'].get('num_workers', 0),
        seed=cfg.get('seed', 42)
    )
    vocab_size = next(iter(train_loader))['vocab_size']

    model_type = args.model
    if model_type == 'gru':
        model = GRUClassifier(vocab_size=vocab_size, emb_dim=cfg['model']['emb_dim'], hid_dim=cfg['model']['hid_dim'], num_values=cfg['data']['num_values'])
    else:
        emma_cfg = cfg.get('emma', {})
        cfg_emma = cfg.get('emma', {})
        model = EMMA(vocab_size=vocab_size,
                     warm_start_epochs=cfg_emma.get('warm_start_epochs', 0),
                     emb_dim=cfg['model']['emb_dim'],
                     hid_dim=cfg['model']['hid_dim'],
                     mem_dim=cfg['model']['mem_dim'],
                     num_values=cfg['data']['num_values'],
                     n_slots=cfg['memory']['n_slots'],
                     k_top=cfg['memory']['k_top'],
                     oracle_write=emma_cfg.get('oracle_write', False),
                     deq_max_iter=emma_cfg.get('deq_max_iter', 15),
                     mem_into_deq=emma_cfg.get('mem_into_deq', False),
                     )
    model.to(device)

    # Optional logit temperature clamp limit
    try:
        if model_type != 'gru' and hasattr(model, 'logit_scale_max'):
            model.logit_scale_max = cfg.get('emma', {}).get('logit_scale_max', model.logit_scale_max)
    except Exception:
        pass

    criterion = nn.CrossEntropyLoss()
    optimizer = (
        optim.SGD(model.parameters(), lr=cfg['train']['lr'], momentum=0.9, weight_decay=cfg['train'].get('wd', 1e-2))
        if (str(device) == 'mps' and model_type != 'gru')
        else optim.AdamW(model.parameters(), lr=cfg['train']['lr'], weight_decay=cfg['train'].get('wd', 1e-2))
    )

    lam_pred = cfg.get('loss', {}).get('lambda_pred_ce', 0.0)
    lam_write = cfg.get('loss', {}).get('lambda_write_cos', 0.0)
    base_lam_nce = cfg.get('loss', {}).get('lambda_write_nce', 0.0)
    warm_start_epochs = cfg.get('emma', {}).get('warm_start_epochs', 0)

    best_val = 0.0
    best_val_postwarm = 0.0
    warm = cfg.get('emma', {}).get('warm_start_epochs', 0)
    best_epoch = -1
    final_val_loss = None
    avg_fp_iters_running = []
    t0 = time.perf_counter()
    # Gate InfoNCE based on write-cos lift-off
    nce_gate = float(cfg.get('emma', {}).get('nce_gate_write_cos', 0.15) or 0.0)
    last_avg_w = 0.0

    # metrics CSV init (Phase 1 demo)
    metrics_csv = args.metrics_out
    wrote_header = False
    for epoch in range(cfg['train']['epochs']):
        # Warm-start schedule: teacher-force writes for first N epochs (0-indexed: epochs [0..warm-1])
        if model_type != 'gru' and hasattr(model, 'oracle_write'):
            warm = cfg.get('emma', {}).get('warm_start_epochs', 0)
            model.oracle_write = (epoch < warm)
            if warm > 0 and epoch < warm:
                rprint(f"[warm-start] epoch={epoch+1} using oracle_write=True")
        # Oracle/pred mix ramp alpha (set on model for non-diff write path)
        if model_type != 'gru' and hasattr(model, 'oracle_mix_alpha'):
            warm = cfg.get('emma', {}).get('warm_start_epochs', 0)
            ramp = cfg.get('emma', {}).get('oracle_mix_ramp_epochs', 0)
            mix_min = float(cfg.get('emma', {}).get('oracle_mix_min', 0.0) or 0.0)
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
        # Log current temperature clamp value once per epoch
        try:
            if model_type != 'gru' and hasattr(model, 'logit_scale'):
                ls = getattr(model, 'logit_scale')
                ls_val = float(ls.detach().cpu().item()) if hasattr(ls, 'detach') else float(ls)
                rprint(f"[temp] epoch={epoch+1} logit_scale={ls_val:.3f}")
        except Exception:
            pass
        # Gate InfoNCE after write head shows life
        lam_nce = base_lam_nce if (nce_gate <= 0.0 or last_avg_w >= nce_gate) else 0.0
        if lam_nce > 0.0:
            rprint(f"[nce] epoch={epoch+1} lambda_nce={lam_nce:.3f}")

        tr_loss, tr_acc, avg_fp_epoch, avg_w_epoch, avg_r_epoch = train_epoch(
            model, train_loader, device, optimizer, criterion, model_type,
            lam_pred, lam_write, lam_nce)
        # Eval-time toggles for causality tests
        if model_type != 'gru':
            try:
                model._eval_disable_write = bool(args.eval_no_write)
                model._eval_shuffle_read = bool(args.eval_shuffle_read)
            except Exception:
                pass
        va_out = eval_epoch(model, val_loader, device, criterion, model_type)
        va_loss, va_acc, va_read_last, va_write_last, va_topk = va_out
        final_val_loss = va_loss
        try:
            last_avg_w = float(avg_w_epoch)
        except Exception:
            pass
        if not (avg_fp_epoch != avg_fp_epoch):  # not NaN
            avg_fp_iters_running.append(avg_fp_epoch)
        logprint(f"[bold green]Epoch {epoch+1}/{cfg['train']['epochs']}[/bold green]  train_loss={tr_loss:.4f} acc={tr_acc:.3f}  val_loss={va_loss:.4f} acc={va_acc:.3f}")
        # Optionally append per-epoch metrics to CSV for Phase 1
        if metrics_csv:
            import csv
            os.makedirs(os.path.dirname(metrics_csv), exist_ok=True)
            write_header = (not wrote_header) and (not os.path.exists(metrics_csv) or os.path.getsize(metrics_csv) == 0)
            with open(metrics_csv, 'a', newline='') as f:
                w = csv.writer(f)
                if write_header:
                    w.writerow(['epoch','train_loss','train_acc','val_loss','val_acc','avg_fp_iters','read_cos_last','write_cos_last','topk_hit_rate'])
                    wrote_header = True
                w.writerow([epoch+1, f"{tr_loss:.6f}", f"{tr_acc:.6f}", f"{va_loss:.6f}", f"{va_acc:.6f}",
                           (f"{avg_fp_epoch:.6f}" if (avg_fp_epoch==avg_fp_epoch) else 'nan'),
                           (f"{va_read_last:.6f}" if (va_read_last==va_read_last) else 'nan'),
                           (f"{va_write_last:.6f}" if (va_write_last==va_write_last) else 'nan'),
                           (f"{va_topk:.6f}" if (va_topk==va_topk) else 'nan')])
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
    logprint(f"METRIC walltime_sec={wall:.3f}")
    if log_fh is not None:
        log_fh.close()

if __name__ == '__main__':
    main()
