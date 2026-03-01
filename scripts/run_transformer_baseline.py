#!/usr/bin/env python3
"""Run a Transformer encoder baseline on EMMA configs.

This launcher mirrors `scripts/run_training.py` in spirit:
- loads a YAML config
- applies `--set` dotted overrides
- writes resolved config and metrics artifacts under `experiments/`
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emma.data import make_dataloaders, make_dataloaders_listops_lite


DEFAULT_EXPERIMENT_DIR = REPO_ROOT / "experiments"


def parse_set_args(values: Iterable[str]) -> list[tuple[list[str], str]]:
    overrides: list[tuple[list[str], str]] = []
    for item in values or []:
        if "=" not in item:
            raise ValueError(f"Invalid --set override '{item}'. Expected key=value.")
        key, val = item.split("=", 1)
        path = key.strip().split(".")
        overrides.append((path, val.strip()))
    return overrides


def coerce_value(raw: str):
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def set_in_dict(cfg: dict, path: list[str], value) -> None:
    cursor = cfg
    for key in path[:-1]:
        if key not in cursor or not isinstance(cursor[key], dict):
            cursor[key] = {}
        cursor = cursor[key]
    cursor[path[-1]] = value


def apply_overrides(cfg: dict, overrides: list[tuple[list[str], str]]) -> dict:
    for path, raw_val in overrides:
        set_in_dict(cfg, path, coerce_value(raw_val))
    return cfg


def make_run_name(base_config: Path, explicit: str | None = None) -> str:
    if explicit:
        return explicit
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_config.stem}_transformer_{timestamp}"


def prepare_paths(run_name: str) -> tuple[Path, Path, Path, Path]:
    run_dir = DEFAULT_EXPERIMENT_DIR / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    config_dir = run_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    resolved = config_dir / "resolved.yaml"
    per_epoch = run_dir / "per_epoch.csv"
    metrics = DEFAULT_EXPERIMENT_DIR / "results" / f"{run_name}_metrics.json"
    metrics.parent.mkdir(parents=True, exist_ok=True)
    return run_dir, resolved, per_epoch, metrics


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TransformerQueryClassifier(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        num_values: int,
        max_len: int,
        emb_dim: int,
        hid_dim: int,
        n_layers: int,
        n_heads: int,
        ff_mult: float,
        dropout: float,
        causal: bool,
        pool: str,
    ) -> None:
        super().__init__()
        self.max_len = max_len
        self.causal = causal
        self.pool = pool
        self.token_embed = nn.Embedding(vocab_size, emb_dim)
        self.pos_embed = nn.Embedding(max_len, emb_dim)
        self.input_proj = nn.Linear(emb_dim, hid_dim) if emb_dim != hid_dim else nn.Identity()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hid_dim,
            nhead=n_heads,
            dim_feedforward=max(hid_dim, int(hid_dim * ff_mult)),
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(hid_dim)
        self.head = nn.Linear(hid_dim, num_values)
        self.drop = nn.Dropout(dropout)

    @staticmethod
    def _causal_mask(length: int, device: torch.device) -> torch.Tensor:
        mask = torch.full((length, length), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask

    def forward(self, tokens: torch.Tensor, query_pos: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = tokens.shape
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds model max_len {self.max_len}.")

        pos = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(bsz, -1)
        x = self.token_embed(tokens) + self.pos_embed(pos)
        x = self.input_proj(self.drop(x))
        attn_mask = self._causal_mask(seq_len, tokens.device) if self.causal else None
        z = self.encoder(x, mask=attn_mask)
        z = self.norm(z)
        idx = query_pos.view(bsz, 1, 1).expand(-1, 1, z.size(-1))
        q = z.gather(1, idx).squeeze(1)
        key_pos = (query_pos - 1).clamp(min=0)
        key_idx = key_pos.view(bsz, 1, 1).expand(-1, 1, z.size(-1))
        q_key = z.gather(1, key_idx).squeeze(1)

        if self.pool == "query":
            pooled = q
        elif self.pool == "query_key":
            pooled = q_key
        elif self.pool == "query_querykey_mean":
            pooled = 0.5 * (q + q_key)
        elif self.pool == "mean":
            pooled = z.mean(dim=1)
        elif self.pool == "last":
            pooled = z[:, -1, :]
        elif self.pool == "max":
            pooled = z.max(dim=1).values
        elif self.pool == "query_mean":
            pooled = 0.5 * (q + z.mean(dim=1))
        else:
            raise ValueError(f"Unknown pooling mode: {self.pool}")
        return self.head(pooled)


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return (logits.argmax(dim=-1) == targets).float().mean().item()


def choose_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Transformer baseline on EMMA configs.")
    parser.add_argument("--config", required=True, help="Base YAML config path (relative to repo root).")
    parser.add_argument("--set", action="append", default=[], help="Override key=value (dotted path).")
    parser.add_argument("--run-name", default=None, help="Custom run name.")
    parser.add_argument("--device", default="cpu", help="torch device (default: cpu).")
    parser.add_argument("--layers", type=int, default=2, help="Transformer encoder layers.")
    parser.add_argument("--heads", type=int, default=4, help="Attention heads.")
    parser.add_argument("--ff-mult", type=float, default=4.0, help="Feedforward multiplier.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Transformer dropout.")
    parser.add_argument(
        "--pool",
        default="query",
        choices=["query", "query_key", "query_querykey_mean", "mean", "last", "max", "query_mean"],
        help="Representation pooling strategy before classification.",
    )
    parser.add_argument("--non-causal", action="store_true", help="Disable causal masking.")
    parser.add_argument("--metrics-out", default=None, help="Optional explicit metrics JSON path.")
    parser.add_argument("--per-epoch-out", default=None, help="Optional explicit per-epoch CSV path.")
    args = parser.parse_args()

    cfg_path = (REPO_ROOT / args.config).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg = apply_overrides(cfg, parse_set_args(args.set))

    task = str(cfg.get("task", "needle") or "needle")
    if task not in {"needle", "listops_lite"}:
        raise ValueError(f"Unsupported task '{task}'. Expected one of: needle, listops_lite.")

    run_name = make_run_name(cfg_path, args.run_name)
    run_dir, resolved_cfg_path, default_per_epoch_out, default_metrics_out = prepare_paths(run_name)
    per_epoch_out = Path(args.per_epoch_out) if args.per_epoch_out else default_per_epoch_out
    metrics_out = Path(args.metrics_out) if args.metrics_out else default_metrics_out
    per_epoch_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)

    with resolved_cfg_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)

    seed = int(cfg.get("seed", 42) or 42)
    set_seed(seed)
    device = choose_device(args.device)
    print(f"[transformer-baseline] run={run_name} device={device} seed={seed}")

    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    model_cfg = cfg["model"]
    warm_start_epochs = int(cfg.get("emma", {}).get("warm_start_epochs", 0) or 0)

    if task == "listops_lite":
        train_loader, val_loader = make_dataloaders_listops_lite(
            length=data_cfg["length"],
            train_size=data_cfg["train_size"],
            val_size=data_cfg["val_size"],
            batch_size=train_cfg["batch_size"],
            num_workers=int(train_cfg.get("num_workers", 0) or 0),
            seed=seed,
        )
        num_classes = 10
    else:
        train_loader, val_loader = make_dataloaders(
            num_values=data_cfg["num_values"],
            vocab_extra=data_cfg["vocab_extra"],
            length=data_cfg["length"],
            train_size=data_cfg["train_size"],
            val_size=data_cfg["val_size"],
            batch_size=train_cfg["batch_size"],
            num_workers=int(train_cfg.get("num_workers", 0) or 0),
            seed=seed,
            n_pairs=int(data_cfg.get("n_pairs", 1) or 1),
            decouple_kv=bool(data_cfg.get("decouple_kv", True)),
        )
        num_classes = int(data_cfg["num_values"])
    vocab_size = next(iter(train_loader))["vocab_size"]

    model = TransformerQueryClassifier(
        vocab_size=vocab_size,
        num_values=num_classes,
        max_len=data_cfg["length"],
        emb_dim=int(model_cfg["emb_dim"]),
        hid_dim=int(model_cfg["hid_dim"]),
        n_layers=max(1, int(args.layers)),
        n_heads=max(1, int(args.heads)),
        ff_mult=max(1.0, float(args.ff_mult)),
        dropout=max(0.0, float(args.dropout)),
        causal=not args.non_causal,
        pool=args.pool,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("wd", 0.0) or 0.0),
    )

    epochs = int(train_cfg["epochs"])
    best_val_acc = 0.0
    best_val_acc_postwarm = 0.0
    best_epoch = 0
    final_val_loss = float("nan")
    epoch_rows: list[list[str]] = []
    t0 = time.perf_counter()

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0
        train_acc_sum = 0.0
        train_count = 0

        for batch in train_loader:
            tokens = batch["tokens"].to(device)
            query_pos = batch.get("query_pos")
            if query_pos is None:
                query_pos = torch.full(
                    (tokens.size(0),),
                    tokens.size(1) - 1,
                    dtype=torch.long,
                    device=tokens.device,
                )
            else:
                query_pos = query_pos.to(device)
            target = batch["target"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(tokens, query_pos)
            loss = criterion(logits, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            bsz = target.size(0)
            train_loss_sum += float(loss.item()) * bsz
            train_acc_sum += accuracy(logits.detach(), target) * bsz
            train_count += bsz

        model.eval()
        val_loss_sum = 0.0
        val_acc_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for batch in val_loader:
                tokens = batch["tokens"].to(device)
                query_pos = batch.get("query_pos")
                if query_pos is None:
                    query_pos = torch.full(
                        (tokens.size(0),),
                        tokens.size(1) - 1,
                        dtype=torch.long,
                        device=tokens.device,
                    )
                else:
                    query_pos = query_pos.to(device)
                target = batch["target"].to(device)
                logits = model(tokens, query_pos)
                loss = criterion(logits, target)
                bsz = target.size(0)
                val_loss_sum += float(loss.item()) * bsz
                val_acc_sum += accuracy(logits, target) * bsz
                val_count += bsz

        train_loss = train_loss_sum / max(1, train_count)
        train_acc = train_acc_sum / max(1, train_count)
        val_loss = val_loss_sum / max(1, val_count)
        val_acc = val_acc_sum / max(1, val_count)
        final_val_loss = val_loss

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
        if epoch >= warm_start_epochs:
            best_val_acc_postwarm = max(best_val_acc_postwarm, val_acc)

        epoch_rows.append(
            [
                str(epoch + 1),
                f"{train_loss:.6f}",
                f"{train_acc:.6f}",
                f"{val_loss:.6f}",
                f"{val_acc:.6f}",
            ]
        )
        print(
            f"[transformer-baseline] epoch={epoch+1}/{epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
            ,
            flush=True,
        )

    walltime_sec = time.perf_counter() - t0

    with per_epoch_out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        writer.writerows(epoch_rows)

    metrics = {
        "run_name": run_name,
        "model": "transformer_baseline",
        "best_val_acc": float(best_val_acc),
        "best_val_acc_postwarm": float(best_val_acc_postwarm),
        "best_epoch": int(best_epoch),
        "final_val_loss": float(final_val_loss),
        "walltime_sec": float(walltime_sec),
        "seed": int(seed),
        "device": str(device),
        "layers": int(args.layers),
        "heads": int(args.heads),
        "ff_mult": float(args.ff_mult),
        "dropout": float(args.dropout),
        "pool": str(args.pool),
        "causal": bool(not args.non_causal),
        "task": task,
        "config": str(cfg_path),
        "resolved_config": str(resolved_cfg_path),
        "per_epoch_out": str(per_epoch_out),
    }
    metrics_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"[transformer-baseline] best_val_acc={best_val_acc:.6f} best_epoch={best_epoch}", flush=True)
    print(f"[transformer-baseline] metrics={metrics_out}", flush=True)


if __name__ == "__main__":
    main()
