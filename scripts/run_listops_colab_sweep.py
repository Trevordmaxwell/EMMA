#!/usr/bin/env python3
"""Run paired EMMA vs Transformer ListOps-Lite sweeps with resume support.

This script is designed for Colab use:
- it runs EMMA and/or Transformer runs across a seed list
- it skips finished runs by default (metrics file exists)
- it writes a compact JSON summary under experiments/results
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "experiments" / "results"
RUN_TRAINING = REPO_ROOT / "scripts" / "run_training.py"
RUN_TRANSFORMER = REPO_ROOT / "scripts" / "run_transformer_baseline.py"
LISTOPS_CONFIG = REPO_ROOT / "configs" / "listops_lite_len512_cpu.yaml"


def parse_seed_list(text: str) -> list[int]:
    values: list[int] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    if not values:
        raise ValueError("No seeds provided.")
    return values


def run_cmd(cmd: list[str]) -> None:
    print("[sweep] running:", " ".join(cmd), flush=True)
    env = os.environ.copy()
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True, env=env)


def emma_metric_path(prefix: str, seed: int) -> Path:
    return RESULTS_DIR / f"{prefix}_emma_s{seed}_metrics.json"


def transformer_metric_path(prefix: str, seed: int) -> Path:
    return RESULTS_DIR / f"{prefix}_transformer_s{seed}_metrics.json"


def run_emma_seed(
    *,
    prefix: str,
    seed: int,
    device: str,
    train_size: int,
    val_size: int,
    epochs: int,
    deq_max_iter: int,
    early_stop_patience: int,
    strict_no_nce: bool,
    skip_existing: bool,
) -> Path:
    metrics_path = emma_metric_path(prefix, seed)
    if skip_existing and metrics_path.exists():
        print(f"[sweep] skip emma seed={seed} (exists: {metrics_path.name})", flush=True)
        return metrics_path

    run_name = f"{prefix}_emma_s{seed}"
    cmd = [
        sys.executable,
        str(RUN_TRAINING),
        "--config",
        str(LISTOPS_CONFIG.relative_to(REPO_ROOT)),
        "--device",
        device,
        "--run-name",
        run_name,
        "--metrics-out",
        str(metrics_path),
        "--set",
        f"seed={seed}",
        "--set",
        f"train.epochs={epochs}",
        "--set",
        f"data.train_size={train_size}",
        "--set",
        f"data.val_size={val_size}",
        "--set",
        f"emma.deq_max_iter={deq_max_iter}",
        "--set",
        f"train.early_stop_patience={early_stop_patience}",
        "--set",
        "train.early_stop_start_epoch=2",
        "--set",
        "train.early_stop_min_delta=0.0",
    ]
    if strict_no_nce:
        cmd.extend(
            [
                "--set",
                "loss.lambda_write_nce=0.0",
                "--set",
                "emma.nce_lambda_floor=0.0",
                "--set",
                "emma.nce_lambda_increment=0.0",
                "--set",
                "emma.nce_lambda_max=0.0",
            ]
        )
    run_cmd(cmd)
    return metrics_path


def run_transformer_seed(
    *,
    prefix: str,
    seed: int,
    device: str,
    train_size: int,
    val_size: int,
    epochs: int,
    skip_existing: bool,
) -> Path:
    metrics_path = transformer_metric_path(prefix, seed)
    if skip_existing and metrics_path.exists():
        print(f"[sweep] skip transformer seed={seed} (exists: {metrics_path.name})", flush=True)
        return metrics_path

    run_name = f"{prefix}_transformer_s{seed}"
    cmd = [
        sys.executable,
        str(RUN_TRANSFORMER),
        "--config",
        str(LISTOPS_CONFIG.relative_to(REPO_ROOT)),
        "--device",
        device,
        "--run-name",
        run_name,
        "--metrics-out",
        str(metrics_path),
        "--pool",
        "mean",
        "--non-causal",
        "--set",
        f"seed={seed}",
        "--set",
        f"train.epochs={epochs}",
        "--set",
        f"data.train_size={train_size}",
        "--set",
        f"data.val_size={val_size}",
    ]
    run_cmd(cmd)
    return metrics_path


def load_metric(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(values: list[float]) -> dict:
    return {
        "n": len(values),
        "mean": statistics.mean(values) if values else None,
        "std": statistics.stdev(values) if len(values) > 1 else (0.0 if len(values) == 1 else None),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def build_summary(prefix: str, seeds: list[int]) -> dict:
    emma_rows: list[dict] = []
    transformer_rows: list[dict] = []
    paired_rows: list[dict] = []

    for seed in seeds:
        emma_path = emma_metric_path(prefix, seed)
        tr_path = transformer_metric_path(prefix, seed)
        emma = load_metric(emma_path) if emma_path.exists() else None
        tr = load_metric(tr_path) if tr_path.exists() else None

        if emma is not None:
            emma_rows.append(
                {
                    "seed": seed,
                    "best_val_acc": emma.get("best_val_acc"),
                    "best_epoch": emma.get("best_epoch"),
                    "early_stop_epoch": emma.get("early_stop_epoch"),
                    "walltime_sec": emma.get("walltime_sec"),
                    "metrics_path": str(emma_path),
                }
            )

        if tr is not None:
            transformer_rows.append(
                {
                    "seed": seed,
                    "best_val_acc": tr.get("best_val_acc"),
                    "best_epoch": tr.get("best_epoch"),
                    "walltime_sec": tr.get("walltime_sec"),
                    "metrics_path": str(tr_path),
                }
            )

        if emma is not None and tr is not None:
            e = float(emma.get("best_val_acc"))
            t = float(tr.get("best_val_acc"))
            paired_rows.append(
                {
                    "seed": seed,
                    "emma_best_val_acc": e,
                    "transformer_best_val_acc": t,
                    "delta_emma_minus_transformer": e - t,
                }
            )

    emma_vals = [float(row["best_val_acc"]) for row in emma_rows if row["best_val_acc"] is not None]
    tr_vals = [float(row["best_val_acc"]) for row in transformer_rows if row["best_val_acc"] is not None]
    paired_vals = [float(row["delta_emma_minus_transformer"]) for row in paired_rows]

    summary = {
        "created_at_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "prefix": prefix,
        "seeds": seeds,
        "emma": {
            "stats_best_val_acc": summarize(emma_vals),
            "runs": emma_rows,
        },
        "transformer": {
            "stats_best_val_acc": summarize(tr_vals),
            "runs": transformer_rows,
        },
        "paired": {
            "stats_delta_emma_minus_transformer": summarize(paired_vals),
            "rows": paired_rows,
        },
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ListOps-Lite Colab sweep.")
    parser.add_argument("--prefix", default="colab_listops_full", help="Run/metric prefix.")
    parser.add_argument("--seeds", default="60,61,62,63,64", help="Comma-separated seeds.")
    parser.add_argument("--mode", default="both", choices=["both", "emma", "transformer"], help="Which model group to run.")
    parser.add_argument("--device", default="cuda", help="Torch device for both runners.")
    parser.add_argument("--train-size", type=int, default=2000, help="ListOps train size.")
    parser.add_argument("--val-size", type=int, default=200, help="ListOps val size.")
    parser.add_argument("--epochs", type=int, default=6, help="Training epochs.")
    parser.add_argument("--deq-max-iter", type=int, default=6, help="EMMA DEQ max iter.")
    parser.add_argument("--early-stop-patience", type=int, default=2, help="EMMA generic early stop patience.")
    parser.add_argument("--strict-no-nce", action="store_true", help="Hard-disable NCE in EMMA runs.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip seeds with existing metrics files.")
    args = parser.parse_args()

    seeds = parse_seed_list(args.seeds)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode in {"both", "emma"}:
        for seed in seeds:
            run_emma_seed(
                prefix=args.prefix,
                seed=seed,
                device=args.device,
                train_size=args.train_size,
                val_size=args.val_size,
                epochs=args.epochs,
                deq_max_iter=args.deq_max_iter,
                early_stop_patience=args.early_stop_patience,
                strict_no_nce=bool(args.strict_no_nce),
                skip_existing=bool(args.skip_existing),
            )

    if args.mode in {"both", "transformer"}:
        for seed in seeds:
            run_transformer_seed(
                prefix=args.prefix,
                seed=seed,
                device=args.device,
                train_size=args.train_size,
                val_size=args.val_size,
                epochs=args.epochs,
                skip_existing=bool(args.skip_existing),
            )

    summary = build_summary(args.prefix, seeds)
    summary_path = RESULTS_DIR / f"{args.prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print("\n[sweep] summary:", summary_path, flush=True)
    print(json.dumps(summary["emma"]["stats_best_val_acc"], indent=2), flush=True)
    print(json.dumps(summary["transformer"]["stats_best_val_acc"], indent=2), flush=True)
    print(json.dumps(summary["paired"]["stats_delta_emma_minus_transformer"], indent=2), flush=True)


if __name__ == "__main__":
    main()
