#!/usr/bin/env python3
"""Convenience launcher for EMMA training with inline overrides.

Usage examples:
    python scripts/run_training.py --config configs/cpu_len256.yaml \
        --set train.lr=0.002 --set emma.deq_max_iter=6 --run-name lr2e3_deq6

    python scripts/run_training.py --config configs/cpu_len512_nceA.yaml --device cpu
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_DIR = REPO_ROOT / "experiments"


def parse_set_args(values: Iterable[str]) -> list[tuple[list[str], str]]:
    overrides = []
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
    if lowered == "null" or lowered == "none":
        return None
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def set_in_dict(cfg: dict, path: list[str], value):
    cursor = cfg
    for key in path[:-1]:
        if key not in cursor or not isinstance(cursor[key], dict):
            cursor[key] = {}
        cursor = cursor[key]
    cursor[path[-1]] = value


def apply_overrides(cfg: dict, overrides: list[tuple[list[str], str]]) -> dict:
    for path, raw_val in overrides:
        value = coerce_value(raw_val)
        set_in_dict(cfg, path, value)
    return cfg


def make_run_name(base_config: Path, explicit: str | None = None) -> str:
    if explicit:
        return explicit
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_config.stem}_{timestamp}"


def prepare_paths(run_name: str) -> tuple[Path, Path, Path]:
    run_dir = DEFAULT_EXPERIMENT_DIR / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"
    per_epoch = run_dir / "per_epoch.csv"
    return run_dir, log_path, per_epoch


def save_config(cfg: dict, run_dir: Path):
    config_dir = run_dir / "config"
    config_dir.mkdir(exist_ok=True)
    path = config_dir / "resolved.yaml"
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
    return path


def main():
    parser = argparse.ArgumentParser(description="Run EMMA training with overrides.")
    parser.add_argument("--config", required=True, help="Base YAML config path.")
    parser.add_argument("--preset", action="append", default=[], help="Path to a preset YAML fragment (relative to repo).")
    parser.add_argument("--set", action="append", default=[], help="Override key=value (dotted path).")
    parser.add_argument("--device", default=None, help="Device override passed to trainer.")
    parser.add_argument("--run-name", default=None, help="Custom run name (used for experiments/runs/<name>).")
    parser.add_argument("--dry-run", action="store_true", help="Print command without executing.")
    parser.add_argument("--extra-arg", action="append", default=[], help="Additional raw args to pass to emma.train.")
    parser.add_argument("--logfile", default=None, help="Override log path (default: run_dir/train.log).")
    parser.add_argument("--per-epoch-out", default=None, help="Override per-epoch CSV path.")
    parser.add_argument("--metrics-out", default=None, help="Override metrics JSON path.")
    parser.add_argument("--sample-log", default=None, help="Base path for per-sample telemetry CSV (suffix _train/_val).")
    parser.add_argument("--sample-log-limit", type=int, default=0, help="Maximum rows to record per split for sample telemetry (0 disables).")
    args = parser.parse_args()

    config_path = (REPO_ROOT / args.config).resolve()
    if not config_path.exists():
        parser.error(f"Config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    def merge_dict(base: dict, update: dict):
        for key, value in (update or {}).items():
            if (
                key in base
                and isinstance(base[key], dict)
                and isinstance(value, dict)
            ):
                merge_dict(base[key], value)
            else:
                base[key] = value

    for preset_path in args.preset:
        preset_abs = (REPO_ROOT / preset_path).resolve()
        if not preset_abs.exists():
            parser.error(f"Preset not found: {preset_abs}")
        with preset_abs.open("r", encoding="utf-8") as pfh:
            preset_cfg = yaml.safe_load(pfh)
        if not isinstance(preset_cfg, dict):
            parser.error(f"Preset must be a mapping at root: {preset_abs}")
        merge_dict(cfg, preset_cfg)

    overrides = parse_set_args(args.set)
    cfg = apply_overrides(cfg, overrides)

    run_name = make_run_name(config_path, args.run_name)
    run_dir, default_log_path, default_per_epoch_path = prepare_paths(run_name)
    resolved_config_path = save_config(cfg, run_dir)

    if args.logfile:
        log_path = Path(args.logfile)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        log_path = default_log_path

    if args.per_epoch_out:
        per_epoch_path = Path(args.per_epoch_out)
        per_epoch_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        per_epoch_path = default_per_epoch_path

    if args.metrics_out:
        metrics_path = Path(args.metrics_out)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        metrics_path = DEFAULT_EXPERIMENT_DIR / "results" / f"{run_name}_metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)

    sample_args: list[str] = []
    sample_limit = max(0, int(args.sample_log_limit or 0))
    if sample_limit > 0:
        sample_base = Path(args.sample_log) if args.sample_log else (run_dir / "samples.csv")
        sample_base.parent.mkdir(parents=True, exist_ok=True)
        sample_args.extend([
            "--sample-log", str(sample_base),
            "--sample-log-limit", str(sample_limit),
        ])

    cmd = [sys.executable, "-m", "emma.train", "--config", str(resolved_config_path)]
    if args.device:
        cmd.extend(["--device", args.device])
    cmd.extend([
        "--logfile", str(log_path),
        "--per-epoch-out", str(per_epoch_path),
        "--metrics-out", str(metrics_path),
    ])
    cmd.extend(sample_args)
    for extra in args.extra_arg:
        cmd.append(extra)

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(REPO_ROOT / "src"))

    print("[run_training] Command:", " ".join(cmd))
    print(f"[run_training] Run directory: {run_dir}")

    if args.dry_run:
        print("[run_training] Dry-run mode; exiting without execution.")
        return

    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT), env=env)


if __name__ == "__main__":
    main()
