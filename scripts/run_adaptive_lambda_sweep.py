#!/usr/bin/env python3
"""Run adaptive InfoNCE λ sweeps using the quickstart configs.

This script wraps ``scripts/run_training.py`` so that we can explore
``nce_lambda`` schedules across the quickstart presets (len256/512/1024).

Example usages:
    python scripts/run_adaptive_lambda_sweep.py --length len256 --mode targeted
    python scripts/run_adaptive_lambda_sweep.py --length len512 --mode full --limit 4
    python scripts/run_adaptive_lambda_sweep.py --profile len256:tuned --profile len512:no_gate
    python scripts/run_adaptive_lambda_sweep.py --device cpu --dry-run

By default the script runs a targeted subset of the grid (per length) to keep
runtime manageable. Use ``--mode full`` to expand to the Cartesian grid defined
for that length. The ``--limit`` flag can cap the number of launches.
"""

from __future__ import annotations

import argparse
import itertools
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_TRAINING = REPO_ROOT / "scripts" / "run_training.py"

QuickstartKey = str
OverrideDict = Dict[str, Any]


# Base overrides ensure the quickstart configs expose the InfoNCE gate knobs the
# sweeps expect. len512/1024 already define these fields, but keeping explicit
# values avoids surprises if the presets change.
BASE_OVERRIDES: Dict[str, OverrideDict] = {
    "len256": {
        "emma.nce_gate_start_epoch": 2,
        "emma.nce_gate_plateau_epochs": 2,
        "emma.nce_gate_plateau_eps": 0.02,
        "emma.nce_gate_read_threshold": 0.42,
        "emma.nce_gate_warmup_epochs": 2,
        "emma.nce_read_hysteresis": 0.03,
        "emma.nce_ema_beta": 0.8,
        "emma.nce_read_ema_beta": 0.8,
        "emma.nce_temperature": 0.1,
        "emma.nce_post_gate_drop_threshold": 0.01,
        "emma.nce_post_gate_drop_patience": 2,
        "emma.nce_lambda_ramp_epochs": 3,
    },
    "len512": {
        "emma.nce_gate_start_epoch": 2,
        "emma.nce_gate_plateau_epochs": 2,
        "emma.nce_gate_plateau_eps": 0.02,
        "emma.nce_gate_read_threshold": 0.45,
        "emma.nce_gate_warmup_epochs": 2,
        "emma.nce_read_hysteresis": 0.03,
        "emma.nce_ema_beta": 0.8,
        "emma.nce_read_ema_beta": 0.8,
        "emma.nce_temperature": 0.1,
        "emma.nce_lambda_ramp_epochs": 4,
    },
    "len1024": {
        "emma.nce_gate_start_epoch": 2,
        "emma.nce_gate_plateau_epochs": 2,
        "emma.nce_gate_plateau_eps": 0.02,
        "emma.nce_gate_read_threshold": 0.45,
        "emma.nce_gate_warmup_epochs": 2,
        "emma.nce_read_hysteresis": 0.03,
        "emma.nce_ema_beta": 0.8,
        "emma.nce_read_ema_beta": 0.8,
        "emma.nce_temperature": 0.1,
        "emma.nce_lambda_ramp_epochs": 4,
    },
}


PROFILE_OVERRIDES: Dict[str, Dict[str, OverrideDict]] = {
    "len256": {
        "tuned": {
            "train.epochs": 4,
            "train.lr_after_warm_factor": 0.7,
            "emma.warm_start_epochs": 2,
            "emma.nce_lambda_max": 0.004,
            "emma.nce_lambda_increment": 0.001,
            "emma.nce_lambda_floor": 0.001,
            "emma.nce_lambda_backoff": 0.5,
            "emma.nce_lambda_ramp_epochs": 4,
            "emma.nce_read_target": 0.43,
            "emma.nce_gate_read_threshold": 0.43,
            "emma.nce_post_gate_drop_threshold": 0.02,
            "emma.nce_post_gate_drop_patience": 1,
        },
    },
    "len512": {
        "no_gate": {
            "train.epochs": 2,
            "emma.nce_gate_start_epoch": 5,
            "emma.nce_force_open_epoch": 0,
            "emma.nce_min_open_epochs": 1,
            "emma.nce_sticky_after_ramp": False,
        },
        "small_lambda": {
            "train.epochs": 6,
            "train.lr_after_warm_factor": 0.7,
            "emma.warm_start_epochs": 2,
            "emma.nce_lambda_max": 0.002,
            "emma.nce_lambda_increment": 0.0005,
            "emma.nce_lambda_floor": 0.0005,
            "emma.nce_lambda_backoff": 0.5,
            "emma.nce_lambda_ramp_epochs": 4,
            "emma.nce_read_target": 0.47,
            "emma.nce_gate_read_threshold": 0.47,
            "emma.nce_post_gate_drop_threshold": 0.03,
            "emma.nce_post_gate_drop_patience": 1,
            "emma.nce_min_open_epochs": 1,
            "emma.nce_sticky_after_ramp": False,
        },
        "cleanup_high": {
            "train.epochs": 6,
            "train.lr_after_warm_factor": 0.7,
            "emma.warm_start_epochs": 2,
            "emma.oracle_mix_min": 0.6,
            "emma.oracle_mix_ramp_epochs": 4,
            "emma.nce_gate_start_epoch": 4,
            "emma.nce_read_target": 0.5,
            "emma.nce_gate_read_threshold": 0.5,
            "emma.nce_lambda_max": 0.002,
            "emma.nce_lambda_increment": 0.0005,
            "emma.nce_lambda_floor": 0.0005,
            "emma.nce_lambda_backoff": 0.4,
            "emma.nce_lambda_ramp_epochs": 4,
            "emma.nce_post_gate_drop_threshold": 0.06,
            "emma.nce_post_gate_drop_patience": 2,
            "emma.nce_min_open_epochs": 1,
            "emma.nce_sticky_after_ramp": False,
            "read.cleanup_temp": 0.6,
            "read.sharpen_temp": 0.7,
        },
        "late_gate": {
            "train.epochs": 8,
            "train.lr_after_warm_factor": 0.65,
            "emma.warm_start_epochs": 3,
            "emma.oracle_mix_min": 0.55,
            "emma.oracle_mix_ramp_epochs": 5,
            "emma.nce_gate_start_epoch": 7,
            "emma.nce_force_open_epoch": 7,
            "emma.nce_read_target": 0.55,
            "emma.nce_gate_read_threshold": 0.55,
            "emma.nce_gate_require_read": True,
            "emma.nce_gate_read_patience": 3,
            "emma.nce_lambda_max": 0.001,
            "emma.nce_lambda_increment": 0.0003,
            "emma.nce_lambda_floor": 0.0005,
            "emma.nce_lambda_backoff": 0.6,
            "emma.nce_lambda_ramp_epochs": 6,
            "emma.nce_post_gate_drop_threshold": 0.04,
            "emma.nce_post_gate_drop_patience": 3,
            "emma.nce_min_open_epochs": 1,
            "emma.nce_sticky_after_ramp": False,
            "read.cleanup_temp": 0.6,
            "read.sharpen_temp": 0.75,
        },
    },
}


TARGETED_GRID: Dict[str, Sequence[OverrideDict]] = {
    "len256": (
        {
            "emma.nce_lambda_max": 0.05,
            "emma.nce_lambda_increment": 0.01,
            "emma.nce_lambda_floor": 0.003,
            "emma.nce_read_target": 0.42,
        },
        {
            "emma.nce_lambda_max": 0.06,
            "emma.nce_lambda_increment": 0.015,
            "emma.nce_lambda_floor": 0.005,
            "emma.nce_read_target": 0.45,
        },
        {
            "emma.nce_lambda_max": 0.05,
            "emma.nce_lambda_increment": 0.015,
            "emma.nce_lambda_floor": 0.003,
            "emma.nce_read_target": 0.45,
        },
    ),
    "len512": (
        {
            "emma.nce_lambda_max": 0.05,
            "emma.nce_lambda_increment": 0.01,
            "emma.nce_lambda_floor": 0.003,
            "emma.nce_read_target": 0.45,
        },
        {
            "emma.nce_lambda_max": 0.06,
            "emma.nce_lambda_increment": 0.015,
            "emma.nce_lambda_floor": 0.003,
            "emma.nce_read_target": 0.47,
        },
        {
            "emma.nce_lambda_max": 0.05,
            "emma.nce_lambda_increment": 0.015,
            "emma.nce_lambda_floor": 0.005,
            "emma.nce_read_target": 0.44,
        },
    ),
    "len1024": (
        {
            "emma.nce_lambda_max": 0.05,
            "emma.nce_lambda_increment": 0.01,
            "emma.nce_lambda_floor": 0.003,
            "emma.nce_read_target": 0.45,
        },
        {
            "emma.nce_lambda_max": 0.06,
            "emma.nce_lambda_increment": 0.015,
            "emma.nce_lambda_floor": 0.003,
            "emma.nce_read_target": 0.48,
        },
        {
            "emma.nce_lambda_max": 0.06,
            "emma.nce_lambda_increment": 0.01,
            "emma.nce_lambda_floor": 0.005,
            "emma.nce_read_target": 0.47,
        },
    ),
}


FULL_GRID_VALUES: Dict[str, Dict[str, Sequence[float]]] = {
    "len256": {
        "emma.nce_lambda_max": (0.05, 0.06),
        "emma.nce_lambda_increment": (0.01, 0.015),
        "emma.nce_lambda_floor": (0.003, 0.005),
        "emma.nce_read_target": (0.42, 0.45),
    },
    "len512": {
        "emma.nce_lambda_max": (0.05, 0.06),
        "emma.nce_lambda_increment": (0.01, 0.015),
        "emma.nce_lambda_floor": (0.003, 0.005),
        "emma.nce_read_target": (0.44, 0.47),
    },
    "len1024": {
        "emma.nce_lambda_max": (0.05, 0.06),
        "emma.nce_lambda_increment": (0.01, 0.015),
        "emma.nce_lambda_floor": (0.003, 0.005),
        "emma.nce_read_target": (0.45, 0.48),
    },
}


CONFIG_PATHS: Dict[str, str] = {
    "len256": "configs/quickstarts/len256_softmax_top2.yaml",
    "len512": "configs/quickstarts/len512_softmax_top2.yaml",
    "len1024": "configs/quickstarts/len1024_softmax_top2.yaml",
}


def iter_grid(length: str, mode: str) -> Iterable[OverrideDict]:
    if mode == "full":
        values = FULL_GRID_VALUES[length]
        keys = sorted(values)
        for combo in itertools.product(*(values[k] for k in keys)):
            yield {k: combo[i] for i, k in enumerate(keys)}
    elif mode == "targeted":
        yield from TARGETED_GRID[length]
    else:
        raise ValueError(f"Unknown mode '{mode}'")


def format_override_value(value: Any) -> str:
    # keep readable floats without excessive precision
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if abs(value) >= 1:
            return f"{value:.3f}".rstrip("0").rstrip(".")
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def build_run_name(length: str, overrides: OverrideDict) -> str:
    pieces = [length]
    for key in ("emma.nce_lambda_max", "emma.nce_lambda_increment", "emma.nce_lambda_floor", "emma.nce_read_target"):
        val = overrides[key]
        pieces.append(
            f"{key.split('.')[-1]}" + f"{val:.3f}".replace(".", "p")
        )
    return "adaptive_" + "_".join(pieces)


def launch(
    length: str,
    grid_overrides: OverrideDict,
    *,
    device: str | None,
    dry_run: bool,
    extra_sets: Sequence[str],
    run_prefix: str | None,
    profile_overrides: Sequence[OverrideDict],
) -> None:
    base_config = CONFIG_PATHS[length]
    overrides = dict(BASE_OVERRIDES.get(length, {}))
    overrides.update(grid_overrides)
    for bundle in profile_overrides:
        overrides.update(bundle)

    run_name = build_run_name(length, grid_overrides)
    if run_prefix:
        run_name = f"{run_prefix}_{run_name}"

    cmd: List[str] = [
        sys.executable,
        str(RUN_TRAINING),
        "--config",
        base_config,
        "--run-name",
        run_name,
    ]
    if device:
        cmd.extend(["--device", device])

    for key, value in overrides.items():
        cmd.extend(["--set", f"{key}={format_override_value(value)}"])

    for raw in extra_sets:
        cmd.extend(["--set", raw])

    if dry_run:
        print("[dry-run]", " ".join(cmd))
        return

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(REPO_ROOT / "src"))
    print("[sweep] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT), env=env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run adaptive λ sweeps on quickstart configs.")
    parser.add_argument(
        "--length",
        action="append",
        choices=sorted(CONFIG_PATHS.keys()),
        help="Lengths to include (default: all)."
    )
    parser.add_argument(
        "--mode",
        choices=("targeted", "full"),
        default="targeted",
        help="Sweep grid size; targeted is a curated subset."
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device handed to run_training.py (default: cpu)."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of launches (after filtering)."
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VAL",
        help="Additional overrides applied to every run."
    )
    parser.add_argument(
        "--run-prefix",
        default=None,
        help="Optional prefix added to each auto-generated run name."
    )
    parser.add_argument(
        "--profile",
        action="append",
        default=[],
        metavar="LENGTH:NAME",
        help=(
            "Apply a predefined override bundle (e.g. len256:tuned, len512:no_gate). "
            "Repeatable; multiple profiles per length are merged in order."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lengths = args.length or sorted(CONFIG_PATHS.keys())
    profile_map: Dict[str, List[OverrideDict]] = {length: [] for length in lengths}
    for raw in args.profile:
        if ":" not in raw:
            raise SystemExit(f"Invalid profile '{raw}'. Expected format length:name.")
        length, name = raw.split(":", 1)
        if length not in CONFIG_PATHS:
            raise SystemExit(f"Unknown length '{length}' for profile '{raw}'.")
        overrides_for_length = PROFILE_OVERRIDES.get(length, {}).get(name)
        if overrides_for_length is None:
            available = ", ".join(sorted(PROFILE_OVERRIDES.get(length, {}))) or "<none>"
            raise SystemExit(
                f"Unknown profile '{name}' for {length}. Available: {available}"
            )
        profile_map.setdefault(length, []).append(overrides_for_length)

    launches = []
    for length in lengths:
        for overrides in iter_grid(length, args.mode):
            launches.append((length, overrides))

    if args.limit is not None:
        launches = launches[: args.limit]

    if not launches:
        print("No sweeps selected; exiting.")
        return

    print(f"Selected {len(launches)} launches (mode={args.mode}, lengths={','.join(lengths)}).")
    for idx, (length, overrides) in enumerate(launches, start=1):
        print(f"Launch {idx}/{len(launches)} → {build_run_name(length, overrides)}")
        launch(
            length,
            overrides,
            device=args.device,
            dry_run=args.dry_run,
            extra_sets=args.set,
            run_prefix=args.run_prefix,
            profile_overrides=profile_map.get(length, []),
        )


if __name__ == "__main__":
    main()
