from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Iterable, Optional


@dataclass
class NCEScheduleConfig:
    gate_start_epoch: int
    plateau_epochs: int
    plateau_eps: float
    read_threshold: float
    gate_warmup_epochs: int
    lambda_floor: float
    lambda_increment: float
    lambda_max: float
    lambda_end: float
    read_target: float
    read_hysteresis: float
    lambda_backoff: float
    lambda_min: float
    open_oracle_hold: int
    enable: bool
    allow_plateau_open: bool
    require_read_ready: bool
    read_ready_patience: int
    pause_below_read: bool
    require_plateau_ready: bool
    lambda_freeze_epochs_after_open: int


@dataclass
class NCEScheduleDecision:
    lambda_value: float
    lambda_prev: float
    gate_state: bool
    opened_this_epoch: bool
    gate_reason: Optional[str]
    plateau_ready: bool
    read_ready: bool
    epoch_ready: bool
    open_steps: int
    lambda_transition: bool


class NCEScheduler:
    """Encapsulates InfoNCE gating schedule logic for readability/reuse."""

    def __init__(self, emma_cfg: dict):
        gate_start_epoch = int(emma_cfg.get('nce_gate_start_epoch', 3) or 3)
        plateau_epochs = int(emma_cfg.get('nce_gate_plateau_epochs', 2) or 2)
        plateau_eps = float(emma_cfg.get('nce_gate_plateau_eps', 0.002) or 0.002)
        read_threshold = float(emma_cfg.get('nce_gate_read_threshold', 0.50) or 0.50)
        gate_warmup_epochs = int(emma_cfg.get('nce_gate_warmup_epochs', gate_start_epoch) or gate_start_epoch)
        lambda_floor = float(emma_cfg.get('nce_lambda_floor', 0.0) or 0.0)
        lambda_increment = float(emma_cfg.get('nce_lambda_increment', 0.01) or 0.01)
        lambda_max_cfg = emma_cfg.get('nce_lambda_max', None)
        if lambda_max_cfg is None:
            base = emma_cfg.get('lambda_write_nce') or emma_cfg.get('lambda_nce') or 0.02
            lambda_max = float(base or 0.02)
        else:
            lambda_max = float(lambda_max_cfg or 0.0)
        lambda_end_cfg = emma_cfg.get('nce_lambda_end', None)
        lambda_end = None
        if lambda_end_cfg is not None:
            try:
                lambda_end = float(lambda_end_cfg)
            except Exception:
                lambda_end = None
        if lambda_end is None or lambda_end != lambda_end:
            lambda_end = lambda_max
        else:
            lambda_end = max(0.0, lambda_end)
            lambda_max = min(lambda_max, lambda_end)
        read_target_cfg = emma_cfg.get('nce_read_target', None)
        if read_target_cfg is None:
            read_target = read_threshold
        else:
            try:
                read_target = float(read_target_cfg)
            except Exception:
                read_target = read_threshold
        read_target = max(0.0, min(1.0, read_target))
        read_hysteresis = float(emma_cfg.get('nce_read_hysteresis', 0.02) or 0.02)
        lambda_backoff = float(emma_cfg.get('nce_lambda_backoff', 0.8) or 0.8)
        lambda_backoff = max(0.0, min(1.0, lambda_backoff))
        open_oracle_hold = int(emma_cfg.get('nce_gate_oracle_hold', 1) or 1)
        allow_plateau_open = bool(emma_cfg.get('nce_gate_allow_plateau', True))
        require_read_ready = bool(emma_cfg.get('nce_gate_require_read', False))
        read_ready_patience = int(emma_cfg.get('nce_gate_read_patience', 0) or 0)
        if read_ready_patience < 0:
            read_ready_patience = 0
        pause_below_read = bool(emma_cfg.get('nce_lambda_pause_below_read', False))
        require_plateau_ready = bool(emma_cfg.get('nce_gate_require_plateau', False))
        lambda_freeze_epochs = int(emma_cfg.get('nce_lambda_freeze_epochs_after_open', 0) or 0)
        if lambda_freeze_epochs < 0:
            lambda_freeze_epochs = 0

        enable = (lambda_max > 0.0) and (lambda_increment > 0.0 or lambda_floor > 0.0)
        lambda_min = max(0.0, lambda_floor)

        self.config = NCEScheduleConfig(
            gate_start_epoch=gate_start_epoch,
            plateau_epochs=max(1, plateau_epochs),
            plateau_eps=max(0.0, plateau_eps),
            read_threshold=read_threshold,
            gate_warmup_epochs=max(0, gate_warmup_epochs),
            lambda_floor=lambda_min,
            lambda_increment=max(0.0, lambda_increment),
            lambda_max=max(0.0, lambda_max),
            lambda_end=max(0.0, lambda_end),
            read_target=read_target,
            read_hysteresis=max(0.0, read_hysteresis),
            lambda_backoff=lambda_backoff,
            lambda_min=lambda_min,
            open_oracle_hold=max(0, open_oracle_hold),
            enable=enable,
            allow_plateau_open=allow_plateau_open,
            require_read_ready=require_read_ready,
            read_ready_patience=read_ready_patience,
            pause_below_read=pause_below_read,
            require_plateau_ready=require_plateau_ready,
            lambda_freeze_epochs_after_open=lambda_freeze_epochs,
        )

        self.gate_state: bool = False
        self.gate_reason: Optional[str] = None
        self.gate_epoch: Optional[int] = None
        self.open_steps: int = 0
        self.lambda_prev: float = 0.0
        self.oracle_hold_remaining: int = 0
        self._read_ready_streak: int = 0
        self.lambda_freeze_remaining: int = 0

    def on_epoch_start(
        self,
        epoch: int,
        val_acc_history: Iterable[float],
        read_ema: float,
    ) -> NCEScheduleDecision:
        cfg = self.config
        plateau_ready = False
        read_ready = False
        epoch_ready = (epoch + 1) >= cfg.gate_start_epoch
        lambda_current = 0.0
        opened_this_epoch = False
        prev_lambda = self.lambda_prev

        if cfg.enable:
            history = list(val_acc_history)
            if len(history) >= cfg.plateau_epochs:
                recent = history[-cfg.plateau_epochs:]
                if len(recent) >= 2:
                    deltas = [abs(recent[i] - recent[i - 1]) for i in range(1, len(recent))]
                    plateau_ready = all(d <= cfg.plateau_eps for d in deltas)
            read_ready_baseline = (not math.isnan(read_ema)) and (read_ema >= cfg.read_threshold)
            if read_ready_baseline:
                self._read_ready_streak += 1
            else:
                self._read_ready_streak = 0
            if cfg.read_ready_patience > 0:
                read_ready = read_ready_baseline and (self._read_ready_streak >= cfg.read_ready_patience)
            else:
                read_ready = read_ready_baseline

            if not self.gate_state:
                can_open = False
                if cfg.require_read_ready:
                    can_open = epoch_ready and read_ready
                    if cfg.require_plateau_ready:
                        can_open = can_open and plateau_ready
                else:
                    plateau_condition = cfg.allow_plateau_open and plateau_ready
                    can_open = epoch_ready and (read_ready or plateau_condition)
                if can_open:
                    self.gate_state = True
                    opened_this_epoch = True
                    self.gate_reason = 'read' if read_ready else 'plateau'
                    if self.gate_epoch is None:
                        self.gate_epoch = epoch + 1
                    self.open_steps = 1
                    if cfg.open_oracle_hold > 0:
                        self.oracle_hold_remaining = max(self.oracle_hold_remaining, cfg.open_oracle_hold)
                    if cfg.lambda_freeze_epochs_after_open > 0:
                        self.lambda_freeze_remaining = cfg.lambda_freeze_epochs_after_open
                    else:
                        self.lambda_freeze_remaining = 0
                else:
                    self.open_steps = 0
            else:
                self.open_steps += 1

            if self.gate_state:
                if (epoch + 1) <= cfg.gate_warmup_epochs:
                    lambda_current = 0.0
                else:
                    lam_candidate = max(prev_lambda, cfg.lambda_floor)
                    freeze_active = self.lambda_freeze_remaining > 0 and self.open_steps > 1
                    if not freeze_active:
                        if not math.isnan(read_ema):
                            upper_band = cfg.read_target + cfg.read_hysteresis
                            lower_band = cfg.read_target - cfg.read_hysteresis
                            if read_ema > upper_band:
                                lam_candidate = min(lam_candidate + cfg.lambda_increment, cfg.lambda_max)
                            elif read_ema < lower_band:
                                lam_candidate = max(cfg.lambda_floor, lam_candidate * cfg.lambda_backoff)
                        if cfg.pause_below_read and not read_ready_baseline:
                            lam_candidate = cfg.lambda_floor
                    lambda_current = max(0.0, min(lam_candidate, cfg.lambda_max))
                    if freeze_active:
                        self.lambda_freeze_remaining = max(0, self.lambda_freeze_remaining - 1)
            else:
                lambda_current = 0.0
                self.lambda_freeze_remaining = 0
        else:
            plateau_ready = False
            read_ready = False
            self.gate_state = False
            self.open_steps = 0
            lambda_current = 0.0
            self.lambda_freeze_remaining = 0

        lambda_transition = (prev_lambda <= 0.0) and (lambda_current > 0.0)
        self.lambda_prev = lambda_current

        return NCEScheduleDecision(
            lambda_value=lambda_current,
            lambda_prev=prev_lambda,
            gate_state=self.gate_state,
            opened_this_epoch=opened_this_epoch,
            gate_reason=self.gate_reason if opened_this_epoch else None,
            plateau_ready=plateau_ready,
            read_ready=read_ready,
            epoch_ready=epoch_ready,
            open_steps=self.open_steps,
            lambda_transition=lambda_transition,
        )

    def decrement_oracle_hold(self) -> None:
        if self.oracle_hold_remaining > 0:
            self.oracle_hold_remaining = max(0, self.oracle_hold_remaining - 1)

    # Convenience accessors -------------------------------------------------
    @property
    def gate_state_int(self) -> int:
        return int(self.gate_state)

    @property
    def gate_epoch_metric(self) -> Optional[int]:
        return self.gate_epoch
