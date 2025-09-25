import math

import pytest

from emma.schedules import NCEScheduler


def make_cfg(**overrides):
    cfg = {
        'nce_gate_start_epoch': 1,
        'nce_gate_plateau_epochs': 1,
        'nce_gate_plateau_eps': 0.0,
        'nce_gate_read_threshold': 0.5,
        'nce_gate_warmup_epochs': 1,
        'nce_lambda_floor': 0.0,
        'nce_lambda_increment': 0.01,
        'nce_lambda_max': 0.05,
        'nce_read_target': 0.6,
        'nce_read_hysteresis': 0.02,
        'nce_lambda_backoff': 0.5,
        'nce_gate_oracle_hold': 2,
    }
    cfg.update(overrides)
    return cfg


def test_scheduler_disabled_when_lambda_max_zero():
    sched = NCEScheduler(make_cfg(nce_lambda_max=0.0))
    decision = sched.on_epoch_start(0, [], read_ema=math.nan)
    assert decision.lambda_value == 0.0
    assert decision.gate_state is False
    assert decision.open_steps == 0


def test_scheduler_opens_on_read_signal_and_produces_lambda():
    sched = NCEScheduler(make_cfg())

    # Epoch 0: gate should open (read_ema above threshold) but stay at lambda=0 due to warmup.
    decision0 = sched.on_epoch_start(0, [], read_ema=0.7)
    assert decision0.opened_this_epoch is True
    assert decision0.lambda_value == 0.0  # warmup epoch keeps lambda at zero
    assert sched.oracle_hold_remaining == 2

    # Epoch 1: warmup complete, lambda should ramp up via increment.
    decision1 = sched.on_epoch_start(1, [], read_ema=0.7)
    assert decision1.gate_state is True
    assert decision1.lambda_value == pytest.approx(0.01, rel=1e-6)
    assert sched.oracle_hold_remaining == 2  # unchanged until decrement

    # Oracle hold decrements each epoch end.
    sched.decrement_oracle_hold()
    assert sched.oracle_hold_remaining == 1


def test_scheduler_backoff_when_read_drops():
    sched = NCEScheduler(make_cfg())
    sched.on_epoch_start(0, [], read_ema=0.7)  # open gate
    sched.on_epoch_start(1, [], read_ema=0.7)  # lambda -> 0.01
    decision_drop = sched.on_epoch_start(2, [], read_ema=0.4)
    assert decision_drop.lambda_value == pytest.approx(0.005, abs=1e-6)


def test_plateau_path_opens_when_read_low():
    sched = NCEScheduler(
        make_cfg(nce_gate_read_threshold=0.9, nce_gate_plateau_epochs=2)
    )
    val_hist = [0.80, 0.80]
    decision = sched.on_epoch_start(1, val_hist, read_ema=0.4)
    assert decision.opened_this_epoch is True
    assert decision.gate_state is True
