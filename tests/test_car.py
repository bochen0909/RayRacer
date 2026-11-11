import numpy as np
import pytest

from rf.car import CarParams, CarState, step, to_mph


def test_to_mph():
    # 3 units = 1 m, so 300 units/s = 100 m/s = 223.7 mph
    assert to_mph(300.0) == pytest.approx(223.69, abs=0.01)


def test_negative_throttle_brakes_and_never_reverses():
    p = CarParams()
    s = CarState(speed=100.0)
    step(s, 0.0, -1.0, p)
    assert s.speed < 100.0 - p.brake * p.dt * 0.9
    for _ in range(200):
        step(s, 0.0, -1.0, p)
    assert s.speed == 0.0


def test_grip_limits_turning_at_speed():
    p = CarParams()
    slow = CarState(speed=40.0)
    fast = CarState(speed=p.max_speed)
    step(slow, 1.0, 0.0, p)
    step(fast, 1.0, 0.0, p)
    slow_radius = slow.speed / (slow.heading / p.dt)
    fast_radius = fast.speed / (fast.heading / p.dt)
    assert slow_radius == pytest.approx(p.min_turn_radius, rel=0.02)  # steering-limited
    assert fast_radius > 3 * slow_radius  # grip-limited: understeers
    assert fast.speed**2 / fast_radius == pytest.approx(p.max_lateral_accel, rel=0.05)
