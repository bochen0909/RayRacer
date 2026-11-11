"""Car kinematics: simple unicycle-style model with drag."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class CarParams:
    length: float = 20.0
    width: float = 10.0
    max_speed: float = 200.0  # units / s
    accel: float = 150.0  # units / s^2 at full throttle
    drag: float = 0.5  # 1/s, linear speed decay
    max_turn_rate: float = 3.0  # rad/s at full steer and full speed
    dt: float = 1.0 / 30.0


@dataclass
class CarState:
    pos: np.ndarray = field(default_factory=lambda: np.zeros(2))
    heading: float = 0.0
    speed: float = 0.0


def step(state: CarState, steer: float, throttle: float, p: CarParams) -> None:
    """Advance `state` in place by one tick. steer, throttle are clipped to [-1, 1]."""
    steer = float(np.clip(steer, -1.0, 1.0))
    throttle = float(np.clip(throttle, -1.0, 1.0))

    state.speed += (throttle * p.accel - p.drag * state.speed) * p.dt
    state.speed = float(np.clip(state.speed, 0.0, p.max_speed))
    # Turning scales with speed so a stationary car cannot spin in place.
    state.heading += steer * p.max_turn_rate * (state.speed / p.max_speed) * p.dt
    state.pos = state.pos + state.speed * p.dt * np.array(
        [np.cos(state.heading), np.sin(state.heading)]
    )


def body_segments(state: CarState, p: CarParams) -> np.ndarray:
    """The car's rectangular outline as 4 segments, shape (4, 2, 2)."""
    c, s = np.cos(state.heading), np.sin(state.heading)
    fwd = np.array([c, s]) * p.length / 2
    side = np.array([-s, c]) * p.width / 2
    corners = np.stack(
        [state.pos + fwd + side, state.pos + fwd - side, state.pos - fwd - side, state.pos - fwd + side]
    )
    return np.stack([corners, np.roll(corners, -1, axis=0)], axis=1)
