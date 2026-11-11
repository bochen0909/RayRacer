"""Car kinematics: unicycle-style model with drag, braking and a tyre-grip limit."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

UNITS_PER_METER = 3.0  # simulation units per metre (real circuits are loaded at this scale)
_MPS_TO_MPH = 2.2369363


def to_mph(speed: float) -> float:
    """Convert a speed in simulation units/s to miles per hour."""
    return speed / UNITS_PER_METER * _MPS_TO_MPH


@dataclass(frozen=True)
class CarParams:
    length: float = 20.0
    width: float = 10.0
    max_speed: float = 300.0  # units / s (about 224 mph)
    accel: float = 120.0  # units / s^2 at full throttle
    brake: float = 150.0  # units / s^2 at full brake (negative throttle)
    drag: float = 0.4  # 1/s, linear speed decay
    min_turn_radius: float = 25.0  # tightest turn at low speed (full steer)
    max_lateral_accel: float = 150.0  # units / s^2 of tyre grip; limits cornering speed
    dt: float = 1.0 / 30.0


@dataclass
class CarState:
    pos: np.ndarray = field(default_factory=lambda: np.zeros(2))
    heading: float = 0.0
    speed: float = 0.0


def step(state: CarState, steer: float, throttle: float, p: CarParams) -> None:
    """Advance `state` in place by one tick. steer, throttle are clipped to [-1, 1].

    Positive throttle accelerates, negative brakes. Curvature is steer / min_turn_radius,
    but never more than the grip limit max_lateral_accel / speed^2, so a car that is too
    fast for a corner understeers into the wall.
    """
    steer = float(np.clip(steer, -1.0, 1.0))
    throttle = float(np.clip(throttle, -1.0, 1.0))

    force = throttle * p.accel if throttle >= 0.0 else throttle * p.brake
    state.speed += (force - p.drag * state.speed) * p.dt
    state.speed = float(np.clip(state.speed, 0.0, p.max_speed))

    grip_curvature = p.max_lateral_accel / max(state.speed**2, 1e-6)
    curvature = float(np.clip(steer / p.min_turn_radius, -grip_curvature, grip_curvature))
    # Turning scales with speed, so a stationary car cannot spin in place.
    state.heading += state.speed * curvature * p.dt
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
