"""Single-car environment: reset/step, collision, checkpoint progress."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rf.car import CarParams, CarState, body_segments, step as car_step
from rf.geometry import segments_intersect_any
from rf.sensors import DEFAULT_RAY_ANGLES, sense
from rf.track import Track


@dataclass(frozen=True)
class EnvConfig:
    max_steps: int = 1500
    stall_steps: int = 150  # end the episode if no checkpoint is passed for this long
    ray_angles: np.ndarray = field(default_factory=lambda: DEFAULT_RAY_ANGLES)
    ray_range: float = 150.0


class CarEnv:
    """Observation: ray distances (normalized) + speed (normalized). Action: (steer, throttle)."""

    def __init__(self, track: Track, car: CarParams | None = None, config: EnvConfig | None = None):
        self.track = track
        self.car = car or CarParams()
        self.config = config or EnvConfig()
        self._walls = track.walls
        self.state = CarState()
        self.reset()

    @property
    def obs_size(self) -> int:
        return len(self.config.ray_angles) + 1

    def reset(self) -> np.ndarray:
        self.state = CarState(self.track.start_pos.copy(), self.track.start_heading, 0.0)
        self.steps = 0
        self.steps_since_checkpoint = 0
        self.checkpoints_passed = 0
        self.crashed = False
        self.done = False
        return self._observe()

    def _observe(self) -> np.ndarray:
        rays = sense(self.state, self._walls, self.config.ray_angles, self.config.ray_range)
        self.rays = rays
        return np.append(rays, self.state.speed / self.car.max_speed)

    def step(self, action: tuple[float, float]) -> tuple[np.ndarray, float, bool, dict]:
        if self.done:
            raise RuntimeError("Episode is done; call reset().")
        prev = self.state.pos.copy()
        car_step(self.state, action[0], action[1], self.car)
        self.steps += 1
        self.steps_since_checkpoint += 1

        reward = 0.0
        gates = self.track.checkpoints
        next_gate = gates[(self.checkpoints_passed + 1) % len(gates)]
        move = np.stack([prev, self.state.pos])[None]
        if segments_intersect_any(move, next_gate[None]):
            self.checkpoints_passed += 1
            self.steps_since_checkpoint = 0
            reward = 1.0

        self.crashed = segments_intersect_any(body_segments(self.state, self.car), self._walls)
        self.done = (
            self.crashed
            or self.steps >= self.config.max_steps
            or self.steps_since_checkpoint >= self.config.stall_steps
        )
        info = {"checkpoints": self.checkpoints_passed, "crashed": self.crashed}
        return self._observe(), reward, self.done, info
