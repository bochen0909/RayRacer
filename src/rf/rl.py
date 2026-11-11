"""Gymnasium wrapper + PPO helpers (optional dependency: `uv sync --extra rl`)."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rf.env import CarEnv
from rf.tracks.oval import make_oval


class CarGymEnv(gym.Env):
    """Continuous control: action = (steer, throttle) in [-1, 1].

    Reward = checkpoint bonus + small speed term (dense signal) - crash penalty.
    """

    def __init__(self, car_env: CarEnv | None = None, crash_penalty: float = 2.0, speed_reward: float = 0.02):
        self.env = car_env or CarEnv(make_oval())
        self.crash_penalty = crash_penalty
        self.speed_reward = speed_reward
        n = self.env.obs_size
        self.observation_space = spaces.Box(0.0, 1.0, shape=(n,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        return self.env.reset().astype(np.float32), {}

    def step(self, action):
        obs, reward, done, info = self.env.step((float(action[0]), float(action[1])))
        reward += self.speed_reward * self.env.state.speed / self.env.car.max_speed
        terminated = bool(info["crashed"])
        if terminated:
            reward -= self.crash_penalty
        truncated = done and not terminated
        return obs.astype(np.float32), float(reward), terminated, truncated, info


class SB3Policy:
    """Adapts a trained Stable-Baselines3 model to the `obs -> (steer, throttle)` interface."""

    def __init__(self, model):
        self.model = model

    def __call__(self, obs) -> tuple[float, float]:
        action, _ = self.model.predict(np.asarray(obs, dtype=np.float32), deterministic=True)
        return float(np.clip(action[0], -1, 1)), float(np.clip(action[1], -1, 1))


def load_policy(path) -> SB3Policy:
    from stable_baselines3 import PPO

    return SB3Policy(PPO.load(str(path), device="cpu"))
