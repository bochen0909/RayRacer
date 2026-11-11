import numpy as np
import pytest

pytest.importorskip("gymnasium")

from rf.rl import CarGymEnv


def test_gym_env_api_and_crash_termination():
    env = CarGymEnv()
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    terminated = truncated = False
    reward = 0.0
    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step(np.array([0.0, 1.0], dtype=np.float32))
    assert terminated and not truncated  # full throttle straight ahead hits the wall
    assert reward < 0.0  # crash penalty dominates
