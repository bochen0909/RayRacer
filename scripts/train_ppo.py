"""PPO training (Stable-Baselines3). Run: uv run --extra rl python scripts/train_ppo.py --steps 200000"""

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from rf.env import CarEnv
from rf.rl import CarGymEnv
from rf.tracks.oval import make_oval


def make() -> CarGymEnv:
    return CarGymEnv(CarEnv(make_oval()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--envs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("runs"))
    args = ap.parse_args()
    args.out.mkdir(exist_ok=True)

    vec = make_vec_env(make, n_envs=args.envs, seed=args.seed, vec_env_cls=SubprocVecEnv)
    model = PPO("MlpPolicy", vec, seed=args.seed, device="cpu", verbose=1)
    model.learn(total_timesteps=args.steps)
    model.save(args.out / "ppo.zip")
    print(f"saved to {args.out}/ppo.zip")


if __name__ == "__main__":
    main()
