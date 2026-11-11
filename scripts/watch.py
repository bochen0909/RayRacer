"""Watch the car.

  uv run --extra viz python scripts/watch.py                 # drive with arrow keys
  uv run --extra viz python scripts/watch.py --policy runs/best.npy
  uv run --extra viz python scripts/watch.py --population runs/population.npy
"""

import argparse
from pathlib import Path

import numpy as np
import pygame

from rf.evolve import make_env, make_policy
from rf.viz.render import Renderer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", type=Path)
    ap.add_argument("--population", type=Path)
    args = ap.parse_args()

    env = make_env()
    renderer = Renderer(env)

    # The first car is the main one (with rays); the rest are drawn dimmed.
    weights = [np.load(args.policy)] if args.policy else list(np.load(args.population)) if args.population else []
    envs = [env] + [make_env() for _ in weights[1:]]
    policies = [make_policy(e, w) for e, w in zip(envs, weights)]

    clock = pygame.time.Clock()
    obs = [e.reset() for e in envs]
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        if policies:
            for i, (e, pol) in enumerate(zip(envs, policies)):
                if not e.done:
                    obs[i], *_ = e.step(pol(obs[i]))
            finished = all(e.done for e in envs)
        else:
            keys = pygame.key.get_pressed()
            steer = float(keys[pygame.K_LEFT]) - float(keys[pygame.K_RIGHT])  # +y is up, CCW positive
            throttle = float(keys[pygame.K_UP]) - float(keys[pygame.K_DOWN])
            env.step((steer, throttle))
            finished = env.done
        renderer.draw(tuple(envs[1:]))
        if finished:
            pygame.time.wait(500)
            obs = [e.reset() for e in envs]
        clock.tick(30)
    pygame.quit()


if __name__ == "__main__":
    main()
