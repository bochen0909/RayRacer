"""Watch the car. Add --track Monza (etc.) to pick a circuit; models are in runs/<track>/.

  uv run --extra viz python scripts/watch.py                 # drive with arrow keys
  uv run --extra viz python scripts/watch.py --policy runs/oval/best.npy
  uv run --extra viz python scripts/watch.py --population runs/oval/population.npy
  uv run --extra viz python scripts/watch.py --neat runs/oval/neat_best.pkl
  uv run --extra viz --extra rl python scripts/watch.py --ppo runs/oval/ppo.zip
"""

import argparse
from pathlib import Path

import numpy as np
import pygame

from rf.evolve import make_env, make_policy
from rf.tracks import track_names
from rf.viz.render import Renderer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", type=Path)
    ap.add_argument("--neat", type=Path)
    ap.add_argument("--ppo", type=Path)
    ap.add_argument("--population", type=Path)
    ap.add_argument("--track", default="oval", choices=track_names())
    ap.add_argument("--follow", action=argparse.BooleanOptionalAction, default=None,
                    help="camera follows the car (default: on for non-oval tracks)")
    args = ap.parse_args()

    # Be forgiving about which flag a model file was passed to: route by extension.
    for flag in ("policy", "neat", "ppo"):
        path = getattr(args, flag)
        if path is not None:
            kind = {".npy": "policy", ".pkl": "neat", ".zip": "ppo"}.get(path.suffix)
            if kind is None:
                ap.error(f"unrecognised model file {path} (expected .npy, .pkl or .zip)")
            if kind != flag:
                setattr(args, flag, None)
                setattr(args, kind, path)
    env = make_env(args.track)
    renderer = Renderer(env, follow=args.track != "oval" if args.follow is None else args.follow)

    # The first car is the main one (with rays); the rest are drawn dimmed.
    if args.neat:
        from rf.neat_train import load_policy

        policies = [load_policy(args.neat)]
        envs = [env]
    elif args.ppo:
        from rf.rl import load_policy

        policies = [load_policy(args.ppo)]
        envs = [env]
    else:
        weights = [np.load(args.policy)] if args.policy else list(np.load(args.population)) if args.population else []
        envs = [env] + [make_env(args.track) for _ in weights[1:]]
        policies = [make_policy(e, w) for e, w in zip(envs, weights)]

    clock = pygame.time.Clock()
    obs = [e.reset() for e in envs]
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                renderer.handle_key(event.key)
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
        hint = "R rays   G gates   M map"
        if not policies:
            hint = "arrows: drive   " + hint if pygame.key.get_focused() else "click this window to give it keyboard focus"
        renderer.draw(tuple(envs[1:]), hint)
        if finished:
            pygame.time.wait(500)
            obs = [e.reset() for e in envs]
        clock.tick(30)
    pygame.quit()


if __name__ == "__main__":
    main()
