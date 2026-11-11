"""Headless GA training. Run: uv run python scripts/train.py --generations 40"""

import argparse
from pathlib import Path

import numpy as np

from rf.evolve import GAConfig, evolve
from rf.tracks import track_names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=40)
    ap.add_argument("--pop", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--track", default="oval", choices=track_names())
    ap.add_argument("--out", type=Path, default=Path("runs"))
    args = ap.parse_args()
    out = args.out / args.track
    out.mkdir(parents=True, exist_ok=True)

    def report(gen: int, pop: np.ndarray, fit: np.ndarray) -> None:
        best = int(np.argmax(fit))
        print(f"gen {gen:3d}  best {fit[best]:6.2f}  mean {fit.mean():6.2f}", flush=True)
        np.save(out / "best.npy", pop[best])

    pop, fit = evolve(args.generations, GAConfig(pop_size=args.pop, seed=args.seed), on_generation=report, track=args.track)
    np.save(out / "population.npy", pop[np.argsort(-fit)])
    print(f"saved to {out}/")


if __name__ == "__main__":
    main()
