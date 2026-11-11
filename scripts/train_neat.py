"""NEAT training. Run: uv run python scripts/train_neat.py --generations 40"""

import argparse
from pathlib import Path

from rf.neat_train import save_genome, train


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=40)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", type=Path, default=Path("runs"))
    args = ap.parse_args()
    args.out.mkdir(exist_ok=True)

    def report(gen, best, mean, n_species):
        nodes, conns = len(best.nodes), sum(c.enabled for c in best.connections.values())
        print(f"gen {gen:3d}  best {best.fitness:6.2f}  mean {mean:6.2f}  species {n_species}  net {nodes} nodes/{conns} conns", flush=True)

    best, config = train(args.generations, args.workers, on_generation=report)
    save_genome(best, config, args.out / "neat_best.pkl")
    print(f"saved to {args.out}/neat_best.pkl")


if __name__ == "__main__":
    main()
