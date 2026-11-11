"""NEAT training (neat-python): evolves network topology and weights."""

from __future__ import annotations

import pickle
from pathlib import Path

import neat

from rf.evolve import make_env, rollout

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "neat.ini"

_worker_env = None


def load_config(path: Path = DEFAULT_CONFIG) -> neat.Config:
    return neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(path),
    )


class NeatPolicy:
    """Wraps a NEAT feed-forward network as a callable policy: obs -> (steer, throttle)."""

    def __init__(self, genome, config: neat.Config):
        self.net = neat.nn.FeedForwardNetwork.create(genome, config)

    def __call__(self, obs) -> tuple[float, float]:
        out = self.net.activate(list(obs))
        return float(out[0]), float(out[1])


def init_worker(track: str) -> None:
    global _worker_env
    _worker_env = make_env(track)


def eval_genome(genome, config) -> float:
    return rollout(_worker_env, NeatPolicy(genome, config))


def train(
    generations: int,
    workers: int | None = None,
    config_path: Path = DEFAULT_CONFIG,
    on_generation=None,
    track: str = "oval",
):
    """Returns (best genome, config)."""
    config = load_config(config_path)
    pop = neat.Population(config)
    if on_generation:
        pop.add_reporter(_Reporter(on_generation))
    with neat.ParallelEvaluator(workers or 4, eval_genome, initializer=init_worker, initargs=(track,)) as evaluator:
        best = pop.run(evaluator.evaluate, generations)
    return best, config


class _Reporter(neat.reporting.BaseReporter):
    def __init__(self, callback):
        self.callback = callback
        self.gen = 0

    def start_generation(self, generation):
        self.gen = generation

    def post_evaluate(self, config, population, species, best_genome):
        fits = [g.fitness for g in population.values()]
        self.callback(self.gen, best_genome, sum(fits) / len(fits), len(species.species))


def save_genome(genome, config: neat.Config, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump((genome, config), f)


def load_policy(path: Path) -> NeatPolicy:
    with open(path, "rb") as f:
        genome, config = pickle.load(f)
    return NeatPolicy(genome, config)
