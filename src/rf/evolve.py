"""Genetic algorithm over flat policy weight vectors."""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import Pool
from typing import Callable

import numpy as np

from rf.env import CarEnv
from rf.policy import MLPPolicy
from rf.tracks import get_track

HIDDEN = 8


def make_env(track: str = "oval") -> CarEnv:
    return CarEnv(get_track(track))


def make_policy(env: CarEnv, weights: np.ndarray | None = None) -> MLPPolicy:
    return MLPPolicy((env.obs_size, HIDDEN, 2), weights)


def rollout(env: CarEnv, policy: Callable[[np.ndarray], tuple[float, float]]) -> float:
    """Checkpoints passed, with a tiny survival bonus to break ties."""
    obs, done = env.reset(), False
    while not done:
        obs, _, done, _ = env.step(policy(obs))
    return env.checkpoints_passed + 0.1 * env.steps / env.config.max_steps


def rollout_fitness(env: CarEnv, weights: np.ndarray) -> float:
    return rollout(env, make_policy(env, weights))


_worker_env: CarEnv | None = None


def init_worker(track: str) -> None:
    global _worker_env
    _worker_env = make_env(track)


def _evaluate(weights: np.ndarray) -> float:
    return rollout_fitness(_worker_env, weights)


@dataclass
class GAConfig:
    pop_size: int = 50
    elite: int = 5
    tournament: int = 3
    mutation_rate: float = 0.1
    mutation_sigma: float = 0.3
    init_sigma: float = 1.0
    seed: int = 0


def next_generation(pop: np.ndarray, fitness: np.ndarray, cfg: GAConfig, rng: np.random.Generator) -> np.ndarray:
    order = np.argsort(-fitness)
    new = [pop[i].copy() for i in order[: cfg.elite]]

    def pick() -> np.ndarray:
        cand = rng.integers(0, len(pop), cfg.tournament)
        return pop[cand[np.argmax(fitness[cand])]]

    while len(new) < cfg.pop_size:
        a, b = pick(), pick()
        child = np.where(rng.random(a.shape) < 0.5, a, b)  # uniform crossover
        mask = rng.random(child.shape) < cfg.mutation_rate
        child = child + mask * rng.normal(0.0, cfg.mutation_sigma, child.shape)
        new.append(child)
    return np.array(new)


def evolve(
    generations: int,
    cfg: GAConfig | None = None,
    workers: int | None = None,
    on_generation: Callable[[int, np.ndarray, np.ndarray], None] | None = None,
    track: str = "oval",
) -> tuple[np.ndarray, np.ndarray]:
    """Run the GA. Returns (final population, its fitness)."""
    cfg = cfg or GAConfig()
    rng = np.random.default_rng(cfg.seed)
    n = make_policy(make_env(track)).n_weights
    pop = rng.normal(0.0, cfg.init_sigma, (cfg.pop_size, n))

    with Pool(workers, initializer=init_worker, initargs=(track,)) as pool:
        for gen in range(generations):
            fitness = np.array(pool.map(_evaluate, list(pop)))
            if on_generation:
                on_generation(gen, pop, fitness)
            if gen < generations - 1:
                pop = next_generation(pop, fitness, cfg, rng)
    return pop, fitness
