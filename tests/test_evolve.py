import numpy as np

from rf.evolve import GAConfig, make_env, make_policy, next_generation, rollout_fitness
from rf.policy import MLPPolicy


def test_policy_output_is_bounded_and_deterministic():
    pol = MLPPolicy((6, 8, 2), np.random.default_rng(0).normal(size=(6 + 1) * 8 + (8 + 1) * 2))
    obs = np.ones(6)
    a = pol(obs)
    assert a == pol(obs)
    assert all(-1.0 <= v <= 1.0 for v in a)


def test_next_generation_keeps_elite_and_size():
    rng = np.random.default_rng(0)
    cfg = GAConfig(pop_size=10, elite=2)
    pop = rng.normal(size=(10, 5))
    fit = np.arange(10, dtype=float)
    new = next_generation(pop, fit, cfg, rng)
    assert new.shape == pop.shape
    assert np.array_equal(new[0], pop[9]) and np.array_equal(new[1], pop[8])


def test_evolution_improves_fitness():
    env = make_env()
    rng = np.random.default_rng(0)
    cfg = GAConfig(pop_size=30, elite=3)
    pop = rng.normal(size=(30, make_policy(env).n_weights))
    first = None
    for _ in range(8):
        fit = np.array([rollout_fitness(env, w) for w in pop])
        first = fit.max() if first is None else first
        pop = next_generation(pop, fit, cfg, rng)
    assert fit.max() >= first
    assert fit.max() > 1.0  # passes at least one checkpoint
