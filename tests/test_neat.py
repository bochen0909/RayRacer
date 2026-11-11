import neat

from rf.evolve import make_env, rollout
from rf.neat_train import NeatPolicy, load_config


def test_neat_policy_drives_env():
    config = load_config()
    genome = next(iter(neat.Population(config).population.values()))
    policy = NeatPolicy(genome, config)
    env = make_env()
    steer, throttle = policy(env.reset())
    assert -1.0 <= steer <= 1.0 and -1.0 <= throttle <= 1.0
    assert rollout(env, policy) >= 0.0
