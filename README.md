# rf

Ray-sensor car racing simulation for evolutionary / RL experiments.

- `src/rf/` — pure NumPy core (geometry, track, car, sensors, env). No pygame import.
- `src/rf/viz/` — optional pygame rendering.
- `src/rf/policy.py`, `src/rf/evolve.py` — MLP policy and genetic algorithm.
- `src/rf/neat_train.py`, `configs/neat.ini` — NEAT version (neat-python), evolves topology too.
- `scripts/train.py` — headless GA training (saves `runs/best.npy`, `runs/population.npy`).
- `scripts/train_neat.py` — headless NEAT training (saves `runs/neat_best.pkl`).
- `src/rf/rl.py`, `scripts/train_ppo.py` — Gymnasium wrapper (dense reward) and PPO via Stable-Baselines3 (optional `rl` extra, saves `runs/ppo.zip`).
- `scripts/watch.py` — drive by hand with arrow keys, or replay `--policy` / `--population` / `--neat` / `--ppo`.

```
uv run pytest
uv run python scripts/train.py --generations 30
uv run python scripts/train_neat.py --generations 20
uv run --extra rl python scripts/train_ppo.py --steps 100000
uv run --extra viz python scripts/watch.py                       # manual driving
uv run --extra viz python scripts/watch.py --population runs/population.npy
uv run --extra viz python scripts/watch.py --neat runs/neat_best.pkl
uv run --extra viz --extra rl python scripts/watch.py --ppo runs/ppo.zip
```
