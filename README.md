# rf

Ray-sensor car racing simulation for evolutionary / RL experiments.

- `src/rf/` — pure NumPy core (geometry, track, car, sensors, env). No pygame import.
- `src/rf/viz/` — optional pygame rendering.
- `src/rf/policy.py`, `src/rf/evolve.py` — MLP policy and genetic algorithm.
- `scripts/train.py` — headless GA training (saves `runs/best.npy`, `runs/population.npy`).
- `scripts/watch.py` — drive by hand with arrow keys, or replay `--policy` / `--population`.

```
uv run pytest
uv run python scripts/train.py --generations 30
uv run --extra viz python scripts/watch.py                       # manual driving
uv run --extra viz python scripts/watch.py --population runs/population.npy
```
