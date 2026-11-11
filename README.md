# rf

Ray-sensor car racing simulation for evolutionary / RL experiments.

- `src/rf/` — pure NumPy core (geometry, track, car, sensors, env). No pygame import.
- `src/rf/tracks/` — `oval` plus real circuits (Monza, Silverstone, Spa, Catalunya, YasMarina) loaded from CSVs in `tracks/data/`. List them with `track_names()`.
- `src/rf/viz/` — optional pygame rendering (the camera follows the car on non-oval tracks).
- `src/rf/policy.py`, `src/rf/evolve.py` — MLP policy and genetic algorithm.
- `src/rf/neat_train.py`, `configs/neat.ini` — NEAT version (neat-python), evolves topology too.
- `src/rf/rl.py`, `scripts/train_ppo.py` — Gymnasium wrapper (dense reward) and PPO via Stable-Baselines3 (optional `rl` extra).
- `scripts/train*.py` save to `runs/<track>/` (`best.npy`, `population.npy`, `neat_best.pkl`, `ppo.zip`).

```
uv run pytest
uv run python scripts/train.py --track Monza --generations 100 --pop 100
uv run python scripts/train_neat.py --track Monza --generations 40
uv run --extra rl python scripts/train_ppo.py --track Monza --steps 300000
uv run --extra viz python scripts/watch.py --track Monza                                 # manual driving
uv run --extra viz python scripts/watch.py --track Monza --population runs/Monza/population.npy
uv run --extra viz python scripts/watch.py --track Monza --neat runs/Monza/neat_best.pkl
uv run --extra viz --extra rl python scripts/watch.py --track Monza --ppo runs/Monza/ppo.zip
```

Omit `--track` for the oval. Track centerlines and widths come from
[TUMFTM/racetrack-database](https://github.com/TUMFTM/racetrack-database) (LGPL-3.0, license copy in
`src/rf/tracks/data/LICENSE-TUMFTM`). Suzuka is not included because its road crosses itself (overpass).

## Car model

Tracks use 3 sim units per metre (`UNITS_PER_METER`), and the HUD shows speed in mph. The car has a top
speed of about 224 mph, separate throttle and brake (negative throttle), and a tyre-grip limit on
cornering (`max_lateral_accel`), so a car that arrives too fast understeers into the wall and must brake
for corners. Models trained before this physics was added need retraining.
