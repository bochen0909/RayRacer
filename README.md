# rf — learning to drive with ray sensors

A small racing simulator where cars "see" the track through distance sensors (rays) and learn to drive it.
The same environment is trained three ways so you can compare them: a **genetic algorithm**, **NEAT**
(neuroevolution that grows its own network), and **PPO** (reinforcement learning). It includes the oval
plus five real F1 circuits, and a pygame viewer to watch the cars.

The simulation core is pure NumPy (no pygame), so training runs headless and fast.

## Quick start

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv run --extra viz python scripts/watch.py --track Monza        # drive it yourself (arrow keys)
uv run python scripts/train.py --track Monza --generations 100 --pop 100   # train (headless)
uv run --extra viz python scripts/watch.py --track Monza --policy runs/Monza/best.npy   # watch the result
```

Click the pygame window once so it receives keyboard focus (the on-screen hint tells you when it has it).

## Training and watching

| Method | Train | Saves | Watch with |
|---|---|---|---|
| Genetic algorithm (fixed 6→8→2 MLP) | `uv run python scripts/train.py --track Monza` | `runs/Monza/best.npy`, `population.npy` | `--policy` / `--population` |
| NEAT (evolves topology + weights) | `uv run python scripts/train_neat.py --track Monza` | `runs/Monza/neat_best.pkl` | `--neat` |
| PPO (Stable-Baselines3) | `uv run --extra rl python scripts/train_ppo.py --track Monza` | `runs/Monza/ppo.zip` | `--ppo` |

```bash
uv run --extra viz python scripts/watch.py --track Monza --population runs/Monza/population.npy  # whole population at once
uv run --extra viz python scripts/watch.py --track Spa --policy runs/Monza/best.npy               # model from another track
uv run --extra viz --extra rl python scripts/watch.py --track Monza --ppo runs/Monza/ppo.zip
```

- `--track` accepts `oval`, `Monza`, `Silverstone`, `Spa`, `Catalunya` or `YasMarina` (default `oval`) on every script.
  Models are saved per track in `runs/<track>/`.
- Models are not tied to a track: the cars only see rays and speed, so any model can be watched on any track.
  Check generalization this way.
- `watch.py` picks the loader from the file extension (`.npy`, `.pkl`, `.zip`), so the flag you use doesn't matter.
- Training options: `--generations`, `--pop`, `--seed` (genetic algorithm); `--generations`, `--workers` (NEAT);
  `--steps`, `--envs` (PPO). Genetic algorithm and NEAT evaluate cars in parallel across your CPU cores.

Rough timings on a laptop: NEAT on Monza for 40 generations is about a minute; the genetic algorithm for
150 generations with 100 cars is about 10 minutes, because good cars survive for thousands of steps.

## Viewer

`watch.py` draws asphalt with sand run-off, kerbs in the corners, a checkered start line, car sprites, tyre
trails, a speed-sensitive follow camera, a speedometer in mph and a minimap.

| Key | Action |
|---|---|
| Arrow keys | Drive (manual mode only): up = throttle, down = brake, left/right = steer |
| R | Toggle sensor rays |
| G | Toggle checkpoint gates |
| M | Toggle minimap |

The camera follows the car on the real circuits; add `--no-follow` for a whole-track view.

## How it works

**Observation** (6 numbers): normalized distances from 5 rays (-60°, -30°, 0°, 30°, 60°, range 300 units) plus
normalized speed. **Action**: `(steer, throttle)`, both in [-1, 1]; negative throttle brakes.

**Episode ends** when the car touches a wall, runs out of steps (about two laps' worth at 100 units/s; 1500 on
the oval), or passes no checkpoint for 150 steps.

**Fitness** (genetic algorithm and NEAT) is the number of checkpoint gates passed, plus a tiny survival bonus
to break ties. Gates repeat every lap, so faster cars score more within the step budget. PPO uses a denser
reward: +1 per gate, a small speed term and a -2 crash penalty (see `src/rf/rl.py`).

**Car model** (`src/rf/car.py`): 3 simulation units per metre, speed shown in mph, top speed about 224 mph.
Throttle and brake are separate, and a tyre-grip limit caps cornering (`max_lateral_accel`), so a car that
arrives at a corner too fast understeers into the wall and has to brake. Models trained under older physics
need retraining.

**Tracks** (`src/rf/tracks/`): the oval is generated; the real circuits are loaded from CSV centerlines and widths
and scaled to the simulation. Gates are placed every 100 units.

## Results so far (Monza)

| Method | Result |
|---|---|
| Genetic algorithm, 150 gens × 100 cars | About 4 laps (747 gates), average 161 mph, braking for chicanes |
| NEAT, 40 gens | About 1.4 laps (247 gates); needs more generations |
| PPO | Solved the oval (about 7 laps) under the earlier, simpler physics; not yet retrained or tuned for the grip physics or Monza |

Earlier experiments showed the genetic algorithm's Monza model transferring to every other track, while a
NEAT model overfit to Monza. Treat these as single runs, not benchmarks.

## Project layout

```
src/rf/
  geometry.py      ray–segment intersection, segment crossing (NumPy)
  car.py           car physics (drag, brake, grip limit), mph conversion
  sensors.py       ray casting
  track.py         Track dataclass: walls, gates, start pose
  tracks/          oval + CSV circuits (data/ holds the CSVs and license)
  env.py           CarEnv: reset/step, collisions, checkpoints
  policy.py        flat-weight MLP policy
  evolve.py        genetic algorithm
  neat_train.py    NEAT (neat-python); config in configs/neat.ini
  rl.py            Gymnasium wrapper and SB3 policy adapter
  viz/render.py    pygame renderer (optional extra)
scripts/           train.py, train_neat.py, train_ppo.py, watch.py
tests/             pytest suite
```

## Development

```bash
uv run --extra viz --extra rl pytest      # run everything; viz/rl tests skip if the extra is missing
```

Optional extras: `viz` (pygame) and `rl` (gymnasium, stable-baselines3, PyTorch). `runs/` is git-ignored.

## Credits

Track centerlines and widths come from [TUMFTM/racetrack-database](https://github.com/TUMFTM/racetrack-database)
(LGPL-3.0; a license copy is in `src/rf/tracks/data/LICENSE-TUMFTM`). Suzuka is not included because its road
crosses itself (overpass), which the wall model can't handle.
