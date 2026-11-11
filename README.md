# rf

Ray-sensor car racing simulation for evolutionary / RL experiments.

- `src/rf/` — pure NumPy core (geometry, track, car, sensors, env). No pygame import.
- `src/rf/viz/` — optional pygame rendering.
- `scripts/watch.py` — drive by hand with arrow keys.

```
uv run pytest
uv run --extra viz python scripts/watch.py
```
