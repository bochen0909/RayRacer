"""Load real circuits from TUMFTM racetrack-database CSVs (LGPL-3.0, see data/LICENSE-TUMFTM).

CSV columns: x_m, y_m, w_tr_right_m, w_tr_left_m (centerline and half-widths in metres).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rf.car import UNITS_PER_METER
from rf.track import Track

DATA_DIR = Path(__file__).parent / "data"


def available() -> list[str]:
    return sorted(p.stem for p in DATA_DIR.glob("*.csv"))


def load_csv_track(
    name: str,
    scale: float = UNITS_PER_METER,
    gate_spacing: float = 100.0,
    step: int = 2,
    car_speed: float = 100.0,
    dt: float = 1.0 / 30.0,
) -> Track:
    """Build a Track from a CSV. `scale` converts metres to sim units.

    `step` subsamples the centerline (fewer wall segments = faster rays/collisions).
    Checkpoint gates are placed every `gate_spacing` sim units along the centerline.
    `max_steps` allows two laps at `car_speed` units/s.
    """
    path = DATA_DIR / f"{name}.csv"
    if not path.exists():
        raise ValueError(f"unknown track {name!r}; available: {available()}")
    data = np.loadtxt(path, delimiter=",", comments="#")[::step]
    c = data[:, :2] * scale
    wr, wl = data[:, 2] * scale, data[:, 3] * scale

    tangent = np.roll(c, -1, axis=0) - np.roll(c, 1, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    left = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    left_wall = c + left * wl[:, None]
    right_wall = c - left * wr[:, None]

    seg = np.linalg.norm(np.roll(c, -1, axis=0) - c, axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)[:-1]])
    lap_length = float(seg.sum())
    n_gates = max(int(lap_length // gate_spacing), 8)
    idx = np.searchsorted(arc, np.arange(n_gates) * lap_length / n_gates)
    idx = np.unique(np.clip(idx, 0, len(c) - 1))
    gates = np.stack([c[idx] + left[idx] * wl[idx, None] * 1.1, c[idx] - left[idx] * wr[idx, None] * 1.1], axis=1)

    heading = float(np.arctan2(tangent[0, 1], tangent[0, 0]))
    max_steps = int(2 * lap_length / (car_speed * dt))
    return Track(left_wall, right_wall, gates, c[0].copy(), heading, max_steps)
