"""Ray-cast distance sensors."""

from __future__ import annotations

import numpy as np

from rf.car import CarState
from rf.geometry import ray_segment_distances

DEFAULT_RAY_ANGLES = np.deg2rad([-60.0, -30.0, 0.0, 30.0, 60.0])


def sense(
    state: CarState,
    walls: np.ndarray,
    ray_angles: np.ndarray = DEFAULT_RAY_ANGLES,
    max_range: float = 150.0,
) -> np.ndarray:
    """Normalized distances in [0, 1] (1 = nothing within `max_range`)."""
    d = ray_segment_distances(state.pos, state.heading + ray_angles, walls, max_range)
    return d / max_range
