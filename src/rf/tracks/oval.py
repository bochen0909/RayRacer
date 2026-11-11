"""Oval track: two concentric ellipses with a road of constant half-width."""

from __future__ import annotations

import numpy as np

from rf.track import Track


def make_oval(
    a: float = 300.0,
    b: float = 180.0,
    half_width: float = 45.0,
    n_points: int = 96,
    n_checkpoints: int = 48,
) -> Track:
    """Counter-clockwise oval centred on the origin. `a`, `b` are centerline semi-axes."""

    def ellipse(ra: float, rb: float, n: int) -> np.ndarray:
        t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        return np.stack([ra * np.cos(t), rb * np.sin(t)], axis=1)

    outer = ellipse(a + half_width, b + half_width, n_points)
    inner = ellipse(a - half_width, b - half_width, n_points)

    # Gates span the road at the same parameter angle (slightly past the walls).
    ci = ellipse(a - half_width * 1.1, b - half_width * 1.1, n_checkpoints)
    co = ellipse(a + half_width * 1.1, b + half_width * 1.1, n_checkpoints)
    checkpoints = np.stack([ci, co], axis=1)

    # Start on the centerline at t=0, heading along the CCW tangent (+y).
    start_pos = np.array([a, 0.0])
    start_heading = np.pi / 2.0
    # Checkpoint 0 sits on the start line; begin the lap expecting checkpoint 1.
    return Track(outer, inner, checkpoints, start_pos, start_heading)
