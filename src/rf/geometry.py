"""Vectorized 2D geometry helpers (NumPy only)."""

from __future__ import annotations

import numpy as np


def ray_segment_distances(
    origin: np.ndarray,
    angles: np.ndarray,
    segments: np.ndarray,
    max_range: float,
) -> np.ndarray:
    """Distance from `origin` along each ray angle to the nearest segment.

    origin: (2,), angles: (R,) radians, segments: (M, 2, 2).
    Returns (R,) distances, capped at `max_range`.
    """
    d = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # (R, 2)
    p = segments[:, 0, :]  # (M, 2)
    e = segments[:, 1, :] - p  # (M, 2)
    w = p - origin  # (M, 2)

    # Solve origin + t*d = p + u*e  ->  t = cross(w, e)/cross(d, e), u = cross(w, d)/cross(d, e)
    denom = d[:, None, 0] * e[None, :, 1] - d[:, None, 1] * e[None, :, 0]  # (R, M)
    wx, wy = w[None, :, 0], w[None, :, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (wx * e[None, :, 1] - wy * e[None, :, 0]) / denom
        u = (wx * d[:, None, 1] - wy * d[:, None, 0]) / denom
    hit = (np.abs(denom) > 1e-12) & (t >= 0.0) & (u >= 0.0) & (u <= 1.0)
    t = np.where(hit, t, np.inf)
    return np.minimum(t.min(axis=1), max_range)


def segments_intersect_any(a: np.ndarray, b: np.ndarray) -> bool:
    """True if any segment in `a` (N, 2, 2) intersects any segment in `b` (M, 2, 2)."""
    p = a[:, None, 0, :]
    r = a[:, None, 1, :] - p
    q = b[None, :, 0, :]
    s = b[None, :, 1, :] - q

    denom = r[..., 0] * s[..., 1] - r[..., 1] * s[..., 0]
    qp = q - p
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (qp[..., 0] * s[..., 1] - qp[..., 1] * s[..., 0]) / denom
        u = (qp[..., 0] * r[..., 1] - qp[..., 1] * r[..., 0]) / denom
    hit = (np.abs(denom) > 1e-12) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
    return bool(hit.any())
