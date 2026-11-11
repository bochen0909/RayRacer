"""Track definition: wall segments, start pose and ordered checkpoint gates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _closed_segments(points: np.ndarray) -> np.ndarray:
    return np.stack([points, np.roll(points, -1, axis=0)], axis=1)


@dataclass(frozen=True)
class Track:
    outer: np.ndarray  # (N, 2) closed polyline
    inner: np.ndarray  # (N, 2) closed polyline
    checkpoints: np.ndarray  # (K, 2, 2) gates, ordered along driving direction
    start_pos: np.ndarray  # (2,)
    start_heading: float  # radians
    max_steps: int | None = None  # suggested episode length (about two laps); None = env default

    @property
    def walls(self) -> np.ndarray:
        """All wall segments, shape (M, 2, 2)."""
        return np.concatenate([_closed_segments(self.outer), _closed_segments(self.inner)])
