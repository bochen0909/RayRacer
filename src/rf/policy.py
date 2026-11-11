"""Small tanh MLP policy whose weights live in one flat vector (easy to evolve)."""

from __future__ import annotations

import numpy as np


class MLPPolicy:
    def __init__(self, sizes: tuple[int, ...], weights: np.ndarray | None = None):
        self.sizes = sizes
        self.n_weights = sum((i + 1) * o for i, o in zip(sizes[:-1], sizes[1:]))
        self.weights = np.zeros(self.n_weights) if weights is None else np.asarray(weights, dtype=float)
        if self.weights.shape != (self.n_weights,):
            raise ValueError(f"expected {self.n_weights} weights, got {self.weights.shape}")

    def __call__(self, obs: np.ndarray) -> tuple[float, float]:
        x = np.asarray(obs, dtype=float)
        k = 0
        for i, o in zip(self.sizes[:-1], self.sizes[1:]):
            w = self.weights[k : k + i * o].reshape(i, o)
            b = self.weights[k + i * o : k + (i + 1) * o]
            k += (i + 1) * o
            x = np.tanh(x @ w + b)
        return float(x[0]), float(x[1])
