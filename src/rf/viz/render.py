"""pygame rendering (optional dependency: `uv sync --extra viz`)."""

from __future__ import annotations

import numpy as np
import pygame

from rf.car import body_segments
from rf.env import CarEnv


class Renderer:
    def __init__(self, env: CarEnv, size: tuple[int, int] = (900, 600), margin: float = 30.0):
        pygame.init()
        self.env = env
        self.screen = pygame.display.set_mode(size)
        pygame.display.set_caption("rf")
        self.font = pygame.font.SysFont(None, 24)
        pts = np.concatenate([env.track.outer, env.track.inner])
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        self.scale = min((size[0] - 2 * margin) / (hi[0] - lo[0]), (size[1] - 2 * margin) / (hi[1] - lo[1]))
        self.offset = np.array(size) / 2 - self.scale * (lo + hi) / 2

    def _to_screen(self, p: np.ndarray) -> tuple[float, float]:
        q = p * np.array([1.0, -1.0]) * self.scale + self.offset
        return float(q[0]), float(q[1])

    def draw(self, others: tuple[CarEnv, ...] = ()) -> None:
        env, scr = self.env, self.screen
        scr.fill((30, 30, 30))
        for poly in (env.track.outer, env.track.inner):
            pygame.draw.polygon(scr, (200, 200, 200), [self._to_screen(p) for p in poly], 2)
        for gate in env.track.checkpoints:
            pygame.draw.line(scr, (60, 60, 90), self._to_screen(gate[0]), self._to_screen(gate[1]), 1)

        for other in others:
            color = (90, 50, 50) if other.crashed else (90, 120, 160)
            for seg in body_segments(other.state, other.car):
                pygame.draw.line(scr, color, self._to_screen(seg[0]), self._to_screen(seg[1]), 2)

        s = env.state
        angles = s.heading + env.config.ray_angles
        rays = env.rays * env.config.ray_range
        for ang, dist in zip(angles, rays):
            end = s.pos + dist * np.array([np.cos(ang), np.sin(ang)])
            pygame.draw.line(scr, (80, 200, 80), self._to_screen(s.pos), self._to_screen(end), 1)
        color = (220, 60, 60) if env.crashed else (80, 160, 255)
        for seg in body_segments(s, env.car):
            pygame.draw.line(scr, color, self._to_screen(seg[0]), self._to_screen(seg[1]), 3)

        text = f"checkpoints {env.checkpoints_passed}  speed {s.speed:5.1f}  step {env.steps}"
        scr.blit(self.font.render(text, True, (230, 230, 230)), (10, 10))
        pygame.display.flip()
