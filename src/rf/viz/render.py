"""pygame rendering (optional dependency: `uv sync --extra viz`)."""

from __future__ import annotations

import numpy as np
import pygame

from rf.car import body_segments, to_mph
from rf.env import CarEnv


class Renderer:
    def __init__(
        self,
        env: CarEnv,
        size: tuple[int, int] = (900, 600),
        margin: float = 30.0,
        follow: bool = False,
        zoom: float = 1.5,
        minimap: bool | None = None,
        minimap_size: int = 200,
    ):
        pygame.init()
        self.env = env
        self.size = size
        self.follow = follow
        self.screen = pygame.display.set_mode(size)
        pygame.display.set_caption("rf")
        self.font = pygame.font.SysFont(None, 24)
        pts = np.concatenate([env.track.outer, env.track.inner])
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        self.scale = zoom if follow else min(
            (size[0] - 2 * margin) / (hi[0] - lo[0]), (size[1] - 2 * margin) / (hi[1] - lo[1])
        )
        self.offset = np.array(size) / 2 - self.scale * (lo + hi) / 2
        self.minimap = follow if minimap is None else minimap
        if self.minimap:
            self._build_minimap(lo, hi, minimap_size)

    def _build_minimap(self, lo: np.ndarray, hi: np.ndarray, side: int, pad: int = 8) -> None:
        """Pre-render the static track outline; cars are drawn on top each frame."""
        self.mm_scale = (side - 2 * pad) / float(max(hi - lo))
        centre = (lo + hi) / 2
        self.mm_offset = np.array([side, side]) / 2 - self.mm_scale * centre * np.array([1.0, -1.0])
        self.mm_surface = pygame.Surface((side, side), pygame.SRCALPHA)
        self.mm_surface.fill((0, 0, 0, 150))
        for poly in (self.env.track.outer, self.env.track.inner):
            pygame.draw.lines(self.mm_surface, (200, 200, 200), True, [self._mm_point(p) for p in poly], 1)
        self.mm_pos = (self.size[0] - side - 10, 10)

    def _mm_point(self, p: np.ndarray) -> tuple[float, float]:
        q = p * np.array([1.0, -1.0]) * self.mm_scale + self.mm_offset
        return float(q[0]), float(q[1])

    def _draw_minimap(self, others: tuple[CarEnv, ...]) -> None:
        panel = self.mm_surface.copy()
        for other in others:
            color = (120, 60, 60) if other.crashed else (90, 130, 200)
            pygame.draw.circle(panel, color, self._mm_point(other.state.pos), 2)
        main = self.env.state
        pygame.draw.circle(panel, (255, 220, 60), self._mm_point(main.pos), 4)
        heading = np.array([np.cos(main.heading), np.sin(main.heading)])
        tip = self._mm_point(main.pos + heading * 8 / self.mm_scale)
        pygame.draw.line(panel, (255, 220, 60), self._mm_point(main.pos), tip, 2)
        self.screen.blit(panel, self.mm_pos)

    def _to_screen(self, p: np.ndarray) -> tuple[float, float]:
        q = p * np.array([1.0, -1.0]) * self.scale + self.offset
        return float(q[0]), float(q[1])

    def draw(self, others: tuple[CarEnv, ...] = (), hint: str = "") -> None:
        env, scr = self.env, self.screen
        if self.follow:
            self.offset = np.array(self.size) / 2 - self.scale * env.state.pos * np.array([1.0, -1.0])
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

        text = f"checkpoints {env.checkpoints_passed}  speed {to_mph(s.speed):3.0f} mph  step {env.steps}"
        scr.blit(self.font.render(text, True, (230, 230, 230)), (10, 10))
        if hint:
            scr.blit(self.font.render(hint, True, (255, 200, 80)), (10, self.size[1] - 30))
        if self.minimap:
            self._draw_minimap(others)
        pygame.display.flip()
