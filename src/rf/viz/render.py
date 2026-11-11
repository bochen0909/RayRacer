"""pygame rendering (optional dependency: `uv sync --extra viz`).

Draws an asphalt road with sand run-off, corner kerbs and a start line, car sprites, tyre trails,
a speed-sensitive follow camera, a HUD and an optional minimap. Keys (see `handle_key`):
R toggles sensor rays, G toggles checkpoint gates, M toggles the minimap.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pygame

from rf.car import CarParams, CarState, to_mph
from rf.env import CarEnv

GRASS = (34, 104, 46)
SAND = (196, 178, 128)
ASPHALT = (58, 58, 64)
EDGE_LINE = (235, 235, 235)
KERB_RED = (200, 40, 40)
KERB_WHITE = (240, 240, 240)
TRAIL = (38, 38, 42)
MAIN_COLOR = (222, 36, 36)
OTHER_COLORS = [
    (60, 120, 230), (240, 200, 40), (60, 190, 120), (180, 90, 220),
    (240, 140, 50), (80, 200, 220), (230, 110, 160), (150, 200, 60),
]

RUNOFF = 14.0  # sand width beyond the road edge, in sim units
KERB_WIDTH = 3.0
KERB_TURN = 0.12  # rad of heading change across neighbouring segments that earns a kerb


class Renderer:
    def __init__(
        self,
        env: CarEnv,
        size: tuple[int, int] = (900, 600),
        margin: float = 30.0,
        follow: bool = False,
        zoom: float = 2.8,
        zoom_out: float = 1.4,
        minimap: bool | None = None,
        minimap_size: int = 200,
    ):
        pygame.init()
        self.env = env
        self.size = size
        self.follow = follow
        self.zoom_in, self.zoom_out = zoom, zoom_out
        self.screen = pygame.display.set_mode(size)
        pygame.display.set_caption("rf")
        self.font = pygame.font.SysFont(None, 24)
        self.font_small = pygame.font.SysFont(None, 20)
        self.font_big = pygame.font.SysFont(None, 56)

        pts = np.concatenate([env.track.outer, env.track.inner])
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        self.scale = zoom if follow else min(
            (size[0] - 2 * margin) / (hi[0] - lo[0]), (size[1] - 2 * margin) / (hi[1] - lo[1])
        )
        self.offset = np.array(size) / 2 - self.scale * (lo + hi) / 2

        self.show_rays = True
        self.show_gates = False
        self.minimap = follow if minimap is None else minimap
        self.trail: deque[np.ndarray] = deque(maxlen=160)
        self._last_steps = -1
        self._build_track_geometry()
        if self.minimap:
            self._build_minimap(lo, hi, minimap_size)

    def handle_key(self, key: int) -> None:
        if key == pygame.K_r:
            self.show_rays = not self.show_rays
        elif key == pygame.K_g:
            self.show_gates = not self.show_gates
        elif key == pygame.K_m and hasattr(self, "mm_surface"):
            self.minimap = not self.minimap

    # ---------------------------------------------------------------- static track geometry
    def _build_track_geometry(self) -> None:
        outer, inner = self.env.track.outer, self.env.track.inner
        n = len(outer)
        across = outer - inner
        across /= np.linalg.norm(across, axis=1, keepdims=True)  # unit vector inner -> outer
        nxt = np.roll(np.arange(n), -1)

        def strip(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            """Quad per segment i -> i+1 between edge polylines a and b: (N, 4, 2)."""
            return np.stack([a, a[nxt], b[nxt], b], axis=1)

        self.q_run = strip(outer + across * RUNOFF, inner - across * RUNOFF)
        self.q_road = strip(outer, inner)
        self.q_kerb_out = strip(outer, outer - across * KERB_WIDTH)
        self.q_kerb_in = strip(inner, inner + across * KERB_WIDTH)

        centre = (outer + inner) / 2
        seg = np.roll(centre, -1, axis=0) - centre
        ang = np.arctan2(seg[:, 1], seg[:, 0])
        turn = np.abs((np.roll(ang, -1) - np.roll(ang, 1) + np.pi) % (2 * np.pi) - np.pi)
        flag = turn > KERB_TURN
        self.kerb = flag | np.roll(flag, 1) | np.roll(flag, -1)

        self.start_cells = self._start_line_cells()

    def _start_line_cells(self) -> list[tuple[np.ndarray, tuple[int, int, int]]]:
        outer, inner = self.env.track.outer[0], self.env.track.inner[0]
        h = self.env.track.start_heading
        along = np.array([np.cos(h), np.sin(h)])
        cols, cell = 10, 4.0
        cells = []
        for row in range(2):
            for c in range(cols):
                p0 = inner + (outer - inner) * c / cols
                p1 = inner + (outer - inner) * (c + 1) / cols
                back = -18.0 + row * cell
                poly = np.array([p0 + along * back, p1 + along * back, p1 + along * (back + cell), p0 + along * (back + cell)])
                cells.append((poly, (245, 245, 245) if (c + row) % 2 == 0 else (20, 20, 20)))
        return cells

    # ---------------------------------------------------------------- minimap
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

    # ---------------------------------------------------------------- coordinate helpers
    def _scr(self, p: np.ndarray) -> np.ndarray:
        return p * np.array([1.0, -1.0]) * self.scale + self.offset

    def _to_screen(self, p: np.ndarray) -> tuple[float, float]:
        q = self._scr(p)
        return float(q[0]), float(q[1])

    # ---------------------------------------------------------------- drawing
    def _draw_track(self) -> None:
        scr, (w, h) = self.screen, self.size
        run = self._scr(self.q_run)
        lo, hi = run.min(axis=1), run.max(axis=1)
        vis = np.nonzero((hi[:, 0] > -60) & (lo[:, 0] < w + 60) & (hi[:, 1] > -60) & (lo[:, 1] < h + 60))[0]
        if len(vis) == 0:
            return
        road = self._scr(self.q_road[vis])
        for k in range(len(vis)):
            pygame.draw.polygon(scr, SAND, run[vis[k]])
        for k in range(len(vis)):
            pygame.draw.polygon(scr, ASPHALT, road[k])
        kerb_out, kerb_in = self._scr(self.q_kerb_out[vis]), self._scr(self.q_kerb_in[vis])
        for k, i in enumerate(vis):
            if self.kerb[i]:
                color = KERB_RED if i % 2 == 0 else KERB_WHITE
                pygame.draw.polygon(scr, color, kerb_out[k])
                pygame.draw.polygon(scr, color, kerb_in[k])
            else:
                pygame.draw.line(scr, EDGE_LINE, road[k][0], road[k][1], 2)
                pygame.draw.line(scr, EDGE_LINE, road[k][3], road[k][2], 2)
        for poly, color in self.start_cells:
            pygame.draw.polygon(scr, color, self._scr(poly))

        if self.show_gates:
            for gate in self.env.track.checkpoints:
                pygame.draw.line(scr, (90, 90, 160), self._to_screen(gate[0]), self._to_screen(gate[1]), 1)

    def _draw_trail(self) -> None:
        if len(self.trail) > 1:
            pygame.draw.lines(self.screen, TRAIL, False, [self._to_screen(p) for p in self.trail], 2)

    def _draw_car(self, state: CarState, car: CarParams, color: tuple[int, int, int], crashed: bool) -> None:
        c, s = np.cos(state.heading), np.sin(state.heading)
        fwd, side = np.array([c, s]), np.array([-s, c])

        def poly(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
            return [self._to_screen(state.pos + fwd * x + side * y) for x, y in points]

        def rect(cx: float, cy: float, hx: float, hy: float) -> list[tuple[float, float]]:
            return poly([(cx - hx, cy - hy), (cx + hx, cy - hy), (cx + hx, cy + hy), (cx - hx, cy + hy)])

        hl, hw = car.length / 2, car.width / 2
        body_color = (110, 110, 110) if crashed else color
        dark = (24, 24, 26)
        scr = self.screen
        for wx, wy, wh in ((hl * 0.62, hw, 1.2), (hl * 0.62, -hw, 1.2), (-hl * 0.7, hw, 1.5), (-hl * 0.7, -hw, 1.5)):
            pygame.draw.polygon(scr, dark, rect(wx, wy, 2.6, wh))
        pygame.draw.polygon(scr, dark, rect(-hl + 0.8, 0.0, 0.8, hw * 0.9))  # rear wing
        pygame.draw.polygon(scr, body_color, poly([
            (-hl, -hw * 0.5), (-hl, hw * 0.5), (hl * 0.35, hw * 0.5),
            (hl, hw * 0.2), (hl, -hw * 0.2), (hl * 0.35, -hw * 0.5),
        ]))
        pygame.draw.polygon(scr, (240, 240, 240) if not crashed else (150, 150, 150), rect(-1.0, 0.0, hl * 0.8, 0.5))
        pygame.draw.polygon(scr, (20, 30, 50), rect(0.5, 0.0, 2.0, 1.3))  # cockpit
        if crashed:
            sx, sy = self._to_screen(state.pos)
            for dx, dy, r in ((0, -6, 7), (6, -12, 6), (-4, -17, 5)):
                pygame.draw.circle(scr, (150, 150, 150), (sx + dx, sy + dy), r, 2)

    def _draw_rays(self) -> None:
        env, s = self.env, self.env.state
        angles = s.heading + env.config.ray_angles
        rays = env.rays * env.config.ray_range
        origin = self._to_screen(s.pos)
        for ang, dist in zip(angles, rays):
            end = s.pos + dist * np.array([np.cos(ang), np.sin(ang)])
            pygame.draw.line(self.screen, (90, 230, 110), origin, self._to_screen(end), 1)
            pygame.draw.circle(self.screen, (255, 90, 90), self._to_screen(end), 3)

    def _panel(self, rect: pygame.Rect) -> None:
        panel = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(panel, (0, 0, 0, 150), panel.get_rect(), border_radius=8)
        self.screen.blit(panel, rect.topleft)

    def _draw_hud(self, hint: str) -> None:
        env, scr, (w, h) = self.env, self.screen, self.size
        gates = len(env.track.checkpoints)
        lap, gate = env.checkpoints_passed // gates + 1, env.checkpoints_passed % gates

        self._panel(pygame.Rect(10, 10, 190, 62))
        scr.blit(self.font_big.render(f"LAP {lap}", True, (255, 255, 255)), (20, 14))
        scr.blit(self.font_small.render(f"gate {gate}/{gates}   step {env.steps}", True, (200, 200, 200)), (20, 52))

        self._panel(pygame.Rect(10, h - 96, 210, 86))
        mph = to_mph(env.state.speed)
        scr.blit(self.font_big.render(f"{mph:3.0f}", True, (255, 255, 255)), (20, h - 92))
        scr.blit(self.font.render("MPH", True, (200, 200, 200)), (100, h - 70))
        frac = min(env.state.speed / env.car.max_speed, 1.0)
        bar = pygame.Rect(20, h - 32, 190, 10)
        pygame.draw.rect(scr, (60, 60, 60), bar, border_radius=4)
        fill = (int(60 + 170 * frac), int(200 - 140 * frac), 60)
        pygame.draw.rect(scr, fill, pygame.Rect(bar.x, bar.y, int(bar.w * frac), bar.h), border_radius=4)

        if env.crashed:
            text = self.font_big.render("CRASHED", True, (255, 70, 70))
            scr.blit(text, text.get_rect(center=(w // 2, 60)))
        if hint:
            scr.blit(self.font.render(hint, True, (255, 210, 90)), (230, h - 30))

    def draw(self, others: tuple[CarEnv, ...] = (), hint: str = "") -> None:
        env, scr = self.env, self.screen
        s = env.state
        if env.steps < self._last_steps:
            self.trail.clear()
        if env.steps != self._last_steps:
            self.trail.append(s.pos.copy())
            self._last_steps = env.steps
        if self.follow:
            target = self.zoom_in - (self.zoom_in - self.zoom_out) * min(s.speed / env.car.max_speed, 1.0)
            self.scale += (target - self.scale) * 0.08
            self.offset = np.array(self.size) / 2 - self.scale * s.pos * np.array([1.0, -1.0])

        scr.fill(GRASS)
        self._draw_track()
        self._draw_trail()
        for i, other in enumerate(others):
            self._draw_car(other.state, other.car, OTHER_COLORS[i % len(OTHER_COLORS)], other.crashed)
        if self.show_rays:
            self._draw_rays()
        self._draw_car(s, env.car, MAIN_COLOR, env.crashed)
        self._draw_hud(hint)
        if self.minimap:
            self._draw_minimap(others)
        pygame.display.flip()
