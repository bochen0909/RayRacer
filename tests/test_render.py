import os

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
pygame = pytest.importorskip("pygame")

from rf.evolve import make_env
from rf.viz.render import Renderer


@pytest.mark.parametrize("track,follow", [("oval", False), ("Monza", True)])
def test_renderer_draws_frames_and_toggles(track, follow):
    env = make_env(track)
    renderer = Renderer(env, follow=follow)
    others = (make_env(track),)
    for _ in range(5):
        env.step((0.0, 1.0))
        renderer.draw(others, "hint")
    renderer.handle_key(pygame.K_r)
    renderer.handle_key(pygame.K_g)
    renderer.draw(others)
    assert renderer.show_rays is False and renderer.show_gates is True
    # The road and car must actually be painted under the car.
    x, y = (int(v) for v in renderer._to_screen(env.state.pos))
    assert renderer.screen.get_at((x, y))[:3] != (34, 104, 46)
