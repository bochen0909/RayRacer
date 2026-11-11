"""Drive the car by hand: arrow keys. Run: uv run --extra viz python scripts/watch.py"""

import pygame

from rf.env import CarEnv
from rf.tracks.oval import make_oval
from rf.viz.render import Renderer


def main() -> None:
    env = CarEnv(make_oval())
    renderer = Renderer(env)
    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        keys = pygame.key.get_pressed()
        steer = float(keys[pygame.K_LEFT]) - float(keys[pygame.K_RIGHT])  # +y is up, CCW positive
        throttle = float(keys[pygame.K_UP]) - float(keys[pygame.K_DOWN])
        env.step((steer, throttle))
        renderer.draw()
        if env.done:
            pygame.time.wait(500)
            env.reset()
        clock.tick(30)
    pygame.quit()


if __name__ == "__main__":
    main()
