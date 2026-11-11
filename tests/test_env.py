import numpy as np
import pytest

from rf.car import CarParams, CarState
from rf.env import CarEnv
from rf.geometry import ray_segment_distances, segments_intersect_any
from rf.sensors import sense
from rf.tracks.oval import make_oval


def test_ray_hits_known_wall():
    wall = np.array([[[10.0, -5.0], [10.0, 5.0]]])
    d = ray_segment_distances(np.zeros(2), np.array([0.0, np.pi]), wall, 100.0)
    assert d[0] == pytest.approx(10.0)
    assert d[1] == pytest.approx(100.0)  # behind the ray: no hit, capped


def test_segment_intersection():
    a = np.array([[[0.0, 0.0], [2.0, 2.0]]])
    assert segments_intersect_any(a, np.array([[[0.0, 2.0], [2.0, 0.0]]]))
    assert not segments_intersect_any(a, np.array([[[5.0, 0.0], [6.0, 1.0]]]))


def test_start_sensors_inside_track():
    track = make_oval()
    state = CarState(track.start_pos.copy(), track.start_heading, 0.0)
    rays = sense(state, track.walls)
    assert np.all(rays > 0.0) and np.all(rays <= 1.0)
    assert rays[0] < 1.0 and rays[-1] < 1.0  # side walls are within range


def test_full_throttle_straight_crashes():
    env = CarEnv(make_oval())
    done = False
    while not done:
        _, _, done, info = env.step((0.0, 1.0))
    assert info["crashed"]


def test_idle_car_stalls_without_crash():
    env = CarEnv(make_oval())
    done = False
    while not done:
        _, _, done, info = env.step((0.0, 0.0))
    assert not info["crashed"] and env.steps == env.config.stall_steps


def test_checkpoints_count_in_order():
    track = make_oval()
    env = CarEnv(track, CarParams())
    # Teleport along the centerline just before gate 1, then step across it.
    gate = track.checkpoints[1]
    mid = gate.mean(axis=0)
    env.state.pos = mid - np.array([0.0, 1.0]) * 0.5 + np.array([0.0, -2.0])
    env.state.heading = np.pi / 2
    env.state.speed = 100.0
    for _ in range(5):
        env.step((0.0, 0.0))
    assert env.checkpoints_passed == 1
