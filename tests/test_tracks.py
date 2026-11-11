import numpy as np
import pytest

from rf.env import CarEnv
from rf.tracks import get_track, track_names


@pytest.mark.parametrize("name", track_names())
def test_track_is_valid(name):
    track = get_track(name)
    env = CarEnv(track)
    assert not env.crashed and not env.done
    assert np.all(env.rays > 0.0)
    # Gates are ordered along the lap: consecutive gate midpoints are close, and the loop closes.
    mids = track.checkpoints.mean(axis=1)
    hops = np.linalg.norm(np.roll(mids, -1, axis=0) - mids, axis=1)
    assert hops.max() < 300.0


def test_driving_along_the_centerline_passes_checkpoints():
    track = get_track("Monza")
    env = CarEnv(track)
    mids = track.checkpoints.mean(axis=1)
    for k in range(1, 6):  # teleport just before each next gate and cross it
        direction = mids[k] - mids[k - 1]
        direction /= np.linalg.norm(direction)
        env.state.pos = mids[k] - direction * 1.0
        env.state.heading = float(np.arctan2(direction[1], direction[0]))
        env.state.speed = 100.0
        env.step((0.0, 0.0))
        assert env.checkpoints_passed == k


def test_unknown_track_raises():
    with pytest.raises(ValueError):
        get_track("Nowhere")
