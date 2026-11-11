"""Track registry: `get_track("oval")`, `get_track("Monza")`, ..."""

from __future__ import annotations

from rf.track import Track
from rf.tracks.csv_tracks import available, load_csv_track
from rf.tracks.oval import make_oval


def track_names() -> list[str]:
    return ["oval", *available()]


def get_track(name: str = "oval") -> Track:
    if name == "oval":
        return make_oval()
    return load_csv_track(name)
