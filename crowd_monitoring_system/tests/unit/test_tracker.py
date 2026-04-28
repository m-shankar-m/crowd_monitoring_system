from src.cv.tracker import Tracker
import pytest

def test_tracker_initialization():
    tracker = Tracker()
    assert tracker.next_id == 0
    assert tracker.objects == {}

def test_tracker_first_update():
    tracker = Tracker()
    boxes = [[0, 0, 10, 10], [20, 20, 30, 30]]
    tracks = tracker.update(boxes)
    assert len(tracks) == 2
    assert tracks[0]["id"] == 0
    assert tracks[1]["id"] == 1
    assert tracker.next_id == 2

def test_tracker_persistence():
    tracker = Tracker()
    # First frame
    tracker.update([[0, 0, 10, 10]])
    # Second frame, object moved slightly
    tracks = tracker.update([[2, 2, 12, 12]])
    assert len(tracks) == 1
    assert tracks[0]["id"] == 0 # Same ID

def test_tracker_new_object():
    tracker = Tracker()
    tracker.update([[0, 0, 10, 10]])
    # Second frame, original object stayed, new one added
    tracks = tracker.update([[0, 0, 10, 10], [500, 500, 510, 510]])
    assert len(tracks) == 2
    ids = [t["id"] for t in tracks]
    assert 0 in ids
    assert 1 in ids

def test_tracker_object_lost():
    tracker = Tracker()
    tracker.update([[0, 0, 10, 10]])
    # Second frame, object disappeared
    tracks = tracker.update([])
    assert len(tracks) == 0
    assert tracker.objects == {}
