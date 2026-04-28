from src.cv.tracker import Tracker

def test_tracker_initialization():
    tracker = Tracker()
    assert tracker is not None
    assert len(tracker.objects) == 0

def test_tracker_update_basic():
    tracker = Tracker()
    # Mock detections: [x1, y1, x2, y2, confidence]
    detections = [
        [100, 100, 200, 200, 0.9],
        [300, 300, 400, 400, 0.8]
    ]
    
    tracks = tracker.update(detections)
    # Should have 2 tracks
    assert len(tracks) == 2
    for track in tracks:
        assert "id" in track
        assert "bbox" in track

def test_tracker_persistence():
    tracker = Tracker()
    detections_v1 = [[100, 100, 200, 200, 0.9]]
    tracks_v1 = tracker.update(detections_v1)
    id_v1 = tracks_v1[0]["id"]
    
    # Slight movement
    detections_v2 = [[105, 105, 205, 205, 0.9]]
    tracks_v2 = tracker.update(detections_v2)
    id_v2 = tracks_v2[0]["id"]
    
    # ID should persist
    assert id_v1 == id_v2
