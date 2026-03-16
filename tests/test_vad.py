import numpy as np
from transcribe.audio.vad import VADTracker
import os

def test_vad_tracker_initialization():
    # Verify we can initialize VADTracker Without crashing
    tracker = VADTracker()
    assert tracker is not None
    assert tracker.model is not None

def test_vad_tracker_process_silence():
    tracker = VADTracker()
    # Feed 1s of silence
    for _ in range(30): # 30 * 512 = 15360 samples
        frame = np.zeros(512, dtype=np.float32)
        res = tracker.process_frame(frame)
        assert res is None # Silence shouldn't trigger

def test_vad_tracker_process_noise():
    tracker = VADTracker()
    # Feed Noise
    for _ in range(30):
        frame = np.random.normal(0, 0.01, 512).astype(np.float32)
        res = tracker.process_frame(frame)
        # Static noise might trigger start, but usually doesn't trigger a completed segment of 0.5s unless continuous volume
        # We just assert it runs
        pass
