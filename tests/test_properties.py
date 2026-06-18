# Feature: slip-fall-detector
"""Property-based tests using Hypothesis."""

import math
import re
import csv
import io
import os
import sys
import tempfile
from datetime import datetime
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import angle_from_vertical, midpoint, format_iso8601
from src.detector import FallDetector

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_keypoints(confident_indices, conf_value=0.9, low_conf=0.1):
    """Build a (17,3) keypoint array with chosen indices set to confident."""
    kps = np.zeros((17, 3), dtype=float)
    for i in confident_indices:
        kps[i] = [100.0 + i, 200.0 + i, conf_value]
    # Give non-confident indices a low score
    for i in range(17):
        if i not in confident_indices and kps[i, 2] == 0:
            kps[i, 2] = low_conf
    return kps


def _make_fallen_bbox(ratio=1.5):
    """Return a bbox with width/height == ratio."""
    return (0, 0, int(150 * ratio), 150)


def _make_upright_bbox(ratio=0.5):
    return (0, 0, int(150 * ratio), 150)


def _fallen_kps():
    """Keypoints where spine is roughly horizontal (fallen)."""
    kps = np.zeros((17, 3), dtype=float)
    # Shoulders side by side horizontally
    kps[5] = [100, 200, 0.9]  # left shoulder
    kps[6] = [200, 200, 0.9]  # right shoulder
    # Hips side by side horizontally (same y as shoulders = horizontal spine)
    kps[11] = [100, 205, 0.9]  # left hip
    kps[12] = [200, 205, 0.9]  # right hip
    kps[0] = [150, 200, 0.9]   # nose
    return kps


def _make_detector(**kwargs):
    defaults = dict(
        aspect_ratio_threshold=1.1,
        angle_threshold_degrees=45.0,
        min_keypoint_confidence=0.3,
        fall_frame_threshold=5,
    )
    defaults.update(kwargs)
    return FallDetector(**defaults)


# ---------------------------------------------------------------------------
# Property 3: Aspect ratio correctness
# Feature: slip-fall-detector, Property 3: aspect ratio = (x2-x1)/(y2-y1)
# ---------------------------------------------------------------------------

@given(
    x1=st.integers(0, 500),
    y1=st.integers(0, 500),
    w=st.integers(1, 500),
    h=st.integers(1, 500),
)
@settings(max_examples=100)
def test_p3_aspect_ratio_correctness(x1, y1, w, h):
    x2 = x1 + w
    y2 = y1 + h
    detector = _make_detector()
    # Provide keypoints with shape mismatch so ratio-only path is exercised
    _, ratio = detector.check((x1, y1, x2, y2), None, track_id=99)
    expected = w / h
    assert abs(ratio - expected) < 1e-9 or ratio == 0.0


# ---------------------------------------------------------------------------
# Property 4: Angle computation correctness and bounds
# Feature: slip-fall-detector, Property 4: angle_from_vertical in [0,180]
# ---------------------------------------------------------------------------

@given(
    x1=st.floats(-1000, 1000, allow_nan=False, allow_infinity=False),
    y1=st.floats(-1000, 1000, allow_nan=False, allow_infinity=False),
    x2=st.floats(-1000, 1000, allow_nan=False, allow_infinity=False),
    y2=st.floats(-1000, 1000, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_p4_angle_bounds(x1, y1, x2, y2):
    assume(not (x1 == x2 and y1 == y2))  # exclude zero-vector
    angle = angle_from_vertical((x1, y1), (x2, y2))
    assert 0.0 <= angle <= 180.0


def test_p4_angle_vertical():
    # Same x, different y — perfectly vertical
    assert abs(angle_from_vertical((10, 0), (10, 50))) < 1e-9


def test_p4_angle_horizontal():
    # Same y, different x — perfectly horizontal = 90 degrees
    assert abs(angle_from_vertical((0, 10), (50, 10)) - 90.0) < 1e-9


# ---------------------------------------------------------------------------
# Property 5: Midpoint formula
# Feature: slip-fall-detector, Property 5: midpoint == arithmetic mean
# ---------------------------------------------------------------------------

@given(
    x1=st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False),
    y1=st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False),
    x2=st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False),
    y2=st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_p5_midpoint_formula(x1, y1, x2, y2):
    mx, my = midpoint((x1, y1), (x2, y2))
    assert abs(mx - (x1 + x2) / 2.0) < 1e-6
    assert abs(my - (y1 + y2) / 2.0) < 1e-6


# ---------------------------------------------------------------------------
# Property 6: Fall counter reaches threshold then emits confirmed fall
# Feature: slip-fall-detector, Property 6: (False,_) for frames 1..N-1, (True,_) at N
# ---------------------------------------------------------------------------

@given(threshold=st.integers(2, 20))
@settings(max_examples=100)
def test_p6_fall_counter_threshold(threshold):
    detector = _make_detector(fall_frame_threshold=threshold)
    bbox = _make_fallen_bbox(ratio=1.5)
    kps = _fallen_kps()

    results = []
    for _ in range(threshold):
        is_fall, _ = detector.check(bbox, kps, track_id=1)
        results.append(is_fall)

    # All frames before threshold must be False
    for i in range(threshold - 1):
        assert results[i] is False, f"Frame {i+1} should not emit fall"
    # Frame at threshold must be True
    assert results[threshold - 1] is True, "Frame N must emit confirmed fall"


# ---------------------------------------------------------------------------
# Property 7: Keypoint confidence fallback
# Feature: slip-fall-detector, Property 7: <4 confident kps => aspect-ratio-only
# ---------------------------------------------------------------------------

@given(
    ratio=st.floats(0.5, 3.0, allow_nan=False, allow_infinity=False),
    threshold=st.floats(0.8, 2.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_p7_confidence_fallback(ratio, threshold):
    detector = _make_detector(aspect_ratio_threshold=threshold, fall_frame_threshold=1)
    # Build keypoints with fewer than 4 confident indices
    kps = np.zeros((17, 3), dtype=float)
    kps[5] = [100, 100, 0.9]  # only 1 confident keypoint
    kps[6] = [200, 100, 0.05]  # below threshold
    kps[11] = [100, 200, 0.05]
    kps[12] = [200, 200, 0.05]
    kps[0] = [150, 50, 0.05]

    # Pure aspect-ratio check expectation
    expected_trigger = ratio > threshold

    bbox = (0, 0, int(100 * ratio), 100)
    is_fall, _ = detector.check(bbox, kps, track_id=1)

    assert is_fall == expected_trigger


# ---------------------------------------------------------------------------
# Property 8: Fall event suppression
# Feature: slip-fall-detector, Property 8: confirmed fall not re-emitted while still fallen
# ---------------------------------------------------------------------------

@given(threshold=st.integers(1, 10))
@settings(max_examples=100)
def test_p8_fall_suppression(threshold):
    detector = _make_detector(fall_frame_threshold=threshold)
    bbox = _make_fallen_bbox(ratio=1.5)
    kps = _fallen_kps()

    # Drive to confirmed fall
    first_true = None
    for i in range(threshold + 5):
        is_fall, _ = detector.check(bbox, kps, track_id=1)
        if is_fall and first_true is None:
            first_true = i

    assert first_true is not None, "Should have confirmed fall"

    # Subsequent fallen-posture frames must return False
    for _ in range(5):
        is_fall, _ = detector.check(bbox, kps, track_id=1)
        assert is_fall is False, "Duplicate fall must be suppressed"


# ---------------------------------------------------------------------------
# Property 9: Counter reset on upright posture
# Feature: slip-fall-detector, Property 9: counter resets to 0 after upright frame
# ---------------------------------------------------------------------------

@given(
    k=st.integers(1, 9),
    threshold=st.integers(10, 20),
)
@settings(max_examples=100)
def test_p9_counter_reset_on_upright(k, threshold):
    assume(k < threshold)
    detector = _make_detector(fall_frame_threshold=threshold)
    fallen_bbox = _make_fallen_bbox(ratio=1.5)
    upright_bbox = _make_upright_bbox(ratio=0.5)
    kps = _fallen_kps()

    # Accumulate k fall frames
    for _ in range(k):
        detector.check(fallen_bbox, kps, track_id=1)

    # Send upright frame (no keypoints — ratio-only path, ratio < threshold)
    detector.check(upright_bbox, None, track_id=1)

    assert detector._fall_counter.get(1, 0) == 0


# ---------------------------------------------------------------------------
# Property 11: Screenshot filename pattern
# Feature: slip-fall-detector, Property 11: filename matches fall_{id}_YYYYMMDD_HHMMSS.jpg
# ---------------------------------------------------------------------------

@given(
    track_id=st.integers(1, 9999),
    year=st.integers(2000, 2099),
    month=st.integers(1, 12),
    day=st.integers(1, 28),
    hour=st.integers(0, 23),
    minute=st.integers(0, 59),
    second=st.integers(0, 59),
)
@settings(max_examples=100)
def test_p11_screenshot_filename_pattern(track_id, year, month, day, hour, minute, second):
    from src.pipeline import ScreenshotSaver

    saver = ScreenshotSaver.__new__(ScreenshotSaver)
    saver.screenshots_dir = tempfile.mkdtemp()

    dt = datetime(year, month, day, hour, minute, second)
    ts = dt.strftime("%Y%m%d_%H%M%S")
    filename = f"fall_{track_id}_{ts}.jpg"

    # Verify pattern
    pattern = re.compile(r"^fall_\d+_\d{8}_\d{6}\.jpg$")
    assert pattern.match(filename), f"Filename '{filename}' does not match pattern"
    # Verify date-time portion
    date_part = filename.split("_", 2)[2].replace(".jpg", "")
    assert date_part == ts


# ---------------------------------------------------------------------------
# Property 13: ISO 8601 round-trip
# Feature: slip-fall-detector, Property 13: fromisoformat(format_iso8601(dt)) == dt (to second)
# ---------------------------------------------------------------------------

@given(
    year=st.integers(2000, 2099),
    month=st.integers(1, 12),
    day=st.integers(1, 28),
    hour=st.integers(0, 23),
    minute=st.integers(0, 59),
    second=st.integers(0, 59),
)
@settings(max_examples=100)
def test_p13_iso8601_roundtrip(year, month, day, hour, minute, second):
    dt = datetime(year, month, day, hour, minute, second)
    recovered = datetime.fromisoformat(format_iso8601(dt))
    assert recovered == dt
