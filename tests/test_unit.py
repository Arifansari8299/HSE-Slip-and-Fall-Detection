"""Unit tests for the Slip and Fall Detection system."""

import csv
import io
import os
import sys
import tempfile
from datetime import datetime
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import angle_from_vertical, midpoint, draw_bbox, format_iso8601
from src.detector import FallDetector


# ---------------------------------------------------------------------------
# utils
# ---------------------------------------------------------------------------

def test_midpoint_basic():
    assert midpoint((0, 0), (4, 4)) == (2.0, 2.0)


def test_midpoint_floats():
    mx, my = midpoint((1.0, 3.0), (3.0, 7.0))
    assert mx == 2.0 and my == 5.0


def test_format_iso8601():
    dt = datetime(2024, 1, 15, 9, 30, 45)
    assert format_iso8601(dt) == "2024-01-15T09:30:45"


def test_draw_bbox_does_not_crash():
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    draw_bbox(frame, (10, 10, 100, 100), "Test", (0, 255, 0))
    # Just verify no exception and frame is still correct shape
    assert frame.shape == (200, 200, 3)


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def test_load_config_valid(tmp_path):
    from src.pipeline import load_config
    cfg_file = tmp_path / "settings.yaml"
    cfg_file.write_text(
        "video_source: rtsp://cam\n"
        "model_weights: weights/model.pt\n"
        "aspect_ratio_threshold: 1.1\n"
        "angle_threshold_degrees: 45.0\n"
        "min_keypoint_confidence: 0.3\n"
        "fall_frame_threshold: 10\n"
        "stream_reconnect_retries: 5\n"
        "csv_log_path: alerts/logs.csv\n"
        "screenshots_dir: alerts/screenshots\n"
    )
    cfg = load_config(str(cfg_file))
    assert cfg["video_source"] == "rtsp://cam"
    assert isinstance(cfg["aspect_ratio_threshold"], float)
    assert isinstance(cfg["fall_frame_threshold"], int)


def test_load_config_missing_file():
    from src.pipeline import load_config
    with pytest.raises(SystemExit) as exc:
        load_config("/nonexistent/path/settings.yaml")
    assert exc.value.code == 1


def test_load_config_missing_key(tmp_path):
    from src.pipeline import load_config
    cfg_file = tmp_path / "settings.yaml"
    cfg_file.write_text("video_source: rtsp://cam\n")  # missing all other keys
    with pytest.raises(SystemExit) as exc:
        load_config(str(cfg_file))
    assert exc.value.code == 1


# ---------------------------------------------------------------------------
# FallDetector
# ---------------------------------------------------------------------------

def _fallen_kps():
    kps = np.zeros((17, 3), dtype=float)
    kps[5] = [100, 200, 0.9]
    kps[6] = [200, 200, 0.9]
    kps[11] = [100, 205, 0.9]
    kps[12] = [200, 205, 0.9]
    kps[0] = [150, 200, 0.9]
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


def test_detector_no_fall_upright():
    detector = _make_detector()
    bbox = (0, 0, 50, 200)  # narrow = upright, ratio=0.25
    is_fall, ratio = detector.check(bbox, None, track_id=1)
    assert is_fall is False
    assert abs(ratio - 0.25) < 1e-9


def test_detector_fall_confirmed_at_threshold():
    detector = _make_detector(fall_frame_threshold=3)
    bbox = (0, 0, 300, 100)  # ratio=3.0 > 1.1
    kps = _fallen_kps()
    results = [detector.check(bbox, kps, track_id=1)[0] for _ in range(3)]
    assert results == [False, False, True]


def test_detector_zero_height_bbox():
    detector = _make_detector()
    is_fall, ratio = detector.check((0, 100, 50, 100), None, track_id=1)
    assert is_fall is False
    assert ratio == 0.0


# ---------------------------------------------------------------------------
# AlertLogger
# ---------------------------------------------------------------------------

def test_alert_logger_creates_header(tmp_path):
    from src.pipeline import AlertLogger
    log_path = str(tmp_path / "logs.csv")
    logger = AlertLogger(log_path)
    logger.log(1, 1.45, "fall_1_20240115_093045.jpg")
    with open(log_path) as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["timestamp", "track_id", "aspect_ratio", "screenshot_filename"]
    assert rows[1][1] == "1"


def test_alert_logger_no_duplicate_header(tmp_path):
    from src.pipeline import AlertLogger
    log_path = str(tmp_path / "logs.csv")
    logger = AlertLogger(log_path)
    logger.log(1, 1.45, "fall_1_a.jpg")
    logger.log(2, 1.55, "fall_2_b.jpg")
    with open(log_path) as f:
        rows = list(csv.reader(f))
    # Only one header row
    headers = [r for r in rows if r == ["timestamp", "track_id", "aspect_ratio", "screenshot_filename"]]
    assert len(headers) == 1
    assert len(rows) == 3  # header + 2 data rows


# ---------------------------------------------------------------------------
# ScreenshotSaver
# ---------------------------------------------------------------------------

def test_screenshot_saver_filename_pattern(tmp_path):
    from src.pipeline import ScreenshotSaver
    saver = ScreenshotSaver(str(tmp_path))
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    with patch("cv2.imwrite", return_value=True):
        filename = saver.save(frame, (10, 10, 50, 50), track_id=3)

    import re
    assert re.match(r"^fall_3_\d{8}_\d{6}\.jpg$", filename)


def test_screenshot_saver_creates_dir(tmp_path):
    from src.pipeline import ScreenshotSaver
    new_dir = str(tmp_path / "new_screenshots")
    saver = ScreenshotSaver(new_dir)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    with patch("cv2.imwrite", return_value=True):
        saver.save(frame, (0, 0, 50, 50), track_id=1)
    assert os.path.isdir(new_dir)


# ---------------------------------------------------------------------------
# Pipeline resource cleanup
# ---------------------------------------------------------------------------

def test_pipeline_releases_resources_on_exit(tmp_path):
    """stream.release() and cv2.destroyAllWindows() must be called on loop exit."""
    cfg_file = tmp_path / "settings.yaml"
    cfg_file.write_text(
        "video_source: rtsp://cam\n"
        "model_weights: weights/model.pt\n"
        "aspect_ratio_threshold: 1.1\n"
        "angle_threshold_degrees: 45.0\n"
        "min_keypoint_confidence: 0.3\n"
        "fall_frame_threshold: 10\n"
        "stream_reconnect_retries: 5\n"
        "csv_log_path: alerts/logs.csv\n"
        "screenshots_dir: alerts/screenshots\n"
    )

    mock_stream = MagicMock()
    mock_model = MagicMock()
    mock_model.infer.return_value = []

    # waitKey returns ord('q') immediately
    with patch("src.pipeline.StreamReader", return_value=mock_stream), \
         patch("src.pipeline.PoseModel", return_value=mock_model), \
         patch("cv2.imshow"), \
         patch("cv2.waitKey", return_value=ord("q")), \
         patch("cv2.destroyAllWindows") as mock_destroy:

        from src.pipeline import Pipeline
        p = Pipeline(config_path=str(cfg_file))
        p.run()

    mock_stream.release.assert_called_once()
    mock_destroy.assert_called_once()
