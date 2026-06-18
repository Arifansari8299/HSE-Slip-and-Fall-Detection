"""Shared utility functions for the Slip and Fall Detection system."""

import math
import cv2
import numpy as np
from datetime import datetime


def angle_from_vertical(p1: tuple, p2: tuple) -> float:
    """Return the angle in degrees between vector p1->p2 and the downward vertical axis."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    # Vertical reference is (0, 1) — downward in image coordinates
    magnitude = math.sqrt(dx * dx + dy * dy)
    if magnitude == 0:
        return 0.0
    # dot product with (0,1) is just dy
    cos_angle = dy / magnitude
    # Clamp to [-1, 1] to guard against floating-point drift
    cos_angle = max(-1.0, min(1.0, cos_angle))
    angle = math.degrees(math.acos(cos_angle))
    return angle


def midpoint(p1: tuple, p2: tuple) -> tuple:
    """Return the Euclidean midpoint of two 2D points."""
    return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)


def draw_bbox(frame: np.ndarray, bbox: tuple, label: str, color: tuple) -> None:
    """Draw a labeled bounding box on frame in-place."""
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def format_iso8601(dt: datetime) -> str:
    """Return ISO 8601 timestamp string (to second precision) from a datetime object."""
    return dt.strftime("%Y-%m-%dT%H:%M:%S")
