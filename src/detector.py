"""FallDetector: evaluates pose data to determine whether a fall has occurred."""

import numpy as np
from src.utils import angle_from_vertical, midpoint

# COCO keypoint indices used for fall detection
_NOSE = 0
_L_SHOULDER = 5
_R_SHOULDER = 6
_L_HIP = 11
_R_HIP = 12
_REQUIRED_INDICES = [_NOSE, _L_SHOULDER, _R_SHOULDER, _L_HIP, _R_HIP]


class FallDetector:
    def __init__(
        self,
        aspect_ratio_threshold: float,
        angle_threshold_degrees: float,
        min_keypoint_confidence: float,
        fall_frame_threshold: int,
    ):
        self.aspect_ratio_threshold = aspect_ratio_threshold
        self.angle_threshold_degrees = angle_threshold_degrees
        self.min_keypoint_confidence = min_keypoint_confidence
        self.fall_frame_threshold = fall_frame_threshold

        # Per-track state
        self._fall_counter: dict[int, int] = {}
        self._fall_confirmed: dict[int, bool] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, bbox, keypoints, track_id: int) -> tuple[bool, float]:
        """Return (is_confirmed_fall, aspect_ratio) for one person in one frame."""
        
        # ------------------------------------------------------------------
        #  CHAIR ELIMINATION FILTER: 
        # Skip the object if it lacks valid human keypoints to filter out empty chairs.
        # ------------------------------------------------------------------
        if keypoints is None or self._count_confident_keypoints(keypoints, _REQUIRED_INDICES) < 3:
            return False, 0.0
        # ------------------------------------------------------------------

        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1

        if height <= 0:
            return False, 0.0

        aspect_ratio = width / float(height)

        # Determine fall condition this frame
        fall_condition = self._evaluate_fall_condition(aspect_ratio, keypoints)

        if fall_condition:
            self._fall_counter[track_id] = self._fall_counter.get(track_id, 0) + 1
        else:
            # Upright posture — reset counter and suppression flag
            self._fall_counter[track_id] = 0
            self._fall_confirmed[track_id] = False

        counter = self._fall_counter.get(track_id, 0)

        # ------------------------------------------------------------------
        # 🔄 VISUAL ALERT CONTINUOUS LOGIC:
        # ------------------------------------------------------------------
        if counter >= self.fall_frame_threshold:
            # First time fall occurs in this episode -> Trigger screenshot & CSV logging
            if not self._fall_confirmed.get(track_id, False):
                self._fall_confirmed[track_id] = True
                return True, aspect_ratio
            
            # If person remains in fall position -> Maintain the visual RED alert 
            # while suppressing duplicate screenshot/logging triggers in pipeline
            return True, aspect_ratio

        return False, aspect_ratio

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _evaluate_fall_condition(self, aspect_ratio: float, keypoints) -> bool:
        """Return True if this frame's data indicates a fallen posture."""
        ratio_trigger = aspect_ratio > self.aspect_ratio_threshold

        if keypoints is None or not hasattr(keypoints, "shape") or keypoints.shape != (17, 3):
            # No usable keypoints — fall back to aspect ratio only
            return ratio_trigger

        confident_count = self._count_confident_keypoints(keypoints, _REQUIRED_INDICES)

        if confident_count < 4:
            # Keypoint fallback: use aspect ratio only (Req 4.5)
            return ratio_trigger

        spine_angle = self._compute_spine_angle(keypoints)
        if spine_angle is None:
            return ratio_trigger

        return ratio_trigger and (spine_angle > self.angle_threshold_degrees)

    def _count_confident_keypoints(self, keypoints: np.ndarray, indices: list) -> int:
        """Count how many of the given keypoint indices have confidence >= threshold."""
        count = 0
        for idx in indices:
            if keypoints[idx, 2] >= self.min_keypoint_confidence:
                count += 1
        return count

    def _compute_spine_angle(self, keypoints: np.ndarray) -> float | None:
        """Angle of spine vector (shoulder_mid -> hip_mid) from vertical."""
        shoulder_mid = midpoint(
            (keypoints[_L_SHOULDER, 0], keypoints[_L_SHOULDER, 1]),
            (keypoints[_R_SHOULDER, 0], keypoints[_R_SHOULDER, 1]),
        )
        hip_mid = midpoint(
            (keypoints[_L_HIP, 0], keypoints[_L_HIP, 1]),
            (keypoints[_R_HIP, 0], keypoints[_R_HIP, 1]),
        )
        # Guard against zero-length vector
        if shoulder_mid == hip_mid:
            return None
        return angle_from_vertical(shoulder_mid, hip_mid)

    def _compute_torso_angle(self, keypoints: np.ndarray) -> float | None:
        """Angle of torso vector (nose -> hip_mid) from vertical."""
        nose = (keypoints[_NOSE, 0], keypoints[_NOSE, 1])
        hip_mid = midpoint(
            (keypoints[_L_HIP, 0], keypoints[_L_HIP, 1]),
            (keypoints[_R_HIP, 0], keypoints[_R_HIP, 1]),
        )
        if nose == hip_mid:
            return None
        return angle_from_vertical(nose, hip_mid)