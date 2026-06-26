"""RunningDetector: Evaluates tracking velocity to identify running or panic behavior."""

import math
import time

class RunningDetector:
    def __init__(self, velocity_threshold: float = 0.45, consecutive_frames: int = 6):
        """
        Args:
            velocity_threshold: Normalized speed threshold. Higher means needs faster running.
            consecutive_frames: Number of continuous frames a person must run to trigger alert.
        """
        self.velocity_threshold = velocity_threshold
        self.consecutive_frames = consecutive_frames

        # Per-track states
        self._prev_positions = {}  # {track_id: (center_x, center_y, timestamp)}
        self._run_counter = {}     # {track_id: count_of_consecutive_fast_frames}

    def check(self, bbox, track_id: int) -> bool:
        """
        Evaluate if a specific Track ID is running.
        Returns: True if running behavior is confirmed, False otherwise.
        """
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        current_time = time.time()
        
        # Bounding box height for distance normalization (Handles perspective scaling)
        bbox_height = float(y2 - y1) if (y2 - y1) > 0 else 1.0

        # If it's a new ID, initialize tracking state and return False
        if track_id not in self._prev_positions:
            self._prev_positions[track_id] = (center_x, center_y, current_time)
            self._run_counter[track_id] = 0
            return False

        # Retrieve historical frame state
        prev_x, prev_y, prev_time = self._prev_positions[track_id]
        
        # Calculate pixel displacement
        dx = center_x - prev_x
        dy = center_y - prev_y
        pixel_distance = math.sqrt(dx**2 + dy**2)
        
        # Normalize distance using height (Fixes: Far away people moving fewer pixels)
        normalized_distance = pixel_distance / bbox_height
        
        dt = current_time - prev_time
        if dt <= 0:
            dt = 0.033  # Prevent division by zero (Default ~30fps frame gap)

        # Final calculated velocity scalar
        velocity = normalized_distance / dt

        # Update historical state cache for next frame
        self._prev_positions[track_id] = (center_x, center_y, current_time)

        # Check against velocity threshold
        if velocity > self.velocity_threshold:
            self._run_counter[track_id] = self._run_counter.get(track_id, 0) + 1
        else:
            # Decay counter slowly instead of instant drop to handle tracking jitters smoothly
            self._run_counter[track_id] = max(0, self._run_counter.get(track_id, 0) - 1)

        # Confirm running only if behavior persists for threshold frames
        if self._run_counter[track_id] >= self.consecutive_frames:
            return True

        return False