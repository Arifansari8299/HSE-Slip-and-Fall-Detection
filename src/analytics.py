# src/analytics.py
import cv2
import numpy as np

class HSEOfficeAnalytics:
    def __init__(self, restricted_zone=None, machine_zone=None):
        """
        restricted_zone: polygon for unauthorized entry detection
        machine_zone:    polygon for machine crowding detection
        """
        if restricted_zone:
            self.restricted_zone = np.array(restricted_zone, dtype=np.int32)
        else:
            self.restricted_zone = np.array([(100, 200), (400, 200), (400, 600), (100, 600)], dtype=np.int32)

        if machine_zone:
            self.machine_zone = np.array(machine_zone, dtype=np.int32)
        else:
            self.machine_zone = np.array([(300, 250), (700, 250), (700, 650), (300, 650)], dtype=np.int32)

    def check_restricted_zone(self, tracks, frame):
        """
        tracks: Ye aapke ByteTrack ya YOLO se nikalne wale active objects hain
        Format assume kar rahe hain: [{'track_id': id, 'bbox': [x1, y1, x2, y2]}]
        """
        alerts = []

        # 1. Screen par Restricted Zone ka Red Box draw karna (Demo me visual dikhane ke liye)
        cv2.polylines(frame, [self.restricted_zone], isClosed=True, color=(0, 0, 255), thickness=2)
        # Zone par label likhna
        cv2.putText(frame, "RESTRICTED ZONE", (self.restricted_zone[0][0], self.restricted_zone[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        for track in tracks:
            track_id = track.get('track_id')
            x1, y1, x2, y2 = track.get('bbox')

            # 2. Insaan ka Base Point (Bottom-Center) nikaalna
            cx = int((x1 + x2) / 2)
            cy = int(y2)

            # 3. OpenCV mathematical test
            is_inside = cv2.pointPolygonTest(self.restricted_zone, (cx, cy), measureDist=False)

            if is_inside >= 0:
                # Inside zone — RED alert box
                color = (0, 0, 255)
                label = f"BREACH! ID: {track_id}"
                alerts.append({
                    "event": "UNAUTHORIZED_ENTRY",
                    "track_id": track_id,
                    "message": f"Breach! Person ID {track_id} entered the restricted zone area."
                })
                cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
            else:
                # Outside zone — GREEN normal box
                color = (0, 255, 0)
                label = f"Safe ID: {track_id}"

            # Always draw bbox and label for every tracked person
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        return alerts, frame

    def check_machine_crowding(self, tracks, frame, max_persons: int = 2):
        """
        Detects if 2 or more persons are inside the machine zone simultaneously.

        Args:
            tracks:      [{'track_id': id, 'bbox': [x1, y1, x2, y2]}, ...]
            frame:       current video frame (drawn on in-place)
            max_persons: alert threshold — default 2

        Returns:
            alert (dict or None): {event, count, track_ids, message} if crowding detected
            frame: annotated frame
        """
        # Draw machine zone — orange border
        cv2.polylines(frame, [self.machine_zone], isClosed=True, color=(0, 165, 255), thickness=2)
        cv2.putText(frame, "MACHINE ZONE",
                    (self.machine_zone[0][0], self.machine_zone[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

        inside_ids = []

        for track in tracks:
            track_id = track.get('track_id')
            x1, y1, x2, y2 = track.get('bbox')

            cx = int((x1 + x2) / 2)
            cy = int(y2)  # bottom-center — feet position

            is_inside = cv2.pointPolygonTest(self.machine_zone, (cx, cy), measureDist=False)

            if is_inside >= 0:
                inside_ids.append(track_id)
                color = (0, 165, 255)   # orange — inside machine zone
                label = f"MACHINE ZONE ID: {track_id}"
                cv2.circle(frame, (cx, cy), 6, (0, 165, 255), -1)
            else:
                color = (0, 255, 0)     # green — safe
                label = f"Safe ID: {track_id}"

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # Count overlay on zone
        count = len(inside_ids)
        count_color = (0, 0, 255) if count >= max_persons else (0, 165, 255)
        cv2.putText(frame, f"Persons in zone: {count}",
                    (self.machine_zone[0][0], self.machine_zone[0][1] - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, count_color, 2)

        alert = None
        if count >= max_persons:
            alert = {
                "event": "MACHINE_CROWDING",
                "count": count,
                "track_ids": inside_ids,
                "message": f"ALERT: {count} persons detected in machine zone! IDs: {inside_ids}",
            }
            # Red overlay on zone when alert
            overlay = frame.copy()
            cv2.fillPoly(overlay, [self.machine_zone], (0, 0, 255))
            cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
            cv2.polylines(frame, [self.machine_zone], True, (0, 0, 255), 3)
            cv2.putText(frame, f"⚠ CROWDING ALERT — {count} PERSONS",
                        (self.machine_zone[0][0], self.machine_zone[2][1] + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

        return alert, frame

    def check_operator_stillness(self, tracks, frame, threshold_secs: float = 120.0,
                                  drift_pixels: float = 5.0):
        """
        Detects operators who haven't moved for >= threshold_secs.

        Tracks Euclidean distance of bottom-center point per track_id across frames.
        If movement drift <= drift_pixels, accumulates elapsed time.
        Resets timer when person moves more than drift_pixels.

        Args:
            tracks:          [{'track_id': id, 'bbox': [x1,y1,x2,y2]}, ...]
            frame:           current video frame (drawn on in-place)
            threshold_secs:  seconds of stillness before alert fires
            drift_pixels:    max movement (px) to still count as stationary

        Returns:
            alerts: list of {event, track_id, elapsed_secs, message}
            frame:  annotated frame
        """
        import time as _time
        import math as _math

        if not hasattr(self, '_still_positions'):
            self._still_positions = {}   # {track_id: (cx, cy, start_time)}
            self._still_alerted = {}     # {track_id: last_alert_time}

        alerts = []
        now = _time.time()
        current_ids = set()

        for track in tracks:
            track_id = track.get('track_id')
            x1, y1, x2, y2 = track.get('bbox')
            cx = int((x1 + x2) / 2)
            cy = int(y2)   # bottom-center
            current_ids.add(track_id)

            if track_id not in self._still_positions:
                # First time seeing this ID
                self._still_positions[track_id] = (cx, cy, now)
                elapsed = 0.0
            else:
                prev_cx, prev_cy, start_time = self._still_positions[track_id]
                dist = _math.sqrt((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2)

                if dist > drift_pixels:
                    # Person moved — reset timer to current position
                    self._still_positions[track_id] = (cx, cy, now)
                    elapsed = 0.0
                else:
                    # Still stationary — keep original start_time
                    elapsed = now - start_time

            # Choose color based on elapsed time
            ratio = min(elapsed / threshold_secs, 1.0)
            if elapsed >= threshold_secs:
                color = (255, 0, 255)    # magenta — alert
                status = f"NO MOVEMENT {int(elapsed)}s!"
            elif ratio > 0.6:
                color = (0, 165, 255)    # orange — warning
                status = f"Still {int(elapsed)}s"
            else:
                color = (0, 255, 0)      # green — normal
                status = f"ID: {track_id}"

            # Draw bbox + label
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, status, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            # Progress bar under bbox
            bar_w = int((x2 - x1) * ratio)
            cv2.rectangle(frame, (int(x1), int(y2) + 4),
                          (int(x1) + bar_w, int(y2) + 10), color, -1)
            cv2.rectangle(frame, (int(x1), int(y2) + 4),
                          (int(x2), int(y2) + 10), (80, 80, 80), 1)

            # Fire alert if threshold exceeded, with 30s re-alert cooldown
            if elapsed >= threshold_secs:
                last_alerted = self._still_alerted.get(track_id, 0)
                if now - last_alerted >= 30:
                    self._still_alerted[track_id] = now
                    alerts.append({
                        "event": "NO_MOVEMENT_OPERATOR",
                        "track_id": track_id,
                        "elapsed_secs": int(elapsed),
                        "bbox": (x1, y1, x2, y2),
                        "message": f"Person ID {track_id} has not moved for {int(elapsed)}s.",
                    })

        # Clean up IDs that left the frame
        gone = set(self._still_positions.keys()) - current_ids
        for tid in gone:
            self._still_positions.pop(tid, None)
            self._still_alerted.pop(tid, None)

        return alerts, frame
