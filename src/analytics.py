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
