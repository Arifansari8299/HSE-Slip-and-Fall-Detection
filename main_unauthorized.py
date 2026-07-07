"""Main script for Unauthorized Entry Detection in Restricted Zones."""

import sys
import os
import cv2
import time
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import load_config, StreamReader, PoseModel, UnauthorizedEntryLogger
from src.analytics import HSEOfficeAnalytics
from src.hse_agent import HSEAgent


# ─────────────────────────────────────────────────────────
#  INTERACTIVE ZONE DRAWING
# ─────────────────────────────────────────────────────────

def draw_zone_interactively(frame: np.ndarray) -> list:
    """
    Show first frame and let user click polygon points.
    Left-click  → add point
    Enter/Space → confirm zone
    Backspace   → remove last point
    R           → reset all points
    ESC         → skip, use settings.yaml zone
    """
    points = []
    win = "Draw Restricted Zone | Click points | Enter=Confirm | R=Reset | ESC=Skip"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    # Mouse callback writes directly into points list
    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))

    cv2.setMouseCallback(win, on_mouse)

    print("\n[Zone Setup] Left-click to place polygon points.")
    print("[Zone Setup] Enter/Space=Confirm  |  Backspace=Undo  |  R=Reset  |  ESC=Skip\n")

    while True:
        # Always redraw from the original frame
        display = frame.copy()

        # Draw placed points and connecting lines
        for i, pt in enumerate(points):
            cv2.circle(display, pt, 7, (0, 255, 255), -1)
            cv2.putText(display, str(i + 1), (pt[0] + 10, pt[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        if len(points) >= 2:
            for i in range(len(points) - 1):
                cv2.line(display, points[i], points[i + 1], (0, 255, 0), 2)
        if len(points) >= 3:
            # Close the polygon preview
            cv2.line(display, points[-1], points[0], (0, 200, 0), 1)
            poly = np.array(points, dtype=np.int32)
            overlay = display.copy()
            cv2.fillPoly(overlay, [poly], (0, 255, 0))
            cv2.addWeighted(overlay, 0.15, display, 0.85, 0, display)
            cv2.polylines(display, [poly], True, (0, 255, 0), 2)

        # Instructions overlay
        lines = [
            "LEFT-CLICK: Add point",
            "ENTER/SPACE: Confirm",
            "BACKSPACE: Undo last",
            "R: Reset all",
            "ESC: Use default zone",
            f"Points: {len(points)}",
        ]
        for i, txt in enumerate(lines):
            cv2.putText(display, txt, (10, 28 + i * 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(display, txt, (10, 28 + i * 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(win, display)
        key = cv2.waitKey(30) & 0xFF

        if key in (13, 32):       # Enter or Space — confirm
            if len(points) < 3:
                print("[Zone Setup] Need at least 3 points. Keep clicking.")
                continue
            break
        elif key == 8:            # Backspace — undo
            if points:
                points.pop()
        elif key in (ord('r'), ord('R')):  # Reset
            points.clear()
            print("[Zone Setup] Reset. Click to start again.")
        elif key == 27:           # ESC — skip
            print("[Zone Setup] Skipped — using zone from settings.yaml")
            points.clear()
            break

    cv2.destroyWindow(win)
    print(f"[Zone Setup] ✅ Zone confirmed: {len(points)} points → {points}")
    return points


def main():
    config_path = "config/settings.yaml"
    cfg = load_config(config_path)

    print("\n" + "="*55)
    print("🚫 INITIALIZING: UNAUTHORIZED ENTRY DETECTION")
    print(f"🎬 Video Source: {cfg['video_source']}")
    print("="*55 + "\n")

    stream = StreamReader(cfg["video_source"], cfg["stream_reconnect_retries"])
    model = PoseModel(cfg["model_weights"])
    stream.open()

    # Grab first frame for zone drawing
    first_frame = stream.read()

    # ── Interactive zone drawing ──
    print("[Zone Setup] Draw your restricted zone on the video frame.")
    print("[Zone Setup] Press ESC to skip and use the zone from settings.yaml.\n")
    drawn_zone = draw_zone_interactively(first_frame)

    # Fall back to settings.yaml if user pressed ESC or drew fewer than 3 points
    if len(drawn_zone) < 3:
        drawn_zone = cfg.get("analytics", {}).get("restricted_zone", None)
        print("[Zone Setup] Using zone from settings.yaml")

    analytics = HSEOfficeAnalytics(restricted_zone=drawn_zone)

    # Dedicated CSV logger for unauthorized events
    unauthorized_logger = UnauthorizedEntryLogger(
        cfg.get("unauthorized_log_path", "alerts/unauthorized_logs.csv")
    )

    # Multi-agent system for email + buzzer alerts
    agent = HSEAgent(
        email_cfg=cfg["email"],
        csv_log_path=cfg["csv_log_path"],
        running_log_path=cfg["running_log_path"],
    )

    # Cooldown tracker — log once per 30s per person to avoid CSV flooding
    _logged_unauthorized: dict = {}

    try:
        while True:
            frame = stream.read()
            if frame is None:
                break

            persons = model.infer(frame)

            tracks = [
                {"track_id": p.track_id, "bbox": list(p.bbox)}
                for p in persons
            ]

            alerts, frame = analytics.check_restricted_zone(tracks, frame)

            now = time.time()
            for alert in alerts:
                tid = alert["track_id"]
                last = _logged_unauthorized.get(tid, 0)

                if now - last >= 30:
                    _logged_unauthorized[tid] = now

                    person_bbox = next(
                        (p.bbox for p in persons if p.track_id == tid), None
                    )
                    screenshot_name = ""
                    if person_bbox:
                        screenshot_name = _save_screenshot(
                            frame, person_bbox, tid, cfg["screenshots_dir"]
                        )

                    unauthorized_logger.log(
                        track_id=tid,
                        zone="restricted_zone",
                        screenshot_name=screenshot_name,
                    )
                    print(f"📝 [CSV LOG] UNAUTHORIZED_ENTRY → Person ID #{tid}")
                    agent.execute_incident_protocol("UNAUTHORIZED_ENTRY", tid)

            cv2.imshow("HSE Unauthorized Entry Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        stream.release()
        cv2.destroyAllWindows()
        print("\nStream session closed cleanly.")


def _save_screenshot(frame, bbox, track_id: int, screenshots_dir: str) -> str:
    """Save annotated screenshot for unauthorized entry event."""
    import cv2 as _cv2
    from datetime import datetime
    try:
        os.makedirs(screenshots_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"unauthorized_{track_id}_{ts}.jpg"
        filepath = os.path.join(screenshots_dir, filename)
        annotated = frame.copy()
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        _cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
        _cv2.putText(
            annotated, f"UNAUTHORIZED ID: {track_id}",
            (x1, y1 - 10), _cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
        )
        if _cv2.imwrite(filepath, annotated):
            return filename
    except Exception as e:
        print(f"[ERROR] Screenshot failed: {e}")
    return ""


if __name__ == "__main__":
    main()
