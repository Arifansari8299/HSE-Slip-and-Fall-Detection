"""
Standalone script: Two or More Persons on Machine Detection.
Mirrors the architecture of main_unauthorized.py exactly.

Run: python more_than_two_person.py
"""

import sys
import os
import cv2
import time
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import load_config, StreamReader, PoseModel, MachineLogger
from src.analytics import HSEOfficeAnalytics
from src.hse_agent import HSEAgent


# ─────────────────────────────────────────────────────────
#  INTERACTIVE ZONE DRAWING  (same pattern as main_unauthorized.py)
# ─────────────────────────────────────────────────────────

def draw_zone_interactively(frame: np.ndarray, zone_name: str = "Machine Zone") -> list:
    """
    Interactive polygon drawing on first frame.
    Left-click → add point | Enter/Space → confirm | Backspace → undo | R → reset | ESC → skip
    """
    points = []
    win = f"Draw {zone_name} | Left-click=Add | Enter=Confirm | R=Reset | ESC=Use default"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))

    cv2.setMouseCallback(win, on_mouse)

    print(f"\n[Zone Setup] Draw the {zone_name} polygon on the frame.")
    print("[Zone Setup] Left-click to place points | Enter=Confirm | Backspace=Undo | R=Reset | ESC=Skip\n")

    while True:
        display = frame.copy()

        for i, pt in enumerate(points):
            cv2.circle(display, pt, 7, (0, 165, 255), -1)
            cv2.putText(display, str(i + 1), (pt[0] + 10, pt[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)
        if len(points) >= 2:
            for i in range(len(points) - 1):
                cv2.line(display, points[i], points[i + 1], (0, 200, 100), 2)
        if len(points) >= 3:
            cv2.line(display, points[-1], points[0], (0, 200, 100), 1)
            poly = np.array(points, dtype=np.int32)
            overlay = display.copy()
            cv2.fillPoly(overlay, [poly], (0, 165, 255))
            cv2.addWeighted(overlay, 0.15, display, 0.85, 0, display)
            cv2.polylines(display, [poly], True, (0, 165, 255), 2)

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

        if key in (13, 32):
            if len(points) < 3:
                print("[Zone Setup] Need at least 3 points. Keep clicking.")
                continue
            break
        elif key == 8:
            if points:
                points.pop()
        elif key in (ord('r'), ord('R')):
            points.clear()
            print("[Zone Setup] Reset.")
        elif key == 27:
            print("[Zone Setup] Skipped — using machine_zone from settings.yaml")
            points.clear()
            break

    cv2.destroyWindow(win)
    print(f"[Zone Setup] ✅ {zone_name} confirmed: {len(points)} points → {points}")
    return points


# ─────────────────────────────────────────────────────────
#  SCREENSHOT HELPER
# ─────────────────────────────────────────────────────────

def _save_screenshot(frame: np.ndarray, count: int,
                     track_ids: list, screenshots_dir: str) -> str:
    from datetime import datetime
    try:
        os.makedirs(screenshots_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"machine_crowding_{count}persons_{ts}.jpg"
        filepath = os.path.join(screenshots_dir, filename)
        annotated = frame.copy()
        cv2.putText(
            annotated,
            f"MACHINE CROWDING: {count} PERSONS  IDs: {track_ids}",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
        )
        if cv2.imwrite(filepath, annotated):
            return filename
    except Exception as e:
        print(f"[ERROR] Screenshot failed: {e}")
    return ""


# ─────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────

def main():
    config_path = "config/settings.yaml"
    cfg = load_config(config_path)

    print("\n" + "=" * 55)
    print("⚙️  INITIALIZING: MACHINE CROWDING DETECTION")
    print(f"🎬 Video Source: {cfg['video_source']}")
    print("    Alert threshold: 2 or more persons in machine zone")
    print("=" * 55 + "\n")

    stream = StreamReader(cfg["video_source"], cfg["stream_reconnect_retries"])
    model = PoseModel(cfg["model_weights"])
    stream.open()

    # Grab first frame for interactive zone drawing
    first_frame = stream.read()

    print("[Zone Setup] Draw your machine zone polygon on the video frame.")
    print("[Zone Setup] Press ESC to skip and use the zone from settings.yaml.\n")
    drawn_zone = draw_zone_interactively(first_frame, zone_name="Machine Zone")

    # Fall back to settings.yaml if user pressed ESC
    if len(drawn_zone) < 3:
        drawn_zone = cfg.get("analytics", {}).get("machine_zone", None)
        print("[Zone Setup] Using machine_zone from settings.yaml")

    analytics = HSEOfficeAnalytics(machine_zone=drawn_zone)

    # Dedicated CSV logger
    machine_logger = MachineLogger(
        cfg.get("machine_log_path", "alerts/machine_logs.csv")
    )

    # Multi-agent system
    agent = HSEAgent(
        email_cfg=cfg["email"],
        csv_log_path=cfg["csv_log_path"],
        running_log_path=cfg["running_log_path"],
    )

    # Cooldown — log once per 30s to avoid CSV flooding
    _last_alert_time = 0

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

            alert, frame = analytics.check_machine_crowding(tracks, frame, max_persons=2)

            now = time.time()
            if alert and (now - _last_alert_time >= 30):
                _last_alert_time = now

                count = alert["count"]
                track_ids = alert["track_ids"]

                # Save screenshot
                screenshot_name = _save_screenshot(
                    frame, count, track_ids, cfg["screenshots_dir"]
                )

                # Log to CSV
                machine_logger.log(
                    person_count=count,
                    track_ids=track_ids,
                    zone="machine_zone",
                    screenshot_name=screenshot_name,
                )
                print(f"📝 [CSV LOG] MACHINE_CROWDING → {count} persons | IDs: {track_ids}")

                # Trigger agentic AI — pass count as track_id so agent knows severity
                agent.execute_incident_protocol("MACHINE_CROWDING", count)

            cv2.imshow("HSE Machine Crowding Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        stream.release()
        cv2.destroyAllWindows()
        print("\nStream session closed cleanly.")


if __name__ == "__main__":
    main()
