"""
Standalone script: No Movement of Operator Detection.
Detects when a tracked person has not moved for >= stillness_threshold_secs.

Run: python main_stillness.py
"""

import sys
import os
import cv2
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import load_config, StreamReader, PoseModel, StillnessLogger
from src.analytics import HSEOfficeAnalytics
from src.hse_agent import HSEAgent


def _save_screenshot(frame, bbox, track_id: int, elapsed: int, screenshots_dir: str) -> str:
    """Save magenta-annotated screenshot for stillness alert."""
    from datetime import datetime
    try:
        os.makedirs(screenshots_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"stillness_{track_id}_{ts}.jpg"
        filepath = os.path.join(screenshots_dir, filename)
        annotated = frame.copy()
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        magenta = (255, 0, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), magenta, 3)
        cv2.putText(annotated, f"NO MOVEMENT ID: {track_id} ({elapsed}s)",
                    (x1, y1 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.65, magenta, 2)
        cv2.putText(annotated, "WELFARE CHECK REQUIRED",
                    (x1, y2 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, magenta, 2)
        if cv2.imwrite(filepath, annotated):
            return filename
    except Exception as e:
        print(f"[ERROR] Screenshot failed: {e}")
    return ""


def main():
    config_path = "config/settings.yaml"
    cfg = load_config(config_path)

    threshold = cfg.get("stillness_threshold_secs", 120)

    print("\n" + "=" * 55)
    print("🟣 INITIALIZING: NO MOVEMENT OPERATOR DETECTION")
    print(f"🎬 Video Source: {cfg['video_source']}")
    print(f"⏱  Alert threshold: {threshold} seconds of no movement")
    print("=" * 55 + "\n")

    stream = StreamReader(cfg["video_source"], cfg["stream_reconnect_retries"])
    model = PoseModel(cfg["model_weights"])
    analytics = HSEOfficeAnalytics()

    stillness_logger = StillnessLogger(
        cfg.get("stillness_log_path", "alerts/stillness_logs.csv")
    )

    agent = HSEAgent(
        email_cfg=cfg["email"],
        csv_log_path=cfg["csv_log_path"],
        running_log_path=cfg["running_log_path"],
    )

    stream.open()

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

            alerts, frame = analytics.check_operator_stillness(
                tracks, frame, threshold_secs=threshold
            )

            for alert in alerts:
                tid = alert["track_id"]
                elapsed = alert["elapsed_secs"]
                bbox = alert["bbox"]

                screenshot_name = _save_screenshot(
                    frame, bbox, tid, elapsed, cfg["screenshots_dir"]
                )

                stillness_logger.log(
                    track_id=tid,
                    elapsed_secs=elapsed,
                    screenshot_name=screenshot_name,
                )
                print(f"📝 [CSV LOG] NO_MOVEMENT_OPERATOR → Person ID #{tid} | {elapsed}s stationary")

                agent.execute_incident_protocol("NO_MOVEMENT_OPERATOR", tid)

            cv2.imshow("HSE No Movement Operator Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        stream.release()
        cv2.destroyAllWindows()
        print("\nStream session closed cleanly.")


if __name__ == "__main__":
    main()
