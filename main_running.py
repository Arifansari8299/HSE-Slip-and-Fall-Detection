"""Main script to execute the Running and Panic Behavior Detection pipeline."""

import sys
import os
import cv2

# Ensure project directories are visible to the interpreter BEFORE imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import load_config, StreamReader, PoseModel
from src.running_detector import RunningDetector
from src.utils import draw_bbox
from src.hse_agent import HSEAgent

def main():
    config_path = "config/settings.yaml"
    cfg = load_config(config_path)

    print("\n" + "="*50)
    print("🚀 INITIALIZING: RUNNING & PANIC DETECTION WITH AGENTIC AI")
    print(f"🎬 Video Source: {cfg['video_source']}")
    print("="*50 + "\n")

    # Reuse existing modular components from your core pipeline
    stream = StreamReader(cfg["video_source"], cfg["stream_reconnect_retries"])
    model = PoseModel(cfg["model_weights"])
    
    # Initialize the new running behavior detector module
    detector = RunningDetector(velocity_threshold=0.45, consecutive_frames=6)
    
    # Initialize Multi-Agent System
    agent = HSEAgent(email_cfg=cfg["email"], csv_log_path=cfg["csv_log_path"])

    # 🛠️ DISPLAY LAG OPTIMIZATION:
    # High-res stream window ko pehle se optimize kar dete hain taaki rendering delay na ho
    cv2.namedWindow("HSE Running & Panic Behavior Detection", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("HSE Running & Panic Behavior Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

    stream.open()

    try:
        while True:
            frame = stream.read()
            if frame is None:
                break
                
            # 🎯 PURE ACCURACY: Model gets the full high-resolution frame
            persons = model.infer(frame)

            for person in persons:
                # Check running state using our new module
                is_running = detector.check(person.bbox, person.track_id)

                if is_running:
                    color = (0, 0, 255)  # Sudden intense RED alert box
                    label = f"RUNNING ALERT! ID: {person.track_id}"
                    
                    # 🔥 AGENTIC AI TRIGGER (Fixed: Using correct object dot notation)
                    agent.execute_incident_protocol("RUNNING_PANIC", person.track_id)
                else:
                    color = (0, 255, 0)  # Safe normal working GREEN box
                    label = f"Walking ID: {person.track_id}"

                draw_bbox(frame, person.bbox, label, color)

            # Optional Zoom Filter integration from your updates
            if "zoom_factor" in cfg and cfg["zoom_factor"] > 1.0:
                from src.utils import zoom
                frame = zoom(frame, cfg["zoom_factor"])

            # Render frame smoothly
            cv2.imshow("HSE Running & Panic Behavior Detection", frame)
            
            # Change waitKey delay to 1ms to prevent software rendering queue buildup
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        stream.release()
        cv2.destroyAllWindows()
        print("\nStream session closed cleanly.")

if __name__ == "__main__":
    main()