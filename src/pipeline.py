"""Pipeline: StreamReader, PoseModel, AlertLogger, ScreenshotSaver, Pipeline."""

import csv
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime

import cv2
import numpy as np
import yaml

from src.utils import draw_bbox, format_iso8601
from src.hse_agent import HSEAgent  # 🔥 Imported the HSE Agentic AI Module

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_REQUIRED_KEYS = {
    "video_source": str,
    "model_weights": str,
    "aspect_ratio_threshold": float,
    "angle_threshold_degrees": float,
    "min_keypoint_confidence": float,
    "fall_frame_threshold": int,
    "stream_reconnect_retries": int,
    "csv_log_path": str,
    "screenshots_dir": str,
}


def load_config(config_path: str) -> dict:
    """Load and validate settings.yaml. Exits with code 1 on any error."""
    if not os.path.exists(config_path):
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)
    try:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
    except yaml.YAMLError as e:
        logger.error("Failed to parse config file: %s", e)
        sys.exit(1)
    if cfg is None:
        logger.error("Config file is empty: %s", config_path)
        sys.exit(1)
    for key, expected_type in _REQUIRED_KEYS.items():
        if key not in cfg:
            logger.error("Missing required config key: '%s'", key)
            sys.exit(1)
        if expected_type is float and isinstance(cfg[key], int):
            cfg[key] = float(cfg[key])
        if not isinstance(cfg[key], expected_type):
            logger.error(
                "Config key '%s' must be %s, got %s",
                key, expected_type.__name__, type(cfg[key]).__name__,
            )
            sys.exit(1)
    return cfg


@dataclass
class PersonData:
    track_id: int
    bbox: tuple
    keypoints: np.ndarray


class StreamReader:
    def __init__(self, url: str, reconnect_retries: int):
        self.url = url
        self.reconnect_retries = reconnect_retries
        self._cap = None

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self.url)
        if self.url.startswith("rtsp://"):
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self._cap.isOpened():
            logger.error("Failed to open video source: %s", self.url)
            sys.exit(1)
        logger.info("Video source opened: %s", self.url)

    def read(self) -> np.ndarray:
        for attempt in range(self.reconnect_retries):
            ret, frame = self._cap.read()
            if ret and frame is not None:
                return frame
            logger.warning(
                "Frame read failed (attempt %d/%d), retrying...",
                attempt + 1, self.reconnect_retries,
            )
            time.sleep(0.5)
            self._cap.release()
            self._cap = cv2.VideoCapture(self.url)
            if self.url.startswith("rtsp://"):
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        logger.error("Exhausted %d retries reading from stream.", self.reconnect_retries)
        sys.exit(1)

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()


class PoseModel:
    def __init__(self, weights_path: str):
        from ultralytics import YOLO
        self._model = YOLO(weights_path)

    def infer(self, frame: np.ndarray) -> list:
        results = self._model.track(frame, persist=True, verbose=False, classes=[0], conf=0.35)
        persons = []
        if not results:
            return persons
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return persons
        boxes = result.boxes.xyxy.cpu().numpy()
        ids = (
            result.boxes.id.cpu().numpy().astype(int)
            if result.boxes.id is not None
            else list(range(len(boxes)))
        )
        keypoints_all = (
            result.keypoints.data.cpu().numpy()
            if result.keypoints is not None
            else [None] * len(boxes)
        )
        for bbox, track_id, kps in zip(boxes, ids, keypoints_all):
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            persons.append(PersonData(
                track_id=int(track_id),
                bbox=(x1, y1, x2, y2),
                keypoints=kps if (kps is not None and hasattr(kps, "shape") and kps.shape == (17, 3)) else None,
            ))
        return persons


class AlertLogger:
    _HEADER = ["timestamp", "track_id", "aspect_ratio", "screenshot_filename"]

    def __init__(self, csv_path: str):
        self.csv_path = csv_path

    def log(self, track_id: int, aspect_ratio: float, screenshot_name: str) -> None:
        try:
            parent = os.path.dirname(self.csv_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            file_exists = os.path.isfile(self.csv_path)
            with open(self.csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(self._HEADER)
                writer.writerow([
                    format_iso8601(datetime.now()),
                    track_id,
                    round(aspect_ratio, 4),
                    screenshot_name,
                ])
                f.flush()
        except IOError as e:
            logger.error("Failed to write CSV log: %s", e)


class ScreenshotSaver:
    def __init__(self, screenshots_dir: str):
        self.screenshots_dir = screenshots_dir

    def save(self, frame: np.ndarray, bbox, track_id: int) -> str:
        try:
            os.makedirs(self.screenshots_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"fall_{track_id}_{ts}.jpg"
            filepath = os.path.join(self.screenshots_dir, filename)
            annotated = frame.copy()
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            red = (0, 0, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), red, 2)
            cv2.putText(
                annotated, f"FALL DETECTED ID: {track_id}",
                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, red, 2,
            )
            cv2.putText(
                annotated, "FALL DETECTED",
                (x1 + 4, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, red, 2,
            )
            if not cv2.imwrite(filepath, annotated):
                logger.error("cv2.imwrite failed for: %s", filepath)
                return ""
            return filename
        except Exception as e:
            logger.error("Screenshot save failed: %s", e)
            return ""


class Pipeline:
    def __init__(self, config_path: str = "config/settings.yaml"):
        cfg = load_config(config_path)
        from src.detector import FallDetector

        self._stream = StreamReader(cfg["video_source"], cfg["stream_reconnect_retries"])
        self._model = PoseModel(cfg["model_weights"])
        self._detector = FallDetector(
            aspect_ratio_threshold=cfg["aspect_ratio_threshold"],
            angle_threshold_degrees=cfg["angle_threshold_degrees"],
            min_keypoint_confidence=cfg["min_keypoint_confidence"],
            fall_frame_threshold=cfg["fall_frame_threshold"],
        )
        self._alert_logger = AlertLogger(cfg["csv_log_path"])
        self._screenshot_saver = ScreenshotSaver(cfg["screenshots_dir"])
        
        # 🔥 Initialize the Multi-Agent System
        self._agent = HSEAgent(email_cfg=cfg["email"], csv_log_path=cfg["csv_log_path"])

    def run(self) -> None:
        self._stream.open()
        try:
            while True:
                frame = self._stream.read()
                persons = self._model.infer(frame)

                for person in persons:
                    is_fall, ratio = self._detector.check(
                        person.bbox, person.keypoints, person.track_id
                    )
                    
                    if is_fall:
                        color = (0, 0, 255)
                        label = f"FALL DETECTED! ID: {person.track_id}"
                        
                        # Trigger system logging and snapshot ONLY once per track session
                        if self._detector.should_save_screenshot(person.track_id):
                            screenshot_name = self._screenshot_saver.save(
                                frame, person.bbox, person.track_id
                            )
                            self._alert_logger.log(person.track_id, ratio, screenshot_name)
                            
                        # 🔥 AGENTIC AI FOR SLIP & FALL TRIGGER
                        # Isko hum frame verification loop me continuous pass karenge,
                        # hse_agent automatic cooldown check karke tool trigger karega.
                        self._agent.execute_incident_protocol("SLIP_FALL", person.track_id)
                    else:
                        color = (0, 255, 0)
                        label = f"Normal ID: {person.track_id}"

                    draw_bbox(frame, person.bbox, label, color)

                cv2.imshow("HSE Slip and Fall Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            self._stream.release()
            cv2.destroyAllWindows()