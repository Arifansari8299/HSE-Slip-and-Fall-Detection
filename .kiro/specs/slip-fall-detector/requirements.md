# Requirements Document

## Introduction

This document defines the requirements for a Slip and Fall Detection system targeting persons in office and industrial environments. The system uses a YOLOv8 pose estimation model to analyze video frames from an RTSP IP camera stream. It detects fall events by combining bounding box aspect ratio analysis with skeletal keypoint geometry, logs confirmed events to a CSV file, saves screenshot evidence, and operates continuously as a production service.

## Glossary

- **System**: The Slip and Fall Detection application as a whole.
- **Pipeline**: The main orchestration component (`pipeline.py`) that coordinates frame ingestion, inference, detection, and alerting.
- **Detector**: The `FallDetector` class responsible for evaluating pose data to determine whether a fall has occurred.
- **PoseModel**: The YOLOv8n-pose model used to perform person detection and keypoint extraction on each frame.
- **StreamReader**: The component responsible for connecting to and reading frames from the configured video source.
- **AlertLogger**: The component responsible for writing fall event records to the CSV log file.
- **ScreenshotSaver**: The component responsible for saving annotated frame images as evidence of detected fall events.
- **Config**: The `settings.yaml` configuration file that controls all runtime parameters.
- **Track_ID**: A unique integer identifier assigned by the YOLOv8 tracker to each detected person across frames.
- **Keypoints**: The 17 COCO-format body landmark coordinates (x, y, confidence) produced by the PoseModel for each detected person.
- **Fall_Event**: A confirmed slip or fall incident for a specific Track_ID, sustained for at least the configured number of consecutive frames.
- **Aspect_Ratio**: The ratio of bounding box width to height for a detected person.
- **Keypoint_Confidence**: The confidence score (0.0–1.0) for each individual keypoint predicted by the PoseModel.

---

## Requirements

### Requirement 1: Configuration Management

**User Story:** As a system administrator, I want all runtime parameters defined in a single YAML file, so that I can reconfigure the system without modifying source code.

#### Acceptance Criteria

1. THE Config SHALL define the video source URL, including support for RTSP stream addresses.
2. THE Config SHALL define the aspect ratio threshold used by the Detector.
3. THE Config SHALL define the minimum number of consecutive frames required to confirm a Fall_Event.
4. THE Config SHALL define the minimum Keypoint_Confidence value below which a keypoint is treated as undetected.
5. THE Config SHALL define the output path for the CSV alert log file.
6. THE Config SHALL define the output directory path for fall screenshots.
7. THE Config SHALL define the YOLOv8 model weights file path.
8. WHEN the Config file is missing or contains invalid values, THE System SHALL log a descriptive error message and exit with a non-zero status code.

---

### Requirement 2: Video Stream Ingestion

**User Story:** As a facility operator, I want the system to read from an RTSP camera stream, so that it can monitor the environment in real time.

#### Acceptance Criteria

1. WHEN the System starts, THE StreamReader SHALL open the video source URL specified in the Config.
2. WHEN the StreamReader successfully opens the video source, THE Pipeline SHALL begin processing frames continuously.
3. IF the video source fails to open, THEN THE System SHALL log a descriptive error message and exit with a non-zero status code.
4. IF a frame cannot be read from an open stream, THEN THE StreamReader SHALL attempt to reconnect to the stream up to the number of retries defined in the Config before exiting.
5. WHERE the video source is an RTSP URL, THE StreamReader SHALL set OpenCV buffer size to 1 to minimize latency.

---

### Requirement 3: Pose Estimation Inference

**User Story:** As a system operator, I want the system to extract skeleton keypoints for every detected person, so that fall geometry can be analyzed accurately.

#### Acceptance Criteria

1. WHEN a frame is received, THE PoseModel SHALL run inference restricted to the person class (COCO class 0).
2. WHEN the PoseModel produces results, THE Pipeline SHALL extract the bounding box coordinates, Track_ID, and Keypoints for each detected person.
3. WHILE a person is being tracked, THE PoseModel SHALL maintain the same Track_ID across consecutive frames using persistent tracking.
4. IF no persons are detected in a frame, THEN THE Pipeline SHALL skip fall analysis for that frame without error.

---

### Requirement 4: Fall Detection Logic

**User Story:** As a safety officer, I want the system to use both bounding box shape and body keypoint geometry to detect falls, so that false positives from crouching or bending are minimized.

#### Acceptance Criteria

1. THE Detector SHALL compute the Aspect_Ratio as bounding box width divided by bounding box height for each tracked person.
2. THE Detector SHALL compute the vertical angle of the spine by using the midpoint of the hip keypoints (indices 11 and 12) and the midpoint of the shoulder keypoints (indices 5 and 6).
3. THE Detector SHALL compute the vertical angle of the torso by using the nose keypoint (index 0) and the midpoint of the hip keypoints (indices 11 and 12).
4. WHEN the Aspect_Ratio exceeds the configured threshold AND the spine angle from vertical exceeds the configured angle threshold, THE Detector SHALL increment the fall frame counter for that Track_ID.
5. WHEN only Keypoints with Keypoint_Confidence above the configured minimum are available for fewer than 4 of the required keypoints, THE Detector SHALL fall back to Aspect_Ratio-only evaluation for that Track_ID in that frame.
6. WHEN the fall frame counter for a Track_ID reaches the configured time threshold, THE Detector SHALL emit a confirmed Fall_Event for that Track_ID.
7. WHEN a Fall_Event has been emitted for a Track_ID, THE Detector SHALL suppress duplicate Fall_Events for that Track_ID until the person returns to an upright posture or disappears from the frame.
8. WHEN a person's posture returns to upright (Aspect_Ratio below threshold AND spine angle below threshold), THE Detector SHALL reset the fall frame counter for that Track_ID to zero.

---

### Requirement 5: Alert Logging

**User Story:** As a safety officer, I want every confirmed fall event recorded with a timestamp and track ID, so that incidents can be reviewed and audited.

#### Acceptance Criteria

1. WHEN a Fall_Event is confirmed, THE AlertLogger SHALL append a record to the CSV log file defined in the Config.
2. THE AlertLogger SHALL write each record with the following fields: timestamp (ISO 8601 format), Track_ID, Aspect_Ratio at time of detection, and screenshot filename.
3. IF the CSV log file does not exist, THEN THE AlertLogger SHALL create the file and write a header row before the first record.
4. IF writing to the CSV log file fails, THEN THE AlertLogger SHALL log a descriptive error to the console without stopping the detection pipeline.
5. THE AlertLogger SHALL flush the CSV file after each write to ensure records are not lost on unexpected shutdown.

---

### Requirement 6: Screenshot Evidence

**User Story:** As a safety investigator, I want an annotated screenshot saved for every confirmed fall event, so that visual evidence is available for review.

#### Acceptance Criteria

1. WHEN a Fall_Event is confirmed, THE ScreenshotSaver SHALL save an annotated copy of the current frame to the screenshots directory defined in the Config.
2. THE ScreenshotSaver SHALL name each screenshot file using the pattern `fall_{Track_ID}_{timestamp}.jpg` where timestamp uses the format `YYYYMMDD_HHMMSS`.
3. THE ScreenshotSaver SHALL annotate the saved frame with the bounding box, the Track_ID label, and the text "FALL DETECTED" in a visually distinct color.
4. IF the screenshots directory does not exist, THEN THE ScreenshotSaver SHALL create it before saving the first file.
5. IF saving a screenshot fails, THEN THE ScreenshotSaver SHALL log a descriptive error to the console without stopping the detection pipeline.

---

### Requirement 7: Frame Annotation and Display

**User Story:** As a monitoring operator, I want the live video feed to show colored bounding boxes and status labels for each tracked person, so that I can visually confirm system operation.

#### Acceptance Criteria

1. WHILE the Pipeline is running, THE System SHALL render a bounding box around each detected person on the display frame.
2. WHEN a person is in a normal posture, THE System SHALL render the bounding box and label in green with the text `Normal ID: {Track_ID}`.
3. WHEN a Fall_Event is active for a Track_ID, THE System SHALL render the bounding box and label in red with the text `FALL DETECTED! ID: {Track_ID}`.
4. THE System SHALL display the annotated frame in a window titled "HSE Slip and Fall Detection".
5. WHEN the user presses the 'q' key, THE System SHALL release the video stream and close all display windows gracefully.

---

### Requirement 8: Pipeline Orchestration

**User Story:** As a developer, I want a dedicated pipeline module that wires together all components, so that `main.py` remains a minimal entry point.

#### Acceptance Criteria

1. THE Pipeline SHALL load the Config from `settings.yaml` at startup.
2. THE Pipeline SHALL initialize the PoseModel, StreamReader, Detector, AlertLogger, and ScreenshotSaver using parameters from the Config.
3. THE Pipeline SHALL process frames in a continuous loop, passing each frame through inference, detection, annotation, logging, and display in that order.
4. WHEN the pipeline loop exits for any reason, THE Pipeline SHALL release all stream and window resources before returning.

---

### Requirement 9: Utility Functions

**User Story:** As a developer, I want shared utility functions available in `utils.py`, so that geometry calculations and other helpers are not duplicated across modules.

#### Acceptance Criteria

1. THE System SHALL provide a utility function that computes the angle between two 2D vectors, returning the result in degrees.
2. THE System SHALL provide a utility function that computes the midpoint between two 2D keypoint coordinates.
3. THE System SHALL provide a utility function that draws a labeled bounding box on a frame given coordinates, a label string, and a color tuple.
4. THE System SHALL provide a utility function that formats a datetime object as an ISO 8601 timestamp string.
