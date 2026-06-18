# Implementation Plan: Slip and Fall Detection System

## Overview

Implement the slip and fall detection pipeline using YOLOv8n-pose, RTSP stream ingestion, keypoint angle analysis, CSV alert logging, and annotated screenshot saving.

## Tasks

- [ ] 1. Set up configuration and utilities
  - [ ] 1.1 Write `config/settings.yaml` with all required keys
    - Include video_source, model_weights, aspect_ratio_threshold, angle_threshold_degrees, min_keypoint_confidence, fall_frame_threshold, stream_reconnect_retries, csv_log_path, screenshots_dir
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7_
  - [ ] 1.2 Implement `src/utils.py` with all four utility functions
    - `angle_from_vertical(p1, p2)` — angle in degrees between p1→p2 and vertical axis
    - `midpoint(p1, p2)` — Euclidean midpoint of two 2D points
    - `draw_bbox(frame, bbox, label, color)` — draw rectangle and label on frame in-place
    - `format_iso8601(dt)` — return ISO 8601 string from datetime object
    - _Requirements: 9.1, 9.2, 9.3, 9.4_
  - [ ]* 1.3 Write property tests for utils (P4, P5, P13)
    - **Property 4: Angle computation correctness and bounds** — angle_from_vertical returns [0,180]; horizontal=90°; vertical=0°
    - **Property 5: Midpoint formula** — result equals arithmetic mean of coordinates
    - **Property 13: ISO 8601 round-trip** — fromisoformat(format_iso8601(dt)) == dt
    - **Validates: Requirements 4.2, 4.3, 9.1, 9.2, 9.4**

- [ ] 2. Implement FallDetector
  - [ ] 2.1 Rewrite `src/detector.py` with keypoint angle logic and suppression
    - Constructor takes aspect_ratio_threshold, angle_threshold_degrees, min_keypoint_confidence, fall_frame_threshold
    - `check(bbox, keypoints, track_id)` returns `(is_confirmed_fall, aspect_ratio)`
    - Spine angle via shoulder midpoint → hip midpoint; torso angle via nose → hip midpoint
    - Keypoint confidence fallback when fewer than 4 confident keypoints
    - Per-track fall_counter and fall_confirmed suppression flag
    - Reset counter on upright posture
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8_
  - [ ]* 2.2 Write property tests for FallDetector (P3, P6, P7, P8, P9)
    - **Property 3: Aspect ratio correctness** — computed ratio equals (x2-x1)/(y2-y1)
    - **Property 6: Fall counter reaches threshold then emits confirmed fall** — (False,_) for frames 1..N-1, (True,_) at frame N
    - **Property 7: Keypoint confidence fallback** — <4 confident keypoints → aspect-ratio-only result
    - **Property 8: Fall event suppression** — second call with fallen posture returns (False,_)
    - **Property 9: Counter reset on upright posture** — after K<N fall frames + upright frame, counter == 0
    - **Validates: Requirements 4.1, 4.4, 4.5, 4.6, 4.7, 4.8**

- [ ] 3. Implement Pipeline (StreamReader, PoseModel, AlertLogger, ScreenshotSaver, Pipeline)
  - [ ] 3.1 Implement `StreamReader` in `src/pipeline.py`
    - `open()` — opens cv2.VideoCapture; sets CAP_PROP_BUFFERSIZE=1 for RTSP; exits on failure
    - `read()` — returns frame; retries on failure; exits after exhausting retries
    - `release()` — releases capture
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_
  - [ ]* 3.2 Write property test for StreamReader retries (P2)
    - **Property 2: StreamReader exhausts retries before exit** — with N retries configured and all reads failing, exactly N retries attempted before SystemExit
    - **Validates: Requirements 2.4**
  - [ ] 3.3 Implement `PoseModel` in `src/pipeline.py`
    - `infer(frame)` returns list of PersonData (track_id, bbox, keypoints)
    - Calls model.track with persist=True, classes=[0]
    - Returns empty list when no persons detected
    - _Requirements: 3.1, 3.2, 3.3, 3.4_
  - [ ] 3.4 Implement `AlertLogger` in `src/pipeline.py`
    - `log(track_id, aspect_ratio, screenshot_name)` appends CSV row
    - Creates file with header on first call; flushes after every write
    - Catches IOError without re-raising
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_
  - [ ]* 3.5 Write property test for AlertLogger CSV fields (P10)
    - **Property 10: CSV record fields and format** — row contains valid ISO 8601 timestamp, correct track_id, aspect_ratio, screenshot_name
    - **Validates: Requirements 5.1, 5.2**
  - [ ] 3.6 Implement `ScreenshotSaver` in `src/pipeline.py`
    - `save(frame, bbox, track_id)` returns filename or empty string on failure
    - Filename pattern: `fall_{track_id}_{YYYYMMDD_HHMMSS}.jpg`
    - Annotates copy of frame with red bbox + "FALL DETECTED ID: {track_id}" label
    - Creates screenshots_dir if missing; catches imwrite failure
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_
  - [ ] 3.7 Write property test for ScreenshotSaver filename (P11)
    - **Property 11: Screenshot filename pattern** — filename matches `fall_{track_id}_\d{8}_\d{6}\.jpg`
    - **Validates: Requirements 6.1, 6.2**
  - [ ] 3.8 Implement `Pipeline` class in `src/pipeline.py`
    - `__init__` loads config, initializes all components
    - `run()` loop: read → infer → detect → annotate → log/save if fall → display
    - Window title "HSE Slip and Fall Detection"; exits on 'q'; finally releases resources
    - Config loader validates required keys and types; exits with error on failure
    - _Requirements: 1.8, 7.1, 7.2, 7.3, 7.4, 7.5, 8.1, 8.2, 8.3, 8.4_
  - [ ]* 3.9 Write property tests for annotation labels and pipeline stage order (P12, P14)
    - **Property 12: Annotation label correctness** — Normal→green+"Normal ID:{id}", Fall→red+"FALL DETECTED! ID:{id}"
    - **Property 14: Pipeline processes stages in order** — infer→detect→annotate→log/save→display
    - **Validates: Requirements 7.2, 7.3, 8.3**

- [ ] 4. Simplify `main.py`
  - Replace body with single `Pipeline().run()` call
  - _Requirements: 8.1, 8.2_

- [ ] 5. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 6. Write config loader validation and unit tests
  - [ ]* 6.1 Write property test for config completeness (P1)
    - **Property 1: Config completeness** — loading valid YAML produces all 9 fields with correct Python types
    - **Validates: Requirements 1.1–1.7**
  - [ ]* 6.2 Write unit tests in `tests/test_unit.py`
    - Config loader: valid YAML loads; missing file raises SystemExit; missing key raises SystemExit
    - FallDetector.check: specific bbox + keypoints produce expected (is_fall, ratio)
    - AlertLogger: header on first call; subsequent calls append without header
    - ScreenshotSaver: filename matches pattern for known datetime
    - utils: midpoint, draw_bbox, format_iso8601 with specific inputs
    - Pipeline exit: stream.release() and cv2.destroyAllWindows() called on loop exit

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests use Hypothesis with @settings(max_examples=100)
- Property tests live in `tests/test_properties.py`; unit tests in `tests/test_unit.py`
