# Design Document: Slip and Fall Detection System

## Overview

The Slip and Fall Detection system is a Python application that continuously monitors an RTSP camera stream, detects fall events using YOLOv8n-pose skeleton keypoint analysis combined with bounding box aspect ratio heuristics, and emits alerts via CSV logging and annotated screenshots. The system is designed as a production service with a clean separation between configuration, inference, detection logic, alerting, and display.

The overall data flow is: **Config → StreamReader → PoseModel → FallDetector → (AlertLogger + ScreenshotSaver) → Display**

All runtime parameters are externalised to `config/settings.yaml` so the system can be reconfigured without touching source code.

---

## Architecture

```mermaid
flowchart TD
    A[settings.yaml] -->|load| B[Pipeline]
    B -->|init| C[StreamReader]
    B -->|init| D[PoseModel]
    B -->|init| E[FallDetector]
    B -->|init| F[AlertLogger]
    B -->|init| G[ScreenshotSaver]

    C -->|frame| D
    D -->|persons: bbox, track_id, keypoints| E
    E -->|Fall_Event| F
    E -->|Fall_Event + frame| G
    E -->|status per track_id| H[Frame Annotator]
    H -->|annotated frame| I[cv2.imshow]
```

### Processing Loop (per frame)

```
frame = StreamReader.read()
  └─> persons = PoseModel.infer(frame)
        └─> for each person:
              is_fall, ratio = FallDetector.check(bbox, keypoints, track_id)
              annotate(frame, bbox, track_id, is_fall)
              if is_fall:
                  AlertLogger.log(track_id, ratio, screenshot_name)
                  ScreenshotSaver.save(frame, bbox, track_id)
display(frame)
```

---

## Components and Interfaces

### Config (`config/settings.yaml` + loader in `src/utils.py`)

Loads and validates YAML config at startup. Exits with a descriptive error if the file is missing or a required key is absent or wrong type.

```yaml
# config/settings.yaml
video_source: "rtsp://admin:iiiot2.com@192.168.10.250:554/1"
model_weights: "weights/yolov8n-pose.pt"
aspect_ratio_threshold: 1.1
angle_threshold_degrees: 45.0
min_keypoint_confidence: 0.3
fall_frame_threshold: 10
stream_reconnect_retries: 5
csv_log_path: "alerts/logs.csv"
screenshots_dir: "alerts/screenshots"
```

Required keys (all must be present): `video_source`, `model_weights`, `aspect_ratio_threshold`, `angle_threshold_degrees`, `min_keypoint_confidence`, `fall_frame_threshold`, `stream_reconnect_retries`, `csv_log_path`, `screenshots_dir`.

### StreamReader (`src/pipeline.py` or inline in `Pipeline`)

```python
class StreamReader:
    def __init__(self, url: str, reconnect_retries: int)
    def open() -> None          # raises SystemExit on failure
    def read() -> np.ndarray    # raises on exhausted retries
    def release() -> None
```

- Sets `CAP_PROP_BUFFERSIZE = 1` when `url` starts with `rtsp://`.
- On a failed `cap.read()`, retries up to `reconnect_retries` times with a short sleep before exiting.

### PoseModel (`src/pipeline.py` or inline)

```python
class PoseModel:
    def __init__(self, weights_path: str)
    def infer(frame: np.ndarray) -> list[PersonData]
```

`PersonData` is a typed dict / dataclass:
```python
@dataclass
class PersonData:
    track_id: int
    bbox: tuple[int, int, int, int]   # x1, y1, x2, y2
    keypoints: np.ndarray             # shape (17, 3) — x, y, conf
```

Calls `model.track(frame, persist=True, verbose=False, classes=[0])`. Returns an empty list when no persons are detected.

### FallDetector (`src/detector.py`)

```python
class FallDetector:
    def __init__(self, aspect_ratio_threshold, angle_threshold_degrees,
                 min_keypoint_confidence, fall_frame_threshold)
    def check(bbox, keypoints, track_id) -> tuple[bool, float]
        # returns (is_confirmed_fall, aspect_ratio)
    def _count_confident_keypoints(keypoints, indices) -> int
    def _compute_spine_angle(keypoints) -> float | None
    def _compute_torso_angle(keypoints) -> float | None
```

Detection algorithm per person per frame:
1. Compute `aspect_ratio = width / height`.
2. Count keypoints with confidence ≥ `min_keypoint_confidence` for indices {0, 5, 6, 11, 12}.
3. If ≥ 4 of those are confident: compute spine angle (shoulder_mid → hip_mid) and torso angle (nose → hip_mid) using `utils.angle_from_vertical`.
4. Fall condition: `aspect_ratio > threshold AND spine_angle > angle_threshold`.
5. If fewer than 4 confident keypoints: fall condition = `aspect_ratio > threshold` only.
6. Increment counter on fall condition; reset to 0 on upright condition.
7. Emit confirmed fall (return `True`) when counter reaches `fall_frame_threshold`.
8. Once confirmed, suppress re-emission until counter resets.

State per track_id: `fall_counter`, `fall_confirmed` flag.

### AlertLogger (`src/pipeline.py` or `src/alert_logger.py`)

```python
class AlertLogger:
    def __init__(self, csv_path: str)
    def log(track_id: int, aspect_ratio: float, screenshot_name: str) -> None
```

- Creates parent directory and writes CSV header on first call if file does not exist.
- CSV columns: `timestamp,track_id,aspect_ratio,screenshot_filename`
- `timestamp` is ISO 8601 via `utils.format_iso8601(datetime.now())`.
- Calls `file.flush()` after every write.
- Catches and logs any `IOError` without re-raising.

### ScreenshotSaver (`src/pipeline.py` or `src/screenshot_saver.py`)

```python
class ScreenshotSaver:
    def __init__(self, screenshots_dir: str)
    def save(frame: np.ndarray, bbox, track_id: int) -> str
        # returns filename or empty string on failure
```

- Filename: `fall_{track_id}_{YYYYMMDD_HHMMSS}.jpg`
- Annotates a copy of the frame with:
  - Bounding box rectangle in red `(0, 0, 255)`
  - Label `"FALL DETECTED ID: {track_id}"` above the box
  - Text `"FALL DETECTED"` in red overlaid on the box
- Creates `screenshots_dir` with `os.makedirs(exist_ok=True)` before saving.
- Catches and logs any `cv2.imwrite` failure without re-raising.

### Pipeline (`src/pipeline.py`)

```python
class Pipeline:
    def __init__(self, config_path: str = "config/settings.yaml")
    def run() -> None
```

`run()` loop:
1. `frame = stream_reader.read()`
2. `persons = pose_model.infer(frame)`
3. For each person: `is_fall, ratio = detector.check(...)` → annotate frame → if fall: log + save screenshot.
4. `cv2.imshow("HSE Slip and Fall Detection", frame)`
5. Exit on `cv2.waitKey(1) & 0xFF == ord('q')`.
6. `finally:` stream_reader.release(); cv2.destroyAllWindows().

### utils (`src/utils.py`)

```python
def angle_from_vertical(p1: tuple, p2: tuple) -> float
    """Angle in degrees between vector p1→p2 and the vertical (downward) axis."""

def midpoint(p1: tuple, p2: tuple) -> tuple
    """Euclidean midpoint of two 2D points."""

def draw_bbox(frame, bbox, label: str, color: tuple) -> None
    """Draw rectangle + label text on frame in-place."""

def format_iso8601(dt: datetime) -> str
    """Return ISO 8601 string e.g. '2024-01-15T09:30:45'."""
```

---

## Data Models

### PersonData

| Field | Type | Description |
|---|---|---|
| `track_id` | `int` | YOLOv8 persistent tracker ID |
| `bbox` | `tuple[int,int,int,int]` | `(x1, y1, x2, y2)` pixel coordinates |
| `keypoints` | `np.ndarray` shape `(17,3)` | COCO keypoints `[x, y, confidence]` |

### COCO Keypoint Indices Used

| Index | Name | Role |
|---|---|---|
| 0 | nose | torso angle top point |
| 5 | left_shoulder | shoulder midpoint |
| 6 | right_shoulder | shoulder midpoint |
| 11 | left_hip | hip midpoint |
| 12 | right_hip | hip midpoint |

### FallDetector Internal State

```python
fall_counter: dict[int, int]          # track_id -> consecutive fall frames
fall_confirmed: dict[int, bool]       # track_id -> suppression flag
```

### CSV Log Row

```
timestamp,track_id,aspect_ratio,screenshot_filename
2024-01-15T09:30:45,3,1.45,fall_3_20240115_093045.jpg
```

---

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system — essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Config completeness

*For any* valid `settings.yaml`, loading it must produce a configuration object where all nine required fields (`video_source`, `model_weights`, `aspect_ratio_threshold`, `angle_threshold_degrees`, `min_keypoint_confidence`, `fall_frame_threshold`, `stream_reconnect_retries`, `csv_log_path`, `screenshots_dir`) are present and hold the correct Python types (str, float, int as specified).

**Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7**

---

### Property 2: StreamReader exhausts retries before exit

*For any* configured retry count N ≥ 1, when every `cap.read()` call returns `(False, None)`, the StreamReader must attempt exactly N retries before raising a SystemExit, and must not exit before all retries are exhausted.

**Validates: Requirements 2.4**

---

### Property 3: Aspect ratio correctness

*For any* bounding box `(x1, y1, x2, y2)` where `y2 > y1`, the computed aspect ratio must equal `(x2 - x1) / (y2 - y1)` exactly.

**Validates: Requirements 4.1**

---

### Property 4: Angle computation correctness and bounds

*For any* two distinct 2D points `p1` and `p2`, `angle_from_vertical(p1, p2)` must return a value in `[0.0, 180.0]` degrees. For a perfectly horizontal vector (same y, different x), the result must be 90°. For a perfectly vertical vector (same x, different y), the result must be 0°.

**Validates: Requirements 4.2, 4.3, 9.1**

---

### Property 5: Midpoint formula

*For any* two 2D points `p1 = (x1, y1)` and `p2 = (x2, y2)`, `midpoint(p1, p2)` must equal `((x1+x2)/2, (y1+y2)/2)`.

**Validates: Requirements 4.2, 4.3, 9.2**

---

### Property 6: Fall counter reaches threshold then emits confirmed fall

*For any* track_id and FallDetector configuration with threshold N, when the fall condition is true for exactly N consecutive frames, the detector must return `(False, _)` for frames 1 through N-1 and `(True, _)` at frame N.

**Validates: Requirements 4.4, 4.6**

---

### Property 7: Keypoint confidence fallback

*For any* set of keypoints where fewer than 4 of the required indices {0, 5, 6, 11, 12} have confidence ≥ `min_keypoint_confidence`, the fall determination must be based solely on aspect ratio, and the result must equal the result of a pure aspect-ratio check.

**Validates: Requirements 4.5**

---

### Property 8: Fall event suppression until reset

*For any* track_id that has a confirmed fall, calling `check()` again in the next frame with a still-fallen posture must return `(False, _)` — i.e., duplicate events are suppressed. The confirmed fall is not re-emitted until the posture returns to upright and falls again.

**Validates: Requirements 4.7**

---

### Property 9: Counter reset on upright posture

*For any* track_id that accumulates K fall frames (K < threshold), if the next frame shows upright posture (aspect_ratio < threshold AND spine_angle < angle_threshold), the internal `fall_counter[track_id]` must be reset to 0.

**Validates: Requirements 4.8**

---

### Property 10: CSV record fields and format

*For any* confirmed fall event with a given `track_id`, `aspect_ratio`, and `screenshot_name`, the row appended to the CSV must contain exactly four fields: a valid ISO 8601 timestamp, the correct `track_id`, the correct `aspect_ratio`, and the correct `screenshot_name`.

**Validates: Requirements 5.1, 5.2**

---

### Property 11: Screenshot filename pattern

*For any* `track_id` and timestamp `dt`, the filename produced by `ScreenshotSaver` must match the regex `fall_{track_id}_\d{8}_\d{6}\.jpg` and the date-time portion must correspond to `dt` formatted as `YYYYMMDD_HHMMSS`.

**Validates: Requirements 6.1, 6.2**

---

### Property 12: Annotation label correctness

*For any* `track_id` and boolean fall status, the label text and bounding box color used in frame annotation must satisfy:
- `is_fall=False` → label contains `"Normal ID: {track_id}"`, color is green `(0, 255, 0)`
- `is_fall=True` → label contains `"FALL DETECTED! ID: {track_id}"`, color is red `(0, 0, 255)`

**Validates: Requirements 7.2, 7.3**

---

### Property 13: ISO 8601 round-trip

*For any* `datetime` object `dt`, `datetime.fromisoformat(format_iso8601(dt))` must equal `dt` (to second precision).

**Validates: Requirements 5.2, 9.4**

---

### Property 14: Pipeline processes stages in order

*For any* frame processed by the pipeline, the sequence of operations must be: infer → detect → annotate → log/save (if fall) → display, and no stage may be skipped even when no falls are present.

**Validates: Requirements 8.3**

---

## Error Handling

| Failure scenario | Behaviour |
|---|---|
| `settings.yaml` missing or invalid key | `logging.error(...)` + `sys.exit(1)` |
| Stream fails to open | `logging.error(...)` + `sys.exit(1)` |
| Frame read fails | Retry up to `stream_reconnect_retries`, then `sys.exit(1)` |
| CSV write fails | `logging.error(...)`, pipeline continues |
| Screenshot save fails | `logging.error(...)`, pipeline continues |
| No persons in frame | Skip detection silently, continue loop |
| Zero-height bounding box | Guard: return `aspect_ratio = 0`, skip fall check |
| Keypoint array missing / wrong shape | Treat as 0 confident keypoints, use aspect-ratio fallback |

All non-fatal errors use Python's `logging` module at `ERROR` level. Fatal errors exit with code 1.

---

## Testing Strategy

### Unit Tests

Focus on concrete examples and error conditions:

- Config loader: valid YAML loads correctly; missing file raises SystemExit; missing required key raises SystemExit.
- `FallDetector.check`: specific bbox + keypoints produce expected `(is_fall, ratio)`.
- `AlertLogger`: header written on first call; subsequent calls append without header.
- `ScreenshotSaver`: filename matches pattern for a known datetime.
- `utils`: `midpoint`, `draw_bbox`, `format_iso8601` with specific inputs.
- Pipeline exit: `stream.release()` and `cv2.destroyAllWindows()` are called on loop exit.

### Property-Based Tests

Use **Hypothesis** (`pip install hypothesis`) for all universally quantified properties.

Each test runs a minimum of **100 iterations** (Hypothesis default is 100; use `@settings(max_examples=100)`).

Each test is tagged with a comment in the format:
`# Feature: slip-fall-detector, Property N: <property text>`

Property tests to implement:

| Property | Test strategy |
|---|---|
| P1: Config completeness | Generate valid YAML dicts, verify all fields present with correct types |
| P2: StreamReader retries | Mock `cap.read` to always fail; parameterise retry count; verify exact retry count |
| P3: Aspect ratio | Generate random bboxes with `y2 > y1`; verify formula holds |
| P4: Angle bounds | Generate random point pairs; verify result ∈ [0, 180]; verify 0° and 90° edge cases |
| P5: Midpoint formula | Generate random point pairs; verify arithmetic |
| P6: Counter threshold | Generate threshold N; simulate N falling frames; verify emission at frame N only |
| P7: Confidence fallback | Generate keypoints with <4 high-confidence; verify result equals aspect-ratio-only check |
| P8: Suppression | Confirm fall; call check again; verify `(False, _)` returned |
| P9: Counter reset | Accumulate K < N fall frames; send upright frame; verify counter == 0 |
| P10: CSV fields | Generate fall events; capture CSV row; verify all four fields correct |
| P11: Screenshot filename | Generate track_id + datetime; verify regex match |
| P12: Annotation labels | Generate track_id + bool; verify label text and color |
| P13: ISO 8601 round-trip | Generate datetime objects; verify fromisoformat(format_iso8601(dt)) == dt |
| P14: Pipeline stage order | Mock all components; process a frame; verify call order via MagicMock |

Property-based tests live in `tests/test_properties.py`. Unit tests live in `tests/test_unit.py`.
