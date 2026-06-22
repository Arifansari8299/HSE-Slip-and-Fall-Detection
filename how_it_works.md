# How It Works — Slip & Fall Detection System

## Quick Demo (30-second pitch)

> "This system watches a live RTSP camera feed 24/7.
> Every frame, it finds people using AI, maps 17 body landmarks on each person,
> and calculates whether the body is upright or horizontal.
> The moment someone falls and stays down — it turns red, saves a photo, and logs the event.
> No cloud needed. Runs locally."

---

## What Happens Every Frame (The Pipeline)

```
RTSP Camera
    │
    ▼
[1] StreamReader         ← Grabs the frame from IP camera
    │
    ▼
[2] PoseModel (YOLOv8)   ← Detects people + maps 17 body keypoints on each
    │
    ▼
[3] FallDetector         ← Checks: is this person standing or fallen?
    │
    ├── Normal?  → Green box, label "Normal ID: X"
    │
    └── Fallen?  → Red box, label "FALL DETECTED! ID: X"
                    │
                    ├── [4] ScreenshotSaver  ← Saves annotated photo (once per event)
                    └── [5] AlertLogger      ← Appends row to alerts/logs.csv
```

---

## How Fall Detection Works (The Core Logic)

### Step 1 — Bounding Box Aspect Ratio

When a person is **standing**, their bounding box is tall and narrow → width/height < 1.0

When a person **falls**, their bounding box becomes wide and flat → width/height > threshold (0.85)

```
Standing:          Fallen:
  ┌──┐             ┌──────────────┐
  │  │  ratio=0.4  │              │  ratio=1.6
  │  │             └──────────────┘
  │  │
  └──┘
```

### Step 2 — Spine Angle (Keypoint Geometry)

YOLOv8 gives us 17 body landmarks per person (COCO format).
We use 5 of them:

```
Index  Keypoint
  0    Nose
  5    Left Shoulder
  6    Right Shoulder
 11    Left Hip
 12    Right Hip
```

We compute the **spine vector**: midpoint(shoulders) → midpoint(hips)

Then measure the **angle from vertical**:

```
Standing person:        Fallen person:
shoulder_mid            shoulder_mid ──────► hip_mid
     │                  angle from vertical ≈ 80-90°
     │  angle ≈ 5°
     ▼
  hip_mid
```

If `spine_angle > 30°` AND `aspect_ratio > 0.85` → **fall condition is TRUE**

### Step 3 — Confidence Fallback

If fewer than 4 keypoints have confidence ≥ 0.3 (person is far away or partially occluded),
the system falls back to **aspect ratio only** — no angle check.

### Step 4 — Frame Counter (Debounce)

A single frame is not enough. Someone bending over would briefly trigger aspect ratio.

The system counts **consecutive frames** where the fall condition is true.
Only after `fall_frame_threshold` frames (default: 5) does it confirm a real fall.

```
Frame 1: fall_condition=True  → counter=1  (no alert)
Frame 2: fall_condition=True  → counter=2  (no alert)
Frame 3: fall_condition=True  → counter=3  (no alert)
Frame 4: fall_condition=True  → counter=4  (no alert)
Frame 5: fall_condition=True  → counter=5  ✅ CONFIRMED FALL → alert triggered
```

If the person stands back up at any point → counter resets to 0.

### Step 5 — Suppression (One Alert Per Fall)

Once a fall is confirmed for Track ID X:
- The box stays **red** on screen for as long as they remain fallen
- But the screenshot and CSV log are written **only once** per fall episode
- When they stand up, the suppression flag resets — next fall will trigger again

---

## Screenshot Saving — What Happens & The Delay

### What gets saved

A copy of the frame at the moment of confirmed fall, annotated with:
- Red bounding box around the person
- Label: `FALL DETECTED ID: {track_id}`
- Filename: `fall_{track_id}_{YYYYMMDD_HHMMSS}.jpg`
- Saved to: `alerts/screenshots/`

### Why there's a delay

The screenshot is **not taken on the first suspicious frame**. Here's the intentional delay:

```
At 30 FPS:
  fall_frame_threshold = 5 frames
  Delay = 5 / 30 = ~0.17 seconds  ← nearly instant

At 15 FPS (typical RTSP):
  Delay = 5 / 15 = ~0.33 seconds  ← still very fast
```

This small delay is a **feature, not a bug** — it filters out:
- Someone bending to pick something up
- Camera glitch / motion blur
- Person crouching momentarily

### The saving code path

```python
# In pipeline.py run() loop:
is_fall, ratio = detector.check(bbox, keypoints, track_id)

if is_fall:
    if detector.should_save_screenshot(track_id):   # ← True only ONCE per fall
        screenshot_name = screenshot_saver.save(frame, bbox, track_id)
        alert_logger.log(track_id, ratio, screenshot_name)
```

`should_save_screenshot()` uses a per-track flag:
```python
def should_save_screenshot(self, track_id):
    if not self._screenshot_taken.get(track_id, False):
        self._screenshot_taken[track_id] = True   # ← flip flag, never fires again
        return True
    return False
    # Resets when person stands up (counter reset path)
```

---

## CSV Log Format

Every confirmed fall appends one row to `alerts/logs.csv`:

```
timestamp,track_id,aspect_ratio,screenshot_filename
2024-06-19T09:30:45,3,1.4500,fall_3_20240619_093045.jpg
2024-06-19T09:31:12,1,1.2300,fall_1_20240619_093112.jpg
```

---

## Config Tuning for Demo

File: `config/settings.yaml`

| Parameter | Default | What it controls |
|---|---|---|
| `aspect_ratio_threshold` | 0.85 | Lower = more sensitive to wide bboxes |
| `angle_threshold_degrees` | 30.0 | Lower = triggers on less tilt |
| `fall_frame_threshold` | 5 | Fewer frames = faster alert, more false positives |
| `zoom_factor` | 1.0 | 1.5 or 2.0 for digital zoom on the stream |

For a demo in a **controlled room** where you'll actually fall:
- Set `fall_frame_threshold: 3` for faster response
- Set `aspect_ratio_threshold: 0.8` for easier triggering
