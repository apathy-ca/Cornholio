# Cornhole CV Scoring System — Implementation Plan

## Overview

A computer vision pipeline that processes cornhole game footage to detect and announce scores. Phase 1 targets a single oblique camera recording. Later phases add live detection and overhead cameras.

---

## Architecture

```
Video Input (recording or live feed)
        │
        ▼
┌─────────────────┐
│  Motion Detector │  ← watches full feed, cheap
│  (round trigger) │
└────────┬────────┘
         │  round-end event
         │  + clip (last N seconds)
         ▼
┌─────────────────────────────────────────┐
│              Scoring Engine              │
│                                         │
│  ┌─────────────────┐  ┌───────────────┐ │
│  │  Static Scorer  │  │ Clip Analyzer │ │
│  │  (end frame)    │  │ (cornhole det)│ │
│  │  HSV + homog.   │  │ blob tracking │ │
│  └────────┬────────┘  └──────┬────────┘ │
│           └────────┬─────────┘          │
└────────────────────┼────────────────────┘
                     │  score delta
                     ▼
             ┌───────────────┐
             │  Score State  │  ← cancel-out logic, running totals
             └───────┬───────┘
                     │
                     ▼
             ┌───────────────┐
             │  TTS Output   │  + optional web dashboard
             └───────────────┘
```

---

## Phase 1 — Recorded Video, Single Oblique Camera

**Goal:** Validate scoring accuracy against known footage before building live infrastructure.

### Step 1 — Calibration Tool

A one-time interactive script to define geometry from a still frame.

- Load first clean frame of the recording
- User clicks 4 corners of each board → stores homography matrices
- User clicks center of each hole → stores hole ROI radius
- User samples red and blue bag pixels → stores HSV ranges with tolerance
- Saves config to `calibration.json`

**Output:** `calibration.json` containing board homographies, hole ROIs, HSV ranges

---

### Step 2 — Motion Detector

Watches the video stream and emits round-end events.

- Frame differencing on mid-court zone (between the two boards)
- Activity detected → round in progress
- Activity ceases for N seconds (tunable, suggest 3s) → round over
- On round-end: extract clean frame + preceding clip (suggest 12 seconds)
- Emit: `{ timestamp, end_frame, clip_path }`

**Key tunable:** stillness threshold and duration — will need adjustment per lighting/wind conditions

---

### Step 3 — Static Scorer

Processes the end-of-round frame to count on-board bags.

1. Apply homography transform → flattened board-plane view
2. HSV mask for red, HSV mask for blue
3. Contour detection, filter by area (removes noise, keeps bags)
4. For each detected controid: classify as **in-hole ROI**, **on-board**, or **off-board**
5. Return: `{ red_on_board, red_in_hole, blue_on_board, blue_in_hole }`

**Known weakness:** Stacked same-color bags may read as one. Flag if contour area is ~2x expected bag size.

---

### Step 4 — Clip Analyzer (Cornhole Detection)

Processes the short video clip to detect bags entering the hole.

- Apply homography to each frame
- Track red and blue blobs frame-by-frame
- Hole ROI defined as fixed circle in transformed space
- Detection logic:
  - Blob enters hole ROI → starts candidate event
  - Blob does not re-emerge within 3 frames → confirmed cornhole
  - Blob re-emerges → rejected (rattled out or passed over)
- Returns: `{ red_cornholes, blue_cornholes }` for the clip

**Why video is needed here:** Static frame cannot distinguish "bag in hole" from "bag slid off back." Temporal trajectory is required.

**Reconciliation:** Compare clip analyzer cornhole count against static scorer in-hole count. If they disagree, flag for review.

---

### Step 5 — Score State Manager

Applies cornhole rules and maintains running totals.

```
Round score:
  red_points  = (3 × red_cornholes)  + (1 × red_on_board)
  blue_points = (3 × blue_cornholes) + (1 × blue_on_board)

Cancel-out (standard rules):
  if red_points > blue_points:  red_nets = red_points - blue_points, blue_nets = 0
  else:                         blue_nets = blue_points - red_points, red_nets = 0

Running total updated, first to 21 wins.
```

Maintains full round history log with timestamps.

---

### Step 6 — Output Layer

- **TTS:** `edge-tts` (better voice quality than pyttsx3, still offline-capable)
  - Announce net points scored and running totals each round
  - Announce game winner
- **Debug overlay:** Optional OpenCV window showing detection visualization during processing
- **Log:** JSON round history, flagged anomalies, clip paths for review

---

## Phase 2 — Live Feed

Minimal changes from Phase 1:

- Replace video file reader with OpenCV `VideoCapture` on camera index or RTSP URL
- Motion detector runs continuously
- Clip buffer maintained as rolling window in memory (no full recording needed)
- Same scoring engine unchanged

---

## Phase 3 — Overhead Cameras

Significant simplification:

- Homography transform no longer needed (already top-down)
- Occlusion problems largely eliminated
- Cornhole detection becomes trivial: blob enters hole ROI and disappears = cornhole, no clip analysis needed
- Static end-of-round frame sufficient for all scoring
- Clip analyzer retired or retained only as edge-case fallback
- Wide/oblique cam retained for round-end motion trigger only

---

## Tech Stack

| Component | Library | Notes |
|---|---|---|
| Video I/O | `opencv-python` | Handles file and live capture |
| Color detection | `opencv-python` HSV | Tuned per calibration |
| Blob tracking | `opencv-python` SimpleBlobDetector or contours | Phase 1 |
| TTS | `edge-tts` | Async, good voice quality |
| Config | `json` + `numpy` | Homography matrices, HSV ranges |
| Logging | `json` newline-delimited | Round history, anomaly flags |
| Optional dashboard | `flask` + plain JS | Live score display, not required |

**GPU note:** 4GB GPU is not needed for classical CV — this runs fine on CPU. GPU becomes relevant only if a vision model fallback is added.

---

## LLM Fallback (Optional, Not Required)

If classical CV produces anomalous results (impossible bag count delta, calibration drift, heavy shadow):

- Flag the round
- Optionally call a vision model (local moondream2 ~2GB VRAM, or cloud API) with the end-of-round frame
- Prompt: "Count red and blue bags. For each bag state: on board, in hole, or off board."
- Use result to override or confirm CV output
- Log discrepancy for calibration tuning

This is an escape hatch, not the primary path.

---

## Calibration Notes

The oblique angle introduces perspective distortion that will affect detection accuracy. Calibration quality is the biggest factor in system reliability. Recommendations:

- Calibrate from a frame with no bags on the board (clean board geometry)
- Re-calibrate if camera position shifts at all
- HSV ranges should be tuned under the actual lighting conditions of the recording
- Outdoor lighting variance (clouds, sun angle) may require wider HSV tolerances or adaptive thresholding

---

## Open Questions for Phase 1

1. Resolution and frame rate of the existing recording?
2. Fixed camera position throughout, or any movement?
3. Bag color — solid red/blue or patterned?
4. Are the boards the standard distance apart and in full frame?
5. What's the acceptable error rate — is a disputed call flag sufficient, or does every round need to be correct?
