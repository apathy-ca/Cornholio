#!/usr/bin/env python3
"""
Automatic calibration: finds board corners, holes, and bag colors without user input.

Strategy:
  Board corners — compute a temporal median frame from ~80 samples across the video.
  Bags and people disappear; both boards appear clean. Threshold at 155 (board ~180-230,
  floor ~100-150), morphological cleanup, largest contour, minAreaRect for 4 exact corners.

  Holes — After rectifying each board, find the largest dark circle using HoughCircles.

  Bag colors — In a frame known to have bags, apply the rectified homography, then
  mask for red and blue pixels to calibrate HSV ranges.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np


# ── approximate board search zones in full-res (2688×1512) ──────────────────
# Near board: x=1000-1520, y=1060-1340 (confirmed from median image inspection)
# Far board:  x=1820-2280, y=530-780   (confirmed from median image inspection)
NEAR_ZONE = (900, 970, 1620, 1400)   # x1,y1,x2,y2
FAR_ZONE  = (1820, 530, 2280, 780)

# Standard board dimensions (inches → rectified pixel coords)
BOARD_W_IN, BOARD_H_IN = 24.0, 48.0
HOLE_X_IN, HOLE_Y_IN, HOLE_R_IN = 12.0, 9.0, 3.0
PX_PER_IN = 10
RECT_W, RECT_H = int(BOARD_W_IN * PX_PER_IN), int(BOARD_H_IN * PX_PER_IN)
RECT_HOLE = (int(HOLE_X_IN * PX_PER_IN), int(HOLE_Y_IN * PX_PER_IN))
RECT_HOLE_R = int(HOLE_R_IN * PX_PER_IN)


def _compute_temporal_median(
    video_path: str,
    n_samples: int = 80,
    end_sec: float = 1800.0,
) -> np.ndarray:
    """
    Sample n_samples frames evenly across the first end_sec seconds and return
    their per-pixel median. Bags, people, and transient objects disappear;
    both boards appear as clean empty surfaces.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end_frame = min(total, int(end_sec * fps))

    indices = np.linspace(0, end_frame - 1, n_samples, dtype=int)
    frames = []
    for fi in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ret, frame = cap.read()
        if ret:
            frames.append(frame.astype(np.float32))
    cap.release()

    if not frames:
        raise RuntimeError("Could not read any frames for median computation")

    median = np.median(np.stack(frames), axis=0).astype(np.uint8)
    print(f"  Temporal median: {len(frames)} samples, end_frame={end_frame}")
    return median


def _find_board_corners(frame: np.ndarray, zone: tuple, label: str) -> np.ndarray | None:
    """
    Find the 4 corners of a cornhole board within zone (x1,y1,x2,y2).

    Strategy: threshold for the bright white playing surface (high V, low S in HSV),
    find the largest white blob, fit a rotated rectangle, then enforce 2:1 aspect ratio
    to account for the dark design pattern that truncates the detected white area.

    First tries convex-hull quad-fit (better for perspectively distorted boards),
    falls back to minAreaRect + 2:1 correction.

    Returns (4,2) float32 array in image coordinates, or None on failure.
    """
    x1, y1, x2, y2 = zone
    crop = frame[y1:y2, x1:x2]
    cw, ch = x2 - x1, y2 - y1

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # Board playing surface: bright white/cream, low saturation
    # S<60 keeps floor reflections out (floor S≈60-80, board S<55)
    white_mask = cv2.inRange(hsv, (0, 0, 155), (180, 60, 255))

    # Morphological cleanup
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, k, iterations=3)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, k, iterations=2)

    # Save debug crops so we can verify what is being detected
    cv2.imwrite(f"debug_crop_{label}.jpg", crop)
    cv2.imwrite(f"debug_mask_{label}.jpg", white_mask)

    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"  [{label}] No white-surface contours found")
        return None

    # Pick the largest contour that's not too small and not spanning the full zone
    zone_area = cw * ch
    valid = [(cv2.contourArea(c), c) for c in contours
             if zone_area * 0.03 < cv2.contourArea(c) < zone_area * 0.90]
    if not valid:
        print(f"  [{label}] No valid-sized white contour found")
        return None

    area, best_cnt = max(valid)

    # Try to fit a proper quadrilateral via convex hull + approxPolyDP.
    # This handles perspective distortion (board appears as trapezoid, not rectangle).
    hull = cv2.convexHull(best_cnt)
    peri = cv2.arcLength(hull, True)
    box = None
    for eps_frac in (0.04, 0.06, 0.08, 0.10):
        poly = cv2.approxPolyDP(hull, eps_frac * peri, True)
        if len(poly) == 4:
            box = poly.reshape(4, 2).astype(np.float32)
            print(f"  [{label}] Quad fit at eps={eps_frac:.2f}: {box.astype(int).tolist()}")
            break

    if box is None:
        # Fallback: minAreaRect + 2:1 aspect correction
        mar = cv2.minAreaRect(best_cnt)
        w_det, h_det = mar[1]
        if min(w_det, h_det) < 10:
            return None
        detected_aspect = max(w_det, h_det) / min(w_det, h_det)
        short = min(w_det, h_det)
        long_corrected = short * 2.0
        if w_det >= h_det:
            corrected_rect = (mar[0], (long_corrected, h_det), mar[2])
        else:
            corrected_rect = (mar[0], (w_det, long_corrected), mar[2])
        box = cv2.boxPoints(corrected_rect).astype(np.float32)
        print(f"  [{label}] minAreaRect fallback: {w_det:.0f}×{h_det:.0f} "
              f"(aspect {detected_aspect:.2f}) → corrected to 2:1")

    # Save debug overlay: contour + detected corners on the crop
    debug_crop = crop.copy()
    cv2.drawContours(debug_crop, [best_cnt], -1, (0, 255, 0), 2)
    for pt in box.astype(int):
        cv2.circle(debug_crop, tuple(pt), 6, (0, 0, 255), -1)
    cv2.imwrite(f"debug_corners_{label}.jpg", debug_crop)

    # Translate to full-image coords
    box[:, 0] += x1
    box[:, 1] += y1

    pts = _sort_corners(box)
    print(f"  [{label}] Found corners: {pts.astype(int).tolist()}")
    return pts


def _sort_corners(pts: np.ndarray) -> np.ndarray:
    """Sort corners: top-left, top-right, bottom-right, bottom-left."""
    s = pts.sum(axis=1)
    d = pts[:, 0] - pts[:, 1]
    return np.array([
        pts[s.argmin()],   # top-left  (small x+y)
        pts[d.argmax()],   # top-right (large x-y)
        pts[s.argmax()],   # bottom-right (large x+y)
        pts[d.argmin()],   # bottom-left (small x-y)
    ], dtype=np.float32)


def _find_hole(rect_img: np.ndarray, label: str) -> tuple[int, int, int]:
    """
    Find the hole in a rectified board image.
    Returns (cx, cy, radius). Falls back to nominal if HoughCircles fails.
    """
    gray = cv2.cvtColor(rect_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=30,
        param1=50,
        param2=20,
        minRadius=15,
        maxRadius=50,
    )

    if circles is None:
        print(f"  [{label}] HoughCircles found no circles; using nominal position")
        return RECT_HOLE[0], RECT_HOLE[1], RECT_HOLE_R

    circles = np.round(circles[0]).astype(int)
    # Pick the darkest circle center — hole should be the darkest region
    best = min(circles, key=lambda c: int(gray[c[1], c[0]]))
    cx, cy, r = int(best[0]), int(best[1]), int(best[2])
    print(f"  [{label}] Hole at ({cx},{cy}) r={r}")
    return cx, cy, r


def _find_bag_colors(
    color_frame: np.ndarray,
    H_near: np.ndarray,
    H_far: np.ndarray,
) -> tuple[list, list, list, list]:
    """
    Find HSV ranges for red and blue bags by analyzing the rectified boards.
    Returns (red_lo, red_hi, blue_lo, blue_hi) as [H,S,V] lists.
    """
    red_pixels = []
    blue_pixels = []

    for H in [H_near, H_far]:
        rect = cv2.warpPerspective(color_frame, H, (RECT_W, RECT_H))
        hsv = cv2.cvtColor(rect, cv2.COLOR_BGR2HSV)

        # Red bags: hue 0-15 or 155-180, high saturation
        red_mask1 = cv2.inRange(hsv, (0, 80, 80), (15, 255, 255))
        red_mask2 = cv2.inRange(hsv, (155, 80, 80), (180, 255, 255))
        red_mask = cv2.erode(red_mask1 | red_mask2, np.ones((3, 3)), iterations=2)

        # Blue bags: hue 95-135
        blue_mask = cv2.inRange(hsv, (95, 80, 60), (135, 255, 255))
        blue_mask = cv2.erode(blue_mask, np.ones((3, 3)), iterations=2)

        if red_mask.any():
            red_pixels.extend(hsv[red_mask > 0].tolist())
        if blue_mask.any():
            blue_pixels.extend(hsv[blue_mask > 0].tolist())

    def hsv_range(pixels, tolerance=(12, 50, 50)):
        if not pixels:
            return [0, 100, 80], [15, 255, 255]
        arr = np.array(pixels, dtype=np.float32)
        lo = np.clip(arr.min(axis=0) - tolerance, 0, [179, 255, 255]).tolist()
        hi = np.clip(arr.max(axis=0) + tolerance, 0, [179, 255, 255]).tolist()
        return lo, hi

    red_lo, red_hi = hsv_range(red_pixels)
    blue_lo, blue_hi = hsv_range(blue_pixels)

    print(f"  Red:  {len(red_pixels)} pixels  HSV {[int(x) for x in red_lo]} → {[int(x) for x in red_hi]}")
    print(f"  Blue: {len(blue_pixels)} pixels  HSV {[int(x) for x in blue_lo]} → {[int(x) for x in blue_hi]}")
    return red_lo, red_hi, blue_lo, blue_hi


def auto_calibrate(
    video_path: str,
    output_path: str = "calibration.json",
    color_frame_idx: int = 50430,
) -> dict:
    """
    Run fully automatic calibration and write calibration.json.
    Returns the config dict.
    """
    # ── temporal median for geometry ─────────────────────────────────────────
    print("Computing temporal median frame (clears bags/people from boards)...")
    geo_img = _compute_temporal_median(video_path)
    cv2.imwrite("debug_median.jpg", geo_img)
    print("  Saved debug_median.jpg")

    # ── color frame with bags on boards ──────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, color_frame_idx)
    ret, color_img = cap.read()
    cap.release()
    if not ret:
        print(f"Warning: cannot read color frame {color_frame_idx}; using median")
        color_img = geo_img

    # ── detect board corners ──────────────────────────────────────────────────
    print("\nDetecting near board corners...")
    near_corners = _find_board_corners(geo_img, NEAR_ZONE, "near")
    print("Detecting far board corners...")
    far_corners = _find_board_corners(geo_img, FAR_ZONE, "far")

    if near_corners is None or far_corners is None:
        raise RuntimeError(
            "Could not detect one or both boards automatically. "
            "Check debug_median.jpg and consider adjusting zone coordinates."
        )

    rect_corners = np.float32([[0, 0], [RECT_W, 0], [RECT_W, RECT_H], [0, RECT_H]])
    H_near = cv2.getPerspectiveTransform(near_corners, rect_corners)
    H_far  = cv2.getPerspectiveTransform(far_corners,  rect_corners)

    # ── detect holes ──────────────────────────────────────────────────────────
    print("\nDetecting holes...")
    near_rect = cv2.warpPerspective(geo_img, H_near, (RECT_W, RECT_H))
    far_rect  = cv2.warpPerspective(geo_img, H_far,  (RECT_W, RECT_H))
    near_hole = _find_hole(near_rect, "near")
    far_hole  = _find_hole(far_rect,  "far")

    # ── detect bag colors ─────────────────────────────────────────────────────
    print("\nDetecting bag colors...")
    red_lo, red_hi, blue_lo, blue_hi = _find_bag_colors(color_img, H_near, H_far)

    # ── assemble config ───────────────────────────────────────────────────────
    config = {
        "board": {
            "width_in": BOARD_W_IN,
            "height_in": BOARD_H_IN,
            "rect_w_px": RECT_W,
            "rect_h_px": RECT_H,
        },
        "hole": {
            "radius_in": HOLE_R_IN,
            "rect_radius_px": RECT_HOLE_R,
            "near_center_rect": list(near_hole[:2]),
            "far_center_rect": list(far_hole[:2]),
        },
        "near_board": {
            "corners_px": near_corners.tolist(),
            "hole_center_px": list(near_hole[:2]),
            "homography": H_near.tolist(),
        },
        "far_board": {
            "corners_px": far_corners.tolist(),
            "hole_center_px": list(far_hole[:2]),
            "homography": H_far.tolist(),
        },
        "colors": {
            "red":  {"hsv_lo": red_lo,  "hsv_hi": red_hi},
            "blue": {"hsv_lo": blue_lo, "hsv_hi": blue_hi},
        },
        "calibration_frame_shape": list(geo_img.shape),
        "color_frame": color_frame_idx,
    }

    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nCalibration saved to {output_path}")

    _save_debug(config, geo_img, color_img)
    return config


def _save_debug(config: dict, geo_img: np.ndarray, color_img: np.ndarray):
    """Save rectified board images with hole circle overlay for visual verification."""
    # Full frame with detected corners drawn on it
    annotated = geo_img.copy()
    for board_key, color in [("near_board", (0, 255, 0)), ("far_board", (0, 128, 255))]:
        corners = np.array(config[board_key]["corners_px"], dtype=np.float32)
        for i, pt in enumerate(corners.astype(int)):
            cv2.circle(annotated, tuple(pt), 12, color, -1)
            cv2.putText(annotated, str(i), tuple(pt), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (255, 255, 255), 2)
        cv2.polylines(annotated, [corners.astype(np.int32).reshape(-1, 1, 2)],
                      True, color, 3)
    cv2.imwrite("debug_full_corners.jpg", annotated)
    print("  Saved debug_full_corners.jpg")

    for board_key, img, label in [
        ("near_board", geo_img,   "geo_near"),
        ("far_board",  geo_img,   "geo_far"),
        ("near_board", color_img, "color_near"),
        ("far_board",  color_img, "color_far"),
    ]:
        H = np.array(config[board_key]["homography"])
        rect = cv2.warpPerspective(img, H, (RECT_W, RECT_H))
        key = "near_center_rect" if "near" in label else "far_center_rect"
        hx, hy = config["hole"][key]
        cv2.circle(rect, (int(hx), int(hy)), RECT_HOLE_R, (0, 255, 0), 2)
        path = f"debug_warp_{label}.jpg"
        cv2.imwrite(path, rect)
        print(f"  Saved {path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Auto-calibrate cornhole CV system")
    parser.add_argument("--video", default="Inside Garage 5-2-2026, 14.01.42 EDT - 5-2-2026, 15.01.42 EDT.mp4")
    parser.add_argument("--output", default="calibration.json")
    parser.add_argument("--color-frame", type=int, default=50430)
    args = parser.parse_args()

    auto_calibrate(
        video_path=args.video,
        output_path=args.output,
        color_frame_idx=args.color_frame,
    )
