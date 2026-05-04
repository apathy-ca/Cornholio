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


def _find_board_corners(
    frame: np.ndarray,
    zone: tuple,
    label: str,
    v_min: int = 155,
) -> np.ndarray | None:
    """
    Find the 4 corners of a cornhole board within zone (x1,y1,x2,y2).

    v_min — minimum HSV V value. Use 170 for near board (filters floor
    contamination that appears at V>155 but S<60). Far board needs 155.

    Strategy:
      1. Threshold for bright white/cream playing surface (high V, low S).
      2. Try convex-hull quad-fit; accept if all 4 corners are distinct and
         at most 1 touches the zone boundary (one corner may legitimately
         sit at the board edge that aligns with the zone boundary).
      3. Fall back to minAreaRect + 2:1 aspect correction.

    Returns (4,2) float32 array in image coordinates, or None on failure.
    """
    x1, y1, x2, y2 = zone
    crop = frame[y1:y2, x1:x2]
    cw, ch = x2 - x1, y2 - y1

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # Low saturation (S<60) excludes floor/wall; v_min separates board from
    # lower-brightness gray floor (near board zone contaminated at V>155).
    white_mask = cv2.inRange(hsv, (0, 0, v_min), (180, 60, 255))

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
    # Accept if all 4 corners are distinct and at most 1 touches the crop boundary
    # (the board's actual corner may sit exactly at the zone edge).
    EDGE_MARGIN = 8  # px from crop border considered "on boundary"
    hull = cv2.convexHull(best_cnt)
    peri = cv2.arcLength(hull, True)
    box = None
    for eps_frac in (0.04, 0.05, 0.06, 0.08, 0.10):
        poly = cv2.approxPolyDP(hull, eps_frac * peri, True)
        if len(poly) != 4:
            continue
        pts4 = poly.reshape(4, 2).astype(np.float32)
        # Reject degenerate quads (two identical corners)
        diffs = [np.linalg.norm(pts4[i] - pts4[j])
                 for i in range(4) for j in range(i + 1, 4)]
        if min(diffs) < 5:
            continue
        on_boundary = int(
            (pts4[:, 0] < EDGE_MARGIN).any()
            + (pts4[:, 0] > cw - EDGE_MARGIN).any()
            + (pts4[:, 1] < EDGE_MARGIN).any()
            + (pts4[:, 1] > ch - EDGE_MARGIN).any()
        )
        if on_boundary <= 1:
            # Verify sorted corners are all distinct (non-rectangular quads
            # can fool _sort_corners into producing duplicate points).
            candidate = pts4.copy()
            candidate[:, 0] += x1
            candidate[:, 1] += y1
            srt = _sort_corners(candidate)
            srt_dists = [np.linalg.norm(srt[i] - srt[j])
                         for i in range(4) for j in range(i + 1, 4)]
            if min(srt_dists) < 5:
                print(f"  [{label}] Quad at eps={eps_frac:.2f} rejected "
                      f"(degenerate after sort)")
                continue
            box = pts4
            print(f"  [{label}] Quad fit at eps={eps_frac:.2f} "
                  f"(boundary_edges={on_boundary}): {box.astype(int).tolist()}")
            break
        else:
            print(f"  [{label}] Quad at eps={eps_frac:.2f} rejected "
                  f"(boundary_edges={on_boundary})")

    if box is None:
        # Fallback: minAreaRect + 2:1 aspect correction.
        # The dark board design truncates the detected white region along the long
        # axis; extend it so the long dimension = 2× the short dimension.
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
              f"(aspect {detected_aspect:.2f}) → 2:1 corrected")

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
    Returns (cx, cy, radius). Falls back to nominal if detection fails.

    Masks the image border (15px) to suppress warp-edge artifacts before
    searching for the darkest circular region.
    """
    gray = cv2.cvtColor(rect_img, cv2.COLOR_BGR2GRAY)
    h_img, w_img = gray.shape

    # Blur on the original gray (unmasked) so edges don't contaminate the blur
    blur = cv2.GaussianBlur(gray, (9, 9), 2)

    # Mask the blurred image AFTER blurring.
    # Bottom third is masked because the warp often extends outside the board
    # into dark background below the playing surface.
    BORDER = 15
    blur_masked = blur.copy()
    blur_masked[:BORDER, :] = 255
    blur_masked[h_img * 2 // 3:, :] = 255   # bottom third
    blur_masked[:, :BORDER] = 255
    blur_masked[:, -BORDER:] = 255

    # Try increasingly permissive params. Hole V is typically <40 in the warp.
    for param1, param2, minR, maxR in [
        (30, 10, 8, 70),
        (25, 8,  6, 80),
        (20, 6,  5, 80),
    ]:
        circles = cv2.HoughCircles(
            blur_masked, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
            param1=param1, param2=param2, minRadius=minR, maxRadius=maxR,
        )
        if circles is None:
            continue
        circles = np.round(circles[0]).astype(int)
        valid = [c for c in circles
                 if BORDER <= c[0] < w_img - BORDER
                 and BORDER <= c[1] < h_img * 2 // 3]
        if not valid:
            continue
        best = min(valid, key=lambda c: int(gray[c[1], c[0]]))
        cx, cy, r = int(best[0]), int(best[1]), int(best[2])
        if gray[cy, cx] < 60:
            print(f"  [{label}] Hole at ({cx},{cy}) r={r} V={gray[cy,cx]} "
                  f"(param2={param2})")
            return cx, cy, r

    # Fallback: darkest pixel within the search region
    search_region = gray[:h_img * 2 // 3, BORDER:w_img - BORDER]
    min_val, _, min_loc, _ = cv2.minMaxLoc(search_region)
    if min_val < 40:
        abs_loc = (min_loc[0] + BORDER, min_loc[1])
        print(f"  [{label}] Hole from darkest pixel: {abs_loc} V={min_val:.0f}")
        return abs_loc[0], abs_loc[1], RECT_HOLE_R

    print(f"  [{label}] No dark hole found; using nominal position")
    return RECT_HOLE[0], RECT_HOLE[1], RECT_HOLE_R


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

    def hsv_range(pixels, tolerance=(10, 40, 40)):
        if not pixels:
            return [0, 100, 80], [15, 255, 255]
        arr = np.array(pixels, dtype=np.float32)
        lo = np.clip(arr.min(axis=0) - tolerance, 0, [179, 255, 255]).tolist()
        hi = np.clip(arr.max(axis=0) + tolerance, 0, [179, 255, 255]).tolist()
        return lo, hi

    def red_hsv_range(pixels):
        """Red hue wraps around 0/180; split into lower (H<30) and upper (H>150)."""
        if not pixels:
            return [0, 100, 80], [15, 255, 255]
        arr = np.array(pixels, dtype=np.float32)
        lower = arr[arr[:, 0] <= 30]   # H≈0-15
        upper = arr[arr[:, 0] >= 150]  # H≈155-180
        # Use whichever cluster has more pixels; combine S/V from all pixels
        sv_arr = np.vstack([lower, upper]) if len(lower) and len(upper) else arr
        sv_lo = sv_arr[:, 1:].min(axis=0) - np.array([40, 40])
        sv_hi = sv_arr[:, 1:].max(axis=0) + np.array([40, 40])
        if len(lower) >= len(upper):
            h_lo = max(0, float(lower[:, 0].min()) - 10)
            h_hi = min(15, float(lower[:, 0].max()) + 10)
        else:
            h_lo = max(150, float(upper[:, 0].min()) - 10)
            h_hi = min(179, float(upper[:, 0].max()) + 10)
        lo = [h_lo, float(np.clip(sv_lo[0], 0, 255)), float(np.clip(sv_lo[1], 0, 255))]
        hi = [h_hi, 255.0, 255.0]
        return lo, hi

    red_lo, red_hi = red_hsv_range(red_pixels)
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
    # Near board: V>170 required — at V>155 the floor contaminates the zone
    # with a huge 70k-px blob; V>170 isolates just the board surface (~26k px).
    print("\nDetecting near board corners...")
    near_corners = _find_board_corners(geo_img, NEAR_ZONE, "near", v_min=170)
    print("Detecting far board corners...")
    far_corners = _find_board_corners(geo_img, FAR_ZONE, "far", v_min=155)

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
