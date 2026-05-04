#!/usr/bin/env python3
"""
Interactive calibration tool for cornhole CV system.

Usage:
    python3 calibrate.py [--video VIDEO] [--frame FRAME_NUM] [--output calibration.json]

Click workflow:
    1. Near board: click 4 corners (top-left, top-right, bottom-right, bottom-left)
    2. Near board: click hole center
    3. Far board: click 4 corners (same order)
    4. Far board: click hole center
    5. Red bags: click 5+ pixels on red bags
    6. Blue bags: click 5+ pixels on blue bags
    Press 'n' to advance to next step, 'u' to undo last click, 'q' to quit/save.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# Standard cornhole board dimensions in inches
BOARD_WIDTH_IN = 24.0
BOARD_HEIGHT_IN = 48.0
HOLE_RADIUS_IN = 3.0  # 6" diameter hole
# Hole center: 9" from top edge, 12" from each side (centered)
HOLE_X_IN = 12.0
HOLE_Y_IN = 9.0

# Rectified board dimensions in pixels (10px per inch)
PX_PER_IN = 10
RECT_W = int(BOARD_WIDTH_IN * PX_PER_IN)  # 240
RECT_H = int(BOARD_HEIGHT_IN * PX_PER_IN)  # 480
RECT_HOLE_X = int(HOLE_X_IN * PX_PER_IN)  # 120
RECT_HOLE_Y = int(HOLE_Y_IN * PX_PER_IN)  # 90
RECT_HOLE_R = int(HOLE_RADIUS_IN * PX_PER_IN)  # 30

STEPS = [
    "NEAR BOARD: Click 4 corners — top-left, top-right, bottom-right, bottom-left",
    "NEAR BOARD: Click hole center",
    "FAR BOARD: Click 4 corners — top-left, top-right, bottom-right, bottom-left",
    "FAR BOARD: Click hole center",
    "RED BAGS: Click 5+ pixels on red bag surfaces",
    "BLUE BAGS: Click 5+ pixels on blue bag surfaces",
]

STEP_COUNTS = [4, 1, 4, 1, 5, 5]  # minimum clicks per step


class CalibrationTool:
    def __init__(self, frame: np.ndarray, output_path: str, color_frame: np.ndarray = None):
        self.orig = frame.copy()
        self.color_orig = color_frame.copy() if color_frame is not None else frame.copy()
        self.output_path = output_path
        self.step = 0
        self.clicks: list[list[tuple[int, int]]] = [[] for _ in range(len(STEPS))]
        self.zoom_factor = 1.0
        self.zoom_center = (frame.shape[1] // 2, frame.shape[0] // 2)
        self.dragging = False
        self.drag_start = None

    def _active_frame(self) -> np.ndarray:
        """Return the frame appropriate for the current step."""
        return self.color_orig if self.step >= 4 else self.orig

    def current_clicks(self):
        return self.clicks[self.step]

    def add_click(self, x: int, y: int):
        img_x, img_y = self._screen_to_img(x, y)
        # Reject clicks within 15px (image coords) of the last click — prevents double-click duplicates
        if self.current_clicks():
            lx, ly = self.current_clicks()[-1]
            if abs(img_x - lx) < 15 and abs(img_y - ly) < 15:
                return
        self.current_clicks().append((img_x, img_y))

    def undo(self):
        if self.current_clicks():
            self.current_clicks().pop()

    def advance_step(self):
        min_clicks = STEP_COUNTS[self.step]
        if len(self.current_clicks()) < min_clicks:
            print(f"Need at least {min_clicks} clicks for this step "
                  f"(have {len(self.current_clicks())})")
            return False

        # After board corners (steps 0 and 2), show a warp preview for verification
        if self.step in (0, 2) and len(self.current_clicks()) >= 4:
            corners = np.float32(self.current_clicks()[:4])
            rect = np.float32([[0, 0], [RECT_W, 0], [RECT_W, RECT_H], [0, RECT_H]])
            try:
                H = cv2.getPerspectiveTransform(corners, rect)
                warped = cv2.warpPerspective(self.orig, H, (RECT_W, RECT_H))
                # Draw hole ROI
                cv2.circle(warped, (RECT_HOLE_X, RECT_HOLE_Y), RECT_HOLE_R, (0, 255, 0), 2)
                label = "NEAR" if self.step == 0 else "FAR"
                cv2.putText(warped, f"{label} board warp — looks right? [n]=yes [u]=undo corners",
                            (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                cv2.imshow(f"Warp preview — {label} board", warped)
                cv2.waitKey(1)
            except cv2.error:
                pass  # degenerate corners — user will see garbage and undo

        self.step += 1
        if self.step >= len(STEPS):
            self.save()
            return True
        return False

    def _screen_to_img(self, sx: int, sy: int) -> tuple[int, int]:
        h, w = self._active_frame().shape[:2]
        cx, cy = self.zoom_center
        left = cx - w / (2 * self.zoom_factor)
        top = cy - h / (2 * self.zoom_factor)
        img_x = int(left + sx / self.zoom_factor)
        img_y = int(top + sy / self.zoom_factor)
        img_x = max(0, min(w - 1, img_x))
        img_y = max(0, min(h - 1, img_y))
        return img_x, img_y

    def _img_to_screen(self, ix: int, iy: int) -> tuple[int, int]:
        h, w = self._active_frame().shape[:2]
        cx, cy = self.zoom_center
        left = cx - w / (2 * self.zoom_factor)
        top = cy - h / (2 * self.zoom_factor)
        sx = int((ix - left) * self.zoom_factor)
        sy = int((iy - top) * self.zoom_factor)
        return sx, sy

    def render(self) -> np.ndarray:
        src = self._active_frame()
        h, w = src.shape[:2]
        cx, cy = self.zoom_center
        left = int(cx - w / (2 * self.zoom_factor))
        top = int(cy - h / (2 * self.zoom_factor))
        right = left + int(w / self.zoom_factor)
        bottom = top + int(h / self.zoom_factor)
        left = max(0, min(left, w))
        top = max(0, min(top, h))
        right = max(0, min(right, w))
        bottom = max(0, min(bottom, h))
        crop = src[top:bottom, left:right]
        if crop.size == 0:
            return self.orig.copy()
        view = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)

        # Draw all clicks for all steps (muted for past steps, bright for current)
        colors = [
            (0, 255, 255),   # near board corners - yellow
            (0, 255, 0),     # near hole - green
            (255, 0, 255),   # far board corners - magenta
            (0, 200, 0),     # far hole - dark green
            (0, 0, 255),     # red bag samples
            (255, 0, 0),     # blue bag samples
        ]
        for s, pts in enumerate(self.clicks):
            col = colors[s] if s == self.step else tuple(c // 3 for c in colors[s])
            for i, (ix, iy) in enumerate(pts):
                sx, sy = self._img_to_screen(ix, iy)
                cv2.circle(view, (sx, sy), 8, col, -1)
                cv2.circle(view, (sx, sy), 8, (255, 255, 255), 1)
                cv2.putText(view, str(i + 1), (sx + 10, sy + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
            # Draw board outline if 4 corners clicked
            if s in (0, 2) and len(pts) == 4:
                screen_pts = [self._img_to_screen(ix, iy) for ix, iy in pts]
                for j in range(4):
                    cv2.line(view, screen_pts[j], screen_pts[(j + 1) % 4], col, 2)

        # HUD
        hud_lines = [
            f"Step {self.step + 1}/{len(STEPS)}: {STEPS[self.step]}",
            f"Clicks: {len(self.current_clicks())}/{STEP_COUNTS[self.step]} min",
            "Keys: [n]=next step  [u]=undo  [+/-]=zoom  [q]=quit/save",
            f"Zoom: {self.zoom_factor:.1f}x",
        ]
        y0 = 30
        cv2.rectangle(view, (5, 5), (w - 5, y0 + len(hud_lines) * 25), (0, 0, 0), -1)
        for i, line in enumerate(hud_lines):
            cv2.putText(view, line, (10, y0 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        return view

    def save(self):
        near_corners = np.float32(self.clicks[0][:4])
        far_corners = np.float32(self.clicks[2][:4])
        rect_corners = np.float32([
            [0, 0], [RECT_W, 0], [RECT_W, RECT_H], [0, RECT_H]
        ])
        H_near = cv2.getPerspectiveTransform(near_corners, rect_corners)
        H_far = cv2.getPerspectiveTransform(far_corners, rect_corners)

        # Compute HSV ranges for bag colors (sampled from color_orig, not the geometry frame)
        hsv = cv2.cvtColor(self.color_orig, cv2.COLOR_BGR2HSV)
        red_samples = [hsv[y, x].tolist() for x, y in self.clicks[4]]
        blue_samples = [hsv[y, x].tolist() for x, y in self.clicks[5]]

        def hsv_range(samples, tolerance=(10, 40, 40)):
            arr = np.array(samples, dtype=np.float32)
            lo = np.clip(arr.min(axis=0) - tolerance, 0, [179, 255, 255]).tolist()
            hi = np.clip(arr.max(axis=0) + tolerance, 0, [179, 255, 255]).tolist()
            return lo, hi

        red_lo, red_hi = hsv_range(red_samples)
        blue_lo, blue_hi = hsv_range(blue_samples)

        config = {
            "board": {
                "width_in": BOARD_WIDTH_IN,
                "height_in": BOARD_HEIGHT_IN,
                "rect_w_px": RECT_W,
                "rect_h_px": RECT_H,
            },
            "hole": {
                "radius_in": HOLE_RADIUS_IN,
                "rect_radius_px": RECT_HOLE_R,
                "near_center_rect": [RECT_HOLE_X, RECT_HOLE_Y],
                "far_center_rect": [RECT_HOLE_X, RECT_HOLE_Y],
            },
            "near_board": {
                "corners_px": self.clicks[0],
                "hole_center_px": self.clicks[1][0] if self.clicks[1] else None,
                "homography": H_near.tolist(),
            },
            "far_board": {
                "corners_px": self.clicks[2],
                "hole_center_px": self.clicks[3][0] if self.clicks[3] else None,
                "homography": H_far.tolist(),
            },
            "colors": {
                "red": {"hsv_lo": red_lo, "hsv_hi": red_hi, "samples": red_samples},
                "blue": {"hsv_lo": blue_lo, "hsv_hi": blue_hi, "samples": blue_samples},
            },
            "calibration_frame_shape": list(self.orig.shape),
        }

        with open(self.output_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"\nCalibration saved to {self.output_path}")

        # Show rectified boards as a sanity check
        self._show_rectified(H_near, H_far)

    def _show_rectified(self, H_near, H_far):
        for name, H in [("near", H_near), ("far", H_far)]:
            rect = cv2.warpPerspective(self.orig, H, (RECT_W, RECT_H))
            cv2.circle(rect, (RECT_HOLE_X, RECT_HOLE_Y), RECT_HOLE_R, (0, 255, 0), 2)
            cv2.imshow(f"Rectified {name} board (press any key)", rect)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def mouse_callback(event, x, y, flags, tool: CalibrationTool):
    if event == cv2.EVENT_LBUTTONDOWN:
        tool.add_click(x, y)
    elif event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:
            tool.zoom_factor = min(tool.zoom_factor * 1.2, 8.0)
        else:
            tool.zoom_factor = max(tool.zoom_factor / 1.2, 1.0)
        tool.zoom_center = tool._screen_to_img(x, y)
        # Re-center if zoom_factor is 1
        if tool.zoom_factor <= 1.0:
            h, w = tool.orig.shape[:2]
            tool.zoom_center = (w // 2, h // 2)


def find_best_frame(video_path: str, search_start: int = 5400, search_end: int = 18000, step: int = 90) -> int:
    """
    Scan for the frame where both board zones are cleanest.
    Uses mean brightness minus std deviation — high score means bright uniform surface (empty board).
    """
    # Near board playing surface zone (full res 2688x1512)
    NB = (690, 1020, 1160, 1240)   # x1, y1, x2, y2
    # Far board playing surface zone
    FB = (1170, 435, 1770, 590)
    cap = cv2.VideoCapture(video_path)
    best_score = float('-inf')
    best_frame = search_start
    fi = search_start
    while fi <= search_end:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            break
        nb = cv2.cvtColor(frame[NB[1]:NB[3], NB[0]:NB[2]], cv2.COLOR_BGR2GRAY)
        fb = cv2.cvtColor(frame[FB[1]:FB[3], FB[0]:FB[2]], cv2.COLOR_BGR2GRAY)
        score = (float(nb.mean()) - float(nb.std())) + (float(fb.mean()) - float(fb.std()))
        if score > best_score:
            best_score = score
            best_frame = fi
        fi += step
    cap.release()
    mins = best_frame // (30 * 60); secs = (best_frame / 30) % 60
    print(f"Best calibration frame: {best_frame} ({mins:02d}:{secs:04.1f}) score={best_score:.1f}")
    return best_frame


def main():
    parser = argparse.ArgumentParser(description="Cornhole calibration tool")
    parser.add_argument("--video", default="Inside Garage 5-2-2026, 14.01.42 EDT - 5-2-2026, 15.01.42 EDT.mp4")
    parser.add_argument("--frame", type=int, default=None,
                        help="Frame for geometry calibration (default: auto-detect cleanest frame)")
    parser.add_argument("--color-frame", type=int, default=50430,
                        help="Frame for color sampling — should have bags on boards (default: 50430 = 28:01)")
    parser.add_argument("--output", default="calibration.json")
    args = parser.parse_args()

    if args.frame is None:
        print("Auto-detecting best calibration frame...")
        args.frame = find_best_frame(args.video)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Cannot open video: {args.video}")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ret, frame = cap.read()
    if not ret:
        print(f"Cannot read frame {args.frame}")
        cap.release()
        sys.exit(1)

    color_frame = None
    if args.color_frame != args.frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.color_frame)
        ret2, color_frame = cap.read()
        if ret2:
            cf_mins = args.color_frame // (30 * 60)
            cf_secs = (args.color_frame / 30) % 60
            print(f"Color frame: {args.color_frame} ({cf_mins:02d}:{cf_secs:04.1f}) — "
                  f"steps 5-6 will switch to this frame automatically")
        else:
            print(f"Warning: cannot read color frame {args.color_frame}, using geometry frame")
            color_frame = None
    cap.release()

    print(f"Geometry frame: {args.frame} ({args.frame//(30*60):02d}:{(args.frame/30)%60:04.1f})")
    print(f"Image size: {frame.shape[1]}x{frame.shape[0]}")
    print("\nCalibration workflow:")
    for i, (step, count) in enumerate(zip(STEPS, STEP_COUNTS)):
        suffix = " ← frame switches here" if i == 4 and color_frame is not None else ""
        print(f"  Step {i+1}: {step} ({count} clicks){suffix}")
    print("\nKeys: [n]=next  [u]=undo  [+/-]=zoom  scroll wheel=zoom  [q]=quit")
    print("Left-click to place points.\n")

    tool = CalibrationTool(frame, args.output, color_frame=color_frame)
    win = "Cornhole Calibration"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1344, 756)
    cv2.setMouseCallback(win, mouse_callback, tool)

    while True:
        view = tool.render()
        cv2.imshow(win, view)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('q'):
            if tool.step >= len(STEPS) - 1 and len(tool.current_clicks()) >= STEP_COUNTS[-1]:
                tool.save()
            break
        elif key == ord('n'):
            done = tool.advance_step()
            if done:
                break
        elif key == ord('u'):
            tool.undo()
        elif key in (ord('+'), ord('=')):
            tool.zoom_factor = min(tool.zoom_factor * 1.3, 8.0)
        elif key == ord('-'):
            tool.zoom_factor = max(tool.zoom_factor / 1.3, 1.0)
            if tool.zoom_factor <= 1.0:
                h, w = frame.shape[:2]
                tool.zoom_center = (w // 2, h // 2)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
