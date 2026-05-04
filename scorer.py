#!/usr/bin/env python3
"""
Static scorer: counts bags on/in-hole for a single end-of-round frame.
Clip analyzer: detects cornholes from a short video clip.

Both use the homography from calibration.json to rectify board views.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


@dataclass
class BoardCount:
    on_board: int
    in_hole: int
    flagged: bool = False  # True if a suspicious large contour was detected


@dataclass
class RoundScore:
    red: BoardCount
    blue: BoardCount
    red_cornholes_from_clip: int = 0
    blue_cornholes_from_clip: int = 0
    clip_static_mismatch: bool = False


def load_calibration(path: str = "calibration.json") -> dict:
    with open(path) as f:
        return json.load(f)


class BoardScorer:
    """Scores a single board in a single frame using HSV + homography."""

    def __init__(self, config: dict, board_key: str):
        self.config = config
        self.board_key = board_key
        board_cfg = config[board_key]

        self.H = np.array(board_cfg["homography"], dtype=np.float64)
        self.rect_w = config["board"]["rect_w_px"]
        self.rect_h = config["board"]["rect_h_px"]
        hole_cfg = config["hole"]
        self.hole_center = tuple(hole_cfg["near_center_rect"])  # same for both boards
        self.hole_radius = hole_cfg["rect_radius_px"]

        color_cfg = config["colors"]
        self.red_lo = np.array(color_cfg["red"]["hsv_lo"], dtype=np.uint8)
        self.red_hi = np.array(color_cfg["red"]["hsv_hi"], dtype=np.uint8)
        self.blue_lo = np.array(color_cfg["blue"]["hsv_lo"], dtype=np.uint8)
        self.blue_hi = np.array(color_cfg["blue"]["hsv_hi"], dtype=np.uint8)

        # Expected bag area in rectified pixels: a 6"x6" bag ~= 60x60px = 3600px²
        # Allow 0.3x to 2.5x that range
        self.bag_area_min = 600
        self.bag_area_max = 9000
        self.bag_area_expected = 3600

    def rectify(self, frame: np.ndarray) -> np.ndarray:
        return cv2.warpPerspective(frame, self.H, (self.rect_w, self.rect_h))

    def _hsv_mask(self, hsv: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        # Handle hue wrap-around for red (hue near 0/180)
        if lo[0] > hi[0]:  # wrap-around case (e.g., hue 170-10)
            m1 = cv2.inRange(hsv, np.array([lo[0], lo[1], lo[2]]),
                             np.array([179, hi[1], hi[2]]))
            m2 = cv2.inRange(hsv, np.array([0, lo[1], lo[2]]),
                             np.array([hi[0], hi[1], hi[2]]))
            return cv2.bitwise_or(m1, m2)
        return cv2.inRange(hsv, lo, hi)

    def _count_bags(self, mask: np.ndarray) -> BoardCount:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        on_board = 0
        in_hole = 0
        flagged = False

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.bag_area_min:
                continue

            if area > self.bag_area_expected * 2.5:
                flagged = True  # Possibly stacked bags

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            dist = np.hypot(cx - self.hole_center[0], cy - self.hole_center[1])

            if dist <= self.hole_radius:
                in_hole += 1
            elif area <= self.bag_area_max:
                on_board += 1

        return BoardCount(on_board=on_board, in_hole=in_hole, flagged=flagged)

    def score_frame(self, frame: np.ndarray) -> tuple[BoardCount, BoardCount]:
        """Returns (red_count, blue_count) for a single frame."""
        rect = self.rectify(frame)
        hsv = cv2.cvtColor(rect, cv2.COLOR_BGR2HSV)
        red_mask = self._hsv_mask(hsv, self.red_lo, self.red_hi)
        blue_mask = self._hsv_mask(hsv, self.blue_lo, self.blue_hi)
        return self._count_bags(red_mask), self._count_bags(blue_mask)

    def debug_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Returns rectified board with detection overlay."""
        rect = self.rectify(frame)
        hsv = cv2.cvtColor(rect, cv2.COLOR_BGR2HSV)
        red_mask = self._hsv_mask(hsv, self.red_lo, self.red_hi)
        blue_mask = self._hsv_mask(hsv, self.blue_lo, self.blue_hi)

        overlay = rect.copy()
        overlay[red_mask > 0] = [0, 0, 200]
        overlay[blue_mask > 0] = [200, 0, 0]
        result = cv2.addWeighted(rect, 0.6, overlay, 0.4, 0)

        cv2.circle(result, self.hole_center, self.hole_radius, (0, 255, 0), 2)
        for mask, color, label in [
            (red_mask, (0, 0, 255), "R"),
            (blue_mask, (255, 0, 0), "B"),
        ]:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < self.bag_area_min:
                    continue
                cv2.drawContours(result, [cnt], -1, color, 2)
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.putText(result, label, (cx - 5, cy + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return result


class ClipAnalyzer:
    """Detects bags entering the hole in a video clip."""

    def __init__(self, config: dict, board_key: str):
        self.scorer = BoardScorer(config, board_key)
        self.hole_center = self.scorer.hole_center
        self.hole_radius = self.scorer.hole_radius
        self.min_track_frames = 3

    def analyze_clip(
        self,
        video_path: str,
        start_frame: int,
        end_frame: int,
    ) -> tuple[int, int]:
        """
        Analyze clip from start_frame to end_frame.
        Returns (red_cornholes, blue_cornholes).
        """
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        red_cornholes = 0
        blue_cornholes = 0

        # Track blobs entering the hole region
        # State: dict of color -> list of (frame_idx, in_hole_bool) events
        red_in_hole_frames: list[int] = []
        blue_in_hole_frames: list[int] = []

        red_was_in = False
        blue_was_in = False
        red_entry_frame = -1
        blue_entry_frame = -1

        prev_red_in_hole = 0
        prev_blue_in_hole = 0

        for fi in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                break

            red_cnt, blue_cnt = self.scorer.score_frame(frame)

            # Detect increase in in-hole count (bag just went in)
            if red_cnt.in_hole > prev_red_in_hole:
                red_cornholes += red_cnt.in_hole - prev_red_in_hole
            if blue_cnt.in_hole > prev_blue_in_hole:
                blue_cornholes += blue_cnt.in_hole - prev_blue_in_hole

            prev_red_in_hole = red_cnt.in_hole
            prev_blue_in_hole = blue_cnt.in_hole

        cap.release()
        return red_cornholes, blue_cornholes


def score_end_frame(
    video_path: str,
    frame_idx: int,
    config: dict,
) -> tuple[RoundScore, np.ndarray, np.ndarray]:
    """
    Score a single end-of-round frame for both boards.
    Returns (RoundScore, near_debug_img, far_debug_img).
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Cannot read frame {frame_idx} from {video_path}")

    near_scorer = BoardScorer(config, "near_board")
    far_scorer = BoardScorer(config, "far_board")

    near_red, near_blue = near_scorer.score_frame(frame)
    far_red, far_blue = far_scorer.score_frame(frame)

    # Aggregate across both boards
    total_red = BoardCount(
        on_board=near_red.on_board + far_red.on_board,
        in_hole=near_red.in_hole + far_red.in_hole,
        flagged=near_red.flagged or far_red.flagged,
    )
    total_blue = BoardCount(
        on_board=near_blue.on_board + far_blue.on_board,
        in_hole=near_blue.in_hole + far_blue.in_hole,
        flagged=near_blue.flagged or far_blue.flagged,
    )

    rs = RoundScore(red=total_red, blue=total_blue)

    near_debug = near_scorer.debug_overlay(frame)
    far_debug = far_scorer.debug_overlay(frame)

    return rs, near_debug, far_debug
