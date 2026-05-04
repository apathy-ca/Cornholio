#!/usr/bin/env python3
"""
Motion detector: watches the mid-court zone between the two boards
and emits round-end events when activity stops.

Emits: RoundEvent(timestamp_sec, end_frame_idx, clip_start_frame, clip_end_frame)
"""

import time
from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np


@dataclass
class RoundEvent:
    timestamp_sec: float
    end_frame_idx: int
    clip_start_frame: int
    clip_end_frame: int


class MotionDetector:
    def __init__(
        self,
        stillness_threshold: float = 300000.0,  # sum of abs diff in flight ROI (blurred, 0.5s stride)
        stillness_duration_sec: float = 8.0,
        clip_before_sec: float = 15.0,
        fps: float = 30.0,
        roi_fracs: tuple[float, float, float, float] = (0.30, 0.38, 0.40, 0.22),
        scale: float = 0.5,
        min_round_gap_sec: float = 20.0,
        diff_stride: int = 15,  # frames between compared pairs (15 = 0.5s at 30fps)
    ):
        self.stillness_threshold = stillness_threshold
        self.stillness_frames = int(stillness_duration_sec * fps)
        self.clip_before_frames = int(clip_before_sec * fps)
        self.fps = fps
        self.roi_fracs = roi_fracs
        self.scale = scale
        self.min_round_gap_frames = int(min_round_gap_sec * fps)
        self.diff_stride = diff_stride

        self._frame_buffer: list[np.ndarray] = []
        self._still_count = 0
        self._was_active = False
        self._last_round_frame = -999999
        self._frame_idx = 0

    def _extract_roi(self, frame: np.ndarray) -> np.ndarray:
        small = cv2.resize(frame, None, fx=self.scale, fy=self.scale,
                           interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (11, 11), 0)
        h, w = gray.shape
        x = int(self.roi_fracs[0] * w)
        y = int(self.roi_fracs[1] * h)
        rw = int(self.roi_fracs[2] * w)
        rh = int(self.roi_fracs[3] * h)
        return gray[y:y+rh, x:x+rw]

    def process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        callback: Optional[Callable[[RoundEvent], None]] = None,
    ) -> Optional[RoundEvent]:
        self._frame_idx = frame_idx
        roi = self._extract_roi(frame)
        self._frame_buffer.append(roi)

        # Keep buffer to diff_stride + 1 frames
        if len(self._frame_buffer) > self.diff_stride + 1:
            self._frame_buffer.pop(0)

        if len(self._frame_buffer) < self.diff_stride + 1:
            return None

        diff = cv2.absdiff(roi, self._frame_buffer[0])
        motion_score = float(diff.sum())

        is_still = motion_score < self.stillness_threshold

        if is_still:
            self._still_count += 1
        else:
            self._was_active = True
            self._still_count = 0

        round_event = None
        if (self._was_active
                and self._still_count >= self.stillness_frames
                and (frame_idx - self._last_round_frame) > self.min_round_gap_frames):
            clip_start = max(0, frame_idx - self._still_count - self.clip_before_frames)
            round_event = RoundEvent(
                timestamp_sec=frame_idx / self.fps,
                end_frame_idx=frame_idx,
                clip_start_frame=clip_start,
                clip_end_frame=frame_idx,
            )
            self._last_round_frame = frame_idx
            self._was_active = False
            self._still_count = 0

            if callback:
                callback(round_event)

        return round_event


def batch_scan(
    video_path: str,
    roi_fracs: tuple[float, float, float, float] = (0.30, 0.38, 0.40, 0.22),
    stillness_threshold: float = 200000.0,
    stillness_duration_sec: float = 8.0,
    clip_before_sec: float = 15.0,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    sample_every: int = 30,  # sample 1 frame per second for speed
    min_round_gap_sec: float = 20.0,
) -> list[RoundEvent]:
    """
    Fast 1fps pre-scan to find round-end events without processing every frame.
    Up to 30x faster than run_detection for recorded video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if end_frame is None:
        end_frame = total

    scale = 0.5
    min_still_samples = int(stillness_duration_sec)  # at 1fps
    min_gap_samples = int(min_round_gap_sec)
    clip_before_frames = int(clip_before_sec * fps)

    prev_roi = None
    scores = []

    print(f"Scanning {video_path} at 1fps ...")
    fi = start_frame
    while fi < end_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            break
        small = cv2.resize(frame, None, fx=scale, fy=scale)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (11, 11), 0)
        h, w = gray.shape
        x = int(roi_fracs[0] * w); y = int(roi_fracs[1] * h)
        rw = int(roi_fracs[2] * w); rh = int(roi_fracs[3] * h)
        roi = gray[y:y+rh, x:x+rw]
        if prev_roi is not None:
            diff = cv2.absdiff(roi, prev_roi)
            scores.append((fi, float(diff.sum())))
        prev_roi = roi
        fi += sample_every

    cap.release()

    if not scores:
        return []

    vals = np.array([s for _, s in scores])
    is_still = vals < stillness_threshold

    events = []
    last_event_sample = -999999
    in_still = False
    still_start_sample = 0

    for i, still in enumerate(is_still):
        if still and not in_still:
            in_still = True
            still_start_sample = i
        elif not still and in_still:
            streak_len = i - still_start_sample
            if (streak_len >= min_still_samples
                    and (still_start_sample - last_event_sample) >= min_gap_samples):
                end_fi = scores[i - 1][0]
                clip_start = max(start_frame, end_fi - clip_before_frames)
                ev = RoundEvent(
                    timestamp_sec=end_fi / fps,
                    end_frame_idx=end_fi,
                    clip_start_frame=clip_start,
                    clip_end_frame=end_fi,
                )
                events.append(ev)
                mins = int(ev.timestamp_sec // 60)
                secs = ev.timestamp_sec % 60
                print(f"  Round-end at {mins:02d}:{secs:05.2f} (frame {ev.end_frame_idx}, "
                      f"still for {streak_len}s)")
                last_event_sample = i
            in_still = False

    print(f"Scan complete. Found {len(events)} round-end events.")
    return events


def run_detection(
    video_path: str,
    callback: Optional[Callable[[RoundEvent], None]] = None,
    roi_fracs: tuple[float, float, float, float] = (0.30, 0.38, 0.40, 0.22),
    stillness_threshold: float = 200000.0,
    stillness_duration_sec: float = 8.0,
    clip_before_sec: float = 15.0,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    show_debug: bool = False,
) -> list[RoundEvent]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if end_frame is None:
        end_frame = total

    detector = MotionDetector(
        stillness_threshold=stillness_threshold,
        stillness_duration_sec=stillness_duration_sec,
        clip_before_sec=clip_before_sec,
        fps=fps,
        roi_fracs=roi_fracs,
        scale=0.5,
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    events = []
    frame_idx = start_frame

    print(f"Running motion detection on {video_path}")
    print(f"Frames {start_frame} - {end_frame} ({(end_frame - start_frame)/fps/60:.1f} min)")

    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        def _cb(ev):
            events.append(ev)
            mins = int(ev.timestamp_sec // 60)
            secs = ev.timestamp_sec % 60
            print(f"  Round-end at {mins:02d}:{secs:05.2f} (frame {ev.end_frame_idx})")
            if callback:
                callback(ev)

        detector.process_frame(frame, frame_idx, _cb)

        if show_debug and frame_idx % 300 == 0:
            progress = (frame_idx - start_frame) / (end_frame - start_frame) * 100
            print(f"\r  Progress: {progress:.1f}%", end="", flush=True)

        frame_idx += 1

    cap.release()
    print(f"\nDetected {len(events)} round-end events.")
    return events
