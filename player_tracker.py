#!/usr/bin/env python3
"""
Player tracker: identifies which player is at each end of the court
during each round, using clothing color histograms extracted from
background-subtracted frames.

Works for 1v1 cornhole: one player at the near end, one at the far end.
Players may swap ends between games.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# Board exclusion zones (x1,y1,x2,y2) — blob in these areas = bag, not person
NEAR_ZONE = (900, 970, 1620, 1400)
FAR_ZONE  = (1820, 530, 2280, 780)

# y-coordinate split between near-end and far-end player zones
PLAYER_ZONE_SPLIT_Y = 900

# Minimum contour area (px²) to be considered a person
MIN_PERSON_AREA = 4000

# Frames to sample from within each round's active clip
SAMPLE_COUNT = 5


@dataclass
class PlayerSighting:
    round_idx: int
    timestamp_sec: float
    end: str          # 'near' or 'far'
    hist: np.ndarray  # flattened normalised HSV histogram
    center: tuple[int, int]
    area: int


def compute_full_median(
    video_path: str,
    n_samples: int = 80,
    end_sec: float = 1800.0,
) -> np.ndarray:
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
        raise RuntimeError("No frames read for median")
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def _changed_mask(frame: np.ndarray, median: np.ndarray, threshold: int = 30) -> np.ndarray:
    diff = np.abs(frame.astype(np.int16) - median.astype(np.int16))
    return (diff.max(axis=2) > threshold).astype(np.uint8) * 255


def _exclude_board_zones(mask: np.ndarray) -> np.ndarray:
    """Zero out the board regions so bag pixels don't pollute player blobs."""
    m = mask.copy()
    for (x1, y1, x2, y2) in [NEAR_ZONE, FAR_ZONE]:
        m[y1:y2, x1:x2] = 0
    return m


def _extract_hist(frame: np.ndarray, contour: np.ndarray) -> np.ndarray:
    """HSV hue+saturation histogram over the upper 60% of the contour bounding box."""
    x, y, w, h = cv2.boundingRect(contour)
    # Use top 60% of blob (torso), skip feet area
    torso_h = max(1, int(h * 0.6))
    roi = frame[y:y + torso_h, x:x + w]
    if roi.size == 0:
        return np.zeros(18 * 8, dtype=np.float32)
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_roi], [0, 1], None, [18, 8], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def _find_player_blobs(
    frame: np.ndarray,
    median: np.ndarray,
) -> list[tuple[int, int, int, np.ndarray]]:
    """
    Returns list of (cx, cy, area, contour) for person-sized moving blobs,
    with board zones excluded.
    """
    mask = _changed_mask(frame, median)
    mask = _exclude_board_zones(mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_PERSON_AREA:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        blobs.append((cx, cy, area, cnt))
    return blobs


def collect_sightings(
    video_path: str,
    events: list,          # list of RoundEvent from motion_detector
    median: np.ndarray,
    video_offset_min: int = 0,
    round_offset: int = 0,
) -> list[PlayerSighting]:
    """
    For each round event, sample SAMPLE_COUNT frames from the active window
    and collect player sightings (one per end per frame).
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    sightings = []

    for ri, ev in enumerate(events):
        # Active window: from clip_start to (end_frame - stillness buffer)
        active_end = max(ev.clip_start_frame + 1,
                         ev.end_frame_idx - int(8 * fps))
        if active_end <= ev.clip_start_frame:
            active_end = ev.clip_start_frame + 1

        sample_frames = np.linspace(
            ev.clip_start_frame, active_end, SAMPLE_COUNT, dtype=int
        )

        near_hists = []
        far_hists  = []

        for fi in sample_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ret, frame = cap.read()
            if not ret:
                continue

            blobs = _find_player_blobs(frame, median)

            # Largest blob in each zone = the player there
            near_blobs = [(cx, cy, a, c) for cx, cy, a, c in blobs if cy >= PLAYER_ZONE_SPLIT_Y]
            far_blobs  = [(cx, cy, a, c) for cx, cy, a, c in blobs if cy <  PLAYER_ZONE_SPLIT_Y]

            if near_blobs:
                cx, cy, area, cnt = max(near_blobs, key=lambda b: b[2])
                near_hists.append(_extract_hist(frame, cnt))

            if far_blobs:
                cx, cy, area, cnt = max(far_blobs, key=lambda b: b[2])
                far_hists.append(_extract_hist(frame, cnt))

        ts = ev.timestamp_sec + video_offset_min * 60
        rnd = round_offset + ri

        for end, hists in [("near", near_hists), ("far", far_hists)]:
            if hists:
                mean_hist = np.mean(np.stack(hists), axis=0)
                sightings.append(PlayerSighting(
                    round_idx=rnd,
                    timestamp_sec=ts,
                    end=end,
                    hist=mean_hist,
                    center=(0, 0),
                    area=0,
                ))

    cap.release()
    return sightings


def _hist_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(cv2.compareHist(
        a.astype(np.float32), b.astype(np.float32),
        cv2.HISTCMP_BHATTACHARYYA,
    ))


def identify_players(
    sightings: list[PlayerSighting],
    n_players: int = 2,
    names: Optional[list[str]] = None,
) -> dict[int, str]:
    """
    Cluster sightings into n_players identities using greedy histogram matching.
    Returns {round_idx: player_name} mapping (keyed by (round_idx, end) tuple).

    Strategy: seed clusters from the first two sightings that look different,
    then assign each subsequent sighting to the nearest cluster centroid.
    """
    if not sightings:
        return {}

    if names is None:
        names = [f"Player {i+1}" for i in range(n_players)]

    # Separate near and far — each end has at most one player per round
    # Find 2 clusters across ALL sightings (players swap ends between games)
    hists = np.stack([s.hist for s in sightings])

    # Seed: pick sighting 0 as cluster 0; find most-different sighting for cluster 1
    centroids = [hists[0]]
    for h in hists[1:]:
        if _hist_distance(h, centroids[0]) > 0.3:
            centroids.append(h)
            break
    while len(centroids) < n_players:
        centroids.append(hists[0])  # degenerate fallback

    # Iterative k-means (5 iterations)
    labels = np.zeros(len(hists), dtype=int)
    for _ in range(5):
        for i, h in enumerate(hists):
            dists = [_hist_distance(h, c) for c in centroids]
            labels[i] = int(np.argmin(dists))
        for k in range(n_players):
            members = hists[labels == k]
            if len(members):
                centroids[k] = members.mean(axis=0)

    # Build result: {(round_idx, end): player_name}
    result = {}
    for i, s in enumerate(sightings):
        result[(s.round_idx, s.end)] = names[labels[i]]

    return result, labels, centroids


def run_player_tracking(
    videos_and_events: list[tuple[str, list, int, int]],  # (path, events, offset_min, round_offset)
    names: Optional[list[str]] = None,
    debug_dir: str = ".",
) -> dict:
    """
    Full pipeline: collect sightings from all videos, cluster into player IDs.

    Returns dict with sightings, labels, and identity map.
    """
    all_sightings = []

    for video_path, events, offset_min, round_offset in videos_and_events:
        print(f"\nComputing median for {Path(video_path).name}...")
        median = compute_full_median(video_path)

        print(f"Collecting player sightings from {len(events)} rounds...")
        sightings = collect_sightings(
            video_path, events, median,
            video_offset_min=offset_min,
            round_offset=round_offset,
        )
        all_sightings.extend(sightings)
        print(f"  Found {len(sightings)} sightings")

    if not all_sightings:
        print("No player sightings found.")
        return {}

    print(f"\nClustering {len(all_sightings)} sightings into 2 players...")
    identity_map, labels, centroids = identify_players(all_sightings, n_players=2, names=names)

    # Summary
    from collections import Counter
    for k, name in enumerate(names or ["Player 1", "Player 2"]):
        count = (labels == k).sum()
        print(f"  {name}: {count} sightings")

    return {
        "sightings": all_sightings,
        "labels": labels,
        "identity_map": identity_map,
    }
