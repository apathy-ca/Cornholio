#!/usr/bin/env python3
"""
Player identification from security camera footage.

Uses YOLOv8 person detection + upper-body color histograms to cluster
the two cornhole players across all round-end frames, then associates
each cluster with a bag color based on which end of the court they stood at.

Output:
  player_id.jpg  — mosaic of the two identified players with bag color labels
  player_id.json — round-by-round player assignments
"""

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

VIDEO = "Inside Garage 5-2-2026, 14.01.42 EDT - 5-2-2026, 15.01.42 EDT.mp4"

# Round-end frames from batch_scan
ROUND_FRAMES = [
    1320, 4710, 6810, 18270, 21900, 24870, 36360, 39780, 42870,
    45390, 47940, 50430, 58560, 60480, 64140, 66420, 68910, 74340,
    80490, 84840, 94560, 97290, 99270, 101400, 106920,
]

# Court zone — exclude spectators on the far left (tractor side)
# Any person detection whose center x < COURT_X_MIN is a spectator, skip it.
COURT_X_MIN = 950

# Throwing-end zones: where the active thrower stands
# Near end: behind the near board (bottom of frame), throws toward far board
# Far end:  behind the far board (upper-right),       throws toward near board
NEAR_END_Y_MIN = 1100   # person center y above this = not near-end thrower
FAR_END_Y_MAX  = 850    # person center y below this = not far-end thrower
FAR_END_X_MIN  = 1700   # far-end throwers are also right of center

# How many frames before the round-end stillness to sample for "who's throwing"
# At still-frame + 0 the round is over; we look back ~5 seconds for a mid-throw sample
LOOKBACK_FRAMES = 5 * 30  # 5s at 30fps

PERSON_CLASS = 0  # COCO class ID for person
MIN_PERSON_HEIGHT = 80  # px — ignore tiny distant detections
CONF_THRESH = 0.50


def load_model() -> YOLO:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO("yolov8s.pt")
    model.to(device)
    return model


def read_frame(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    return frame if ret else None


def detect_people(model, frame, conf=CONF_THRESH):
    """Return list of (x1,y1,x2,y2,confidence) for all people in frame."""
    results = model.predict(frame, conf=conf, classes=[PERSON_CLASS], verbose=False)[0]
    people = []
    for box in results.boxes:
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
        h = y2 - y1
        if h < MIN_PERSON_HEIGHT:
            continue
        cx = (x1 + x2) / 2
        if cx < COURT_X_MIN:
            continue
        people.append((x1, y1, x2, y2, float(box.conf[0])))
    return people


def upper_body_hist(frame, x1, y1, x2, y2) -> np.ndarray:
    """16-bin HSV histogram of the upper 55% of a person bounding box."""
    h = y2 - y1
    crop = frame[int(y1):int(y1 + h * 0.55), int(x1):int(x2)]
    if crop.size == 0:
        return np.zeros(16 * 3, dtype=np.float32)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = []
    for ch in range(3):
        h_ch = cv2.calcHist([hsv], [ch], None, [16], [0, 256])
        hist.append(h_ch.flatten())
    vec = np.concatenate(hist).astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def cluster_players(samples: list[dict], n_clusters=2):
    """
    K-means on appearance histograms. Returns cluster labels (0 or 1) per sample.
    Seeded with the two most dissimilar histograms for stability.
    """
    hists = np.stack([s["hist"] for s in samples])  # (N, D)

    # Find seed pair: maximum cosine distance
    best_dist, best_i, best_j = -1, 0, 1
    for i in range(min(len(samples), 40)):
        for j in range(i + 1, min(len(samples), 40)):
            d = 1 - float(hists[i] @ hists[j])
            if d > best_dist:
                best_dist, best_i, best_j = d, i, j

    centers = hists[[best_i, best_j]].copy()

    for _ in range(50):
        dists = np.stack([np.linalg.norm(hists - c, axis=1) for c in centers], axis=1)
        labels = np.argmin(dists, axis=1)
        new_centers = np.stack([
            hists[labels == k].mean(axis=0) if (labels == k).any() else centers[k]
            for k in range(n_clusters)
        ])
        if np.allclose(centers, new_centers, atol=1e-6):
            break
        centers = new_centers

    return labels


def sample_position(cy, cx):
    """Return 'near_end', 'far_end', or 'mid' based on image position."""
    if cy > NEAR_END_Y_MIN:
        return "near_end"
    if cy < FAR_END_Y_MAX and cx > FAR_END_X_MIN:
        return "far_end"
    return "mid"


def make_mosaic(samples: list[dict], frame_map: dict, label_name: str,
                max_tiles=12, tile_size=(96, 160)) -> np.ndarray:
    tiles = []
    for s in samples[:max_tiles]:
        frame = frame_map.get(s["frame_idx"])
        if frame is None:
            continue
        x1, y1, x2, y2 = s["x1"], s["y1"], s["x2"], s["y2"]
        crop = frame[int(y1):int(y2), int(x1):int(x2)]
        if crop.size == 0:
            continue
        crop = cv2.resize(crop, tile_size)
        cv2.putText(crop, f"R{s['round_num']}", (2, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        tiles.append(crop)
    if not tiles:
        return np.zeros((tile_size[1], tile_size[0] * 3, 3), dtype=np.uint8)
    n = len(tiles)
    cols = min(n, 6)
    rows = (n + cols - 1) // cols
    canvas = np.zeros((rows * tile_size[1], cols * tile_size[0], 3), dtype=np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        canvas[r*tile_size[1]:(r+1)*tile_size[1], c*tile_size[0]:(c+1)*tile_size[0]] = t
    # Label bar
    bar = np.zeros((28, canvas.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, label_name, (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return np.vstack([bar, canvas])


def main():
    print("Loading YOLOv8s person detector ...")
    model = load_model()

    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    samples = []
    frame_cache = {}

    print(f"Scanning {len(ROUND_FRAMES)} round-end frames + lookback windows ...")
    for rnd_num, fi in enumerate(ROUND_FRAMES, 1):
        # Sample at the round-end frame and a few frames before (mid-round action)
        check_frames = [fi, max(0, fi - LOOKBACK_FRAMES // 2)]
        for sample_fi in check_frames:
            frame = read_frame(cap, sample_fi)
            if frame is None:
                continue
            people = detect_people(model, frame)
            for (x1, y1, x2, y2, conf) in people:
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                pos = sample_position(cy, cx)
                hist = upper_body_hist(frame, x1, y1, x2, y2)
                samples.append({
                    "round_num": rnd_num,
                    "frame_idx": sample_fi,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "cx": cx, "cy": cy,
                    "conf": conf,
                    "pos": pos,
                    "hist": hist,
                })
            frame_cache[sample_fi] = frame

        mins = fi // int(fps) // 60
        secs = (fi // int(fps)) % 60
        n = len([s for s in samples if s["round_num"] == rnd_num])
        print(f"  Rnd {rnd_num:>2} [{mins:02d}:{secs:02d}]  {n} people detected")

    cap.release()

    if len(samples) < 4:
        print("Not enough person detections to cluster. Exiting.")
        return

    print(f"\nClustering {len(samples)} person samples into 2 player identities ...")
    labels = cluster_players(samples)
    for i, s in enumerate(samples):
        s["player"] = int(labels[i])

    # Count appearances and position distribution per cluster
    for p in [0, 1]:
        psamp = [s for s in samples if s["player"] == p]
        positions = [s["pos"] for s in psamp]
        near = positions.count("near_end")
        far  = positions.count("far_end")
        mid  = positions.count("mid")
        rounds = sorted(set(s["round_num"] for s in psamp))
        print(f"\n  Player {p}: {len(psamp)} appearances in {len(rounds)} rounds")
        print(f"    Position breakdown: near_end={near}  far_end={far}  mid={mid}")
        print(f"    Rounds seen: {rounds}")

    # Bag color association heuristic:
    # We know from YOLO scoring which board had which color bags each round.
    # Thrower at near_end -> bags land on far board.
    # Thrower at far_end  -> bags land on near board.
    # For simplicity: the player more often at near_end is the one whose bags
    # go to the far board. Cross-reference with which color dominated the far board.
    # (Here we hardcode the rough result from our YOLO scan: far board had more red.)
    p0_near = sum(1 for s in samples if s["player"] == 0 and s["pos"] == "near_end")
    p1_near = sum(1 for s in samples if s["player"] == 1 and s["pos"] == "near_end")

    # The player seen more at near_end throws toward the far board.
    # From YOLO results, far-board detections were predominantly red.
    # So near_end player → red bags is a reasonable first guess.
    if p0_near >= p1_near:
        bag_color = {0: "red?", 1: "blue?"}
    else:
        bag_color = {0: "blue?", 1: "red?"}

    print(f"\n  Near-end appearances: player0={p0_near}  player1={p1_near}")
    print(f"  Tentative color assignment: player0={bag_color[0]}  player1={bag_color[1]}")
    print("  (Marked '?' — verify visually from the mosaic image)")

    # Build mosaic
    mosaics = []
    for p in [0, 1]:
        psamp = [s for s in samples if s["player"] == p]
        label = f"Player {p}  ({bag_color[p]} bags)"
        mosaic = make_mosaic(psamp, frame_cache, label)
        mosaics.append(mosaic)

    # Pad to same width
    w = max(m.shape[1] for m in mosaics)
    padded = []
    for m in mosaics:
        if m.shape[1] < w:
            pad = np.zeros((m.shape[0], w - m.shape[1], 3), dtype=np.uint8)
            m = np.hstack([m, pad])
        padded.append(m)

    combined = np.vstack(padded)
    cv2.imwrite("player_id.jpg", combined)
    print(f"\nSaved player_id.jpg")

    # Save JSON
    result = {
        "player0_bag_color": bag_color[0],
        "player1_bag_color": bag_color[1],
        "rounds": [
            {
                "round_num": rnd_num,
                "frame": fi,
                "people": [
                    {"player": s["player"], "pos": s["pos"], "conf": round(s["conf"], 2)}
                    for s in samples if s["round_num"] == rnd_num and s["frame_idx"] == fi
                ]
            }
            for rnd_num, fi in enumerate(ROUND_FRAMES, 1)
        ]
    }
    with open("player_id.json", "w") as f:
        json.dump(result, f, indent=2)
    print("Saved player_id.json")


if __name__ == "__main__":
    main()
