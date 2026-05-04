#!/usr/bin/env python3
"""
YOLO-World prototype scorer for cornhole bag detection.

Uses crop-based detection: each board region is cropped and fed to YOLO
separately so bags appear large enough to detect reliably (especially far board).

Usage:
    python3 yolo_test.py
    python3 yolo_test.py --conf 0.20 --frames 1320 4710 50430
    python3 yolo_test.py --benchmark
    python3 yolo_test.py --video "path/to/video.mp4"
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ── Crop regions for each board (in full-frame pixel coords) ─────────────────
# Near board corners: ~(1164,1204)(1405,1094)(1460,1214)(1219,1324)
# Generous padding so bags partially off the board surface are included.
NEAR_X1, NEAR_Y1 = 1050, 980
NEAR_X2, NEAR_Y2 = 1580, 1480

# Far board corners: ~(1904,571)(2239,658)(2182,779)(1820,779)
FAR_X1, FAR_Y1  = 1750, 470
FAR_X2, FAR_Y2  = 2320, 870

CALIB_PATH = "calibration.json"
VIDEO_PATH = "Inside Garage 5-2-2026, 14.01.42 EDT - 5-2-2026, 15.01.42 EDT.mp4"
WEIGHTS = "yolov8s-worldv2.pt"

# Best zero-shot prompts found through testing:
# "red cornhole bag" catches far-board red bags better than generic "red bag"
# "blue bean bag" gives 0.9+ confidence on near-board blue bags
CLASSES = ["red cornhole bag", "blue bean bag"]

# Representative round-end frames from batch_scan (includes far-board-only rounds)
DEFAULT_FRAMES = [4710, 6810, 50430, 80490, 97290, 101400]


def load_calibration(path):
    with open(path) as f:
        return json.load(f)


def compute_hole_center_image(H: np.ndarray, rect_cx: int, rect_cy: int) -> tuple[int, int]:
    """Map rectified-space hole center back to image coordinates."""
    H_inv = np.linalg.inv(H)
    pt = H_inv @ np.array([rect_cx, rect_cy, 1.0])
    pt /= pt[2]
    return int(pt[0]), int(pt[1])


def load_model_once(weights: str, device: str) -> YOLO:
    """
    Load YOLO-World and set classes BEFORE first inference.
    Must not call set_classes() again after predict() — doing so causes
    a CUDA/CPU device mismatch in the CLIP text encoder on subsequent calls.
    """
    model = YOLO(weights)
    # CLIP text encoding happens here on CPU; model.to(device) moves detector
    # head but CLIP stays on CPU — that's fine as long as we don't re-encode.
    model.set_classes(CLASSES)
    model.to(device)
    return model


def _infer_crop(model, crop: np.ndarray, conf: float,
                offset_x: int, offset_y: int, board_label: str,
                hole_img_x: int, hole_img_y: int, hole_radius_px: int) -> list[dict]:
    """Run inference on a cropped region; map boxes back to full-frame coords."""
    results = model.predict(crop, conf=conf, verbose=False)[0]
    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf_val = float(box.conf[0])
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
        # Map back to full-frame coordinates
        fx1 = x1 + offset_x; fy1 = y1 + offset_y
        fx2 = x2 + offset_x; fy2 = y2 + offset_y
        cx = (fx1 + fx2) / 2
        cy = (fy1 + fy2) / 2
        color = "red" if cls_id == 0 else "blue"
        dist_to_hole = np.hypot(cx - hole_img_x, cy - hole_img_y)
        in_hole = dist_to_hole <= hole_radius_px
        detections.append({
            "color": color,
            "conf": conf_val,
            "cx": cx, "cy": cy,
            "bbox": (fx1, fy1, fx2, fy2),
            "board": board_label,
            "in_hole": in_hole,
        })
    return detections


def run_inference(model, frame: np.ndarray, conf: float,
                  near_hole: tuple, far_hole: tuple,
                  near_hole_r: int = 55, far_hole_r: int = 28) -> tuple[list[dict], float]:
    """
    Crop each board region and run YOLO on each crop separately.
    Returns (detections, elapsed_ms). All bbox coords are in full-frame space.
    """
    near_crop = frame[NEAR_Y1:NEAR_Y2, NEAR_X1:NEAR_X2]
    far_crop  = frame[FAR_Y1:FAR_Y2,   FAR_X1:FAR_X2]

    t0 = time.perf_counter()
    near_dets = _infer_crop(model, near_crop, conf, NEAR_X1, NEAR_Y1, "near",
                             near_hole[0], near_hole[1], near_hole_r)
    far_dets  = _infer_crop(model, far_crop,  conf, FAR_X1,  FAR_Y1,  "far",
                             far_hole[0],  far_hole[1],  far_hole_r)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return near_dets + far_dets, elapsed_ms


def count_bags(detections: list[dict]) -> dict:
    counts = {"near_red": 0, "near_blue": 0, "far_red": 0, "far_blue": 0,
              "near_red_hole": 0, "near_blue_hole": 0, "far_red_hole": 0, "far_blue_hole": 0}
    for d in detections:
        board = d["board"]
        color = d["color"]
        counts[f"{board}_{color}"] += 1
        if d.get("in_hole"):
            counts[f"{board}_{color}_hole"] += 1
    return counts


def draw_debug(frame: np.ndarray, detections: list[dict],
               near_hole_img: tuple, far_hole_img: tuple,
               near_hole_r: int, far_hole_r: int) -> np.ndarray:
    """Draw bounding boxes, board crops, and hole markers on a downscaled copy."""
    scale = 0.4
    vis = cv2.resize(frame, None, fx=scale, fy=scale)

    def sc(pt):
        return (int(pt[0] * scale), int(pt[1] * scale))

    # Board crop rectangles
    cv2.rectangle(vis, sc((NEAR_X1, NEAR_Y1)), sc((NEAR_X2, NEAR_Y2)), (200, 200, 0), 2)
    cv2.rectangle(vis, sc((FAR_X1,  FAR_Y1)),  sc((FAR_X2,  FAR_Y2)),  (200, 200, 0), 2)

    # Hole markers
    cv2.circle(vis, sc(near_hole_img), int(near_hole_r * scale), (0, 255, 0), 2)
    cv2.circle(vis, sc(far_hole_img),  int(far_hole_r  * scale), (0, 255, 0), 2)
    cv2.putText(vis, "near", sc((NEAR_X1 + 5, NEAR_Y1 + 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
    cv2.putText(vis, "far", sc((FAR_X1 + 5, FAR_Y1 + 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)

    # Detections
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        color_bgr = (0, 0, 220) if d["color"] == "red" else (220, 80, 0)
        hole_tag = "●" if d.get("in_hole") else ""
        label = f"{d['color'][0].upper()}{hole_tag} {d['conf']:.2f}"
        cv2.rectangle(vis, sc((x1, y1)), sc((x2, y2)), color_bgr, 2)
        cv2.putText(vis, label, sc((x1, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_bgr, 1)

    return vis


def read_frame(video_path: str, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Cannot read frame {frame_idx}")
    return frame


def benchmark_fps(model, frame: np.ndarray, conf: float,
                  near_hole: tuple, far_hole: tuple, n: int = 20) -> float:
    """Run n inferences and return mean ms/frame."""
    times = []
    for _ in range(n):
        _, ms = run_inference(model, frame, conf, near_hole, far_hole)
        times.append(ms)
    times = times[5:]  # discard warmup
    return sum(times) / len(times)


def main():
    parser = argparse.ArgumentParser(description="YOLO-World cornhole bag detection test")
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--frames", type=int, nargs="+", default=DEFAULT_FRAMES)
    parser.add_argument("--video", default=VIDEO_PATH)
    parser.add_argument("--weights", default=WEIGHTS)
    parser.add_argument("--benchmark", action="store_true",
                        help="Measure GPU inference speed (two crops/frame)")
    parser.add_argument("--save-debug", action="store_true",
                        help="Save per-frame debug images as yolo_debug_NNNNNN.jpg")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        free, total = torch.cuda.mem_get_info()
        print(f"  VRAM: {free/1e9:.1f} GB free / {total/1e9:.1f} GB total")

    print(f"\nLoading {args.weights} ...")
    model = load_model_once(args.weights, device)
    print(f"  Model loaded, classes: {CLASSES}")

    config = load_calibration(CALIB_PATH)
    H_near = np.array(config["near_board"]["homography"], dtype=np.float64)
    H_far  = np.array(config["far_board"]["homography"],  dtype=np.float64)
    near_hole = compute_hole_center_image(
        H_near, config["hole"]["near_center_rect"][0], config["hole"]["near_center_rect"][1])
    far_hole  = compute_hole_center_image(
        H_far,  config["hole"]["far_center_rect"][0],  config["hole"]["far_center_rect"][1])
    # Hole radius in image coords: rectified radius=30px over board width 240px.
    # Near board ~300px wide in image → scale 1.25 → radius ≈ 38px (use 55 with margin).
    # Far board ~340px wide but perspective-shrunk → use 28px.
    NEAR_HOLE_R, FAR_HOLE_R = 55, 28
    print(f"  Near hole: {near_hole}  r={NEAR_HOLE_R}px")
    print(f"  Far  hole: {far_hole}   r={FAR_HOLE_R}px")

    if args.benchmark:
        print(f"\nBenchmarking 2-crop inference (frame {args.frames[0]}) ...")
        frame = read_frame(args.video, args.frames[0])
        mean_ms = benchmark_fps(model, frame, args.conf, near_hole, far_hole)
        fps_est = 1000.0 / mean_ms
        print(f"  Mean: {mean_ms:.1f} ms/frame → ~{fps_est:.0f} fps")
        print(f"  Real-time (30fps = 33ms): "
              f"{'YES' if mean_ms < 33 else 'BORDERLINE' if mean_ms < 60 else 'NO'}")

    print(f"\nCrop-based detection on {len(args.frames)} frames  conf≥{args.conf}:")
    print(f"  Near crop: ({NEAR_X1},{NEAR_Y1})→({NEAR_X2},{NEAR_Y2})  "
          f"{NEAR_X2-NEAR_X1}×{NEAR_Y2-NEAR_Y1}px")
    print(f"  Far  crop: ({FAR_X1},{FAR_Y1})→({FAR_X2},{FAR_Y2})  "
          f"{FAR_X2-FAR_X1}×{FAR_Y2-FAR_Y1}px")
    print()
    print(f"{'Frame':>8}  {'nR':>4} {'nRh':>4} {'nB':>4} {'nBh':>4}  "
          f"{'fR':>4} {'fRh':>4} {'fB':>4} {'fBh':>4}  {'ms':>7}  Detections")
    print("-" * 100)

    for fi in args.frames:
        try:
            frame = read_frame(args.video, fi)
        except RuntimeError as e:
            print(f"  {fi:>8}: {e}")
            continue

        detections, ms = run_inference(model, frame, args.conf, near_hole, far_hole,
                                       NEAR_HOLE_R, FAR_HOLE_R)
        c = count_bags(detections)

        det_str = "  ".join(
            f"{d['color'][0].upper()}{'●' if d['in_hole'] else ''}({d['conf']:.2f},{d['board']})"
            for d in detections
        )

        mins = fi // 30 // 60
        secs = (fi // 30) % 60
        print(f"{fi:>8} [{mins:02d}:{secs:02d}]  "
              f"{c['near_red']:>4} {c['near_red_hole']:>4} {c['near_blue']:>4} {c['near_blue_hole']:>4}  "
              f"{c['far_red']:>4} {c['far_red_hole']:>4} {c['far_blue']:>4} {c['far_blue_hole']:>4}  "
              f"{ms:>6.1f}ms  {det_str}")

        if args.save_debug:
            vis = draw_debug(frame, detections, near_hole, far_hole, NEAR_HOLE_R, FAR_HOLE_R)
            out_path = f"yolo_debug_{fi:06d}.jpg"
            cv2.imwrite(out_path, vis)

    # Always save one debug image
    frame = read_frame(args.video, args.frames[-2] if len(args.frames) >= 2 else args.frames[0])
    detections, _ = run_inference(model, frame, args.conf, near_hole, far_hole,
                                  NEAR_HOLE_R, FAR_HOLE_R)
    vis = draw_debug(frame, detections, near_hole, far_hole, NEAR_HOLE_R, FAR_HOLE_R)
    cv2.imwrite("yolo_debug_sample.jpg", vis)
    print(f"\nSaved yolo_debug_sample.jpg")


if __name__ == "__main__":
    main()
