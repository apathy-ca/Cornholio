#!/usr/bin/env python3
"""
Main pipeline: process one or more recorded video files for cornhole scoring.

Usage:
    python3 process_video.py [options] video1.mp4 [video2.mp4 ...]

Options:
    --calibration FILE   Path to calibration.json (default: calibration.json)
    --output DIR         Output directory for clips and logs (default: output/)
    --threshold FLOAT    Motion stillness threshold (default: 2000)
    --stillness FLOAT    Seconds of stillness to trigger round-end (default: 3.0)
    --clip-before FLOAT  Seconds of clip to capture before stillness (default: 12.0)
    --no-tts             Disable text-to-speech output
    --debug              Show OpenCV debug windows for each round
    --log FILE           JSON log output file (default: output/score_log.json)
    --start-min FLOAT    Start processing at this minute in the video
    --end-min FLOAT      Stop processing at this minute in the video
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

from motion_detector import MotionDetector, RoundEvent
from scorer import BoardScorer, score_end_frame, ClipAnalyzer, RoundScore
from score_state import ScoreState, RoundResult


def process_video(
    video_paths: list[str],
    calibration_path: str = "calibration.json",
    output_dir: str = "output",
    stillness_threshold: float = 2000.0,
    stillness_duration_sec: float = 3.0,
    clip_before_sec: float = 12.0,
    use_tts: bool = True,
    show_debug: bool = False,
    log_path: str = "output/score_log.json",
    start_min: float = 0.0,
    end_min: float = float("inf"),
):
    os.makedirs(output_dir, exist_ok=True)

    with open(calibration_path) as f:
        config = json.load(f)

    state = ScoreState()
    near_scorer = BoardScorer(config, "near_board")
    far_scorer = BoardScorer(config, "far_board")
    near_clip = ClipAnalyzer(config, "near_board")
    far_clip = ClipAnalyzer(config, "far_board")

    if use_tts:
        from tts_output import speak
    else:
        def speak(text, **kwargs):
            print(f"[TTS] {text}")

    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Cannot open {video_path}, skipping.")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start_frame = int(start_min * 60 * fps)
        end_frame = int(min(end_min * 60 * fps, total_frames)) if end_min < float("inf") else total_frames
        cap.release()

        print(f"\nProcessing: {video_path}")
        print(f"  {total_frames} frames @ {fps:.1f}fps = {total_frames/fps/60:.1f} min")
        print(f"  Range: {start_min:.1f}–{end_min if end_min < float('inf') else 'end'} min")

        detector = MotionDetector(
            stillness_threshold=stillness_threshold,
            stillness_duration_sec=stillness_duration_sec,
            clip_before_sec=clip_before_sec,
            fps=fps,
            scale=0.5,
        )

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_idx = start_frame

        while frame_idx < end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            event = detector.process_frame(frame, frame_idx)

            if event is not None:
                _handle_round_end(
                    event=event,
                    video_path=video_path,
                    config=config,
                    state=state,
                    near_scorer=near_scorer,
                    far_scorer=far_scorer,
                    near_clip=near_clip,
                    far_clip=far_clip,
                    output_dir=output_dir,
                    show_debug=show_debug,
                    speak=speak,
                )

                if state.game_over:
                    print(f"\nGame over! {state.winner.title()} wins!")
                    break

            if frame_idx % 300 == 0:
                progress = (frame_idx - start_frame) / max(1, end_frame - start_frame) * 100
                mins = int(frame_idx / fps / 60)
                secs = (frame_idx / fps) % 60
                print(f"\r  {mins:02d}:{secs:04.1f}  {progress:.0f}%  "
                      f"Rounds: {len(state.rounds)}  "
                      f"Red: {state.red_total}  Blue: {state.blue_total}",
                      end="", flush=True)

            frame_idx += 1

        cap.release()

    print(f"\n\n{state.summary()}")
    state.save_log(log_path)
    return state


def _handle_round_end(
    event: RoundEvent,
    video_path: str,
    config: dict,
    state: ScoreState,
    near_scorer: BoardScorer,
    far_scorer: BoardScorer,
    near_clip: ClipAnalyzer,
    far_clip: ClipAnalyzer,
    output_dir: str,
    show_debug: bool,
    speak,
):
    mins = int(event.timestamp_sec // 60)
    secs = event.timestamp_sec % 60
    print(f"\n\n=== Round {state.round_num} at {mins:02d}:{secs:04.1f} ===")

    # Score the end frame
    try:
        rs, near_debug, far_debug = score_end_frame(
            video_path, event.end_frame_idx, config
        )
    except Exception as e:
        print(f"  Error scoring frame: {e}")
        return

    print(f"  Static scorer: Red on={rs.red.on_board} hole={rs.red.in_hole}  "
          f"Blue on={rs.blue.on_board} hole={rs.blue.in_hole}")

    # Clip analysis for cornhole detection
    red_ch = rs.red.in_hole
    blue_ch = rs.blue.in_hole
    try:
        near_r_ch, near_b_ch = near_clip.analyze_clip(
            video_path, event.clip_start_frame, event.end_frame_idx
        )
        far_r_ch, far_b_ch = far_clip.analyze_clip(
            video_path, event.clip_start_frame, event.end_frame_idx
        )
        clip_red_ch = near_r_ch + far_r_ch
        clip_blue_ch = near_b_ch + far_b_ch
        print(f"  Clip analyzer: Red cornholes={clip_red_ch}  Blue cornholes={clip_blue_ch}")

        mismatch = (clip_red_ch != red_ch or clip_blue_ch != blue_ch)
        if mismatch:
            print(f"  WARNING: Static vs clip mismatch — using clip values")
            red_ch = clip_red_ch
            blue_ch = clip_blue_ch
    except Exception as e:
        print(f"  Clip analysis error: {e}")
        mismatch = False

    # Determine flags
    flags = []
    if rs.red.flagged:
        flags.append("Possible stacked red bags")
    if rs.blue.flagged:
        flags.append("Possible stacked blue bags")
    if mismatch:
        flags.append("Static/clip cornhole count mismatch")

    flagged = bool(flags)
    flag_reason = "; ".join(flags) if flags else ""

    result = state.record_round(
        timestamp_sec=event.timestamp_sec,
        red_cornholes=red_ch,
        red_on_board=rs.red.on_board,
        blue_cornholes=blue_ch,
        blue_on_board=rs.blue.on_board,
        flagged=flagged,
        flag_reason=flag_reason,
    )

    print(f"  Net: Red +{result.red_net}  Blue +{result.blue_net}")
    print(f"  Running: Red {result.red_total}  Blue {result.blue_total}")
    if flagged:
        print(f"  FLAG: {flag_reason}")

    # Save debug images
    round_dir = Path(output_dir) / f"round_{result.round_num:03d}"
    round_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(round_dir / "near_board.jpg"), near_debug)
    cv2.imwrite(str(round_dir / "far_board.jpg"), far_debug)

    # Optional debug display
    if show_debug:
        combined = np.hstack([near_debug, far_debug])
        combined = cv2.resize(combined, None, fx=1.5, fy=1.5)
        cv2.putText(combined, f"Round {result.round_num}: R{result.red_total} B{result.blue_total}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.imshow("Debug", combined)
        cv2.waitKey(2000)

    # TTS announcement
    text = state.announce_text(result)
    speak(text, blocking=False)


def main():
    parser = argparse.ArgumentParser(description="Cornhole video scorer")
    parser.add_argument("videos", nargs="+", help="Input video file(s)")
    parser.add_argument("--calibration", default="calibration.json")
    parser.add_argument("--output", default="output")
    parser.add_argument("--threshold", type=float, default=2000.0,
                        help="Motion stillness threshold (lower=more sensitive)")
    parser.add_argument("--stillness", type=float, default=3.0,
                        help="Seconds of stillness before round-end trigger")
    parser.add_argument("--clip-before", type=float, default=12.0,
                        help="Seconds of clip before stillness onset")
    parser.add_argument("--no-tts", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--log", default="output/score_log.json")
    parser.add_argument("--start-min", type=float, default=0.0)
    parser.add_argument("--end-min", type=float, default=float("inf"))
    args = parser.parse_args()

    if not Path(args.calibration).exists():
        print(f"Calibration file not found: {args.calibration}")
        print("Run calibrate.py first.")
        sys.exit(1)

    process_video(
        video_paths=args.videos,
        calibration_path=args.calibration,
        output_dir=args.output,
        stillness_threshold=args.threshold,
        stillness_duration_sec=args.stillness,
        clip_before_sec=args.clip_before,
        use_tts=not args.no_tts,
        show_debug=args.debug,
        log_path=args.log,
        start_min=args.start_min,
        end_min=args.end_min,
    )


if __name__ == "__main__":
    main()
