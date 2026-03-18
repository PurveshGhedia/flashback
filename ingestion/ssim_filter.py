"""
ingestion/ssim_filter.py
=========================
Stage 1 of the frame filtering cascade — Adaptive SSIM Filter.

What this module does:
  Takes the raw 1fps frame stream and removes frames that are visually
  redundant at the pixel level. "Redundant" is defined adaptively per
  video using a rolling window of SSIM scores rather than a hardcoded
  threshold.

Why adaptive?
  A hardcoded threshold (e.g. 0.95) breaks across video types:
    - A screen-captured slide deck will have SSIM ~0.99 between consecutive
      frames (almost nothing changes between seconds)
    - A shaky handheld camera recording will have SSIM ~0.70 even for
      "static" moments
  The adaptive threshold (rolling mean - std_multiplier * rolling_std)
  automatically calibrates to the natural variance of each video.

How the adaptive threshold works:
  1. Compute SSIM between every consecutive pair of frames
  2. Maintain a rolling window of the last N SSIM scores
  3. Threshold = rolling_mean - SSIM_STD_MULTIPLIER * rolling_std
  4. If SSIM(frame[i], frame[i-1]) > threshold → frame[i] is redundant → drop
  5. Hard floor (SSIM_ABSOLUTE_FLOOR) and ceiling (SSIM_ABSOLUTE_CEILING)
     override the adaptive threshold for extreme cases

Scene-awareness:
  The rolling window is RESET at scene boundaries. This prevents statistics
  from a long static slide segment from contaminating the threshold for the
  next scene.

Input:  list/generator of FrameData (from frame_extractor.py)
        list of SceneSegment (from scene_detector.py)
Output: list of FrameData that survived the filter, with filter metadata
"""

import sys
import logging
import numpy as np
from collections import deque
from pathlib import Path
from typing import Generator

# Path fix for running as standalone script
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    SSIM_ROLLING_WINDOW,
    SSIM_STD_MULTIPLIER,
    SSIM_ABSOLUTE_FLOOR,
    SSIM_ABSOLUTE_CEILING,
)
from ingestion.frame_extractor import FrameData
from ingestion.scene_detector import SceneSegment, get_scene_for_timestamp

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _to_grayscale(frame_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR frame to grayscale for SSIM computation."""
    import cv2
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)


def _compute_ssim(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """
    Compute SSIM between two BGR frames.
    Converts to grayscale first — color SSIM is slower and not meaningfully
    better for the redundancy detection use case.

    Returns a float in [0, 1]. Higher = more similar.
    """
    from skimage.metrics import structural_similarity as ssim

    gray_a = _to_grayscale(frame_a)
    gray_b = _to_grayscale(frame_b)

    score, _ = ssim(gray_a, gray_b, full=True)
    return float(score)


class _RollingStats:
    """
    Maintains a rolling window of SSIM scores and computes
    mean and std on demand.

    Resets cleanly at scene boundaries via reset().
    """

    def __init__(self, window_size: int):
        self.window_size = window_size
        self._window: deque = deque(maxlen=window_size)

    def update(self, value: float) -> None:
        self._window.append(value)

    def mean(self) -> float:
        if not self._window:
            return 0.85      # Sensible default before window fills
        return float(np.mean(self._window))

    def std(self) -> float:
        if len(self._window) < 2:
            return 0.05      # Sensible default before window fills
        return float(np.std(self._window))

    def threshold(self, std_multiplier: float) -> float:
        """Adaptive threshold = mean - std_multiplier * std."""
        return self.mean() - std_multiplier * self.std()

    def reset(self) -> None:
        """Call this at scene boundaries."""
        self._window.clear()

    def __len__(self) -> int:
        return len(self._window)


# ------------------------------------------------------------------
# Main filter function
# ------------------------------------------------------------------

def filter_frames_ssim(
    frames: list[FrameData],
    scenes: list[SceneSegment],
    rolling_window: int       = SSIM_ROLLING_WINDOW,
    std_multiplier: float     = SSIM_STD_MULTIPLIER,
    absolute_floor: float     = SSIM_ABSOLUTE_FLOOR,
    absolute_ceiling: float   = SSIM_ABSOLUTE_CEILING,
) -> tuple[list[FrameData], dict]:
    """
    Apply adaptive SSIM filtering to a list of FrameData objects.

    Parameters
    ----------
    frames          : list of FrameData from frame_extractor
    scenes          : list of SceneSegment from scene_detector
                      (used to reset rolling window at scene boundaries)
    rolling_window  : number of recent SSIM scores to use for threshold
    std_multiplier  : controls aggressiveness (higher = keep more frames)
    absolute_floor  : always KEEP a frame if SSIM drops below this
                      (catches hard cuts before window fills)
    absolute_ceiling: always DROP a frame if SSIM exceeds this
                      (handles pixel-perfect duplicates immediately)

    Returns
    -------
    (kept_frames, stats)
      kept_frames : list of FrameData that survived the filter
      stats       : dict with filter diagnostics
                    {
                      total_input      : int,
                      total_kept       : int,
                      total_dropped    : int,
                      reduction_pct    : float,
                      ssim_scores      : list[float],
                      thresholds_used  : list[float],
                      scene_resets     : int,
                    }

    Example
    -------
    >>> frames = extract_frames_list("lecture.mp4")
    >>> scenes = detect_scenes("lecture.mp4")
    >>> kept, stats = filter_frames_ssim(frames, scenes)
    >>> print(f"Kept {stats['total_kept']} / {stats['total_input']} frames")
    """

    if not frames:
        logger.warning("filter_frames_ssim received empty frame list.")
        return [], {}

    kept_frames     : list[FrameData] = []
    ssim_scores     : list[float]     = []
    thresholds_used : list[float]     = []
    scene_resets    : int             = 0

    rolling = _RollingStats(window_size=rolling_window)
    prev_frame_data : FrameData | None = None
    current_scene_id: int | None       = None

    logger.info(
        "SSIM filter starting — %d input frames | window=%d | "
        "std_mult=%.2f | floor=%.2f | ceiling=%.2f",
        len(frames), rolling_window, std_multiplier,
        absolute_floor, absolute_ceiling
    )

    for frame_data in frames:

        # ----------------------------------------------------------------
        # Scene boundary check — reset rolling window if scene changed
        # ----------------------------------------------------------------
        scene = get_scene_for_timestamp(frame_data.timestamp, scenes)
        scene_id = scene.scene_id if scene else -1

        if scene_id != current_scene_id:
            if current_scene_id is not None:
                # Genuine scene change (not just the first frame)
                rolling.reset()
                scene_resets += 1
                logger.debug(
                    "Scene boundary at t=%.2fs — rolling window reset "
                    "(scene %d → %d)",
                    frame_data.timestamp, current_scene_id, scene_id
                )
            current_scene_id = scene_id

        # ----------------------------------------------------------------
        # Always keep the first frame of the video / each scene
        # ----------------------------------------------------------------
        if prev_frame_data is None:
            kept_frames.append(frame_data)
            prev_frame_data = frame_data
            logger.debug("t=%.2fs — KEPT (first frame)", frame_data.timestamp)
            continue

        # ----------------------------------------------------------------
        # Compute SSIM against the last KEPT frame
        # (comparing against last kept, not last seen, avoids drift)
        # ----------------------------------------------------------------
        ssim_score = _compute_ssim(prev_frame_data.frame, frame_data.frame)
        ssim_scores.append(ssim_score)

        # Update rolling stats BEFORE computing threshold so the current
        # frame's similarity informs future decisions
        rolling.update(ssim_score)
        adaptive_threshold = rolling.threshold(std_multiplier)
        thresholds_used.append(adaptive_threshold)

        # ----------------------------------------------------------------
        # Decision logic (order matters — absolute rules first)
        # ----------------------------------------------------------------
        if ssim_score < absolute_floor:
            # Hard scene cut or major content change — always keep
            decision = "KEPT (below floor)"
            keep = True

        elif ssim_score > absolute_ceiling:
            # Pixel-perfect duplicate — always drop
            decision = "DROPPED (above ceiling)"
            keep = False

        elif ssim_score > adaptive_threshold:
            # Above adaptive threshold — redundant, drop
            decision = f"DROPPED (ssim={ssim_score:.4f} > thresh={adaptive_threshold:.4f})"
            keep = False

        else:
            # Below adaptive threshold — meaningful change, keep
            decision = f"KEPT (ssim={ssim_score:.4f} <= thresh={adaptive_threshold:.4f})"
            keep = True

        logger.debug("t=%.2fs — %s", frame_data.timestamp, decision)

        if keep:
            kept_frames.append(frame_data)
            prev_frame_data = frame_data   # Update reference to last KEPT frame

    # ----------------------------------------------------------------
    # Build stats
    # ----------------------------------------------------------------
    total_input   = len(frames)
    total_kept    = len(kept_frames)
    total_dropped = total_input - total_kept
    reduction_pct = (total_dropped / total_input * 100) if total_input > 0 else 0.0

    stats = {
        "total_input"     : total_input,
        "total_kept"      : total_kept,
        "total_dropped"   : total_dropped,
        "reduction_pct"   : round(reduction_pct, 2),
        "ssim_scores"     : ssim_scores,
        "thresholds_used" : thresholds_used,
        "scene_resets"    : scene_resets,
        "mean_ssim"       : round(float(np.mean(ssim_scores)), 4) if ssim_scores else 0.0,
        "std_ssim"        : round(float(np.std(ssim_scores)), 4)  if ssim_scores else 0.0,
    }

    logger.info(
        "SSIM filter complete — kept %d / %d frames (%.1f%% reduction) | "
        "%d scene resets | mean SSIM=%.4f",
        total_kept, total_input, reduction_pct,
        scene_resets, stats["mean_ssim"]
    )

    return kept_frames, stats


# ------------------------------------------------------------------
# Quick smoke test
# (run: python ingestion/ssim_filter.py <video_path> [max_seconds])
# ------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python ssim_filter.py <video_path> [max_seconds]")
        sys.exit(1)

    video_path = sys.argv[1]
    max_seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 120.0

    print(f"\nTesting SSIM filter on first {max_seconds}s of '{video_path}'")
    print("=" * 60)

    from ingestion.frame_extractor import extract_frames_list
    from ingestion.scene_detector import detect_scenes

    print("Step 1: Extracting frames...")
    frames = extract_frames_list(video_path, fps=1, end_sec=max_seconds)
    print(f"  Extracted {len(frames)} frames")

    print("\nStep 2: Detecting scenes...")
    scenes = detect_scenes(video_path)
    relevant_scenes = [s for s in scenes if s.start_sec < max_seconds]
    print(f"  Found {len(relevant_scenes)} scenes in first {max_seconds}s")

    print("\nStep 3: Applying SSIM filter...")
    kept, stats = filter_frames_ssim(frames, relevant_scenes)

    print(f"\nResults:")
    print(f"  Input frames  : {stats['total_input']}")
    print(f"  Kept frames   : {stats['total_kept']}")
    print(f"  Dropped       : {stats['total_dropped']}")
    print(f"  Reduction     : {stats['reduction_pct']}%")
    print(f"  Mean SSIM     : {stats['mean_ssim']}")
    print(f"  Std SSIM      : {stats['std_ssim']}")
    print(f"  Scene resets  : {stats['scene_resets']}")

    print(f"\nKept frame timestamps:")
    for fd in kept:
        print(f"  t={fd.timestamp:.2f}s  (frame_idx={fd.frame_idx})")
