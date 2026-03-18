"""
ingestion/scene_detector.py
============================
Detects scene boundaries in a video using PySceneDetect.

Why scenes matter for this pipeline:
  SSIM filtering compares *consecutive* frames. If we do this globally,
  a hard cut from "slide A" to "slide B" will correctly survive the filter
  (SSIM will be low). But within a static slide segment, dozens of frames
  may be near-identical. By detecting scene boundaries first, we can:
    1. Reset the SSIM rolling window at each scene boundary so the
       adaptive threshold doesn't inherit statistics from a prior scene.
    2. (Future) Process each scene in parallel on a cluster.

This module wraps PySceneDetect's ContentDetector, which measures
frame-to-frame differences in the HSV color space — well-suited for
lecture videos where scene changes typically mean a new slide appearing.

Output:
  A list of SceneSegment named tuples, each with:
    .scene_id   : int   — 0-indexed scene number
    .start_sec  : float — start time in seconds
    .end_sec    : float — end time in seconds
"""

import sys
import logging
from pathlib import Path
from collections import namedtuple
from typing import Optional

# Path fix for running as a standalone script
sys.path.append(str(Path(__file__).parent.parent))
from config import SCENE_DETECTION_THRESHOLD

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Data container
# ------------------------------------------------------------------
SceneSegment = namedtuple("SceneSegment", ["scene_id", "start_sec", "end_sec"])


# ------------------------------------------------------------------
# Scene detection
# ------------------------------------------------------------------
def detect_scenes(
    video_path: str | Path,
    threshold: float = SCENE_DETECTION_THRESHOLD,
    min_scene_duration_sec: float = 2.0,
) -> list[SceneSegment]:
    """
    Detect scene boundaries and return a list of SceneSegment objects.

    Parameters
    ----------
    video_path              : path to the video file
    threshold               : ContentDetector sensitivity (higher = fewer cuts)
    min_scene_duration_sec  : discard scenes shorter than this (avoids noise
                              from rapid flashes / transitions)

    Returns
    -------
    List of SceneSegment, sorted by start time.
    Always returns at least one segment covering the full video.

    Example
    -------
    >>> scenes = detect_scenes("lecture.mp4")
    >>> for s in scenes:
    ...     print(f"Scene {s.scene_id}: {s.start_sec:.1f}s → {s.end_sec:.1f}s")
    """
    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    try:
        from scenedetect import open_video, SceneManager
        from scenedetect.detectors import ContentDetector
    except ImportError:
        raise ImportError(
            "PySceneDetect is required. Install with: pip install scenedetect"
        )

    logger.info("Running scene detection on '%s' (threshold=%.1f)...", video_path.name, threshold)

    video       = open_video(str(video_path))
    scene_mgr   = SceneManager()
    scene_mgr.add_detector(ContentDetector(threshold=threshold))

    # detect_scenes() reads the whole video — this is the expensive step
    scene_mgr.detect_scenes(video=video, show_progress=False)
    raw_scene_list = scene_mgr.get_scene_list()

    # get_scene_list() returns list of (start_timecode, end_timecode) tuples
    # Each timecode has a .get_seconds() method.
    segments = []
    for idx, (start_tc, end_tc) in enumerate(raw_scene_list):
        start_sec = start_tc.get_seconds()
        end_sec   = end_tc.get_seconds()
        duration  = end_sec - start_sec

        if duration < min_scene_duration_sec:
            logger.debug(
                "Skipping short scene %d (%.2fs < %.2fs min)",
                idx, duration, min_scene_duration_sec
            )
            continue

        segments.append(SceneSegment(
            scene_id  = len(segments),   # re-index after filtering short scenes
            start_sec = round(start_sec, 3),
            end_sec   = round(end_sec, 3),
        ))

    # Edge case: PySceneDetect found no cuts (e.g. a completely static video
    # or a very short clip). Fall back to treating the whole video as one scene.
    if not segments:
        logger.warning(
            "No scene boundaries detected. Treating entire video as one scene. "
            "Consider lowering SCENE_DETECTION_THRESHOLD if this seems wrong."
        )
        total_duration = _get_video_duration(video_path)
        segments = [SceneSegment(scene_id=0, start_sec=0.0, end_sec=total_duration)]

    logger.info(
        "Scene detection complete — %d scenes detected.", len(segments)
    )
    for seg in segments:
        logger.debug(
            "  Scene %d: %.1fs → %.1fs (%.1fs)",
            seg.scene_id, seg.start_sec, seg.end_sec, seg.end_sec - seg.start_sec
        )

    return segments


# ------------------------------------------------------------------
# Utility: look up which scene a timestamp belongs to
# ------------------------------------------------------------------
def get_scene_for_timestamp(
    timestamp_sec: float,
    scenes: list[SceneSegment],
) -> Optional[SceneSegment]:
    """
    Return the SceneSegment that contains `timestamp_sec`, or None.

    Uses linear scan — fine for typical lecture video scene counts (~10–100).
    """
    for scene in scenes:
        if scene.start_sec <= timestamp_sec < scene.end_sec:
            return scene
    # Timestamp exactly at the end of the last scene
    if scenes and timestamp_sec == scenes[-1].end_sec:
        return scenes[-1]
    return None


# ------------------------------------------------------------------
# Internal helper
# ------------------------------------------------------------------
def _get_video_duration(video_path: Path) -> float:
    """Get total duration in seconds using OpenCV (avoid re-importing scenedetect)."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return frames / fps if fps > 0 else 0.0


# ------------------------------------------------------------------
# Quick smoke test  (run: python ingestion/scene_detector.py <video>)
# ------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python scene_detector.py <video_path>")
        sys.exit(1)

    scenes = detect_scenes(sys.argv[1])
    print(f"\nDetected {len(scenes)} scenes:")
    for s in scenes:
        print(f"  Scene {s.scene_id:3d}: {s.start_sec:7.2f}s → {s.end_sec:7.2f}s "
              f"(duration: {s.end_sec - s.start_sec:.2f}s)")

    # Test lookup
    test_ts = scenes[len(scenes)//2].start_sec + 1.0
    found   = get_scene_for_timestamp(test_ts, scenes)
    print(f"\nTimestamp {test_ts:.2f}s belongs to: Scene {found.scene_id if found else 'None'}")
