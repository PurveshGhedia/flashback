"""
ingestion/frame_extractor.py
=============================
Extracts frames from a video file at a fixed FPS using OpenCV.

Responsibilities:
  - Open a video file and read its metadata (fps, total frames, duration)
  - Sample frames at FRAME_EXTRACTION_FPS (default: 1 fps)
  - Yield (frame_bgr, timestamp_seconds) tuples — lazy generator, never
    loads the entire video into memory at once
  - Optionally restrict extraction to a time range [start_sec, end_sec]
    so the ingestion pipeline can process scene segments independently

Output format:
  Each yielded item is a FrameData named tuple:
    .frame      : np.ndarray  — BGR image (OpenCV native format)
    .timestamp  : float       — seconds from the start of the video
    .frame_idx  : int         — original frame index in the video stream

Note on color space:
  OpenCV reads frames as BGR. CLIP and SSIM both expect RGB. Conversion
  happens in the respective filter modules, NOT here. This module is
  intentionally format-agnostic.
"""

import cv2
import numpy as np
from collections import namedtuple
from pathlib import Path
from typing import Generator, Optional
import logging

# Bring in config — use importlib so this module works whether run from
# the project root or from inside the ingestion/ subdirectory
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import FRAME_EXTRACTION_FPS

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Data container
# ------------------------------------------------------------------
FrameData = namedtuple("FrameData", ["frame", "timestamp", "frame_idx"])


# ------------------------------------------------------------------
# Video metadata
# ------------------------------------------------------------------
def get_video_metadata(video_path: str | Path) -> dict:
    """
    Return basic metadata for a video file without reading any frames.

    Returns
    -------
    dict with keys:
        fps           : float  — native frame rate of the video
        total_frames  : int    — total frame count
        duration_sec  : float  — total duration in seconds
        width         : int
        height        : int
        path          : str
    """
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {video_path}")

    fps           = cap.get(cv2.CAP_PROP_FPS)
    total_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width         = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec  = total_frames / fps if fps > 0 else 0.0

    cap.release()

    metadata = {
        "fps"          : fps,
        "total_frames" : total_frames,
        "duration_sec" : duration_sec,
        "width"        : width,
        "height"       : height,
        "path"         : video_path,
    }

    logger.info(
        "Video metadata — duration: %.1fs | native fps: %.2f | "
        "resolution: %dx%d | total frames: %d",
        duration_sec, fps, width, height, total_frames
    )

    return metadata


# ------------------------------------------------------------------
# Frame extraction generator
# ------------------------------------------------------------------
def extract_frames(
    video_path: str | Path,
    fps: float = FRAME_EXTRACTION_FPS,
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None,
) -> Generator[FrameData, None, None]:
    """
    Lazily yield frames sampled at `fps` from a video file.

    Parameters
    ----------
    video_path : path to the video file
    fps        : sampling rate in frames per second (default from config)
    start_sec  : if set, skip frames before this timestamp
    end_sec    : if set, stop after this timestamp

    Yields
    ------
    FrameData(frame, timestamp, frame_idx)

    Example
    -------
    >>> for fd in extract_frames("lecture.mp4"):
    ...     process(fd.frame, fd.timestamp)
    """
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {video_path}")

    native_fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec  = total_frames / native_fps if native_fps > 0 else 0.0

    # How many native frames to skip between each sampled frame
    # e.g. native 30fps, target 1fps → step = 30
    frame_step = max(1, round(native_fps / fps))

    # Resolve time bounds
    start_sec = start_sec if start_sec is not None else 0.0
    end_sec   = end_sec   if end_sec   is not None else duration_sec

    # Jump directly to the start frame to avoid reading through the whole video
    start_frame_idx = int(start_sec * native_fps)
    if start_frame_idx > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)

    current_frame_idx = start_frame_idx
    frames_yielded    = 0

    logger.info(
        "Extracting frames — target: %.1f fps | step: %d native frames | "
        "range: [%.1fs, %.1fs]",
        fps, frame_step, start_sec, end_sec
    )

    while True:
        ret, frame = cap.read()

        if not ret:
            # End of video or read error
            break

        timestamp = current_frame_idx / native_fps

        if timestamp > end_sec:
            break

        # Only yield frames that land on our sampling grid
        relative_idx = current_frame_idx - start_frame_idx
        if relative_idx % frame_step == 0:
            yield FrameData(
                frame      = frame,          # BGR ndarray
                timestamp  = round(timestamp, 3),
                frame_idx  = current_frame_idx,
            )
            frames_yielded += 1

        current_frame_idx += 1

    cap.release()
    logger.info("Frame extraction complete — yielded %d frames.", frames_yielded)


# ------------------------------------------------------------------
# Convenience: extract all frames into a list (for small videos / tests)
# ------------------------------------------------------------------
def extract_frames_list(
    video_path: str | Path,
    fps: float = FRAME_EXTRACTION_FPS,
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None,
) -> list[FrameData]:
    """
    Same as extract_frames() but returns a list instead of a generator.
    Only use this when the full frame set fits comfortably in memory.
    For a 2-hour video at 1fps that's ~7200 frames — fine for a laptop.
    """
    return list(extract_frames(video_path, fps, start_sec, end_sec))


# ------------------------------------------------------------------
# Utility: save a single frame to disk
# ------------------------------------------------------------------
def save_frame(
    frame: np.ndarray,
    output_path: str | Path,
    quality: int = 95,
) -> None:
    """
    Save a BGR frame as a JPEG.

    Parameters
    ----------
    frame       : BGR ndarray from OpenCV
    output_path : destination file path (should end in .jpg or .jpeg)
    quality     : JPEG quality 0–100
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
        str(output_path),
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, quality]
    )


# ------------------------------------------------------------------
# Quick smoke test  (run: python ingestion/frame_extractor.py <video>)
# ------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import hashlib

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python frame_extractor.py <video_path>")
        sys.exit(1)

    path = sys.argv[1]
    meta = get_video_metadata(path)
    print("\nMetadata:", meta)

    print("\nExtracting first 10 seconds...")
    frames = extract_frames_list(path, fps=1, end_sec=10)
    for fd in frames:
        h, w, _ = fd.frame.shape
        print(f"  Frame idx={fd.frame_idx:5d} | t={fd.timestamp:.2f}s | shape=({h}x{w})")

    print(f"\nExtracted {len(frames)} frames from first 10 seconds.")
