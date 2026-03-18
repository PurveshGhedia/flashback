"""
ingestion/transcriber.py
=========================
Transcribes the audio track of a lecture video using OpenAI Whisper,
producing timestamped transcript chunks ready for ChromaDB indexing.

Why transcription is a first-class component:
  For lecture content, spoken words are the primary retrieval signal.
  A student searching for "backpropagation" is almost certainly looking
  for a moment the professor *said* that word — not a frame where it
  appears visually. Whisper gives us word-level timestamps so we can
  link every chunk of text back to an exact moment in the video.

Pipeline:
  1. Extract audio from video using ffmpeg (via Whisper's built-in loader)
  2. Run Whisper inference → get word/segment level timestamps
  3. Chunk the transcript into overlapping windows of ~30 seconds
     (overlap prevents a concept from being split across two chunks)
  4. Return a list of TranscriptChunk named tuples

Output format:
  Each TranscriptChunk has:
    .text       : str   — the spoken text in this chunk
    .start_time : float — chunk start in seconds
    .end_time   : float — chunk end in seconds
    .chunk_id   : int   — sequential chunk index

Chunking strategy:
  We chunk by time window (not by sentence/word count) because:
    - Time-based chunks align naturally with video timestamps
    - Students think in terms of "around the 20-minute mark", not
      "in the 400th word"
    - Overlap ensures concepts that straddle chunk boundaries are
      still retrievable
"""

import sys
import logging
from pathlib import Path
from typing import NamedTuple

sys.path.append(str(Path(__file__).parent.parent))

from config import (
    WHISPER_MODEL_SIZE,
    TRANSCRIPT_CHUNK_SECONDS,
    TRANSCRIPT_CHUNK_OVERLAP_SECONDS,
    DEVICE,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Data container
# ------------------------------------------------------------------
class TranscriptChunk(NamedTuple):
    """
    A time-bounded chunk of transcript text, ready for embedding
    and storage in ChromaDB.
    """
    text       : str    # spoken text in this chunk
    start_time : float  # seconds from video start
    end_time   : float  # seconds from video start
    chunk_id   : int    # sequential index (0-based)


# ------------------------------------------------------------------
# Whisper model loader (cached — only loads once per process)
# ------------------------------------------------------------------
_whisper_model = None


def load_whisper_model(model_size: str = WHISPER_MODEL_SIZE):
    """
    Load Whisper model. Cached globally so repeated calls don't
    reload from disk.

    Parameters
    ----------
    model_size : one of "tiny", "base", "small", "medium", "large"
                 Override via WHISPER_MODEL_SIZE env var or config.py
    """
    global _whisper_model

    if _whisper_model is not None:
        return _whisper_model

    try:
        import whisper
    except ImportError:
        raise ImportError(
            "OpenAI Whisper is required. Install with:\n"
            "  pip install openai-whisper"
        )

    logger.info(
        "Loading Whisper model '%s' on device '%s'...",
        model_size, DEVICE
    )

    _whisper_model = whisper.load_model(model_size, device=DEVICE)
    logger.info("Whisper model loaded successfully.")

    return _whisper_model


# ------------------------------------------------------------------
# Core transcription function
# ------------------------------------------------------------------
def transcribe_video(
    video_path: str | Path,
    model_size: str = WHISPER_MODEL_SIZE,
    language:   str = "en",
) -> list[dict]:
    """
    Run Whisper on a video file and return raw segments with timestamps.

    Whisper handles audio extraction internally — no need to extract
    audio as a separate step.

    Parameters
    ----------
    video_path : path to the video file
    model_size : Whisper model size (default from config)
    language   : language code — set explicitly to avoid Whisper spending
                 time on language detection for English lectures

    Returns
    -------
    List of raw Whisper segment dicts, each containing:
        {
          "id"    : int,
          "start" : float,   # seconds
          "end"   : float,   # seconds
          "text"  : str,
        }

    Note: This returns RAW segments, not chunks. Call
    chunk_transcript() on the result to get ChromaDB-ready chunks.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    model = load_whisper_model(model_size)

    logger.info(
        "Transcribing '%s' with Whisper '%s'...",
        video_path.name, model_size
    )
    logger.info(
        "Note: This may take several minutes for a long lecture on CPU. "
        "Use WHISPER_MODEL_SIZE=large on Kaggle/GPU for better accuracy."
    )

    result = model.transcribe(
        str(video_path),
        language=language,
        verbose=False,          # suppress per-segment stdout output
        word_timestamps=True,   # enables word-level timestamps
        fp16=False,             # fp16 not supported on CPU
    )

    segments = result.get("segments", [])

    logger.info(
        "Transcription complete — %d segments | "
        "detected language: %s",
        len(segments),
        result.get("language", "unknown")
    )

    return segments


# ------------------------------------------------------------------
# Chunking function
# ------------------------------------------------------------------
def chunk_transcript(
    segments       : list[dict],
    chunk_seconds  : int   = TRANSCRIPT_CHUNK_SECONDS,
    overlap_seconds: int   = TRANSCRIPT_CHUNK_OVERLAP_SECONDS,
) -> list[TranscriptChunk]:
    """
    Convert raw Whisper segments into overlapping time-window chunks
    suitable for embedding and ChromaDB storage.

    Strategy:
      - Slide a window of `chunk_seconds` across the transcript
      - Advance by (chunk_seconds - overlap_seconds) each step
      - All segments whose start_time falls within the window are
        merged into one chunk
      - Empty windows (no speech) are skipped

    Parameters
    ----------
    segments        : raw Whisper segments from transcribe_video()
    chunk_seconds   : window size in seconds (default 30s)
    overlap_seconds : overlap between consecutive windows (default 5s)

    Returns
    -------
    List of TranscriptChunk, sorted by start_time.

    Example
    -------
    With chunk_seconds=30, overlap_seconds=5, step=25:
      Chunk 0: t=0s   → t=30s
      Chunk 1: t=25s  → t=55s
      Chunk 2: t=50s  → t=80s
      ...
    """
    if not segments:
        logger.warning("chunk_transcript received empty segment list.")
        return []

    # Find the total duration from the last segment
    total_duration = max(seg["end"] for seg in segments)
    step_seconds   = chunk_seconds - overlap_seconds

    if step_seconds <= 0:
        raise ValueError(
            f"overlap_seconds ({overlap_seconds}) must be less than "
            f"chunk_seconds ({chunk_seconds})"
        )

    chunks     : list[TranscriptChunk] = []
    chunk_id   : int = 0
    window_start = 0.0

    while window_start < total_duration:
        window_end = window_start + chunk_seconds

        # Collect all segments that start within this window
        window_text_parts = []
        actual_start = None
        actual_end   = None

        for seg in segments:
            seg_start = seg["start"]
            seg_end   = seg["end"]

            # Include segment if it overlaps with the window
            if seg_start < window_end and seg_end > window_start:
                window_text_parts.append(seg["text"].strip())
                if actual_start is None:
                    actual_start = seg_start
                actual_end = seg_end

        # Skip empty windows (silence, no speech)
        if not window_text_parts:
            window_start += step_seconds
            continue

        combined_text = " ".join(window_text_parts).strip()

        chunks.append(TranscriptChunk(
            text       = combined_text,
            start_time = round(actual_start, 3),
            end_time   = round(min(actual_end, window_end), 3),
            chunk_id   = chunk_id,
        ))

        chunk_id    += 1
        window_start += step_seconds

    logger.info(
        "Chunking complete — %d chunks from %d segments "
        "(window=%ds, overlap=%ds, step=%ds)",
        len(chunks), len(segments),
        chunk_seconds, overlap_seconds, step_seconds
    )

    return chunks


# ------------------------------------------------------------------
# Convenience: transcribe + chunk in one call
# ------------------------------------------------------------------
def transcribe_and_chunk(
    video_path      : str | Path,
    model_size      : str = WHISPER_MODEL_SIZE,
    language        : str = "en",
    chunk_seconds   : int = TRANSCRIPT_CHUNK_SECONDS,
    overlap_seconds : int = TRANSCRIPT_CHUNK_OVERLAP_SECONDS,
) -> tuple[list[TranscriptChunk], list[dict]]:
    """
    Full pipeline: video → timestamped transcript chunks.

    Returns
    -------
    (chunks, raw_segments)
      chunks       : list of TranscriptChunk — ready for ChromaDB
      raw_segments : raw Whisper output — saved to disk by indexer.py
                     for debugging and re-chunking without re-running Whisper

    Example
    -------
    >>> chunks, segments = transcribe_and_chunk("lecture.mp4")
    >>> for chunk in chunks:
    ...     print(f"[{chunk.start_time:.1f}s - {chunk.end_time:.1f}s] {chunk.text[:80]}")
    """
    segments = transcribe_video(video_path, model_size, language)
    chunks   = chunk_transcript(segments, chunk_seconds, overlap_seconds)

    return chunks, segments


# ------------------------------------------------------------------
# Utility: save raw segments to JSON (so Whisper doesn't need to
# re-run if we want to experiment with different chunk sizes)
# ------------------------------------------------------------------
def save_segments_json(
    segments  : list[dict],
    output_path: str | Path,
) -> None:
    """
    Save raw Whisper segments to a JSON file.
    Useful for re-chunking experiments without re-running Whisper.
    """
    import json

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Whisper segments may contain non-serialisable objects — clean them
    clean_segments = []
    for seg in segments:
        clean_segments.append({
            "id"    : seg.get("id"),
            "start" : seg.get("start"),
            "end"   : seg.get("end"),
            "text"  : seg.get("text", "").strip(),
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(clean_segments, f, indent=2, ensure_ascii=False)

    logger.info("Saved %d segments to '%s'", len(clean_segments), output_path)


def load_segments_json(input_path: str | Path) -> list[dict]:
    """Load previously saved Whisper segments from JSON."""
    import json

    with open(input_path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    logger.info("Loaded %d segments from '%s'", len(segments), input_path)
    return segments


# ------------------------------------------------------------------
# Quick smoke test
# (run: python ingestion/transcriber.py <video_path> [max_seconds])
# ------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python transcriber.py <video_path> [max_seconds]")
        sys.exit(1)

    video_path  = sys.argv[1]
    max_seconds = int(sys.argv[2]) if len(sys.argv) > 2 else 300

    # Whisper doesn't natively support end_sec — we trim the audio using
    # a temp file for the smoke test only
    import tempfile, subprocess, os
    print(f"\nTranscribing first {max_seconds}s of '{video_path}'")
    print("=" * 60)

    # Create a trimmed audio clip for faster smoke testing
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    print(f"Trimming video to first {max_seconds}s for smoke test...")
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path,
        "-t", str(max_seconds),
        "-c", "copy", tmp_path
    ], capture_output=True)

    print("Running Whisper transcription...")
    chunks, segments = transcribe_and_chunk(
        tmp_path,
        chunk_seconds=30,
        overlap_seconds=5,
    )

    os.unlink(tmp_path)  # Clean up temp file

    print(f"\nResults:")
    print(f"  Raw segments : {len(segments)}")
    print(f"  Chunks       : {len(chunks)}")
    print(f"\nFirst 5 chunks:")
    for chunk in chunks[:5]:
        mins  = int(chunk.start_time // 60)
        secs  = int(chunk.start_time % 60)
        print(f"\n  Chunk {chunk.chunk_id} [{mins:02d}:{secs:02d} - {chunk.end_time:.1f}s]")
        print(f"  {chunk.text[:200]}{'...' if len(chunk.text) > 200 else ''}")
