"""
ingestion/indexer.py
=====================
The final stage of ingestion — orchestrates the full pipeline and
writes everything into ChromaDB.

What this module does:
  1. Accepts a video file path
  2. Runs frame extraction → SSIM filter → CLIP filter
  3. Runs Whisper transcription → chunking
  4. Saves keyframe images to data/frames/<video_hash>/
  5. Saves raw Whisper segments to data/transcripts/<video_hash>.json
  6. Writes two ChromaDB collections:
       <video_hash>_frames      — CLIP embeddings + frame metadata
       <video_hash>_transcript  — MiniLM embeddings + chunk metadata
  7. Returns an IndexResult summary

ChromaDB collection schema:

  frames collection:
    id        : "<video_hash>_frame_<frame_idx>"
    embedding : 512-dim CLIP image embedding (float32)
    document  : "" (no text — visual only)
    metadata  : {
      video_hash  : str,
      timestamp   : float,   ← seconds from video start
      frame_idx   : int,
      frame_path  : str,     ← relative path to saved JPEG
    }

  transcript collection:
    id        : "<video_hash>_chunk_<chunk_id>"
    embedding : 384-dim MiniLM text embedding (float32)
    document  : chunk text (used by ChromaDB for BM25 if needed)
    metadata  : {
      video_hash  : str,
      start_time  : float,   ← seconds from video start
      end_time    : float,
      chunk_id    : int,
    }

Video hash:
  We use the first 8 chars of MD5(video filename + file size) as a
  short, stable identifier. This namespaces ChromaDB collections and
  frame directories so multiple videos can coexist.
"""

import sys
import json
import hashlib
import logging
from pathlib import Path
from typing import NamedTuple

sys.path.append(str(Path(__file__).parent.parent))

from config import (
    FRAMES_DIR,
    TRANSCRIPTS_DIR,
    CHROMA_DIR,
    CHROMA_FRAMES_COLLECTION_SUFFIX,
    CHROMA_TRANSCRIPT_COLLECTION_SUFFIX,
    TEXT_EMBEDDING_MODEL,
    FRAME_EXTRACTION_FPS,
    DEVICE,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Data container for indexing result
# ------------------------------------------------------------------
class IndexResult(NamedTuple):
    """Summary of a completed indexing run."""
    video_hash          : str
    video_path          : str
    frames_collection   : str   # ChromaDB collection name
    transcript_collection: str  # ChromaDB collection name
    total_raw_frames    : int
    total_keyframes     : int   # after SSIM + CLIP filtering
    total_chunks        : int   # transcript chunks
    frames_dir          : str   # where keyframe JPEGs are saved
    transcript_path     : str   # where raw segments JSON is saved
    ssim_reduction_pct  : float
    clip_reduction_pct  : float


# ------------------------------------------------------------------
# Video hash
# ------------------------------------------------------------------
def get_video_hash(video_path: str | Path) -> str:
    """
    Generate a short stable hash for a video file.
    Uses filename + file size — fast (no need to hash file contents).
    Returns first 8 hex characters, e.g. "a3f2c891"
    """
    video_path = Path(video_path)
    raw = f"{video_path.name}_{video_path.stat().st_size}"
    return hashlib.md5(raw.encode()).hexdigest()[:8]


# ------------------------------------------------------------------
# ChromaDB helpers
# ------------------------------------------------------------------
def get_chroma_client():
    """Return a persistent ChromaDB client pointed at CHROMA_DIR."""
    try:
        import chromadb
    except ImportError:
        raise ImportError("ChromaDB is required: pip install chromadb")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client


def get_or_create_collection(client, name: str):
    """
    Get a ChromaDB collection by name, creating it if it doesn't exist.
    Uses cosine similarity (appropriate for both CLIP and MiniLM embeddings).
    """
    import chromadb.utils.embedding_functions as ef

    collection = client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},  # cosine similarity for retrieval
    )
    return collection


# ------------------------------------------------------------------
# Text embedding function (MiniLM)
# ------------------------------------------------------------------
_text_embedder = None


def get_text_embedder():
    """
    Load sentence-transformers MiniLM model. Cached globally.
    Returns a callable that takes a list of strings and returns
    a list of 384-dim embeddings.
    """
    global _text_embedder

    if _text_embedder is not None:
        return _text_embedder

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers is required: pip install sentence-transformers"
        )

    logger.info("Loading text embedding model '%s'...", TEXT_EMBEDDING_MODEL)
    model = SentenceTransformer(TEXT_EMBEDDING_MODEL, device=DEVICE)
    _text_embedder = model
    logger.info("Text embedding model loaded.")

    return _text_embedder


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of text strings using MiniLM.
    Returns list of 384-dim float lists (ChromaDB expects plain lists).
    """
    model = get_text_embedder()
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=False,
        normalize_embeddings=True,   # L2 normalise for cosine similarity
    )
    return embeddings.tolist()


# ------------------------------------------------------------------
# Frame saving helper
# ------------------------------------------------------------------
def save_keyframe(
    frame,           # np.ndarray BGR
    video_hash: str,
    frame_idx: int,
    timestamp: float,
) -> str:
    """
    Save a keyframe JPEG to data/frames/<video_hash>/.
    Returns the relative path string for storage in ChromaDB metadata.
    """
    import cv2

    frame_dir = FRAMES_DIR / video_hash
    frame_dir.mkdir(parents=True, exist_ok=True)

    filename    = f"frame_{frame_idx:06d}_t{timestamp:.2f}.jpg"
    output_path = frame_dir / filename

    cv2.imwrite(
        str(output_path),
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, 95]
    )

    # Return relative path (portable across machines)
    return str(output_path.relative_to(Path(__file__).parent.parent))


# ------------------------------------------------------------------
# Main indexing function
# ------------------------------------------------------------------
def index_video(
    video_path         : str | Path,
    force_reindex      : bool = False,
    whisper_language   : str  = "en",
) -> IndexResult:
    """
    Full ingestion pipeline: video → ChromaDB.

    Parameters
    ----------
    video_path       : path to the lecture video file
    force_reindex    : if True, re-process even if collections exist.
                       If False and collections already exist, skip and
                       return existing index info.
    whisper_language : language code for Whisper (default "en")

    Returns
    -------
    IndexResult with summary of the indexing run.

    Example
    -------
    >>> result = index_video("data/videos/lecture.mp4")
    >>> print(f"Indexed {result.total_keyframes} keyframes and "
    ...       f"{result.total_chunks} transcript chunks")
    """
    import numpy as np

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    video_hash = get_video_hash(video_path)
    frames_collection_name     = f"{video_hash}{CHROMA_FRAMES_COLLECTION_SUFFIX}"
    transcript_collection_name = f"{video_hash}{CHROMA_TRANSCRIPT_COLLECTION_SUFFIX}"

    logger.info(
        "Starting indexing — video: '%s' | hash: %s",
        video_path.name, video_hash
    )

    # ----------------------------------------------------------------
    # Check if already indexed
    # ----------------------------------------------------------------
    client = get_chroma_client()

    existing_collections = [c.name for c in client.list_collections()]

    if not force_reindex and frames_collection_name in existing_collections:
        logger.info(
            "Video '%s' already indexed (hash: %s). "
            "Use force_reindex=True to re-process.",
            video_path.name, video_hash
        )
        frames_col     = client.get_collection(frames_collection_name)
        transcript_col = client.get_collection(transcript_collection_name)

        return IndexResult(
            video_hash             = video_hash,
            video_path             = str(video_path),
            frames_collection      = frames_collection_name,
            transcript_collection  = transcript_collection_name,
            total_raw_frames       = -1,   # unknown (not re-processed)
            total_keyframes        = frames_col.count(),
            total_chunks           = transcript_col.count(),
            frames_dir             = str(FRAMES_DIR / video_hash),
            transcript_path        = str(TRANSCRIPTS_DIR / f"{video_hash}.json"),
            ssim_reduction_pct     = -1.0,
            clip_reduction_pct     = -1.0,
        )

    # ----------------------------------------------------------------
    # Step 1: Scene detection
    # ----------------------------------------------------------------
    logger.info("Step 1/5 — Detecting scenes...")
    from ingestion.scene_detector import detect_scenes
    scenes = detect_scenes(str(video_path))
    logger.info("  %d scenes detected.", len(scenes))

    # ----------------------------------------------------------------
    # Step 2: Frame extraction + SSIM filter
    # ----------------------------------------------------------------
    logger.info("Step 2/5 — Extracting and filtering frames (SSIM)...")
    from ingestion.frame_extractor import extract_frames_list
    from ingestion.ssim_filter import filter_frames_ssim

    raw_frames = extract_frames_list(str(video_path), fps=FRAME_EXTRACTION_FPS)
    logger.info("  Extracted %d raw frames.", len(raw_frames))

    ssim_frames, ssim_stats = filter_frames_ssim(raw_frames, scenes)
    logger.info(
        "  SSIM filter: %d → %d frames (%.1f%% reduction)",
        ssim_stats["total_input"],
        ssim_stats["total_kept"],
        ssim_stats["reduction_pct"],
    )

    # Free raw frames from memory — they're large
    del raw_frames

    # ----------------------------------------------------------------
    # Step 3: CLIP filter
    # ----------------------------------------------------------------
    logger.info("Step 3/5 — Applying CLIP semantic filter...")
    from ingestion.clip_filter import filter_frames_clip

    keyframes, clip_stats = filter_frames_clip(ssim_frames)
    logger.info(
        "  CLIP filter: %d → %d frames (%.1f%% reduction)",
        clip_stats["total_input"],
        clip_stats["total_kept"],
        clip_stats["reduction_pct"],
    )

    del ssim_frames  # Free memory

    # ----------------------------------------------------------------
    # Step 4: Whisper transcription
    # ----------------------------------------------------------------
    logger.info("Step 4/5 — Transcribing audio with Whisper...")
    from ingestion.transcriber import transcribe_and_chunk, save_segments_json

    chunks, raw_segments = transcribe_and_chunk(
        video_path,
        language=whisper_language,
    )

    # Save raw segments so Whisper doesn't need to re-run
    transcript_path = TRANSCRIPTS_DIR / f"{video_hash}.json"
    save_segments_json(raw_segments, transcript_path)
    logger.info(
        "  Transcription: %d segments → %d chunks | saved to %s",
        len(raw_segments), len(chunks), transcript_path
    )

    # ----------------------------------------------------------------
    # Step 5: Index into ChromaDB
    # ----------------------------------------------------------------
    logger.info("Step 5/5 — Indexing into ChromaDB...")

    frames_col     = get_or_create_collection(client, frames_collection_name)
    transcript_col = get_or_create_collection(client, transcript_collection_name)

    # -- 5a: Index keyframes --
    logger.info("  Indexing %d keyframes...", len(keyframes))

    frame_ids        = []
    frame_embeddings = []
    frame_metadatas  = []
    frame_documents  = []

    for kf in keyframes:
        # Save JPEG to disk
        frame_path = save_keyframe(kf.frame, video_hash, kf.frame_idx, kf.timestamp)

        frame_id = f"{video_hash}_frame_{kf.frame_idx:06d}"
        frame_ids.append(frame_id)
        frame_embeddings.append(kf.embedding.tolist())   # ChromaDB needs list
        frame_documents.append("")                        # No text for frames
        frame_metadatas.append({
            "video_hash" : video_hash,
            "timestamp"  : kf.timestamp,
            "frame_idx"  : kf.frame_idx,
            "frame_path" : frame_path,
        })

    # Upsert in batches of 100 (ChromaDB recommended batch size)
    _batch_upsert(frames_col, frame_ids, frame_embeddings,
                  frame_documents, frame_metadatas, batch_size=100)

    logger.info("  Keyframes indexed: %d", len(keyframes))

    # -- 5b: Index transcript chunks --
    logger.info("  Embedding and indexing %d transcript chunks...", len(chunks))

    chunk_texts = [c.text for c in chunks]
    chunk_embeddings = embed_texts(chunk_texts)

    chunk_ids       = []
    chunk_metadatas = []

    for chunk in chunks:
        chunk_ids.append(f"{video_hash}_chunk_{chunk.chunk_id:04d}")
        chunk_metadatas.append({
            "video_hash" : video_hash,
            "start_time" : chunk.start_time,
            "end_time"   : chunk.end_time,
            "chunk_id"   : chunk.chunk_id,
        })

    _batch_upsert(
        transcript_col,
        chunk_ids,
        chunk_embeddings,
        chunk_texts,       # store text as document for potential BM25
        chunk_metadatas,
        batch_size=100,
    )

    logger.info("  Transcript chunks indexed: %d", len(chunks))

    # ----------------------------------------------------------------
    # Done
    # ----------------------------------------------------------------
    result = IndexResult(
        video_hash             = video_hash,
        video_path             = str(video_path),
        frames_collection      = frames_collection_name,
        transcript_collection  = transcript_collection_name,
        total_raw_frames       = ssim_stats["total_input"],
        total_keyframes        = len(keyframes),
        total_chunks           = len(chunks),
        frames_dir             = str(FRAMES_DIR / video_hash),
        transcript_path        = str(transcript_path),
        ssim_reduction_pct     = ssim_stats["reduction_pct"],
        clip_reduction_pct     = clip_stats["reduction_pct"],
    )

    logger.info(
        "Indexing complete — hash: %s | keyframes: %d | chunks: %d",
        video_hash, result.total_keyframes, result.total_chunks,
    )

    return result


# ------------------------------------------------------------------
# Internal: batch upsert helper
# ------------------------------------------------------------------
def _batch_upsert(
    collection,
    ids        : list[str],
    embeddings : list[list[float]],
    documents  : list[str],
    metadatas  : list[dict],
    batch_size : int = 100,
) -> None:
    """Upsert records into a ChromaDB collection in batches."""
    total = len(ids)
    for i in range(0, total, batch_size):
        batch_slice = slice(i, i + batch_size)
        collection.upsert(
            ids        = ids[batch_slice],
            embeddings = embeddings[batch_slice],
            documents  = documents[batch_slice],
            metadatas  = metadatas[batch_slice],
        )
        logger.debug(
            "Upserted batch %d/%d (%d records)",
            i // batch_size + 1,
            (total + batch_size - 1) // batch_size,
            len(ids[batch_slice]),
        )


# ------------------------------------------------------------------
# Quick smoke test
# (run: python ingestion/indexer.py <video_path>)
# ------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s"
    )

    if len(sys.argv) < 2:
        print("Usage: python indexer.py <video_path> [--force]")
        sys.exit(1)

    video_path   = sys.argv[1]
    force        = "--force" in sys.argv

    print(f"\nIndexing '{video_path}' {'(force reindex)' if force else ''}")
    print("=" * 60)
    print("Warning: Full indexing of an 80-min lecture takes ~20-30 min on CPU.")
    print("This smoke test runs on the full video.\n")

    result = index_video(video_path, force_reindex=force)

    print(f"\n{'='*60}")
    print(f"Index Summary")
    print(f"{'='*60}")
    print(f"  Video hash            : {result.video_hash}")
    print(f"  Raw frames extracted  : {result.total_raw_frames}")
    print(f"  Keyframes (after SSIM): after {result.ssim_reduction_pct}% reduction")
    print(f"  Keyframes (after CLIP): {result.total_keyframes} final keyframes")
    print(f"  CLIP reduction        : {result.clip_reduction_pct}%")
    print(f"  Transcript chunks     : {result.total_chunks}")
    print(f"  Frames saved to       : {result.frames_dir}")
    print(f"  Transcript saved to   : {result.transcript_path}")
    print(f"  ChromaDB frames       : {result.frames_collection}")
    print(f"  ChromaDB transcript   : {result.transcript_collection}")
    print(f"\n  ✓ Ready for retrieval.")
