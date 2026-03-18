"""
ingestion/clip_filter.py
=========================
Stage 2 of the frame filtering cascade — CLIP Semantic Filter.

What this module does:
  Takes the keyframes that survived the SSIM filter (Stage 1) and removes
  any that are still semantically redundant even if their pixels differ.

Why do we need this after SSIM?
  SSIM is a pixel-level metric. It can miss cases like:
    - The professor moves slightly but the slide content is identical
    - A camera zoom/pan changes pixel values but not slide content
    - Two different slides that look visually similar (same layout, similar colors)
      but have different text — CLIP catches the text semantics, SSIM doesn't
  CLIP operates in a high-dimensional semantic embedding space, so it
  filters based on *meaning* rather than *appearance*.

How it works:
  1. Load CLIP ViT-B/32 locally (runs on CPU or GPU)
  2. For each frame, compute a 512-dim CLIP image embedding
  3. Compare consecutive frame embeddings using cosine similarity
  4. Drop frames whose embedding is too similar to the previous kept frame
  5. Store the embeddings — they are reused during ChromaDB indexing (no
     need to re-run CLIP inference at index time)

Key design decision:
  We compare against the last KEPT frame's embedding (same as SSIM filter),
  not the last seen frame. This prevents semantic drift.

Input:  list of FrameData that survived ssim_filter
Output: list of FrameData that survived clip_filter, with their CLIP
        embeddings attached as a parallel list
"""

import sys
import logging
import numpy as np
from pathlib import Path
from typing import NamedTuple

sys.path.append(str(Path(__file__).parent.parent))

from config import (
    CLIP_MODEL_NAME,
    CLIP_SIMILARITY_THRESHOLD,
    DEVICE,
)
from ingestion.frame_extractor import FrameData

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Data container — FrameData + its CLIP embedding
# ------------------------------------------------------------------
class KeyFrame(NamedTuple):
    """
    A frame that has survived both SSIM and CLIP filtering.
    Carries its CLIP embedding so indexer.py doesn't need to
    re-run inference.
    """
    frame      : np.ndarray   # BGR image (OpenCV format)
    timestamp  : float        # seconds from video start
    frame_idx  : int          # original frame index in video stream
    embedding  : np.ndarray   # 512-dim CLIP image embedding (L2-normalised)


# ------------------------------------------------------------------
# CLIP model loader (cached — only loads once per process)
# ------------------------------------------------------------------
_clip_model  = None
_clip_preprocess = None


def load_clip_model():
    """
    Load CLIP ViT-B/32 model and preprocessing pipeline.
    Cached globally so repeated calls don't reload from disk.
    """
    global _clip_model, _clip_preprocess

    if _clip_model is not None:
        return _clip_model, _clip_preprocess

    try:
        import clip
    except ImportError:
        raise ImportError(
            "OpenAI CLIP is required. Install with:\n"
            "  pip install git+https://github.com/openai/CLIP.git"
        )

    logger.info(
        "Loading CLIP model '%s' on device '%s'...",
        CLIP_MODEL_NAME, DEVICE
    )

    _clip_model, _clip_preprocess = clip.load(CLIP_MODEL_NAME, device=DEVICE)
    _clip_model.eval()  # Inference only — no gradient tracking needed

    logger.info("CLIP model loaded successfully.")
    return _clip_model, _clip_preprocess


# ------------------------------------------------------------------
# Embedding helpers
# ------------------------------------------------------------------
def embed_frame(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Compute a L2-normalised CLIP embedding for a single BGR frame.

    Parameters
    ----------
    frame_bgr : np.ndarray — BGR image from OpenCV

    Returns
    -------
    np.ndarray of shape (512,) — L2-normalised embedding
    """
    import torch
    import cv2
    from PIL import Image

    model, preprocess = load_clip_model()

    # OpenCV BGR → RGB → PIL Image (CLIP expects PIL)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)

    # Preprocess and move to device
    image_tensor = preprocess(pil_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        embedding = model.encode_image(image_tensor)
        # L2 normalise so cosine similarity = dot product
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)

    return embedding.cpu().numpy().squeeze()   # shape: (512,)


def embed_frames_batch(
    frames_bgr: list[np.ndarray],
    batch_size: int = 32,
) -> np.ndarray:
    """
    Compute CLIP embeddings for a list of BGR frames in batches.
    More efficient than calling embed_frame() in a loop for large lists.

    Parameters
    ----------
    frames_bgr : list of BGR frames
    batch_size : number of frames to process per forward pass
                 (reduce if running out of memory on CPU)

    Returns
    -------
    np.ndarray of shape (N, 512) — one L2-normalised embedding per frame
    """
    import torch
    import cv2
    from PIL import Image

    model, preprocess = load_clip_model()

    all_embeddings = []

    for i in range(0, len(frames_bgr), batch_size):
        batch_bgr = frames_bgr[i : i + batch_size]

        tensors = []
        for frame_bgr in batch_bgr:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            tensors.append(preprocess(pil_image))

        batch_tensor = torch.stack(tensors).to(DEVICE)

        with torch.no_grad():
            embeddings = model.encode_image(batch_tensor)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        all_embeddings.append(embeddings.cpu().numpy())

        logger.debug(
            "Embedded batch %d/%d (%d frames)",
            i // batch_size + 1,
            (len(frames_bgr) + batch_size - 1) // batch_size,
            len(batch_bgr)
        )

    return np.vstack(all_embeddings)   # shape: (N, 512)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two L2-normalised vectors.
    Since both are normalised, this is just the dot product.
    Returns a float in [-1, 1]. Higher = more similar.
    """
    return float(np.dot(a, b))


# ------------------------------------------------------------------
# Main filter function
# ------------------------------------------------------------------
def filter_frames_clip(
    frames: list[FrameData],
    similarity_threshold: float = CLIP_SIMILARITY_THRESHOLD,
    batch_size: int = 32,
) -> tuple[list[KeyFrame], dict]:
    """
    Apply CLIP semantic filtering to frames that survived SSIM filtering.

    Parameters
    ----------
    frames               : list of FrameData from ssim_filter
    similarity_threshold : cosine similarity above which a frame is
                           considered semantically redundant and dropped.
                           Default 0.95 — only drop near-identical semantics.
    batch_size           : frames per CLIP forward pass (reduce if OOM)

    Returns
    -------
    (keyframes, stats)
      keyframes : list of KeyFrame (FrameData + embedding)
      stats     : dict with filter diagnostics
                  {
                    total_input          : int,
                    total_kept           : int,
                    total_dropped        : int,
                    reduction_pct        : float,
                    similarity_scores    : list[float],
                    mean_similarity      : float,
                    std_similarity       : float,
                  }

    Example
    -------
    >>> ssim_frames, _ = filter_frames_ssim(raw_frames, scenes)
    >>> keyframes, stats = filter_frames_clip(ssim_frames)
    >>> print(f"Final keyframes: {len(keyframes)}")
    """

    if not frames:
        logger.warning("filter_frames_clip received empty frame list.")
        return [], {}

    logger.info(
        "CLIP filter starting — %d input frames | "
        "similarity_threshold=%.3f | device=%s",
        len(frames), similarity_threshold, DEVICE
    )

    # ----------------------------------------------------------------
    # Step 1: Embed all frames in batches (more efficient than one-by-one)
    # ----------------------------------------------------------------
    logger.info("Computing CLIP embeddings for %d frames...", len(frames))
    all_frame_arrays = [fd.frame for fd in frames]
    all_embeddings   = embed_frames_batch(all_frame_arrays, batch_size=batch_size)
    logger.info("Embeddings computed. Shape: %s", all_embeddings.shape)

    # ----------------------------------------------------------------
    # Step 2: Filter based on cosine similarity to last kept frame
    # ----------------------------------------------------------------
    kept_keyframes      : list[KeyFrame] = []
    similarity_scores   : list[float]    = []
    prev_embedding      : np.ndarray | None = None

    for i, frame_data in enumerate(frames):
        embedding = all_embeddings[i]

        # Always keep the first frame
        if prev_embedding is None:
            kept_keyframes.append(KeyFrame(
                frame     = frame_data.frame,
                timestamp = frame_data.timestamp,
                frame_idx = frame_data.frame_idx,
                embedding = embedding,
            ))
            prev_embedding = embedding
            logger.debug("t=%.2fs — KEPT (first frame)", frame_data.timestamp)
            continue

        sim = cosine_similarity(embedding, prev_embedding)
        similarity_scores.append(sim)

        if sim >= similarity_threshold:
            logger.debug(
                "t=%.2fs — DROPPED (cosine=%.4f >= threshold=%.4f)",
                frame_data.timestamp, sim, similarity_threshold
            )
        else:
            kept_keyframes.append(KeyFrame(
                frame     = frame_data.frame,
                timestamp = frame_data.timestamp,
                frame_idx = frame_data.frame_idx,
                embedding = embedding,
            ))
            prev_embedding = embedding
            logger.debug(
                "t=%.2fs — KEPT (cosine=%.4f < threshold=%.4f)",
                frame_data.timestamp, sim, similarity_threshold
            )

    # ----------------------------------------------------------------
    # Step 3: Build stats
    # ----------------------------------------------------------------
    total_input   = len(frames)
    total_kept    = len(kept_keyframes)
    total_dropped = total_input - total_kept
    reduction_pct = (total_dropped / total_input * 100) if total_input > 0 else 0.0

    stats = {
        "total_input"       : total_input,
        "total_kept"        : total_kept,
        "total_dropped"     : total_dropped,
        "reduction_pct"     : round(reduction_pct, 2),
        "similarity_scores" : similarity_scores,
        "mean_similarity"   : round(float(np.mean(similarity_scores)), 4) if similarity_scores else 0.0,
        "std_similarity"    : round(float(np.std(similarity_scores)), 4)  if similarity_scores else 0.0,
    }

    logger.info(
        "CLIP filter complete — kept %d / %d frames (%.1f%% reduction) | "
        "mean cosine similarity=%.4f",
        total_kept, total_input, reduction_pct, stats["mean_similarity"]
    )

    return kept_keyframes, stats


# ------------------------------------------------------------------
# Utility: embed a text query (used during retrieval)
# ------------------------------------------------------------------
def embed_text(text: str) -> np.ndarray:
    """
    Compute a L2-normalised CLIP text embedding for a query string.
    Used by retrieval/embedder.py to embed user queries into the same
    space as the frame embeddings stored in ChromaDB.

    Parameters
    ----------
    text : natural language query string

    Returns
    -------
    np.ndarray of shape (512,) — L2-normalised text embedding
    """
    import torch
    import clip

    model, _ = load_clip_model()
    text_tokens = clip.tokenize([text]).to(DEVICE)

    with torch.no_grad():
        embedding = model.encode_text(text_tokens)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)

    return embedding.cpu().numpy().squeeze()   # shape: (512,)


# ------------------------------------------------------------------
# Quick smoke test
# (run: python ingestion/clip_filter.py <video_path> [max_seconds])
# ------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python clip_filter.py <video_path> [max_seconds]")
        sys.exit(1)

    video_path  = sys.argv[1]
    max_seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 120.0

    print(f"\nTesting CLIP filter on first {max_seconds}s of '{video_path}'")
    print("=" * 60)

    from ingestion.frame_extractor import extract_frames_list
    from ingestion.scene_detector import detect_scenes
    from ingestion.ssim_filter import filter_frames_ssim

    print("Step 1: Extracting frames...")
    frames = extract_frames_list(video_path, fps=1, end_sec=max_seconds)
    print(f"  Extracted {len(frames)} raw frames")

    print("\nStep 2: Detecting scenes...")
    scenes = detect_scenes(video_path)
    print(f"  Found {len(scenes)} scenes total")

    print("\nStep 3: SSIM filter (Stage 1)...")
    ssim_kept, ssim_stats = filter_frames_ssim(frames, scenes)
    print(f"  SSIM kept {ssim_stats['total_kept']} / {ssim_stats['total_input']} "
          f"({ssim_stats['reduction_pct']}% reduction)")

    print("\nStep 4: CLIP filter (Stage 2)...")
    keyframes, clip_stats = filter_frames_clip(ssim_kept)

    print(f"\nResults:")
    print(f"  After SSIM  : {ssim_stats['total_kept']} frames")
    print(f"  After CLIP  : {clip_stats['total_kept']} frames")
    print(f"  CLIP reduction : {clip_stats['reduction_pct']}%")
    print(f"  Mean cosine similarity : {clip_stats['mean_similarity']}")
    print(f"  Std cosine similarity  : {clip_stats['std_similarity']}")

    print(f"\nFinal keyframe timestamps:")
    for kf in keyframes:
        print(f"  t={kf.timestamp:.2f}s  (embedding shape: {kf.embedding.shape})")

    # Test text embedding
    print(f"\nTesting text embedding...")
    text_emb = embed_text("backpropagation neural network")
    print(f"  Text embedding shape  : {text_emb.shape}")
    print(f"  Text embedding norm   : {np.linalg.norm(text_emb):.4f}  (should be ~1.0)")
