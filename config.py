"""
config.py
=========
Single source of truth for all tuneable constants and paths.

When running heavy ingestion on Kaggle or a university cluster,
override WHISPER_MODEL_SIZE and DEVICE here (or via env vars)
without touching any other module.
"""

import os
import torch
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR        = Path(__file__).parent
DATA_DIR        = BASE_DIR / "data"
VIDEOS_DIR      = DATA_DIR / "videos"
FRAMES_DIR      = DATA_DIR / "frames"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
CHROMA_DIR      = DATA_DIR / "chroma_db"

# Ensure all data directories exist on import
for _dir in [VIDEOS_DIR, FRAMES_DIR, TRANSCRIPTS_DIR, CHROMA_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Device  (auto-detects GPU; override by setting env var LECTURE_RAG_DEVICE)
# ---------------------------------------------------------------------------
DEVICE = os.environ.get(
    "LECTURE_RAG_DEVICE",
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ---------------------------------------------------------------------------
# Frame Extraction
# ---------------------------------------------------------------------------
FRAME_EXTRACTION_FPS = 1          # Extract 1 frame per second

# ---------------------------------------------------------------------------
# Scene Detection  (PySceneDetect)
# ---------------------------------------------------------------------------
# ContentDetector threshold — higher = less sensitive (fewer scene cuts detected)
# 27.0 is PySceneDetect's recommended default for most content
SCENE_DETECTION_THRESHOLD = 27.0

# ---------------------------------------------------------------------------
# Stage 1 Filter — Adaptive SSIM
# ---------------------------------------------------------------------------
# Rolling window size (in frames) for computing mean/std of SSIM scores.
# A window of 30 means we look at the last 30 SSIM scores to set the threshold.
SSIM_ROLLING_WINDOW     = 30

# Threshold = rolling_mean - SSIM_STD_MULTIPLIER * rolling_std
# Lower multiplier → more aggressive filtering (keep fewer frames)
# Higher multiplier → more conservative filtering (keep more frames)
SSIM_STD_MULTIPLIER     = 1.0

# Absolute floor: never drop a frame if SSIM is below this, regardless of
# adaptive threshold. Catches hard scene cuts that adaptive logic might miss.
SSIM_ABSOLUTE_FLOOR     = 0.50

# Absolute ceiling: always drop a frame if SSIM is above this — pixel-perfect
# duplicates that we never want to keep even on the first few frames before
# the rolling window has enough data.
SSIM_ABSOLUTE_CEILING   = 0.98

# ---------------------------------------------------------------------------
# Stage 2 Filter — CLIP Semantic Similarity
# ---------------------------------------------------------------------------
CLIP_MODEL_NAME             = "ViT-B/32"

# Frames with cosine similarity ABOVE this threshold are considered
# semantically redundant and dropped. ~0.95 is conservative (only drop
# near-identical semantics); lower values filter more aggressively.
CLIP_SIMILARITY_THRESHOLD   = 0.95

# ---------------------------------------------------------------------------
# Whisper Transcription
# ---------------------------------------------------------------------------
# Override to "large" on Kaggle / cluster for better accuracy.
# Supported: "tiny", "base", "small", "medium", "large"
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "medium")

# Transcript chunking: how many seconds per chunk stored in ChromaDB.
# Smaller = more granular retrieval. ~30s is a good balance for lectures.
TRANSCRIPT_CHUNK_SECONDS = 30

# Overlap between consecutive transcript chunks (in seconds) to avoid
# splitting a concept across two non-retrievable boundaries.
TRANSCRIPT_CHUNK_OVERLAP_SECONDS = 5

# ---------------------------------------------------------------------------
# Embedding Models
# ---------------------------------------------------------------------------
# Used for transcript chunk embeddings in ChromaDB
TEXT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ChromaDB collection names (namespaced per video using video hash as prefix)
CHROMA_FRAMES_COLLECTION_SUFFIX      = "_frames"
CHROMA_TRANSCRIPT_COLLECTION_SUFFIX  = "_transcript"

# ---------------------------------------------------------------------------
# Retrieval — Dynamic-k
# ---------------------------------------------------------------------------
# Initial candidate pool before threshold filtering
RETRIEVAL_INITIAL_K     = 10

# Cosine similarity threshold for keeping a candidate.
# CLIP embedding space: ~0.25–0.30 is a reasonable starting point.
RETRIEVAL_SIMILARITY_THRESHOLD = 0.27

# After threshold filtering, clamp result count to [MIN_K, MAX_K]
RETRIEVAL_MIN_K = 2
RETRIEVAL_MAX_K = 10

# Weight for blending frame vs. transcript retrieval scores (must sum to 1.0)
RETRIEVAL_FRAME_WEIGHT      = 0.4
RETRIEVAL_TRANSCRIPT_WEIGHT = 0.6   # Transcripts weighted higher for lecture content

# ---------------------------------------------------------------------------
# Gemini (Re-ranking + Generation)
# ---------------------------------------------------------------------------
GEMINI_MODEL_NAME       = "gemini-1.5-pro"
GEMINI_MAX_OUTPUT_TOKENS = 2048

# Max keyframe images to send to Gemini in a single generation call.
# Gemini 1.5 Pro supports many images but we cap to control cost.
GEMINI_MAX_FRAMES       = 5

# Max transcript characters to include in the Gemini prompt context.
# ~4000 chars ≈ ~1000 tokens — keeps cost predictable.
GEMINI_MAX_TRANSCRIPT_CHARS = 4000

# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
SUPPORTED_VIDEO_EXTENSIONS = [".mp4", ".mov", ".avi", ".mkv", ".webm"]
MAX_UPLOAD_SIZE_MB = 500
