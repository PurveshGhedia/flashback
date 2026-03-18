"""
retrieval/embedder.py
======================
Unified query embedding module.

Converts a user's natural language query into embeddings suitable
for searching both ChromaDB collections:
  - CLIP embedding (512-dim)  → search the frames collection
  - MiniLM embedding (384-dim) → search the transcript collection

Why two embeddings per query?
  The two ChromaDB collections live in different vector spaces:
    - Frames were indexed with CLIP image embeddings
    - Transcript chunks were indexed with MiniLM text embeddings
  A single query needs to be embedded into BOTH spaces to search both.

  CLIP has a joint image-text embedding space — the same model that
  embeds frames also embeds text queries, so CLIP text embeddings are
  directly comparable to CLIP image embeddings. This is the key property
  that makes cross-modal retrieval possible.

Design:
  This module is intentionally thin — it just calls the embedding
  functions that already exist in ingestion/clip_filter.py and
  indexer.py, keeping all model loading logic in one place.
"""

import sys
import logging
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


def embed_query_for_frames(query: str) -> np.ndarray:
    """
    Embed a text query into CLIP's joint text-image space.
    The result is directly comparable to frame embeddings stored
    in the <hash>_frames ChromaDB collection.

    Parameters
    ----------
    query : natural language query string

    Returns
    -------
    np.ndarray of shape (512,) — L2-normalised CLIP text embedding
    """
    from ingestion.clip_filter import embed_text
    embedding = embed_text(query)
    logger.debug("Query embedded for frames — shape: %s", embedding.shape)
    return embedding


def embed_query_for_transcript(query: str) -> list[float]:
    """
    Embed a text query using MiniLM sentence-transformers.
    The result is directly comparable to transcript chunk embeddings
    stored in the <hash>_transcript ChromaDB collection.

    Parameters
    ----------
    query : natural language query string

    Returns
    -------
    list[float] of length 384 — L2-normalised MiniLM embedding
    """
    from ingestion.indexer import embed_texts
    embeddings = embed_texts([query])
    embedding  = embeddings[0]
    logger.debug("Query embedded for transcript — length: %d", len(embedding))
    return embedding


def embed_query(query: str) -> dict:
    """
    Embed a query into both vector spaces in one call.

    Returns
    -------
    dict with keys:
      "frames"     : np.ndarray (512,)  — for CLIP frame search
      "transcript" : list[float] (384)  — for MiniLM transcript search

    Example
    -------
    >>> embeddings = embed_query("explain backpropagation")
    >>> print(embeddings["frames"].shape)      # (512,)
    >>> print(len(embeddings["transcript"]))   # 384
    """
    logger.info("Embedding query: '%s'", query)

    return {
        "frames"     : embed_query_for_frames(query),
        "transcript" : embed_query_for_transcript(query),
    }


# ------------------------------------------------------------------
# Quick smoke test
# (run: python retrieval/embedder.py)
# ------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    queries = [
        "explain backpropagation",
        "what is word2vec",
        "attention mechanism transformer",
    ]

    for q in queries:
        print(f"\nQuery: '{q}'")
        embeddings = embed_query(q)
        print(f"  CLIP embedding  shape : {embeddings['frames'].shape}")
        print(f"  CLIP embedding  norm  : {np.linalg.norm(embeddings['frames']):.4f}")
        print(f"  MiniLM embedding len  : {len(embeddings['transcript'])}")
        print(f"  MiniLM embedding norm : {np.linalg.norm(embeddings['transcript']):.4f}")
