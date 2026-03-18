"""
retrieval/searcher.py
======================
Hybrid dynamic-k search across both ChromaDB collections.

What this module does:
  1. Takes a user query and a video hash
  2. Searches the frames collection (CLIP embeddings)
  3. Searches the transcript collection (MiniLM embeddings)
  4. Applies dynamic-k filtering: start with top-10, apply cosine
     similarity threshold, clamp to [MIN_K, MAX_K]
  5. Merges results from both collections with weighted scoring
  6. Returns a ranked list of SearchResult objects

Dynamic-k explained:
  Fixed k (e.g. always return top 3) breaks for two reasons:
    - Specific queries ("when does the professor write the chain rule")
      have one clear answer — returning 3 results adds noise
    - Broad queries ("explain neural networks") have many relevant
      moments — returning only 3 misses useful context
  Dynamic-k adapts: retrieve 10 candidates, keep those above the
  similarity threshold, then clamp to [2, 10].

Hybrid retrieval weighting:
  Transcript results are weighted 60%, frame results 40%.
  This reflects the reality that for lecture content, spoken words
  are a stronger retrieval signal than visual frames.

Result deduplication:
  A frame and a transcript chunk may point to the same moment in the
  video. We deduplicate by merging results within a 30-second window,
  taking the higher score when there's overlap.
"""

from config import (
    RETRIEVAL_INITIAL_K,
    RETRIEVAL_SIMILARITY_THRESHOLD,
    RETRIEVAL_MIN_K,
    RETRIEVAL_MAX_K,
    RETRIEVAL_FRAME_WEIGHT,
    RETRIEVAL_TRANSCRIPT_WEIGHT,
    CHROMA_FRAMES_COLLECTION_SUFFIX,
    CHROMA_TRANSCRIPT_COLLECTION_SUFFIX,
)
import sys
import logging
import numpy as np
from pathlib import Path
from typing import NamedTuple, Optional

sys.path.append(str(Path(__file__).parent.parent))


logger = logging.getLogger(__name__)

# Window within which a frame and transcript result are considered
# to be about the "same moment" and should be merged
DEDUP_WINDOW_SECONDS = 30.0


# ------------------------------------------------------------------
# Data containers
# ------------------------------------------------------------------
class SearchResult(NamedTuple):
    """
    A single retrieved result, normalised across both collections.
    """
    result_id: str          # ChromaDB document id
    timestamp: float        # seconds from video start (primary sort key)
    score: float        # weighted similarity score [0, 1]
    source: str          # "frames" or "transcript"
    text: str          # transcript text (empty for frame results)
    frame_path: str          # relative path to JPEG (empty for transcript)
    metadata: dict         # full ChromaDB metadata dict


class SearchResponse(NamedTuple):
    """
    Full response from a search query.
    """
    query: str
    results: list         # list[SearchResult], sorted by score desc
    total_frames_candidates: int
    total_transcript_candidates: int
    dynamic_k: int          # how many results were kept after threshold


# ------------------------------------------------------------------
# ChromaDB client helper
# ------------------------------------------------------------------
def _get_collection(video_hash: str, suffix: str):
    """Get a ChromaDB collection by video hash and suffix."""
    from ingestion.indexer import get_chroma_client
    client = get_chroma_client()
    collection = client.get_collection(f"{video_hash}{suffix}")
    return collection


# ------------------------------------------------------------------
# Individual collection search
# ------------------------------------------------------------------
def _search_collection(
    collection,
    query_embedding: list,
    initial_k: int,
    threshold: float,
    weight: float,
    source_label: str,
) -> list[SearchResult]:
    """
    Search a single ChromaDB collection and return filtered results.

    Parameters
    ----------
    collection      : ChromaDB collection object
    query_embedding : query embedding as a list of floats
    initial_k       : how many candidates to retrieve before filtering
    threshold       : minimum similarity score to keep a result
    weight          : score multiplier for this collection
    source_label    : "frames" or "transcript"

    Returns
    -------
    List of SearchResult, filtered by threshold, scores weighted.
    """
    # ChromaDB returns distances in cosine space — we convert to similarity
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(initial_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    if not results or not results["ids"][0]:
        return []

    search_results = []

    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for rid, doc, meta, dist in zip(ids, documents, metadatas, distances):
        # ChromaDB cosine distance is in [0, 2]. Convert to similarity [0, 1]:
        # similarity = 1 - (distance / 2)
        # For well-normalised embeddings distance is in [0, 1] so:
        similarity = 1.0 - dist

        if similarity < threshold:
            logger.debug(
                "  %s: %s — score %.4f below threshold %.4f, skipping",
                source_label, rid, similarity, threshold
            )
            continue

        weighted_score = similarity * weight

        # Extract timestamp — different field names for frames vs transcript
        if source_label == "frames":
            timestamp = meta.get("timestamp", 0.0)
            frame_path = meta.get("frame_path", "")
            text = ""
        else:
            timestamp = meta.get("start_time", 0.0)
            frame_path = ""
            text = doc or ""

        search_results.append(SearchResult(
            result_id=rid,
            timestamp=timestamp,
            score=round(weighted_score, 4),
            source=source_label,
            text=text,
            frame_path=frame_path,
            metadata=meta,
        ))

        logger.debug(
            "  %s: t=%.1fs | raw_sim=%.4f | weighted=%.4f",
            source_label, timestamp, similarity, weighted_score
        )

    return search_results


# ------------------------------------------------------------------
# Deduplication
# ------------------------------------------------------------------
def _deduplicate_results(
    results: list[SearchResult],
    window_seconds: float = DEDUP_WINDOW_SECONDS,
) -> list[SearchResult]:
    """
    Merge results that point to the same moment in the video.

    Two results are considered duplicates if their timestamps are
    within `window_seconds` of each other. When merging, keep the
    result with the higher score but combine their sources.

    This handles the common case where a frame at t=120s and a
    transcript chunk spanning t=110-140s both get retrieved for
    the same query moment.
    """
    if not results:
        return []

    # Sort by timestamp for windowed dedup
    sorted_results = sorted(results, key=lambda r: r.timestamp)
    deduped = []
    used = set()

    for i, result in enumerate(sorted_results):
        if i in used:
            continue

        # Find all results within the dedup window
        group = [result]
        for j, other in enumerate(sorted_results[i + 1:], start=i + 1):
            if j in used:
                continue
            if abs(other.timestamp - result.timestamp) <= window_seconds:
                group.append(other)
                used.add(j)

        used.add(i)

        # Keep the highest-scoring result from the group
        best = max(group, key=lambda r: r.score)
        deduped.append(best)

    return deduped


# ------------------------------------------------------------------
# Dynamic-k filtering
# ------------------------------------------------------------------
def _apply_dynamic_k(
    results: list[SearchResult],
    threshold: float = RETRIEVAL_SIMILARITY_THRESHOLD,
    min_k: int = RETRIEVAL_MIN_K,
    max_k: int = RETRIEVAL_MAX_K,
) -> list[SearchResult]:
    """
    Apply dynamic-k: filter by threshold, clamp to [min_k, max_k].

    Always returns at least min_k results (even if below threshold)
    and never more than max_k results.
    """
    # Sort by score descending
    sorted_results = sorted(results, key=lambda r: r.score, reverse=True)

    # Apply threshold filter
    above_threshold = [r for r in sorted_results if r.score >= threshold]

    # Clamp: ensure at least min_k, at most max_k
    if len(above_threshold) < min_k:
        # Not enough above threshold — take top min_k regardless
        final = sorted_results[:min_k]
        logger.debug(
            "Dynamic-k: only %d above threshold (%.3f), "
            "falling back to top %d",
            len(above_threshold), threshold, min_k
        )
    else:
        final = above_threshold[:max_k]
        logger.debug(
            "Dynamic-k: %d above threshold → clamped to %d",
            len(above_threshold), len(final)
        )

    return final


# ------------------------------------------------------------------
# Main search function
# ------------------------------------------------------------------
def search(
    query: str,
    video_hash: str,
    initial_k: int = RETRIEVAL_INITIAL_K,
    threshold: float = RETRIEVAL_SIMILARITY_THRESHOLD,
    min_k: int = RETRIEVAL_MIN_K,
    max_k: int = RETRIEVAL_MAX_K,
) -> SearchResponse:
    """
    Hybrid search across frames + transcript collections for a video.

    Parameters
    ----------
    query      : natural language query string
    video_hash : video identifier (from indexer.get_video_hash())
    initial_k  : candidates to retrieve per collection before filtering
    threshold  : minimum weighted score to keep a result
    min_k      : minimum results to return (overrides threshold if needed)
    max_k      : maximum results to return

    Returns
    -------
    SearchResponse with ranked SearchResult list.

    Example
    -------
    >>> response = search("explain backpropagation", "a3f2c891")
    >>> for r in response.results:
    ...     print(f"t={r.timestamp:.1f}s | score={r.score:.4f} | {r.source}")
    """
    from retrieval.embedder import embed_query

    logger.info("Searching for: '%s' (video: %s)", query, video_hash)

    # Embed query into both spaces
    embeddings = embed_query(query)

    # -- Search frames collection --
    frame_results = []
    try:
        frames_col = _get_collection(
            video_hash, CHROMA_FRAMES_COLLECTION_SUFFIX)
        frame_results = _search_collection(
            collection=frames_col,
            query_embedding=embeddings["frames"].tolist(),
            initial_k=initial_k,
            threshold=0.0,   # threshold applied after merging
            weight=RETRIEVAL_FRAME_WEIGHT,
            source_label="frames",
        )
        logger.info("  Frames collection: %d candidates", len(frame_results))
    except Exception as e:
        logger.warning("Could not search frames collection: %s", e)

    # -- Search transcript collection --
    transcript_results = []
    try:
        transcript_col = _get_collection(
            video_hash, CHROMA_TRANSCRIPT_COLLECTION_SUFFIX)
        transcript_results = _search_collection(
            collection=transcript_col,
            query_embedding=embeddings["transcript"],
            initial_k=initial_k,
            threshold=0.0,   # threshold applied after merging
            weight=RETRIEVAL_TRANSCRIPT_WEIGHT,
            source_label="transcript",
        )
        logger.info("  Transcript collection: %d candidates",
                    len(transcript_results))
    except Exception as e:
        logger.warning("Could not search transcript collection: %s", e)

    # -- Merge, deduplicate, apply dynamic-k --
    all_results = frame_results + transcript_results

    if not all_results:
        logger.warning("No results found for query: '%s'", query)
        return SearchResponse(
            query=query,
            results=[],
            total_frames_candidates=0,
            total_transcript_candidates=0,
            dynamic_k=0,
        )

    deduped = _deduplicate_results(all_results)
    final = _apply_dynamic_k(deduped, threshold, min_k, max_k)

    logger.info(
        "Search complete — %d frame + %d transcript → "
        "%d after dedup → %d after dynamic-k",
        len(frame_results), len(transcript_results),
        len(deduped), len(final)
    )

    return SearchResponse(
        query=query,
        results=final,
        total_frames_candidates=len(frame_results),
        total_transcript_candidates=len(transcript_results),
        dynamic_k=len(final),
    )


# ------------------------------------------------------------------
# Timestamp-range filtered search
# ------------------------------------------------------------------
def search_in_range(
    query: str,
    video_hash: str,
    start_sec: float,
    end_sec: float,
    **kwargs,
) -> SearchResponse:
    """
    Search within a specific time range of the video.
    Useful for queries like "what happened in the first 30 minutes".

    Runs a normal search then filters results by timestamp range.
    ChromaDB supports metadata filtering natively but for simplicity
    we post-filter here — acceptable for our result set sizes.
    """
    response = search(query, video_hash, **kwargs)

    filtered = [
        r for r in response.results
        if start_sec <= r.timestamp <= end_sec
    ]

    logger.info(
        "Range filter [%.1fs, %.1fs]: %d → %d results",
        start_sec, end_sec, len(response.results), len(filtered)
    )

    return SearchResponse(
        query=response.query,
        results=filtered,
        total_frames_candidates=response.total_frames_candidates,
        total_transcript_candidates=response.total_transcript_candidates,
        dynamic_k=len(filtered),
    )


# ------------------------------------------------------------------
# Quick smoke test (requires an indexed video)
# (run: python retrieval/searcher.py <video_hash>)
# ------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s: %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python searcher.py <video_hash>")
        print("Get video_hash from: python -c \"from ingestion.indexer import "
              "get_video_hash; print(get_video_hash('path/to/video.mp4'))\"")
        sys.exit(1)

    video_hash = sys.argv[1]

    test_queries = [
        "explain word2vec",
        "what is backpropagation",
        "attention mechanism",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: '{query}'")
        response = search(query, video_hash)
        print(f"  Dynamic-k: {response.dynamic_k} results")
        for r in response.results:
            mins = int(r.timestamp // 60)
            secs = int(r.timestamp % 60)
            print(f"  [{mins:02d}:{secs:02d}] score={r.score:.4f} "
                  f"source={r.source:10s} "
                  f"text='{r.text[:60]}...'" if r.text else
                  f"  [{mins:02d}:{secs:02d}] score={r.score:.4f} "
                  f"source={r.source:10s} frame={r.frame_path}")


# ------------------------------------------------------------------
# Cross-video search — searches ALL indexed videos
# ------------------------------------------------------------------
def search_all_videos(
    query: str,
    top_n: int = 3,
    **kwargs,
) -> list[dict]:
    """
    Search across ALL indexed videos and return the most relevant
    results grouped by video.

    Parameters
    ----------
    query  : natural language query string
    top_n  : number of top videos to return results from

    Returns
    -------
    List of dicts sorted by best score, each containing:
      {
        "video_hash"  : str,
        "video_info"  : dict,        — registry entry
        "best_score"  : float,       — highest scoring result
        "results"     : list[SearchResult],
      }

    Example
    -------
    >>> results = search_all_videos("explain backpropagation")
    >>> for r in results:
    ...     print(r["video_info"]["video_name"], r["best_score"])
    """
    from ingestion.indexer import list_indexed_videos

    videos = list_indexed_videos()
    if not videos:
        logger.warning("No indexed videos found.")
        return []

    logger.info(
        "Cross-video search: '%s' across %d videos", query, len(videos)
    )

    video_results = []

    for video in videos:
        hash_ = video["video_hash"]
        try:
            response = search(query, hash_, **kwargs)
            if response.results:
                best_score = max(r.score for r in response.results)
                video_results.append({
                    "video_hash": hash_,
                    "video_info": video,
                    "best_score": best_score,
                    "results": response.results,
                })
                logger.info(
                    "  %s: best_score=%.4f (%d results)",
                    video.get("video_name", hash_),
                    best_score,
                    len(response.results),
                )
        except Exception as e:
            logger.warning("Search failed for video %s: %s", hash_, e)
            continue

    # Sort by best score descending
    video_results.sort(key=lambda x: x["best_score"], reverse=True)

    # Return top_n most relevant videos
    top = video_results[:top_n]

    logger.info(
        "Cross-video search complete — %d/%d videos had results",
        len(video_results), len(videos)
    )

    return top
