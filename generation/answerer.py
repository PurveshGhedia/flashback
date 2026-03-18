"""
generation/answerer.py
=======================
Final answer generation using Gemini 1.5 Pro.

What this module does:
  1. Takes re-ranked SearchResults from reranker.py
  2. Loads the corresponding keyframe images from disk
  3. Finds the most relevant transcript segments for context
  4. Builds a multimodal prompt (text + images) for Gemini
  5. Returns a structured Answer with the generated response,
     timestamps, and keyframe images for display in the UI

Why multimodal?
  Sending both images AND transcript text gives Gemini the richest
  possible context:
    - Images show what was on the slide/board at that moment
    - Transcript text shows what the professor said
  Together they let Gemini give answers that reference both the
  visual and spoken content of the lecture.

Cost control:
  - Max GEMINI_MAX_FRAMES images sent per query (default 5)
  - Max GEMINI_MAX_TRANSCRIPT_CHARS transcript text per query
  - Re-ranking already filtered to the most relevant candidates,
    so we're not sending noise to Gemini

Prompt design:
  The prompt is structured as:
    1. System context (you are a lecture assistant)
    2. Relevant transcript segments
    3. Keyframe images with their timestamps
    4. The student's question
    5. Instructions for the response format
"""

from retrieval.searcher import SearchResult
from config import (
    GEMINI_MODEL_NAME,
    GEMINI_MAX_OUTPUT_TOKENS,
    GEMINI_MAX_FRAMES,
    GEMINI_MAX_TRANSCRIPT_CHARS,
)
import sys
import logging
import base64
from pathlib import Path
from typing import NamedTuple, Optional

sys.path.append(str(Path(__file__).parent.parent))


logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Data container
# ------------------------------------------------------------------
class Answer(NamedTuple):
    """
    The complete response to a student's query.
    Contains everything needed to render the UI response.
    """
    query: str
    response_text: str           # Gemini's generated answer
    timestamps: list[float]   # relevant timestamps in seconds
    keyframe_paths: list[str]     # paths to keyframe JPEGs for display
    transcript_used: list[str]     # transcript excerpts used as context
    sources: list[dict]    # source metadata for citations
    model_used: str           # which Gemini model generated this


# ------------------------------------------------------------------
# Gemini client (reuses loader from reranker if already loaded)
# ------------------------------------------------------------------
def _load_gemini():
    """Load Gemini model — reuses cached instance from reranker if available."""
    try:
        from retrieval.reranker import load_gemini_model
        return load_gemini_model()
    except Exception as e:
        logger.error("Failed to load Gemini model: %s", e)
        raise


# ------------------------------------------------------------------
# Image loading helper
# ------------------------------------------------------------------
def _load_image_for_gemini(frame_path: str) -> Optional[dict]:
    """
    Load a keyframe JPEG and encode it for the Gemini API.

    Gemini's Python SDK accepts PIL Images directly — we use that
    rather than base64 encoding for cleaner code.

    Parameters
    ----------
    frame_path : relative path to the JPEG (as stored in ChromaDB metadata)

    Returns
    -------
    PIL Image object, or None if the file cannot be loaded.
    """
    try:
        from PIL import Image

        # frame_path is relative to project root
        abs_path = Path(__file__).parent.parent / frame_path

        if not abs_path.exists():
            logger.warning("Keyframe not found: %s", abs_path)
            return None

        img = Image.open(abs_path)
        # Resize to reduce token usage — Gemini handles 1080p but it's wasteful
        img.thumbnail((1280, 720), Image.LANCZOS)
        return img

    except Exception as e:
        logger.warning("Failed to load keyframe '%s': %s", frame_path, e)
        return None


# ------------------------------------------------------------------
# Prompt builder
# ------------------------------------------------------------------
def _build_generation_prompt(
    query: str,
    results: list[SearchResult],
    transcript_context: str,
) -> list:
    """
    Build a multimodal prompt for Gemini.

    Returns a list of content parts (text strings and PIL Images)
    that can be passed directly to model.generate_content().

    Structure:
      [system_text, transcript_text, image_1, caption_1, ..., question_text]
    """
    parts = []

    # -- System context --
    system_text = (
        "You are a helpful lecture assistant. A student is watching a recorded "
        "lecture and needs help understanding the content. You have been provided "
        "with relevant keyframes from the video and transcript excerpts.\n\n"
        "Your role is to:\n"
        "1. Answer the student's question clearly and accurately\n"
        "2. Reference specific timestamps when relevant (use MM:SS format)\n"
        "3. Connect visual content from the frames with spoken explanations "
        "from the transcript\n"
        "4. Be concise but complete — this is educational content\n\n"
    )
    parts.append(system_text)

    # -- Transcript context --
    if transcript_context.strip():
        transcript_text = (
            f"RELEVANT TRANSCRIPT EXCERPTS:\n"
            f"{transcript_context}\n\n"
        )
        parts.append(transcript_text)

    # -- Keyframes with timestamps --
    frame_results = [r for r in results if r.source ==
                     "frames" and r.frame_path]

    if frame_results:
        parts.append("RELEVANT KEYFRAMES FROM THE LECTURE:\n")

        for r in frame_results[:GEMINI_MAX_FRAMES]:
            timestamp_str = _format_timestamp(r.timestamp)
            img = _load_image_for_gemini(r.frame_path)

            if img is not None:
                parts.append(f"[Frame at {timestamp_str}]\n")
                parts.append(img)

    # -- Student question --
    question_text = (
        f"\nSTUDENT QUESTION:\n{query}\n\n"
        "Please provide a clear, helpful answer based on the lecture content above. "
        "Include specific timestamps (MM:SS) when referring to moments in the video."
    )
    parts.append(question_text)

    return parts


def _build_transcript_context(
    results: list[SearchResult],
    max_chars: int = GEMINI_MAX_TRANSCRIPT_CHARS,
) -> str:
    """
    Collect and format transcript text from search results.
    Sorted by timestamp, truncated to max_chars.
    """
    transcript_results = [
        r for r in results if r.source == "transcript" and r.text]

    if not transcript_results:
        return ""

    # Sort by timestamp so context reads chronologically
    sorted_results = sorted(transcript_results, key=lambda r: r.timestamp)

    parts = []
    total_chars = 0

    for r in sorted_results:
        timestamp_str = _format_timestamp(r.timestamp)
        entry = f"[{timestamp_str}] {r.text}"

        if total_chars + len(entry) > max_chars:
            # Truncate last entry to fit within budget
            remaining = max_chars - total_chars
            if remaining > 50:  # Only add if meaningful content remains
                parts.append(entry[:remaining] + "...")
            break

        parts.append(entry)
        total_chars += len(entry)

    return "\n\n".join(parts)


# ------------------------------------------------------------------
# Main answer generation function
# ------------------------------------------------------------------
def generate_answer(
    query: str,
    results: list[SearchResult],
) -> Answer:
    """
    Generate a final answer using Gemini 1.5 Pro.

    Parameters
    ----------
    query   : the student's original question
    results : re-ranked SearchResults from reranker.rerank()

    Returns
    -------
    Answer with response text, timestamps, keyframe paths, and sources.

    Example
    -------
    >>> response  = search("explain backpropagation", video_hash)
    >>> reranked  = rerank(response.query, response.results)
    >>> answer    = generate_answer(query, reranked)
    >>> print(answer.response_text)
    """
    if not results:
        return Answer(
            query=query,
            response_text=(
                "I couldn't find relevant content in the lecture for your query. "
                "Try rephrasing or asking about a different topic."
            ),
            timestamps=[],
            keyframe_paths=[],
            transcript_used=[],
            sources=[],
            model_used=GEMINI_MODEL_NAME,
        )

    logger.info(
        "Generating answer for: '%s' | %d results", query, len(results)
    )

    # -- Collect context --
    transcript_context = _build_transcript_context(results)
    prompt_parts = _build_generation_prompt(query, results, transcript_context)

    # -- Metadata for the Answer object --
    all_timestamps = sorted(set(r.timestamp for r in results))
    keyframe_paths = [
        r.frame_path for r in results
        if r.source == "frames" and r.frame_path
    ][:GEMINI_MAX_FRAMES]
    transcript_texts = [
        r.text for r in results
        if r.source == "transcript" and r.text
    ]
    sources = [
        {
            "result_id": r.result_id,
            "source": r.source,
            "timestamp": r.timestamp,
            "score": r.score,
        }
        for r in results
    ]

    # -- Call Gemini --
    try:
        import google.genai as genai
        import google.genai.types as types

        client = _load_gemini()

        # Gemini new SDK requires all parts to be strings or inline data
        # PIL Images need to be converted to inline image parts
        clean_parts = []
        for part in prompt_parts:
            if isinstance(part, str):
                clean_parts.append(part)
            else:
                # PIL Image — convert to bytes for inline data
                import io
                buf = io.BytesIO()
                part.save(buf, format="JPEG", quality=85)
                clean_parts.append(
                    types.Part.from_bytes(
                        data=buf.getvalue(),
                        mime_type="image/jpeg",
                    )
                )

        response = client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=clean_parts,
            config=types.GenerateContentConfig(
                max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
                temperature=0.2,
            ),
        )
        response_text = response.text
        logger.info("Answer generated successfully (%d chars).",
                    len(response_text))

    except Exception as e:
        logger.error("Gemini generation failed: %s", e)
        # Graceful degradation — return transcript context as plain answer
        response_text = (
            f"I found relevant content at these timestamps: "
            f"{', '.join(_format_timestamp(t) for t in all_timestamps)}.\n\n"
            f"Here's what was discussed:\n\n{transcript_context}\n\n"
            f"(Note: AI-generated summary unavailable — showing raw transcript)"
        )

    return Answer(
        query=query,
        response_text=response_text,
        timestamps=all_timestamps,
        keyframe_paths=keyframe_paths,
        transcript_used=transcript_texts,
        sources=sources,
        model_used=GEMINI_MODEL_NAME,
    )


# ------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------
def _format_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS format."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


# ------------------------------------------------------------------
# Full pipeline convenience function
# ------------------------------------------------------------------
def ask(query: str, video_hash: str) -> Answer:
    """
    One-call convenience function for the full RAG pipeline:
    query + video_hash → Answer

    Runs: embed → search → rerank → generate

    Parameters
    ----------
    query      : student's natural language question
    video_hash : video identifier from indexer.get_video_hash()

    Returns
    -------
    Answer ready for display in the Streamlit UI.

    Example
    -------
    >>> answer = ask("explain backpropagation", "a3f2c891")
    >>> print(answer.response_text)
    """
    from retrieval.searcher import search
    from retrieval.reranker import rerank

    logger.info("=== RAG Pipeline: '%s' ===", query)

    # Step 1: Search
    search_response = search(query, video_hash)

    if not search_response.results:
        return generate_answer(query, [])

    # Step 2: Re-rank
    reranked = rerank(query, search_response.results)

    # Step 3: Generate
    answer = generate_answer(query, reranked)

    return answer


# ------------------------------------------------------------------
# Quick smoke test (requires indexed video + Gemini API key)
# (run: python generation/answerer.py <video_hash> "<query>")
# ------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s: %(message)s")

    if len(sys.argv) < 3:
        print("Usage: python answerer.py <video_hash> \"<query>\"")
        print("Example: python answerer.py a3f2c891 \"explain word2vec\"")
        sys.exit(1)

    video_hash = sys.argv[1]
    query = sys.argv[2]

    print(f"\nQuery    : '{query}'")
    print(f"Video    : {video_hash}")
    print("=" * 60)

    answer = ask(query, video_hash)

    print(f"\nAnswer ({len(answer.response_text)} chars):")
    print("-" * 60)
    print(answer.response_text)
    print("-" * 60)
    print(
        f"\nTimestamps  : {[_format_timestamp(t) for t in answer.timestamps]}")
    print(f"Keyframes   : {answer.keyframe_paths}")
    print(f"Model used  : {answer.model_used}")


# Minimum score of the top re-ranked result to proceed to Gemini.
# This is the last gate before an API call is made.
PRE_GEMINI_MIN_SCORE = 0.20

# ------------------------------------------------------------------
# Cross-video pipeline convenience function
# ------------------------------------------------------------------


def ask_across_videos(query: str) -> list:
    """
    Search ALL indexed videos and return answers from the most
    relevant ones.

    Returns
    -------
    List of dicts, one per relevant video found:
      {
        "video_hash"  : str,
        "video_info"  : dict,    — registry entry (name, thumbnail etc.)
        "answer"      : Answer,  — generated answer for this video
        "best_score"  : float,
      }
    Sorted by relevance score descending.

    Example
    -------
    >>> results = ask_across_videos("explain backpropagation")
    >>> for r in results:
    ...     print(r["video_info"]["video_name"])
    ...     print(r["answer"].response_text)
    """
    from retrieval.searcher import search_all_videos
    from retrieval.reranker import rerank

    logger.info("=== Cross-video RAG: '%s' ===", query)

    # Step 1: Search all videos
    video_results = search_all_videos(query)

    if not video_results:
        return []

    answers = []

    for vr in video_results:
        # Step 2: Re-rank per video
        reranked = rerank(query, vr["results"])

        if not reranked:
            logger.info("No results after re-ranking for %s — skipping.",
                        vr["video_hash"])
            continue

        # Step 3: Pre-Gemini score gate
        # Check the best score among re-ranked results.
        # If it's still too weak, skip — no API call made.
        top_score = max(r.score for r in reranked)
        if top_score < PRE_GEMINI_MIN_SCORE:
            logger.info(
                "Pre-Gemini gate: skipping video %s — "
                "top re-ranked score %.4f < %.2f threshold.",
                vr["video_hash"], top_score, PRE_GEMINI_MIN_SCORE
            )
            continue

        # Step 4: Generate answer — only called if content is relevant
        answer = generate_answer(query, reranked)

        answers.append({
            "video_hash": vr["video_hash"],
            "video_info": vr["video_info"],
            "answer": answer,
            "best_score": vr["best_score"],
        })

    return answers
