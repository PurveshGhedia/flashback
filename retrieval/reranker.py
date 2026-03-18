"""
retrieval/reranker.py
======================
Re-ranks search candidates using Gemini before final answer generation.

Why re-rank?
  Vector similarity search retrieves the most *similar* results, not
  necessarily the most *relevant* ones. For example:
    - A query about "backpropagation" may retrieve a frame showing the
      word "propagation" in a different context
    - Two chunks may be retrieved that cover the same concept — Gemini
      can pick the more complete one
    - Transcript chunks retrieved by MiniLM may match keywords but not
      the actual concept the student is asking about

  Gemini re-ranking adds a semantic understanding layer that vector
  search alone cannot provide. This step meaningfully improves the
  quality of the final answer.

What this module does:
  1. Takes the top-k SearchResults from searcher.py
  2. Formats them into a structured prompt for Gemini
  3. Asks Gemini to select the most relevant subset and rank them
  4. Returns the re-ranked list of SearchResults

Cost control:
  Re-ranking only sends TEXT to Gemini (no images yet — that's in
  answerer.py). This keeps re-ranking cheap. We send:
    - The user query
    - Transcript text for transcript results
    - Timestamp + frame description for frame results
  The actual keyframe images are only sent once, in the final
  generation step.
"""

import sys
import json
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from config import GEMINI_MODEL_NAME
from retrieval.searcher import SearchResult

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Gemini client loader (cached)
# ------------------------------------------------------------------
_gemini_model = None


def load_gemini_model():
    """Load and cache the Gemini model."""
    global _gemini_model

    if _gemini_model is not None:
        return _gemini_model

    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError(
            "google-generativeai is required: pip install google-generativeai"
        )

    import os
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found. Set it in your .env file:\n"
            "  GEMINI_API_KEY=your_key_here"
        )

    genai.configure(api_key=api_key)
    _gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)

    logger.info("Gemini model '%s' loaded.", GEMINI_MODEL_NAME)
    return _gemini_model


# ------------------------------------------------------------------
# Re-ranking prompt builder
# ------------------------------------------------------------------
def _build_rerank_prompt(
    query   : str,
    results : list[SearchResult],
) -> str:
    """
    Build a structured re-ranking prompt for Gemini.

    The prompt asks Gemini to:
      1. Review the candidates
      2. Select the most relevant ones for the query
      3. Return a JSON list of selected indices in ranked order

    Returns plain text prompt string.
    """
    candidates_text = []

    for i, result in enumerate(results):
        if result.source == "transcript":
            entry = (
                f"Candidate {i}:\n"
                f"  Source    : transcript\n"
                f"  Timestamp : {_format_timestamp(result.timestamp)}\n"
                f"  Text      : {result.text[:500]}"
                f"{'...' if len(result.text) > 500 else ''}\n"
            )
        else:
            entry = (
                f"Candidate {i}:\n"
                f"  Source    : video frame\n"
                f"  Timestamp : {_format_timestamp(result.timestamp)}\n"
                f"  (Visual frame — no text available)\n"
            )
        candidates_text.append(entry)

    candidates_block = "\n".join(candidates_text)

    prompt = f"""You are helping a student find relevant moments in a lecture video.

Student query: "{query}"

Below are candidate results retrieved from the lecture. Each has a timestamp and content.
Your task is to select the most relevant candidates that would help answer the student's query.

{candidates_block}

Instructions:
- Select between 1 and 5 candidates that are most relevant to the query
- Rank them from most to least relevant
- Return ONLY a JSON object in this exact format, nothing else:
  {{"selected": [0, 2, 1], "reasoning": "brief explanation"}}
- The numbers in "selected" are the Candidate indices (0-based)
- If no candidates are relevant, return: {{"selected": [], "reasoning": "explanation"}}
- Do not include any text outside the JSON object"""

    return prompt


# ------------------------------------------------------------------
# Response parser
# ------------------------------------------------------------------
def _parse_rerank_response(response_text: str, num_candidates: int) -> list[int]:
    """
    Parse Gemini's JSON re-ranking response.
    Returns a list of valid candidate indices, or [] on parse failure.
    """
    # Strip any markdown code fences if Gemini adds them
    clean = response_text.strip()
    if clean.startswith("```"):
        lines = clean.split("\n")
        clean = "\n".join(lines[1:-1])

    try:
        parsed = json.loads(clean)
        selected = parsed.get("selected", [])
        reasoning = parsed.get("reasoning", "")

        logger.debug("Gemini re-ranking reasoning: %s", reasoning)

        # Validate indices
        valid = [
            int(idx) for idx in selected
            if isinstance(idx, (int, float)) and 0 <= int(idx) < num_candidates
        ]

        return valid

    except (json.JSONDecodeError, ValueError, TypeError) as e:
        logger.warning(
            "Failed to parse Gemini re-ranking response: %s\n"
            "Raw response: %s",
            e, response_text[:200]
        )
        # Fallback: return original order
        return list(range(min(num_candidates, 3)))


# ------------------------------------------------------------------
# Main re-ranking function
# ------------------------------------------------------------------
def rerank(
    query   : str,
    results : list[SearchResult],
    max_reranked: int = 5,
) -> list[SearchResult]:
    """
    Re-rank search results using Gemini.

    Parameters
    ----------
    query        : the user's original query
    results      : list of SearchResult from searcher.search()
    max_reranked : maximum results to return after re-ranking

    Returns
    -------
    Re-ranked list of SearchResult, most relevant first.
    Falls back to original score-based order if Gemini call fails.

    Example
    -------
    >>> response  = search("explain backpropagation", video_hash)
    >>> reranked  = rerank(response.query, response.results)
    >>> for r in reranked:
    ...     print(f"t={r.timestamp:.1f}s — {r.text[:80]}")
    """
    if not results:
        return []

    # If only 1-2 results, re-ranking adds no value — return as-is
    if len(results) <= 2:
        logger.info("Skipping re-ranking — only %d results.", len(results))
        return results

    logger.info(
        "Re-ranking %d candidates for query: '%s'",
        len(results), query
    )

    try:
        model  = load_gemini_model()
        prompt = _build_rerank_prompt(query, results)

        response      = model.generate_content(prompt)
        response_text = response.text

        selected_indices = _parse_rerank_response(response_text, len(results))

        if not selected_indices:
            logger.warning(
                "Gemini returned no selected indices — "
                "falling back to score-based order."
            )
            return results[:max_reranked]

        # Build re-ranked list
        reranked = [results[i] for i in selected_indices]
        reranked = reranked[:max_reranked]

        logger.info(
            "Re-ranking complete — %d → %d results | "
            "selected indices: %s",
            len(results), len(reranked), selected_indices
        )

        return reranked

    except Exception as e:
        logger.error(
            "Re-ranking failed: %s — falling back to score-based order.", e
        )
        # Graceful degradation: return original order
        return sorted(results, key=lambda r: r.score, reverse=True)[:max_reranked]


# ------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------
def _format_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS format."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


# ------------------------------------------------------------------
# Quick smoke test
# (run: python retrieval/reranker.py)
# ------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Mock results for testing without a real indexed video
    mock_results = [
        SearchResult(
            result_id  = "test_chunk_0",
            timestamp  = 120.0,
            score      = 0.42,
            source     = "transcript",
            text       = "So backpropagation is the algorithm we use to compute "
                         "gradients in a neural network. The key idea is the chain rule.",
            frame_path = "",
            metadata   = {},
        ),
        SearchResult(
            result_id  = "test_chunk_1",
            timestamp  = 245.0,
            score      = 0.38,
            source     = "transcript",
            text       = "Word2Vec uses a shallow neural network to learn word "
                         "embeddings from a large corpus of text.",
            frame_path = "",
            metadata   = {},
        ),
        SearchResult(
            result_id  = "test_frame_0",
            timestamp  = 118.0,
            score      = 0.35,
            source     = "frames",
            text       = "",
            frame_path = "data/frames/abc123/frame_003540_t118.00.jpg",
            metadata   = {},
        ),
    ]

    query = "explain backpropagation"
    print(f"Query: '{query}'")
    print(f"Input candidates: {len(mock_results)}")

    reranked = rerank(query, mock_results)

    print(f"\nRe-ranked results ({len(reranked)}):")
    for i, r in enumerate(reranked):
        print(f"  {i+1}. [{_format_timestamp(r.timestamp)}] "
              f"score={r.score:.4f} source={r.source}")
        if r.text:
            print(f"     {r.text[:100]}...")
