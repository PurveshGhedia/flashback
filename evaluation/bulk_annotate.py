"""
evaluation/bulk_annotate.py
============================
Bulk-writes ground-truth annotation files from Q&A pairs you've
already collected by watching the video yourself — skips the
interactive prompts in annotator.py.

Reuses the same timestamp parsing and video-hash-matching logic as
annotator.py, so the output is byte-for-byte compatible with what
metrics.py expects.

Usage:
    1. Update the JOBS list below: correct video_path for each video,
       correct output_path, and your (query, timestamp string) pairs.
    2. Run: python evaluation/bulk_annotate.py
"""

from evaluation.annotator import parse_timestamps, TIMESTAMP_TOLERANCE_SECONDS
from ingestion.indexer import get_video_hash, list_indexed_videos
import sys
from pathlib import Path
from datetime import datetime
import json

sys.path.append(str(Path(__file__).parent.parent))


# ---------------------------------------------------------------
# Fill in one block per video. Timestamp strings accept the same
# formats as the interactive tool: "MM:SS", "H:MM:SS", comma-
# separated for multiple relevant moments.
# ---------------------------------------------------------------
JOBS = [
    {
        "video_path": "data/videos/vidssave.com \u201cAI and Our Economic Future\u201d with Professor Chad Jones 720P.mp4 ",   # <-- update filename
        "output_path": "evaluation/annotations_chad_jones.json",
        "qa": [
            ("What are the two extreme scenarios for AI's economic impact?",
             "0:01:35, 0:02:26"),
            ("How does the 'weak link' model apply to business success and technological bottlenecks?", "0:11:08, 0:14:06"),
            ("Why is AI not necessarily replacing radiologists despite predictions?",
             "0:31:32, 0:33:05"),
            ("What are the potential catastrophic risks associated with advanced AI?",
             "0:36:36, 0:38:51"),
            ("Why does the presenter believe that AI's impact might take decades rather than years?", "0:33:26, 0:34:07"),
            ("What happens to the labor share of GDP in the provided growth models?",
             "0:23:15, 0:24:35"),
            ("How can retirees be an analogy for a future world of AI-driven abundance?",
             "0:36:04, 0:36:35"),
        ],
    },
    {
        "video_path": "data/videos/CHANGE_ME_llm_training.mp4",  # <-- update filename
        "output_path": "evaluation/annotations_llm_training.json",
        "qa": [
            ("What are the core components required to train LLMs?", "0:00:58, 0:01:47"),
            ("How does autoregressive modeling function in language models?",
             "0:06:36, 0:07:49"),
            ("Why is tokenization a crucial process for LLMs?", "0:10:45, 0:13:30"),
            ("How is perplexity utilized to evaluate these models?", "0:16:00, 0:20:50"),
            ("What is the purpose of the MMLU benchmark?", "0:22:41, 0:24:30"),
            ("How is training data processed and filtered before use?", "0:31:01, 0:32:24"),
            ("What are scaling laws and how do they predict model performance?",
             "0:45:00, 0:51:20"),
            ("What is the primary objective of Supervised Fine-Tuning (SFT)?",
             "1:02:26, 1:03:30"),
            ("Why is Reinforcement Learning from Human Feedback (RLHF) implemented in post-training?", "1:09:59, 1:12:14"),
            ("How can synthetic data generation reduce LLM training costs?",
             "1:25:53, 1:28:40"),
        ],
    },
]


def resolve_hash(video_path: Path) -> str:
    """Match against the ChromaDB registry first, same logic as annotator.py."""
    indexed = list_indexed_videos()
    for v in indexed:
        if (v["video_name"] == video_path.name or
            Path(v["video_name"]).stem in video_path.stem or
                video_path.stem in Path(v["video_name"]).stem):
            print(
                f"  Matched registry entry: {v['video_name']} (hash: {v['video_hash']})")
            return v["video_hash"]
    h = get_video_hash(video_path)
    print(
        f"  No registry match found for {video_path.name} — computed hash: {h}")
    print("  WARNING: this video may not be indexed yet — run ingestion first.")
    return h


def build(job: dict) -> None:
    video_path = Path(job["video_path"])
    output_path = Path(job["output_path"])

    print(f"\nProcessing: {video_path.name}")
    video_hash = resolve_hash(video_path)

    annotations = []
    for query, raw_ts in job["qa"]:
        timestamps = parse_timestamps(raw_ts)
        if not timestamps:
            print(
                f"  ! Could not parse timestamps for query: {query!r} — skipping.")
            continue
        annotations.append({
            "query": query,
            "relevant_timestamps": timestamps,
            "notes": "",
        })

    result = {
        "video_hash": video_hash,
        "video_name": video_path.name,
        "video_path": str(video_path),
        "annotated_at": datetime.now().isoformat(timespec="seconds"),
        "timestamp_tolerance_sec": TIMESTAMP_TOLERANCE_SECONDS,
        "total_annotations": len(annotations),
        "annotations": annotations,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"  \u2713 Wrote {len(annotations)} annotations to {output_path}")


if __name__ == "__main__":
    for job in JOBS:
        build(job)

    print(f"\nDone. Total videos processed: {len(JOBS)}")
