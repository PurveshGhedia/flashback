"""
evaluation/annotator.py
========================
CLI tool for creating ground-truth timestamp annotations.

Usage:
    python evaluation/annotator.py --video data/videos/lecture.mp4
                                   --output evaluation/annotations.json

What this does:
    Walks you through a list of concept queries interactively.
    For each concept you manually enter the correct timestamp(s)
    where that concept is explained in the video.
    Saves a JSON file used by metrics.py for evaluation.

Annotation format:
    {
      "video_hash"  : "a3f2c891",
      "video_name"  : "Stanford_CS224N_NLP.mp4",
      "annotations" : [
        {
          "query"              : "explain word2vec",
          "relevant_timestamps": [73.0, 2246.0],
          "notes"              : "main explanation at 1:13, revisited at 37:26"
        },
        ...
      ]
    }

The relevant_timestamps are the ground-truth seconds where the concept
is discussed. A retrieval result is considered correct if it falls
within TIMESTAMP_TOLERANCE_SECONDS of any ground-truth timestamp.
"""

import json
import argparse
import sys
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

# Tolerance window — a retrieved timestamp is "correct" if it falls
# within this many seconds of a ground-truth timestamp
TIMESTAMP_TOLERANCE_SECONDS = 30

# Default concept queries for a typical NLP lecture
# Edit or replace these with concepts from your specific video
DEFAULT_QUERIES = [
    "what is word2vec",
    "explain the skip-gram model",
    "what is a word embedding",
    "explain backpropagation",
    "what is gradient descent",
    "explain the attention mechanism",
    "what is a transformer",
    "explain self-attention",
    "what are the limitations of RNN",
    "what is the softmax function",
    "explain cross entropy loss",
    "what is a language model",
    "explain the encoder decoder architecture",
    "what is beam search",
    "what is BLEU score",
    "explain positional encoding",
    "what is multi-head attention",
    "explain the feed forward network in transformers",
    "what is transfer learning in NLP",
    "explain fine tuning",
]


def parse_timestamps(raw: str) -> list[float]:
    """
    Parse a comma-separated string of timestamps.
    Accepts both MM:SS and raw seconds formats.

    Examples:
        "1:13, 37:26"   → [73.0, 2246.0]
        "73, 2246"      → [73.0, 2246.0]
        "1:13:05"       → [4385.0]
    """
    timestamps = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            segments = part.split(":")
            if len(segments) == 2:
                t = int(segments[0]) * 60 + float(segments[1])
            elif len(segments) == 3:
                t = int(segments[0]) * 3600 + \
                    int(segments[1]) * 60 + float(segments[2])
            else:
                print(f"  Could not parse '{part}' — skipping.")
                continue
        else:
            try:
                t = float(part)
            except ValueError:
                print(f"  Could not parse '{part}' — skipping.")
                continue
        timestamps.append(round(t, 1))
    return timestamps


def fmt_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{h}:{mins:02d}:{secs:02d}" if h > 0 else f"{mins:02d}:{secs:02d}"


def annotate(
    video_path: str,
    output_path: str,
    queries: list[str] = None,
    resume: bool = True,
) -> dict:
    """
    Interactive CLI annotation session.

    Parameters
    ----------
    video_path  : path to the video being annotated
    output_path : where to save the annotations JSON
    queries     : list of concept queries to annotate
                  (defaults to DEFAULT_QUERIES)
    resume      : if True and output_path exists, load existing
                  annotations and skip already-annotated queries

    Returns
    -------
    Annotations dict.
    """
    from ingestion.indexer import get_video_hash, list_indexed_videos

    video_path = Path(video_path)
    output_path = Path(output_path)
    queries = queries or DEFAULT_QUERIES

    # Try to find the hash from the registry first — avoids mismatch
    # when the indexed filename differs slightly from the given path
    video_hash = None
    indexed = list_indexed_videos()
    for v in indexed:
        # Match on stem (filename without extension) to handle _1 suffixes
        if (v["video_name"] == video_path.name or
            Path(v["video_name"]).stem in video_path.stem or
                video_path.stem in Path(v["video_name"]).stem):
            video_hash = v["video_hash"]
            print(
                f"Matched registry entry: {v['video_name']} (hash: {video_hash})")
            break

    # Fall back to computing hash from the given path
    if not video_hash:
        video_hash = get_video_hash(video_path)
        print(f"No registry match found — computed hash: {video_hash}")
        print("Warning: make sure this video is indexed before running evaluation.")

    # Load existing annotations if resuming
    existing_annotations = []
    if resume and output_path.exists():
        with open(output_path) as f:
            existing = json.load(f)
        existing_annotations = existing.get("annotations", [])
        already_done = {a["query"] for a in existing_annotations}
        queries = [q for q in queries if q not in already_done]
        print(f"\nResuming — {len(already_done)} already annotated, "
              f"{len(queries)} remaining.\n")

    print("=" * 60)
    print(f"  Flashback — Ground Truth Annotator")
    print(f"  Video : {video_path.name}")
    print(f"  Hash  : {video_hash}")
    print("=" * 60)
    print("\nFor each query, watch the video and enter the timestamp(s)")
    print("where the concept is explained.")
    print("Format: MM:SS or HH:MM:SS, comma-separated for multiple.")
    print("Enter 's' to skip a query, 'q' to quit and save.\n")

    new_annotations = []

    for i, query in enumerate(queries):
        print(f"[{i+1}/{len(queries)}] Query: \"{query}\"")

        while True:
            raw = input("  Timestamp(s): ").strip()

            if raw.lower() == 'q':
                print("\nSaving and quitting...")
                break

            if raw.lower() == 's':
                print("  Skipped.\n")
                break

            if not raw:
                print("  Enter a timestamp or 's' to skip.")
                continue

            timestamps = parse_timestamps(raw)
            if not timestamps:
                print("  Could not parse timestamps. Try again.")
                continue

            notes = input("  Notes (optional): ").strip()

            new_annotations.append({
                "query": query,
                "relevant_timestamps": timestamps,
                "notes": notes,
            })

            print(f"  ✓ Saved: {[fmt_ts(t) for t in timestamps]}\n")
            break

        if raw.lower() == 'q':
            break

    # Merge with existing
    all_annotations = existing_annotations + new_annotations

    result = {
        "video_hash": video_hash,
        "video_name": video_path.name,
        "video_path": str(video_path),
        "annotated_at": datetime.now().isoformat(timespec="seconds"),
        "timestamp_tolerance_sec": TIMESTAMP_TOLERANCE_SECONDS,
        "total_annotations": len(all_annotations),
        "annotations": all_annotations,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n✓ Saved {len(all_annotations)} annotations to '{output_path}'")
    return result


def load_annotations(path: str) -> dict:
    """Load annotations from a JSON file."""
    with open(path) as f:
        return json.load(f)


def show_annotations(path: str) -> None:
    """Pretty-print an annotations file."""
    data = load_annotations(path)
    print(f"\nAnnotations for: {data['video_name']}")
    print(f"Hash: {data['video_hash']} | Total: {data['total_annotations']}")
    print(f"Tolerance: ±{data['timestamp_tolerance_sec']}s\n")
    for i, ann in enumerate(data["annotations"]):
        ts_str = ", ".join(fmt_ts(t) for t in ann["relevant_timestamps"])
        print(f"  {i+1:2d}. {ann['query']}")
        print(f"      → {ts_str}")
        if ann.get("notes"):
            print(f"      ℹ {ann['notes']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create ground-truth annotations for Flashback evaluation"
    )
    parser.add_argument("--video",  required=True,
                        help="Path to the video file")
    parser.add_argument("--output", required=True,  help="Output JSON path")
    parser.add_argument("--show",   action="store_true",
                        help="Show existing annotations and exit")
    parser.add_argument("--no-resume", action="store_true",
                        help="Start fresh, ignore existing annotations")
    args = parser.parse_args()

    if args.show:
        show_annotations(args.output)
    else:
        annotate(
            video_path=args.video,
            output_path=args.output,
            resume=not args.no_resume,
        )
