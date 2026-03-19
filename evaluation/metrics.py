"""
evaluation/metrics.py
======================
Evaluates Flashback retrieval quality against ground-truth annotations.

Metrics computed:
  - Precision@k  : of the top-k retrieved results, what fraction are relevant?
  - Recall@k     : of all relevant timestamps, what fraction did we retrieve?
  - MRR          : Mean Reciprocal Rank — how high does the first correct
                   result appear? MRR=1.0 means always first, 0.5 means
                   always second, etc.
  - Hit@k        : did at least one relevant result appear in the top-k?
                   (binary version of Recall)

A retrieved result is "correct" if its timestamp falls within
TIMESTAMP_TOLERANCE_SECONDS of any ground-truth timestamp.

Usage:
    python evaluation/metrics.py
        --video   data/videos/lecture.mp4
        --annotations evaluation/annotations.json
        --k 5

Output:
    Per-query results table + aggregate metrics summary.
    Optionally saves a detailed JSON report.
"""

import json
import argparse
import sys
from pathlib import Path
from typing import NamedTuple

sys.path.append(str(Path(__file__).parent.parent))


# ------------------------------------------------------------------
# Data containers
# ------------------------------------------------------------------
class QueryResult(NamedTuple):
    query               : str
    retrieved_timestamps: list[float]   # timestamps returned by the system
    relevant_timestamps : list[float]   # ground-truth timestamps
    hits                : list[bool]    # per-result relevance flags
    precision_at_k      : float
    recall_at_k         : float
    reciprocal_rank     : float         # 1/rank of first correct result, 0 if none
    hit_at_k            : bool


class EvaluationReport(NamedTuple):
    video_hash          : str
    video_name          : str
    k                   : int
    num_queries         : int
    mean_precision_at_k : float
    mean_recall_at_k    : float
    mrr                 : float         # Mean Reciprocal Rank
    hit_rate_at_k       : float         # fraction of queries with at least 1 hit
    query_results       : list          # list[QueryResult]


# ------------------------------------------------------------------
# Core relevance check
# ------------------------------------------------------------------
def is_relevant(
    retrieved_ts    : float,
    ground_truth_ts : list[float],
    tolerance_sec   : float,
) -> bool:
    """
    Return True if retrieved_ts is within tolerance_sec of any
    ground-truth timestamp.
    """
    return any(
        abs(retrieved_ts - gt_ts) <= tolerance_sec
        for gt_ts in ground_truth_ts
    )


# ------------------------------------------------------------------
# Per-query evaluation
# ------------------------------------------------------------------
def evaluate_query(
    query               : str,
    relevant_timestamps : list[float],
    k                   : int,
    video_hash          : str,
    tolerance_sec       : float,
) -> QueryResult:
    """
    Run the full RAG pipeline for a query and evaluate the results.

    Parameters
    ----------
    query               : natural language query string
    relevant_timestamps : ground-truth timestamps from annotations
    k                   : evaluate top-k results
    video_hash          : which video to search
    tolerance_sec       : timestamp match tolerance in seconds

    Returns
    -------
    QueryResult with all metrics computed.
    """
    from retrieval.searcher import search
    from retrieval.reranker import rerank

    # Run retrieval
    response = search(query, video_hash)
    reranked = rerank(query, response.results) if response.results else []

    # Take top-k
    top_k = reranked[:k]

    retrieved_timestamps = [r.timestamp for r in top_k]

    # Compute per-result relevance flags
    hits = [
        is_relevant(ts, relevant_timestamps, tolerance_sec)
        for ts in retrieved_timestamps
    ]

    # Precision@k — fraction of retrieved that are relevant
    precision_at_k = sum(hits) / k if k > 0 else 0.0

    # Recall@k — fraction of relevant timestamps retrieved
    # A ground-truth timestamp is "found" if any retrieved result hits it
    found_gt = set()
    for ts in retrieved_timestamps:
        for gt_ts in relevant_timestamps:
            if abs(ts - gt_ts) <= tolerance_sec:
                found_gt.add(gt_ts)
    recall_at_k = len(found_gt) / len(relevant_timestamps) \
                  if relevant_timestamps else 0.0

    # Reciprocal Rank — 1 / rank of first correct result
    reciprocal_rank = 0.0
    for rank, hit in enumerate(hits, start=1):
        if hit:
            reciprocal_rank = 1.0 / rank
            break

    hit_at_k = any(hits)

    return QueryResult(
        query                = query,
        retrieved_timestamps = retrieved_timestamps,
        relevant_timestamps  = relevant_timestamps,
        hits                 = hits,
        precision_at_k       = round(precision_at_k, 4),
        recall_at_k          = round(recall_at_k, 4),
        reciprocal_rank      = round(reciprocal_rank, 4),
        hit_at_k             = hit_at_k,
    )


# ------------------------------------------------------------------
# Full evaluation run
# ------------------------------------------------------------------
def evaluate(
    annotations_path : str,
    k                : int   = 5,
    verbose          : bool  = True,
) -> EvaluationReport:
    """
    Run evaluation against all annotated queries.

    Parameters
    ----------
    annotations_path : path to annotations JSON from annotator.py
    k                : evaluate top-k results per query
    verbose          : print per-query results as they run

    Returns
    -------
    EvaluationReport with aggregate metrics.
    """
    from evaluation.annotator import load_annotations

    data         = load_annotations(annotations_path)
    video_hash   = data["video_hash"]
    video_name   = data["video_name"]
    annotations  = data["annotations"]
    tolerance    = data.get("timestamp_tolerance_sec", 30)

    if not annotations:
        raise ValueError("No annotations found in file.")

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Flashback Evaluation")
        print(f"  Video    : {video_name}")
        print(f"  Queries  : {len(annotations)}")
        print(f"  k        : {k}")
        print(f"  Tolerance: ±{tolerance}s")
        print(f"{'='*60}\n")

    query_results = []

    for i, ann in enumerate(annotations):
        query   = ann["query"]
        gt_ts   = ann["relevant_timestamps"]

        if verbose:
            print(f"[{i+1:2d}/{len(annotations)}] \"{query}\"")

        result = evaluate_query(
            query               = query,
            relevant_timestamps = gt_ts,
            k                   = k,
            video_hash          = video_hash,
            tolerance_sec       = tolerance,
        )
        query_results.append(result)

        if verbose:
            retrieved_str = ", ".join(_fmt_ts(t) for t in result.retrieved_timestamps)
            gt_str        = ", ".join(_fmt_ts(t) for t in gt_ts)
            hits_str      = " ".join("✓" if h else "✗" for h in result.hits)
            print(f"  GT       : {gt_str}")
            print(f"  Retrieved: {retrieved_str}")
            print(f"  Hits     : {hits_str}")
            print(f"  P@{k}={result.precision_at_k:.3f} "
                  f"R@{k}={result.recall_at_k:.3f} "
                  f"RR={result.reciprocal_rank:.3f}\n")

    # Aggregate metrics
    n = len(query_results)
    mean_precision  = sum(r.precision_at_k   for r in query_results) / n
    mean_recall     = sum(r.recall_at_k      for r in query_results) / n
    mrr             = sum(r.reciprocal_rank  for r in query_results) / n
    hit_rate        = sum(r.hit_at_k         for r in query_results) / n

    report = EvaluationReport(
        video_hash          = video_hash,
        video_name          = video_name,
        k                   = k,
        num_queries         = n,
        mean_precision_at_k = round(mean_precision, 4),
        mean_recall_at_k    = round(mean_recall, 4),
        mrr                 = round(mrr, 4),
        hit_rate_at_k       = round(hit_rate, 4),
        query_results       = query_results,
    )

    if verbose:
        _print_summary(report)

    return report


# ------------------------------------------------------------------
# Report saving
# ------------------------------------------------------------------
def save_report(report: EvaluationReport, output_path: str) -> None:
    """Save evaluation report as JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert NamedTuples to dicts for JSON serialisation
    data = {
        "video_hash"          : report.video_hash,
        "video_name"          : report.video_name,
        "k"                   : report.k,
        "num_queries"         : report.num_queries,
        "mean_precision_at_k" : report.mean_precision_at_k,
        "mean_recall_at_k"    : report.mean_recall_at_k,
        "mrr"                 : report.mrr,
        "hit_rate_at_k"       : report.hit_rate_at_k,
        "query_results"       : [
            {
                "query"               : r.query,
                "retrieved_timestamps": r.retrieved_timestamps,
                "relevant_timestamps" : r.relevant_timestamps,
                "hits"                : r.hits,
                "precision_at_k"      : r.precision_at_k,
                "recall_at_k"         : r.recall_at_k,
                "reciprocal_rank"     : r.reciprocal_rank,
                "hit_at_k"            : r.hit_at_k,
            }
            for r in report.query_results
        ],
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\n✓ Report saved to '{output_path}'")


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------
def _fmt_ts(seconds: float) -> str:
    h    = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{h}:{mins:02d}:{secs:02d}" if h > 0 else f"{mins:02d}:{secs:02d}"


def _print_summary(report: EvaluationReport) -> None:
    print(f"\n{'='*60}")
    print(f"  EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Video             : {report.video_name}")
    print(f"  Queries evaluated : {report.num_queries}")
    print(f"  k                 : {report.k}")
    print(f"  ─────────────────────────────────────────")
    print(f"  Precision@{report.k:<2}       : {report.mean_precision_at_k:.4f}"
          f"  ({report.mean_precision_at_k*100:.1f}%)")
    print(f"  Recall@{report.k:<2}          : {report.mean_recall_at_k:.4f}"
          f"  ({report.mean_recall_at_k*100:.1f}%)")
    print(f"  MRR               : {report.mrr:.4f}")
    print(f"  Hit Rate@{report.k:<2}        : {report.hit_rate_at_k:.4f}"
          f"  ({report.hit_rate_at_k*100:.1f}%)")
    print(f"{'='*60}\n")

    # Quick interpretation
    if report.hit_rate_at_k >= 0.8:
        print("  ✓ Strong retrieval — most queries find relevant content.")
    elif report.hit_rate_at_k >= 0.6:
        print("  ~ Moderate retrieval — room for improvement.")
    else:
        print("  ✗ Weak retrieval — consider tuning thresholds.")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Flashback retrieval quality"
    )
    parser.add_argument(
        "--annotations", required=True,
        help="Path to annotations JSON from annotator.py"
    )
    parser.add_argument(
        "--k", type=int, default=5,
        help="Evaluate top-k results (default: 5)"
    )
    parser.add_argument(
        "--output",
        help="Optional path to save detailed JSON report"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-query output, show summary only"
    )
    args = parser.parse_args()

    report = evaluate(
        annotations_path = args.annotations,
        k                = args.k,
        verbose          = not args.quiet,
    )

    if args.output:
        save_report(report, args.output)
