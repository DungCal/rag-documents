"""Hybrid search over the Pinecone index with parent-section merging.

Usage:
    python -m rag_index.query [--index-type {hybrid,dense}] "how do I adjust the seat belt"
"""
import argparse
import json
import sys
import time
from datetime import datetime

from .logging_config import query_logger
from .searcher import DenseSearcher, HybridSearcher


def log_query(
    query: str,
    index_type: str,
    merge: bool,
    top_k: int,
    elapsed: float,
    results: list,
) -> None:
    entry = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": "cli",
        "query": query,
        "index_type": index_type,
        "merge": merge,
        "top_k": top_k,
        "elapsed_seconds": round(elapsed, 3),
        "num_results": len(results),
        "results": [],
    }
    for rank, r in enumerate(results, 1):
        md = r["metadata"]
        entry["results"].append(
            {
                "rank": rank,
                "score": r["score"],
                "heading": md.get("heading"),
                "heading_level": md.get("heading_level"),
                "chunk_type": md.get("chunk_type"),
                "chunk_file": md.get("chunk_file"),
                "parent_chunk_file": md.get("parent_chunk_file"),
                "page_numbers": md.get("page_numbers"),
                "sources": md.get("sources"),
                "merged": r.get("merged", False),
                "num_merged": r.get("num_merged", 1),
                "content": md.get("content", ""),
            }
        )
    try:
        query_logger.info(json.dumps(entry, indent=2, ensure_ascii=False) + "\n")
    except Exception as exc:
        print(f"warning: failed to write query log: {exc}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", default=None)
    parser.add_argument("--index-type", choices=["hybrid", "dense"], default="hybrid")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--merge", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    query = args.query or sys.stdin.read().strip()

    searcher = DenseSearcher() if args.index_type == "dense" else HybridSearcher()
    t0 = time.perf_counter()
    results = searcher.search(query, top_k=args.top_k)
    if args.merge:
        results = searcher.merge_by_parent(results, top_k=args.top_k)
    elapsed = time.perf_counter() - t0

    for i, r in enumerate(results, 1):
        md = r["metadata"]
        print(f"#{i} score={r['score']:.4f} merged={r.get('merged', False)} ({r.get('num_merged', 1)} parts)")
        print(f"  heading: {md.get('heading')}")
        print(f"  level: {md.get('heading_level')}  type: {md.get('chunk_type')}")
        print(f"  pages: {md.get('page_numbers')}  sources: {md.get('sources')}")
        print(f"  content: {json.dumps(md.get('content', '')[:200])}")
        print()

    log_query(query, args.index_type, args.merge, args.top_k, elapsed, results)


if __name__ == "__main__":
    main()
