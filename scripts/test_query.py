"""Smoke-test the rag_index query path against the Pinecone indexes.

Runs a battery of manual-related queries through a searcher (dense by default)
with parent-section merging, prints ranked results, and exits non-zero if any
query returns no hits.

Usage (from repo root):
    python scripts\test_query.py                       # dense index
    python scripts\test_query.py --index-type hybrid   # hybrid index
    python scripts\test_query.py --top-k 3 --verbose
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_index.logging_config import query_logger  # noqa: E402
from rag_index.searcher import DenseSearcher, HybridSearcher  # noqa: E402

TEST_QUERIES = [
    "how do I adjust the seat belt",
    "how to replace the fuel filter",
    "DPF regeneration warning lamp",
    "urea tank capacity and refilling",
    "tire pressure specification",
    "jump start the tractor battery",
    "bluetooth phone pairing",
    "PTO switch operation",
    "transmission oil change interval",
    "remote engine start from the app",
]

CONTENT_SNIPPET_LEN = 160


def log_test_query(
    qi: int,
    query: str,
    index_type: str,
    merge: bool,
    top_k: int,
    elapsed: float,
    status: str,
    results: list,
) -> None:
    entry = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": "test_query",
        "test_index": qi,
        "status": status,
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


def print_result(rank: int, r: dict) -> None:
    md = r["metadata"]
    merged = r.get("merged", False)
    parts = r.get("num_merged", 1)
    tag = f"merged({parts})" if merged else "single"
    print(f"  #{rank} score={r['score']:.4f} [{tag}]")
    print(f"     heading: {md.get('heading')}")
    print(f"     level: {md.get('heading_level')}  type: {md.get('chunk_type')}")
    print(f"     pages: {md.get('page_numbers')}  sources: {md.get('sources')}")
    snippet = json.dumps((md.get("content") or "")[:CONTENT_SNIPPET_LEN])
    print(f"     content: {snippet}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-type", choices=["hybrid", "dense"], default="dense")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-merge", action="store_true", help="disable parent merging")
    parser.add_argument("--verbose", action="store_true", help="print every result instead of top-3")
    args = parser.parse_args()

    searcher = HybridSearcher() if args.index_type == "hybrid" else DenseSearcher()
    show = args.top_k if args.verbose else min(3, args.top_k)

    failures = []
    total_hits = 0
    t_all = time.perf_counter()

    for qi, query in enumerate(TEST_QUERIES, 1):
        t0 = time.perf_counter()
        results = searcher.search(query, top_k=args.top_k)
        if not args.no_merge:
            results = searcher.merge_by_parent(results, top_k=args.top_k)
        elapsed = time.perf_counter() - t0

        status = "OK" if results else "NO HITS"
        print(f"[{qi}/{len(TEST_QUERIES)}] {status} ({elapsed:.2f}s) {query!r}")
        log_test_query(qi, query, args.index_type, not args.no_merge, args.top_k, elapsed, status, results)
        if not results:
            failures.append(query)
            continue

        total_hits += len(results)
        for rank, r in enumerate(results[:show], 1):
            print_result(rank, r)
        print()

    print("=" * 60)
    print(
        f"{len(TEST_QUERIES) - len(failures)}/{len(TEST_QUERIES)} queries passed, "
        f"{total_hits} total hits, took {time.perf_counter() - t_all:.1f}s "
        f"(index-type={args.index_type}, merge={not args.no_merge})"
    )
    if failures:
        print("FAILED queries:")
        for q in failures:
            print(f"  - {q}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
