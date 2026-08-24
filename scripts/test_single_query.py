"""Run one query and verify it lands as a pretty-printed JSON entry in the query log.

Usage (from repo root):
    python scripts\test_single_query.py
    python scripts\test_single_query.py "DPF regeneration warning lamp" --index-type hybrid
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_index import config  # noqa: E402
from rag_index.query import log_query  # noqa: E402
from rag_index.searcher import DenseSearcher, HybridSearcher  # noqa: E402

DEFAULT_QUERY = "how do I adjust the seat belt"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", default=DEFAULT_QUERY)
    parser.add_argument("--index-type", choices=["hybrid", "dense"], default="dense")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--no-merge", action="store_true")
    args = parser.parse_args()

    log_path = Path(config.QUERY_LOG_FILE)

    searcher = HybridSearcher() if args.index_type == "hybrid" else DenseSearcher()
    t0 = time.perf_counter()
    results = searcher.search(args.query, top_k=args.top_k)
    if not args.no_merge:
        results = searcher.merge_by_parent(results, top_k=args.top_k)
    elapsed = time.perf_counter() - t0

    print(f"query: {args.query!r} ({elapsed:.2f}s, {len(results)} hits)")
    for rank, r in enumerate(results, 1):
        md = r["metadata"]
        tag = f"merged({r.get('num_merged', 1)})" if r.get("merged") else "single"
        print(f"  #{rank} score={r['score']:.4f} [{tag}] {md.get('heading')}")

    log_query(args.query, args.index_type, not args.no_merge, args.top_k, elapsed, results)

    if not log_path.exists():
        print(f"FAIL: log file not created at {log_path}")
        return 1
    lines = log_path.read_text(encoding="utf-8").splitlines()
    start = None
    for i in range(len(lines) - 1, -1, -1):
        if lines[i] == "{":
            start = i
            break
    if start is None:
        print("FAIL: no pretty-printed JSON entry found in query log")
        return 1
    try:
        entry = json.loads("\n".join(lines[start:]))
    except json.JSONDecodeError as exc:
        print(f"FAIL: last log entry is not valid JSON: {exc}")
        return 1
    if entry.get("query") != args.query or entry.get("source") != "cli":
        print(f"FAIL: unexpected entry: {str(entry)[:200]}")
        return 1

    print(f"LOG OK: new JSON entry appended to {log_path}")
    print(f"  timestamp={entry['timestamp']}  num_results={entry['num_results']}  merge={entry['merge']}")
    if entry["results"]:
        top = entry["results"][0]
        preview = " ".join(top["content"][:120].split())
        print(f"  top result full content length: {len(top['content'])} chars")
        print(f"  top result preview: {preview}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
