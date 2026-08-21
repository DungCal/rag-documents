"""Hybrid search over the Pinecone index with parent-section merging.

Usage:
    python -m rag_index.query [--index-type {hybrid,dense}] "how do I adjust the seat belt"
"""
import argparse
import json
import sys

from .searcher import DenseSearcher, HybridSearcher


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", default=None)
    parser.add_argument("--index-type", choices=["hybrid", "dense"], default="hybrid")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--merge", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    query = args.query or sys.stdin.read().strip()

    searcher = DenseSearcher() if args.index_type == "dense" else HybridSearcher()
    results = searcher.search(query, top_k=args.top_k)
    if args.merge:
        results = searcher.merge_by_parent(results, top_k=args.top_k)

    for i, r in enumerate(results, 1):
        md = r["metadata"]
        print(f"#{i} score={r['score']:.4f} merged={r.get('merged', False)} ({r.get('num_merged', 1)} parts)")
        print(f"  heading: {md.get('heading')}")
        print(f"  level: {md.get('heading_level')}  type: {md.get('chunk_type')}")
        print(f"  pages: {md.get('page_numbers')}  sources: {md.get('sources')}")
        print(f"  content: {json.dumps(md.get('content', '')[:200])}")
        print()


if __name__ == "__main__":
    main()