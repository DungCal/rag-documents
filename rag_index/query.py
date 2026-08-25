"""Hybrid search over the Pinecone index with parent-section merging.

Usage:
    python -m rag_index.query [--index-type {hybrid,dense}] "how do I adjust the seat belt"
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from . import config
from .embedder import get_embedder
from .searcher import DenseSearcher, HybridSearcher

QUERY_LOG_FILE = "logs/queries.log"
SNIPPET_LEN = 200


def format_result(r: dict, idx: int) -> str:
    md = r["metadata"]
    lines = [
        f"#{idx} score={r['score']:.4f} merged={r.get('merged', False)} "
        f"({r.get('num_merged', 1)} part{'s' if r.get('num_merged', 1) != 1 else ''})",
        f"  heading: {md.get('heading')}",
        f"  level: {md.get('heading_level')}  type: {md.get('chunk_type')}",
        f"  pages: {md.get('page_numbers')}  sources: {md.get('sources')}",
    ]

    subsections = md.get("subsections")
    if subsections:
        lines.append(f"  subsections ({len(subsections)}):")
        for j, sub in enumerate(subsections, 1):
            snippet = (sub.get("snippet") or "").replace("\n", " ").strip()
            lines.append(f"    {j}. ### {sub.get('heading')}")
            lines.append(f"       \"{snippet}\"")
    else:
        content = (md.get("content") or "")[:SNIPPET_LEN].replace("\n", " ")
        lines.append(f"  content: {json.dumps(content)}")

    return "\n".join(lines)


def write_query_log(
    output_file: str,
    query: str,
    results: list[dict],
    index_type: str,
    embedder: str,
    merge: bool,
    top_k: int,
) -> None:
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"=== QUERY {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        f.write(f"Query: {json.dumps(query)}\n")
        f.write(
            f"Index: {index_type} | Embedder: {embedder} | "
            f"Merge: {merge} | Top-K: {top_k}\n"
        )
        for i, r in enumerate(results, 1):
            f.write(format_result(r, i) + "\n\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", default=None)
    parser.add_argument("--index-type", choices=["hybrid", "dense"], default="hybrid")
    parser.add_argument(
        "--embedder",
        choices=["local", "hf"],
        default=config.EMBEDDER_TYPE,
        help="Embedding backend: 'local' (SentenceTransformer) or 'hf' (InferenceClient API)",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Custom path to local BGE-M3 model weights (used when --embedder=local)",
    )
    parser.add_argument(
        "--device",
        default=config.EMBEDDER_DEVICE,
        help="Device to run local model on: 'cuda', 'cuda:0', 'cpu', or 'auto' (default from config)",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--merge", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--host",
        default=config.PINECONE_HOST,
        help="Pinecone host URL (e.g. 'http://localhost:5080' for Pinecone Local emulator)",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Shorthand to force using local Pinecone emulator at http://localhost:5080",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        default=QUERY_LOG_FILE,
        help=f"Append results to this log file (default: {QUERY_LOG_FILE})",
    )
    args = parser.parse_args()

    pinecone_host = "http://localhost:5080" if args.local else (args.host or None)
    query = args.query or sys.stdin.read().strip()

    embedder_kwargs = {}
    if args.model_path:
        embedder_kwargs["model_path"] = args.model_path
    if args.device:
        embedder_kwargs["device"] = args.device

    embedder = get_embedder(embedder_type=args.embedder, **embedder_kwargs)

    searcher = (
        DenseSearcher(embedder=embedder, host=pinecone_host)
        if args.index_type == "dense"
        else HybridSearcher(embedder=embedder, host=pinecone_host)
    )
    results = searcher.search(query, top_k=args.top_k)
    if args.merge:
        results = searcher.merge_by_parent(results, top_k=args.top_k)

    formatted = "\n\n".join(format_result(r, i) for i, r in enumerate(results, 1))
    print(formatted)

    write_query_log(
        output_file=args.output_file,
        query=query,
        results=results,
        index_type=args.index_type,
        embedder=args.embedder,
        merge=args.merge,
        top_k=args.top_k,
    )
    print(f"\nResults appended to {args.output_file}")


if __name__ == "__main__":
    main()