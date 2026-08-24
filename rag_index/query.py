"""Hybrid search over the Pinecone index with parent-section merging.

Usage:
    python -m rag_index.query [--index-type {hybrid,dense}] "how do I adjust the seat belt"
"""
import argparse
import json
import sys

from . import config
from .embedder import get_embedder
from .searcher import DenseSearcher, HybridSearcher


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