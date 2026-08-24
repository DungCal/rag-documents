"""Index all hierarchical chunks from output/parsed/chunks into Pinecone.

Usage:
    python -m rag_index.index [--index-type {hybrid,dense}] [--limit N]
"""
import argparse
import time

from . import config
from .chunker import build_all_chunks
from .embedder import get_embedder
from .indexer import DenseIndexer, PineconeIndexer
from .logging_config import logger


def main() -> None:
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for embedding documents")
    parser.add_argument("--limit", type=int, default=None)
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
        "--metric",
        choices=["cosine", "dotproduct", "euclidean"],
        default=config.METRIC,
        help="Distance metric for index (default: from config/env)",
    )
    args = parser.parse_args()

    pinecone_host = "http://localhost:5080" if args.local else (args.host or None)

    logger.info(
        "=== Pipeline START (index-type=%s, embedder=%s, host=%s, metric=%s) ===",
        args.index_type,
        args.embedder,
        pinecone_host or "cloud",
        args.metric,
    )
    pipeline_start = time.perf_counter()

    t0 = time.perf_counter()
    chunks = build_all_chunks()
    if args.limit:
        chunks = chunks[: args.limit]
    logger.info("Chunker: built %d hierarchical chunks (took=%.3fs)", len(chunks), time.perf_counter() - t0)

    def embed_text(c: dict) -> str:
        parts = [p for p in (c.get("heading"), c.get("content")) if p]
        return "\n\n".join(parts).strip()

    # Drop chunks with no usable text (would produce a zero vector).
    kept, texts = [], []
    for c in chunks:
        t = embed_text(c)
        if t:
            kept.append(c)
            texts.append(t)
    logger.info(
        "Filter: %d chunks with text (%d empty skipped)", len(kept), len(chunks) - len(kept)
    )

    embedder_kwargs = {}
    if args.model_path:
        embedder_kwargs["model_path"] = args.model_path
    if args.device:
        embedder_kwargs["device"] = args.device

    embedder = get_embedder(embedder_type=args.embedder, **embedder_kwargs)
    indexer = (
        DenseIndexer(host=pinecone_host, metric=args.metric)
        if args.index_type == "dense"
        else PineconeIndexer(host=pinecone_host, metric=args.metric)
    )

    t0 = time.perf_counter()
    logger.info("Embedding %d chunks with %s ...", len(texts), args.embedder)
    if hasattr(embedder, "embed_documents"):
        try:
            embeddings = embedder.embed_documents(texts, batch_size=args.batch_size)
        except TypeError:
            embeddings = embedder.embed_documents(texts)
    else:
        embeddings = [embedder.embed(t) for t in texts]
    logger.info("Embedder: done, %d vectors (took=%.3fs)", len(embeddings), time.perf_counter() - t0)

    t0 = time.perf_counter()
    indexer.upsert_chunks(kept, embeddings)
    logger.info("Indexer: done, %d chunks upserted (took=%.3fs)", len(kept), time.perf_counter() - t0)

    logger.info("=== Pipeline END: total %.3fs, upserted=%d ===", time.perf_counter() - pipeline_start, len(kept))


if __name__ == "__main__":
    main()