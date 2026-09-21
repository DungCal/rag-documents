"""Index all hierarchical chunks from output/parsed/chunks into Pinecone.

Usage:
    python -m rag_index.index [--index-type {hybrid,dense}] [--limit N]
"""
import argparse
import time

from .chunker import build_all_chunks
from .embedder import BGE_M3_Embedder
from .indexer import DenseIndexer, PineconeIndexer
from .logging_config import logger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-type", choices=["hybrid", "dense"], default="hybrid")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    logger.info("=== Pipeline START (index-type=%s) ===", args.index_type)
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

    embedder = BGE_M3_Embedder()
    indexer = DenseIndexer() if args.index_type == "dense" else PineconeIndexer()

    t0 = time.perf_counter()
    logger.info("Embedding %d chunks with BGE-M3 ...", len(texts))
    embeddings = embedder.embed_documents(texts)
    logger.info("Embedder: done, %d vectors (took=%.3fs)", len(embeddings), time.perf_counter() - t0)

    t0 = time.perf_counter()
    indexer.upsert_chunks(kept, embeddings)
    logger.info("Indexer: done, %d chunks upserted (took=%.3fs)", len(kept), time.perf_counter() - t0)

    logger.info("=== Pipeline END: total %.3fs, upserted=%d ===", time.perf_counter() - pipeline_start, len(kept))


if __name__ == "__main__":
    main()