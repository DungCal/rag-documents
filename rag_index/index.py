"""Index all hierarchical chunks from output/parsed/chunks into Pinecone.

Usage:
    python -m rag_index.index [--limit N]
"""
import argparse

from .chunker import build_all_chunks
from .embedder import BGE_M3_Embedder
from .indexer import PineconeIndexer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    chunks = build_all_chunks()
    if args.limit:
        chunks = chunks[: args.limit]
    print(f"Built {len(chunks)} hierarchical chunks")

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
    print(f"Indexing {len(kept)} chunks with text ({len(chunks) - len(kept)} empty skipped)")

    embedder = BGE_M3_Embedder()
    indexer = PineconeIndexer()

    print("Embedding documents with BGE-M3 ...")
    embeddings = embedder.embed_documents(texts)

    print("Upserting into Pinecone ...")
    indexer.upsert_chunks(kept, embeddings)
    print(f"Indexed {len(kept)} chunks into {indexer.pc.list_indexes().names()}")


if __name__ == "__main__":
    main()