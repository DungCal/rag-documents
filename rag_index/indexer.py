from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import SpladeEncoder

from . import config


def _slugify(text: str) -> str:
    import re

    t = re.sub(r"[^A-Za-z0-9/._-]+", "_", text).strip("_")
    return t or "chunk"


def _sparse_values(splade: SpladeEncoder, text: str) -> dict:
    res = splade.encode_documents(text)
    return {"indices": res["indices"], "values": res["values"]}


def _clean_metadata(metadata: dict) -> dict:
    """Pinecone rejects None/null metadata values; drop them and stringify non-list scalars."""
    cleaned = {}
    for k, v in metadata.items():
        if v is None:
            continue
        if isinstance(v, list):
            cleaned[k] = [str(x) for x in v]
        else:
            cleaned[k] = v
    return cleaned


class PineconeIndexer:
    """Create/use a Pinecone index and upsert hierarchical chunk vectors."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or config.PINECONE_API_KEY
        self.pc = Pinecone(api_key=self.api_key)
        self.splade = SpladeEncoder()

    def ensure_index(self, index_name: str | None = None) -> str:
        name = index_name or config.PINECONE_INDEX_NAME
        if name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=name,
                dimension=config.EMBEDDING_DIM,
                metric=config.METRIC,
                spec=ServerlessSpec(
                    cloud="aws",
                    region=config.PINECONE_ENVIRONMENT or "us-east-1",
                ),
                vector_type="hybrid",
            )
        return name

    def upsert_chunks(
        self,
        chunks: list[dict],
        embeddings: list[list[float]],
        namespace: str = "",
        index_name: str | None = None,
    ) -> dict:
        name = self.ensure_index(index_name)
        idx = self.pc.Index(name)

        vectors = []
        for chunk, emb in zip(chunks, embeddings):
            sparse = _sparse_values(self.splade, chunk.get("content", ""))
            vectors.append(
                {
                    "id": _slugify(chunk["chunk_file"]),
                    "values": emb,
                    "sparse_values": sparse,
                    "metadata": _clean_metadata(
                        {
                            "chunk_file": chunk.get("chunk_file"),
                            "chunk_type": chunk.get("chunk_type"),
                            "heading_level": chunk.get("heading_level"),
                            "heading": chunk.get("heading"),
                            "parent_heading": chunk.get("parent_heading"),
                            "parent_chunk_file": chunk.get("parent_chunk_file"),
                            "content": chunk.get("content"),
                            "page_numbers": chunk.get("page_numbers"),
                            "sources": chunk.get("sources"),
                        }
                    ),
                }
            )
        return idx.upsert(vectors=vectors, namespace=namespace)