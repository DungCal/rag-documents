import time

from pinecone import Pinecone, ServerlessSpec
from pinecone.errors import PineconeError
from pinecone_text.sparse import SpladeEncoder

from . import config
from .logging_config import logger

UPSERT_BATCH_SIZE = 100
UPSERT_MAX_RETRIES = 5


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


def _upsert_with_retry(idx, vectors: list[dict], namespace: str) -> dict:
    """Upsert one batch with exponential backoff on transient Pinecone errors."""
    last_exc: Exception | None = None
    for attempt in range(UPSERT_MAX_RETRIES):
        try:
            return idx.upsert(vectors=vectors, namespace=namespace)
        except PineconeError as e:
            last_exc = e
            if attempt < UPSERT_MAX_RETRIES - 1:
                delay = 2.0 * (2**attempt) + (time.perf_counter() % 1.0)
                logger.warning(
                    "Upsert failed (%s), retry %d/%d in %.1fs",
                    str(e)[:120], attempt + 1, UPSERT_MAX_RETRIES, delay,
                )
                time.sleep(delay)
    raise last_exc


def _upsert_in_batches(idx, vectors: list[dict], namespace: str) -> dict:
    """Upsert vectors in batches to stay under request-size limits."""
    total_resp = {"upserted_count": 0}
    total = len(vectors)
    for start in range(0, total, UPSERT_BATCH_SIZE):
        batch = vectors[start : start + UPSERT_BATCH_SIZE]
        resp = _upsert_with_retry(idx, batch, namespace)
        total_resp["upserted_count"] += resp.get("upserted_count", 0)
        logger.info(
            "Upserted batch %d-%d/%d (upserted_count=%s)",
            start + 1, start + len(batch), total, resp.get("upserted_count"),
        )
    return total_resp


class PineconeIndexer:
    """Create/use a Pinecone index and upsert hierarchical chunk vectors."""

    def __init__(
        self,
        api_key: str | None = None,
        host: str | None = None,
        metric: str | None = None,
    ):
        self.host = host or config.PINECONE_HOST or None
        self.api_key = api_key or config.PINECONE_API_KEY or ("pclocal" if self.host else "")
        self.metric = metric or config.METRIC

        kwargs = {"api_key": self.api_key}
        if self.host:
            kwargs["host"] = self.host
        self.pc = Pinecone(**kwargs)
        self.splade = SpladeEncoder()

    def get_index(self, name: str):
        if self.host:
            desc = self.pc.describe_index(name)
            target_host = desc.host if desc.host.startswith("http") else f"http://{desc.host}"
            return self.pc.Index(host=target_host)
        return self.pc.Index(name)

    def ensure_index(self, index_name: str | None = None) -> str:
        name = index_name or config.PINECONE_INDEX_NAME
        has_idx = (
            self.pc.has_index(name)
            if hasattr(self.pc, "has_index")
            else (name in self.pc.list_indexes().names())
        )
        if not has_idx:
            logger.info(
                "Creating hybrid index '%s' (metric=%s, dim=%d) on %s ...",
                name,
                self.metric,
                config.EMBEDDING_DIM,
                self.host or "cloud",
            )
            self.pc.create_index(
                name=name,
                dimension=config.EMBEDDING_DIM,
                metric=self.metric,
                spec=ServerlessSpec(
                    cloud="aws",
                    region=config.PINECONE_ENVIRONMENT or "us-east-1",
                ),
                vector_type="dense",
                deletion_protection="disabled",
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
        idx = self.get_index(name)
        logger.info("Upserting %d chunks into index '%s' (namespace=%r)", len(chunks), name, namespace)

        vectors = []
        total = len(chunks)
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings), 1):
            start = time.perf_counter()
            sparse = _sparse_values(self.splade, chunk.get("content", ""))
            elapsed = time.perf_counter() - start
            cid = _slugify(chunk["chunk_file"])
            vectors.append(
                {
                    "id": cid,
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
            logger.info(
                "Upserting chunk %d/%d: id=%s, sparse_dim=%d, took=%.3fs",
                i,
                total,
                cid,
                len(sparse["indices"]),
                elapsed,
            )

        start = time.perf_counter()
        resp = _upsert_in_batches(idx, vectors, namespace)
        elapsed = time.perf_counter() - start
        logger.info("Pinecone upsert response: %s (took=%.3fs)", resp, elapsed)
        return resp


class DenseIndexer:
    """Create/use a dense-only Pinecone index and upsert chunk vectors without sparse values."""

    def __init__(
        self,
        api_key: str | None = None,
        host: str | None = None,
        metric: str | None = None,
    ):
        self.host = host or config.PINECONE_HOST or None
        self.api_key = api_key or config.PINECONE_API_KEY or ("pclocal" if self.host else "")
        self.metric = metric or config.METRIC

        kwargs = {"api_key": self.api_key}
        if self.host:
            kwargs["host"] = self.host
        self.pc = Pinecone(**kwargs)

    def get_index(self, name: str):
        if self.host:
            desc = self.pc.describe_index(name)
            target_host = desc.host if desc.host.startswith("http") else f"http://{desc.host}"
            return self.pc.Index(host=target_host)
        return self.pc.Index(name)

    def ensure_index(self, index_name: str | None = None) -> str:
        name = index_name or config.PINECONE_DENSE_INDEX_NAME
        has_idx = (
            self.pc.has_index(name)
            if hasattr(self.pc, "has_index")
            else (name in self.pc.list_indexes().names())
        )
        if not has_idx:
            logger.info(
                "Creating dense index '%s' (metric=%s, dim=%d) on %s ...",
                name,
                self.metric,
                config.EMBEDDING_DIM,
                self.host or "cloud",
            )
            self.pc.create_index(
                name=name,
                dimension=config.EMBEDDING_DIM,
                metric=self.metric,
                spec=ServerlessSpec(
                    cloud="aws",
                    region=config.PINECONE_ENVIRONMENT or "us-east-1",
                ),
                vector_type="dense",
                deletion_protection="disabled",
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
        idx = self.get_index(name)
        logger.info("Upserting %d chunks into dense index '%s' (namespace=%r)", len(chunks), name, namespace)

        vectors = []
        total = len(chunks)
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings), 1):
            cid = _slugify(chunk["chunk_file"])
            vectors.append(
                {
                    "id": cid,
                    "values": emb,
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
            logger.info(
                "Upserting chunk %d/%d: id=%s (dense-only)",
                i,
                total,
                cid,
            )

        start = time.perf_counter()
        resp = _upsert_in_batches(idx, vectors, namespace)
        elapsed = time.perf_counter() - start
        logger.info("Pinecone upsert response: %s (took=%.3fs)", resp, elapsed)
        return resp