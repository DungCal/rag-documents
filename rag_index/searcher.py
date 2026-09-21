from collections import defaultdict

from pinecone import Pinecone
from pinecone_text.sparse import SpladeEncoder

from . import config
from .embedder import BGE_M3_Embedder


def merge_by_parent(results: list[dict], top_k: int = 10) -> list[dict]:
    """
    Group level-3 chunks by parent_chunk_file and merge their content
    into a single result block (per the hierarchical retrieval strategy).
    Level-2/special chunks pass through untouched.
    """
    groups: dict[str, dict] = {}
    singles: list[dict] = []

    for r in results:
        md = r["metadata"]
        parent_file = md.get("parent_chunk_file")
        if parent_file:
            g = groups.setdefault(
                parent_file,
                {
                    "id": parent_file,
                    "scores": [],
                    "content_parts": [],
                    "page_numbers": set(),
                    "sources": set(),
                    "parent_heading": md.get("parent_heading"),
                    "chunk_type": md.get("chunk_type"),
                    "heading_level": md.get("heading_level"),
                },
            )
            g["scores"].append(r["score"])
            g["content_parts"].append(md.get("content", ""))
            g["page_numbers"].update(md.get("page_numbers") or [])
            g["sources"].update(md.get("sources") or [])
        else:
            singles.append(r)

    merged = []
    for g in groups.values():
        merged.append(
            {
                "id": g["id"],
                "score": max(g["scores"]),
                "merged": True,
                "num_merged": len(g["content_parts"]),
                "metadata": {
                    "chunk_type": g["chunk_type"],
                    "heading_level": g["heading_level"],
                    "heading": g["parent_heading"],
                    "parent_heading": g["parent_heading"],
                    "content": "\n\n".join(g["content_parts"]),
                    "page_numbers": sorted(g["page_numbers"]),
                    "sources": sorted(g["sources"]),
                },
            }
        )

    merged.extend(singles)
    merged.sort(key=lambda x: x["score"], reverse=True)
    return merged[:top_k]


class HybridSearcher:
    """Hybrid (dense + sparse) search over Pinecone, then merge by parent section."""

    def __init__(
        self,
        embedder: BGE_M3_Embedder | None = None,
        api_key: str | None = None,
        index_name: str | None = None,
    ):
        self.embedder = embedder or BGE_M3_Embedder()
        self.index_name = index_name or config.PINECONE_INDEX_NAME
        self.pc = Pinecone(api_key=api_key or config.PINECONE_API_KEY)
        self.splade = SpladeEncoder()
        self.idx = self.pc.Index(self.index_name)

    def search(self, query: str, top_k: int = 10, namespace: str = "") -> list[dict]:
        dense = self.embedder.embed(query)
        sparse = self.splade.encode_queries(query)

        res = self.idx.query(
            vector=dense,
            sparse_vector=sparse,
            top_k=top_k,
            namespace=namespace,
            include_metadata=True,
        )
        return [
            {
                "id": hit["id"],
                "score": hit["score"],
                "metadata": hit.get("metadata", {}),
            }
            for hit in res.get("matches", [])
        ]

    @staticmethod
    def merge_by_parent(results: list[dict], top_k: int = 10) -> list[dict]:
        return merge_by_parent(results, top_k=top_k)


class DenseSearcher:
    """Dense-only search over Pinecone, then merge by parent section."""

    def __init__(
        self,
        embedder: BGE_M3_Embedder | None = None,
        api_key: str | None = None,
        index_name: str | None = None,
    ):
        self.embedder = embedder or BGE_M3_Embedder()
        self.index_name = index_name or config.PINECONE_DENSE_INDEX_NAME
        self.pc = Pinecone(api_key=api_key or config.PINECONE_API_KEY)
        self.idx = self.pc.Index(self.index_name)

    def search(self, query: str, top_k: int = 10, namespace: str = "") -> list[dict]:
        dense = self.embedder.embed(query)

        res = self.idx.query(
            vector=dense,
            top_k=top_k,
            namespace=namespace,
            include_metadata=True,
        )
        return [
            {
                "id": hit["id"],
                "score": hit["score"],
                "metadata": hit.get("metadata", {}),
            }
            for hit in res.get("matches", [])
        ]

    @staticmethod
    def merge_by_parent(results: list[dict], top_k: int = 10) -> list[dict]:
        return merge_by_parent(results, top_k=top_k)