import time
import random

from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError

from . import config
from .logging_config import logger


def _retry_embedding(func, text: str, max_retries: int = 5, base_delay: float = 2.0) -> list[float]:
    """Retry embedding with exponential backoff on transient HF API errors."""
    for attempt in range(max_retries):
        try:
            return func(text)
        except HfHubHTTPError as e:
            status = getattr(e.response, "status_code", 0)
            if status in (429, 502, 503, 504):
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(
                        "Embedding failed (status=%d), retry %d/%d in %.1fs: %s",
                        status, attempt + 1, max_retries, delay, str(e)[:100],
                    )
                    time.sleep(delay)
                    continue
            raise
        except Exception as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(
                    "Embedding error, retry %d/%d in %.1fs: %s",
                    attempt + 1, max_retries, delay, str(e)[:100],
                )
                time.sleep(delay)
                continue
            raise
    raise RuntimeError("Max retries exceeded")


class BGE_M3_Embedder:
    """Embed text with BGE-M3 via HuggingFace InferenceClient."""

    def __init__(self, token: str | None = None, model: str | None = None):
        self.token = token or config.HF_TOKEN
        self.model = model or config.BGE_M3_MODEL
        # Per-request timeout so a hung connection fails into the retry path
        # instead of blocking forever.
        self.client = InferenceClient(model=self.model, token=self.token, timeout=60.0)

    def embed(self, text: str) -> list[float]:
        if not text:
            return [0.0] * config.EMBEDDING_DIM

        def _do_embed(t: str) -> list[float]:
            result = self.client.feature_extraction(text=t)
            arr = (
                result.tolist() if hasattr(result, "tolist") else list(result)
            )
            if len(arr) != config.EMBEDDING_DIM:
                # Model may return a batch wrapper (nested); unwrap the first row.
                if arr and len(arr[0]) == config.EMBEDDING_DIM:
                    arr = arr[0]
                else:
                    raise ValueError(
                        f"Unexpected embedding dim {len(arr)}, expected {config.EMBEDDING_DIM}"
                    )
            return arr

        return _retry_embedding(_do_embed, text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        total = len(texts)
        embeddings = []
        for i, text in enumerate(texts, 1):
            start = time.perf_counter()
            emb = self.embed(text)
            elapsed = time.perf_counter() - start
            embeddings.append(emb)
            logger.info(
                "Embedding chunk %d/%d: dim=%d, took=%.3fs",
                i,
                total,
                len(emb),
                elapsed,
            )
        return embeddings