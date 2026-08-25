import os
import random
import time
from pathlib import Path
from typing import Literal

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


class Local_BGE_M3_Embedder:
    """Embed text locally using BGE-M3 model weights via SentenceTransformer."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        device: str | None = None,
    ):
        raw_path = model_path or config.LOCAL_BGE_M3_PATH
        p = Path(raw_path)
        if not p.exists():
            # Try resolving relative to repository root
            repo_root = Path(__file__).resolve().parents[1]
            alt_path = repo_root / raw_path
            if alt_path.exists():
                p = alt_path
            else:
                raise FileNotFoundError(
                    f"Local BGE-M3 model path not found: {raw_path} (also checked {alt_path})"
                )

        self.model_path = str(p.resolve())
        dev = (device or config.EMBEDDER_DEVICE or "auto").strip().lower()
        if dev.startswith("cuda"):
            try:
                import torch

                if not torch.cuda.is_available():
                    raise RuntimeError(
                        f"Device '{dev}' requested for local BGE-M3 model, but CUDA is not available in the current PyTorch environment.\n"
                        "To enable GPU acceleration on NVIDIA GPUs, install CUDA-enabled PyTorch:\n"
                        "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124"
                    )
                self.device = dev
                gpu_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "GPU"
                logger.info("Forcing GPU execution on %s (device='%s')", gpu_name, self.device)
            except ImportError:
                raise RuntimeError("PyTorch is not installed in the environment.")
        elif dev == "auto":
            try:
                import torch

                if torch.cuda.is_available():
                    self.device = "cuda"
                    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "GPU"
                    logger.info("Auto-detected GPU: using %s (device='cuda')", gpu_name)
                else:
                    self.device = "cpu"
                    logger.info("CUDA not available: using CPU")
            except ImportError:
                self.device = "cpu"
        else:
            self.device = dev

        logger.info(
            "Initializing Local_BGE_M3_Embedder from '%s' on device='%s'",
            self.model_path,
            self.device,
        )
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(self.model_path, device=self.device)

    def embed(self, text: str) -> list[float]:
        if not text or not text.strip():
            return [0.0] * config.EMBEDDING_DIM
        emb = self.model.encode(text, normalize_embeddings=True, show_progress_bar=False)
        return emb.tolist() if hasattr(emb, "tolist") else list(emb)

    def embed_documents(
        self,
        texts: list[str],
        batch_size: int = 16,
    ) -> list[list[float]]:
        if not texts:
            return []
        total = len(texts)
        logger.info("Local embedding %d texts with batch_size=%d ...", total, batch_size)
        start = time.perf_counter()
        embs = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        elapsed = time.perf_counter() - start
        logger.info("Local embedding done: %d vectors in %.3fs", total, elapsed)
        if hasattr(embs, "tolist"):
            return embs.tolist()
        return [e.tolist() if hasattr(e, "tolist") else list(e) for e in embs]


class HF_BGE_M3_Embedder:
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


# Alias for backward compatibility
BGE_M3_Embedder = HF_BGE_M3_Embedder


def get_embedder(
    embedder_type: Literal["local", "hf"] | str | None = None,
    **kwargs,
) -> Local_BGE_M3_Embedder | HF_BGE_M3_Embedder:
    """Factory function to return either a local or Hugging Face BGE-M3 embedder."""
    selected_type = (embedder_type or config.EMBEDDER_TYPE or "local").lower()
    if selected_type in ("local", "local_bge_m3", "sentence_transformers"):
        return Local_BGE_M3_Embedder(**kwargs)
    elif selected_type in ("hf", "huggingface", "api"):
        # HF embedder only accepts token/model; drop CLI-only kwargs (model_path, device).
        hf_kwargs = {k: v for k, v in kwargs.items() if k in ("token", "model")}
        return HF_BGE_M3_Embedder(**hf_kwargs)
    else:
        raise ValueError(
            f"Unknown embedder_type: '{selected_type}'. Supported values are 'local' and 'hf'."
        )