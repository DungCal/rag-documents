from huggingface_hub import InferenceClient

from . import config


class BGE_M3_Embedder:
    """Embed text with BGE-M3 via HuggingFace InferenceClient."""

    def __init__(self, token: str | None = None, model: str | None = None):
        self.token = token or config.HF_TOKEN
        self.model = model or config.BGE_M3_MODEL
        self.client = InferenceClient(model=self.model, token=self.token)

    def embed(self, text: str) -> list[float]:
        if not text:
            return [0.0] * config.EMBEDDING_DIM
        result = self.client.feature_extraction(text=text)
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

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]