from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Optional, Any

import requests


@dataclass(frozen=True)
class OllamaEmbeddingConfig:
    base_url: str = "http://localhost:11434"
    model: str = "nomic-embed-text"
    timeout_s: int = 120


class OllamaEmbedder:
    # Embedding adapter for Ollama's HTTP API.
    # Prefers the batch `/api/embed` endpoint and falls back to legacy `/api/embeddings`.
    def __init__(self, config: Optional[OllamaEmbeddingConfig] = None):
        self.config = config or OllamaEmbeddingConfig()

    @property
    def embedding_id(self) -> str:
        # Included in the manifest so index refresh triggers when embedding backend/model changes.
        return f"ollama:{self.config.model}"

    def embed_query(self, query: str) -> List[float]:
        return self.embed_texts([query])[0]

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        # Batch embedding via Ollama when available.
        items = [str(t) for t in texts]
        if not items:
            return []

        embed_url = f"{self.config.base_url}/api/embed"
        payload = {"model": self.config.model, "input": items}
        try:
            r = requests.post(embed_url, json=payload, timeout=self.config.timeout_s)
            if r.status_code == 404:
                raise RuntimeError("embed endpoint not available")
            r.raise_for_status()
            data = r.json()
            embeddings = data.get("embeddings")
            if isinstance(embeddings, list) and embeddings:
                return embeddings
        except Exception:
            return self._embed_via_legacy_endpoint(items)

        return self._embed_via_legacy_endpoint(items)

    def _embed_via_legacy_endpoint(self, items: Sequence[str]) -> List[List[float]]:
        # Legacy per-text embedding endpoint kept for backward compatibility.
        legacy_url = f"{self.config.base_url}/api/embeddings"
        out: List[List[float]] = []
        for text in items:
            payload = {"model": self.config.model, "prompt": text}
            r = requests.post(legacy_url, json=payload, timeout=self.config.timeout_s)
            r.raise_for_status()
            data = r.json()
            emb = data.get("embedding")
            if not isinstance(emb, list):
                raise RuntimeError("Invalid Ollama embeddings response")
            out.append(emb)
        return out


class SentenceTransformerEmbedder:
    # Adapter so callers can inject a sentence-transformers model while keeping the same interface.
    def __init__(self, model: Any, *, embedding_id: str = "sentence-transformers"):
        self._model = model
        self._embedding_id = embedding_id

    @property
    def embedding_id(self) -> str:
        return self._embedding_id

    def embed_query(self, query: str) -> List[float]:
        return self.embed_texts([query])[0]

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        arr = self._model.encode(list(texts), show_progress_bar=True)
        return arr.tolist()
