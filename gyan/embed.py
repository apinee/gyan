"""Embedders. `hash` is a dependency-free dev embedder; use sentence-transformers for real use."""
from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np

from .config import Settings

_TOKEN = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashEmbedder:
    """Feature-hashing bag of words (+ bigrams). Deterministic, offline, good enough for tests."""

    def __init__(self, dim: int = 768):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        toks = _TOKEN.findall(text.lower())
        for tok in toks + [a + "_" + b for a, b in zip(toks, toks[1:])]:
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 64) & 1 else -1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t).tolist() for t in texts]


class SentenceTransformerEmbedder:
    def __init__(self, model: str):
        from sentence_transformers import SentenceTransformer  # optional dep

        self._m = SentenceTransformer(model)
        self.dim = self._m.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._m.encode(texts, normalize_embeddings=True).tolist()


def get_embedder(s: Settings) -> Embedder:
    if s.embedder == "sentence-transformers":
        return SentenceTransformerEmbedder(s.embed_model)
    return HashEmbedder(s.embed_dim)


def cosine(a: list[float], b: list[float]) -> float:
    va, vb = np.asarray(a), np.asarray(b)
    d = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(va @ vb / d) if d else 0.0
