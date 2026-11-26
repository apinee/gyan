"""Wire everything together from Settings."""
from __future__ import annotations

from functools import lru_cache

from .agent.distiller import Distiller
from .agent.llm import get_llm
from .config import Settings, get_settings
from .embed import get_embedder
from .recall.engine import RecallEngine
from .store import get_store


def build(s: Settings | None = None) -> RecallEngine:
    s = s or get_settings()
    embedder = get_embedder(s)
    return RecallEngine(get_store(s), embedder, Distiller(embedder, get_llm(s)))


@lru_cache(maxsize=1)
def engine() -> RecallEngine:
    return build()
