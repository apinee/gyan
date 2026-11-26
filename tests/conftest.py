from pathlib import Path

import pytest

from gyan.agent.distiller import Distiller
from gyan.embed import HashEmbedder
from gyan.recall.engine import RecallEngine
from gyan.store.local import LocalStore

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures() -> Path:
    return FIX


@pytest.fixture
def engine() -> RecallEngine:
    emb = HashEmbedder(128)
    return RecallEngine(LocalStore(":memory:", dim=128), emb, Distiller(emb, llm=None))
