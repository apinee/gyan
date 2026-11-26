"""Settings, read from environment variables (see .env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Tiny .env loader so we don't need python-dotenv."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


@dataclass
class Settings:
    store: str = field(default_factory=lambda: _env("GYAN_STORE", "local"))
    db_path: str = field(default_factory=lambda: _env("GYAN_DB_PATH", "./data/gyan.db"))

    monkdb_host: str = field(default_factory=lambda: _env("MONKDB_HOST", "localhost"))
    monkdb_port: int = field(default_factory=lambda: int(_env("MONKDB_API_PORT", "4200")))
    monkdb_user: str = field(default_factory=lambda: _env("MONKDB_USER", "testuser"))
    monkdb_password: str = field(default_factory=lambda: _env("MONKDB_PASSWORD", ""))
    monkdb_schema: str = field(default_factory=lambda: _env("MONKDB_SCHEMA", "gyan"))
    monkdb_native_graph: bool = field(default_factory=lambda: _env("MONKDB_NATIVE_GRAPH", "0") == "1")

    llm: str = field(default_factory=lambda: _env("GYAN_LLM", "heuristic"))
    model: str = field(default_factory=lambda: _env("GYAN_MODEL", "claude-sonnet-5-5"))
    card_model: str = field(default_factory=lambda: _env("GYAN_CARD_MODEL", "claude-haiku-5-5"))

    embedder: str = field(default_factory=lambda: _env("GYAN_EMBEDDER", "hash"))
    embed_model: str = field(default_factory=lambda: _env("GYAN_EMBED_MODEL", "sentence-transformers/all-mpnet-base-v2"))
    embed_dim: int = field(default_factory=lambda: int(_env("GYAN_EMBED_DIM", "768")))


def get_settings() -> Settings:
    _load_dotenv()
    return Settings()
