"""Importers: each turns a vendor export into a list of `Conversation`s.

Use `load(path)` to auto-detect the format, or `load(path, source="chatgpt")`.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any, Callable

from ..models import Chunk, Conversation, stable_id
from . import chatgpt, claude, gemini, markdown

PARSERS: dict[str, Callable[[Any], list[Conversation]]] = {
    "chatgpt": chatgpt.parse,
    "claude": claude.parse,
    "gemini": gemini.parse,
}


def _read_json(path: Path) -> Any:
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            for name in ("conversations.json", "MyActivity.json"):
                hit = [n for n in z.namelist() if n.endswith(name)]
                if hit:
                    return json.loads(z.read(hit[0]))
        raise ValueError(f"No known export file inside {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def detect(data: Any) -> str:
    sample = data[0] if isinstance(data, list) and data else data
    if isinstance(sample, dict):
        if "mapping" in sample:
            return "chatgpt"
        if "chat_messages" in sample:
            return "claude"
        if "header" in sample and "time" in sample:
            return "gemini"
    raise ValueError("Could not detect export format; pass source= explicitly")


def load(path: str | Path, source: str | None = None) -> list[Conversation]:
    path = Path(path)
    if path.suffix in (".md", ".txt") or source == "markdown":
        return markdown.parse_file(path)
    data = _read_json(path)
    return PARSERS[source or detect(data)](data)


def chunk(conv: Conversation) -> list[Chunk]:
    """Split a conversation into user-question -> assistant-answer pairs."""
    chunks: list[Chunk] = []
    pending_q: list[str] = []
    q_ts = None
    for m in conv.messages:
        if m.role == "user":
            pending_q.append(m.text)
            q_ts = q_ts or m.ts
        elif m.role == "assistant" and pending_q:
            q = "\n".join(pending_q)
            chunks.append(Chunk(id=stable_id(conv.id, str(len(chunks)), q), conv_id=conv.id,
                                idx=len(chunks), question=q, answer=m.text, ts=q_ts or conv.created_at))
            pending_q, q_ts = [], None
    return chunks
