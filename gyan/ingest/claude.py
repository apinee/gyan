"""Claude export (Settings -> Privacy -> Export data): conversations.json."""
from __future__ import annotations

from typing import Any

from ..models import Conversation, Message
from ._util import to_dt

ROLE = {"human": "user", "assistant": "assistant"}


def _text(m: dict) -> str:
    if m.get("text"):
        return m["text"].strip()
    parts = [b.get("text", "") for b in m.get("content") or [] if b.get("type") == "text"]
    return "\n".join(parts).strip()


def parse(data: list[dict[str, Any]]) -> list[Conversation]:
    convs = []
    for c in data:
        msgs = [Message(role=ROLE.get(m.get("sender"), m.get("sender", "")), text=_text(m),
                        ts=to_dt(m.get("created_at")))
                for m in c.get("chat_messages") or [] if _text(m)]
        convs.append(Conversation(id=str(c.get("uuid")), source="claude", title=c.get("name") or "Untitled",
                                  created_at=to_dt(c.get("created_at")), messages=msgs))
    return convs
