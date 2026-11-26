"""ChatGPT export (Settings -> Data controls -> Export): conversations.json.

Each conversation has a `mapping` tree of nodes; we walk from `current_node`
back to the root to recover the visible branch.
"""
from __future__ import annotations

from typing import Any

from ..models import Conversation, Message
from ._util import to_dt


def _text(msg: dict) -> str:
    content = msg.get("content") or {}
    parts = content.get("parts") or []
    out = [p if isinstance(p, str) else p.get("text", "") for p in parts if p]
    if not out and content.get("text"):
        out = [content["text"]]
    return "\n".join(x for x in out if x).strip()


def parse(data: list[dict[str, Any]]) -> list[Conversation]:
    convs = []
    for c in data:
        mapping = c.get("mapping") or {}
        node_id = c.get("current_node")
        path = []
        while node_id and node_id in mapping:
            path.append(mapping[node_id])
            node_id = mapping[node_id].get("parent")
        if not c.get("current_node"):  # fallback: insertion order
            path = list(mapping.values())[::-1]
        msgs = []
        for node in reversed(path):
            m = node.get("message")
            if not m:
                continue
            role = (m.get("author") or {}).get("role", "")
            text = _text(m)
            if role in ("user", "assistant") and text:
                msgs.append(Message(role=role, text=text, ts=to_dt(m.get("create_time"))))
        convs.append(Conversation(id=str(c.get("id") or c.get("conversation_id") or c.get("title")),
                                  source="chatgpt", title=c.get("title") or "Untitled",
                                  created_at=to_dt(c.get("create_time")), messages=msgs))
    return convs
