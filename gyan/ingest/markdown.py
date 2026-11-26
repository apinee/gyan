"""Generic markdown/text transcripts. Lines starting with `User:` / `Assistant:`
(or `## User` / `## Assistant`, or `You:` / `ChatGPT:` / `Claude:` / `Gemini:`) start a new turn."""
from __future__ import annotations

import re
from pathlib import Path

from ..models import Conversation, Message, stable_id

_TURN = re.compile(r"^(?:#+\s*)?(user|you|human|me|assistant|ai|chatgpt|claude|gemini|bot)\s*:?\s*(.*)$", re.I)
_USER = {"user", "you", "human", "me"}


def parse_text(text: str, title: str = "Untitled", source: str = "markdown") -> Conversation:
    msgs: list[Message] = []
    for line in text.splitlines():
        m = _TURN.match(line.strip())
        if m and (line.strip().endswith(":") or ":" in line[:20] or line.lstrip().startswith("#")):
            role = "user" if m.group(1).lower() in _USER else "assistant"
            msgs.append(Message(role=role, text=m.group(2)))
        elif msgs:
            msgs[-1].text = (msgs[-1].text + "\n" + line).strip()
    return Conversation(id=stable_id(source, title, text[:200]), source=source, title=title, messages=msgs)


def parse_file(path: Path) -> list[Conversation]:
    return [parse_text(path.read_text(encoding="utf-8"), title=path.stem)]
