"""Gemini via Google Takeout -> "My Activity" -> Gemini Apps -> MyActivity.json.

Takeout stores one activity entry per prompt (no conversation grouping), with the
prompt in `title` ("Prompted ...") and the response as HTML in `safeHtmlItem`.
We group entries from the same day into one conversation.
"""
from __future__ import annotations

import html
import re
from collections import defaultdict
from typing import Any

from ..models import Conversation, Message, stable_id
from ._util import to_dt

_TAG = re.compile(r"<[^>]+>")


def _strip_html(s: str) -> str:
    return html.unescape(_TAG.sub(" ", s)).strip()


def parse(data: list[dict[str, Any]]) -> list[Conversation]:
    by_day: dict[str, list[dict]] = defaultdict(list)
    for e in data:
        if "Gemini" not in e.get("header", "") and "Bard" not in e.get("header", ""):
            continue
        by_day[str(e.get("time", ""))[:10]].append(e)
    convs = []
    for day, entries in sorted(by_day.items()):
        msgs = []
        for e in sorted(entries, key=lambda x: x.get("time", "")):
            ts = to_dt(e.get("time"))
            prompt = re.sub(r"^Prompted\s+", "", e.get("title", "")).strip()
            answer = " ".join(_strip_html(i.get("html", "")) for i in e.get("safeHtmlItem") or [])
            if prompt:
                msgs.append(Message(role="user", text=prompt, ts=ts))
            if answer:
                msgs.append(Message(role="assistant", text=answer, ts=ts))
        convs.append(Conversation(id=stable_id("gemini", day), source="gemini", title=f"Gemini — {day}",
                                  created_at=to_dt(day), messages=msgs))
    return convs
