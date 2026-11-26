"""Offline fallback distiller: no LLM, just keyword statistics. Lower quality, zero cost."""
from __future__ import annotations

import re
from collections import Counter
from typing import Any

from ..models import Chunk

STOP = set("""a an the and or but if then else of to in on for with by from as at is are was were be been being
it its this that these those i you he she we they me my your our their what which who whom how why when where
can could should would will shall do does did done have has had not no yes so than too very just also about into
over under more most some any each other such only own same here there all both few many much use using used
please thanks thank like get got make made want need know think one two way thing things really ok okay sure
let lets let's example e.g i.e etc""".split())

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9+#\-]{2,}")


def keywords(text: str, k: int = 5) -> list[str]:
    words = [w.lower() for w in _WORD.findall(text)]
    counts = Counter(w for w in words if w not in STOP)
    return [w for w, _ in counts.most_common(k)]


def _first_sentence(text: str, limit: int = 220) -> str:
    s = re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)[0]
    return s[:limit].rstrip() + ("…" if len(s) > limit else "")


def extract(chunks: list[Chunk]) -> dict[str, Any]:
    atoms, relations = [], []
    for c in chunks:
        kws = keywords(c.question + " " + c.answer)
        if not kws:
            continue
        summary = _first_sentence(c.answer)
        atoms.append({
            "type": "concept",
            "title": _first_sentence(c.question, 80),
            "summary": summary,
            "hook": " → ".join(kws[:3]),
            "concepts": kws[:4],
            "chunk_idx": [c.idx],
            "cards": [{"kind": "qa", "front": _first_sentence(c.question, 200), "back": summary}],
        })
        relations += [{"src": kws[0], "dst": k, "type": "related_to"} for k in kws[1:4]]
    return {"atoms": atoms, "relations": relations}
