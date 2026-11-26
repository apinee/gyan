"""The distiller agent: Conversation -> Distillation (atoms, cards, concepts, relations)."""
from __future__ import annotations

from typing import Any

from ..embed import Embedder
from ..ingest import chunk as chunk_conv
from ..models import (Atom, AtomType, Card, CardKind, Chunk, Concept, Conversation, Distillation,
                      Relation, stable_id)
from . import heuristic
from .llm import LLM

MAX_CHARS = 60_000  # keep a single request comfortably small


def concept_id(name: str) -> str:
    return stable_id("concept", name.strip().lower())


def _transcript(conv: Conversation, chunks: list[Chunk]) -> str:
    parts = [f"Chat title: {conv.title}\nSource: {conv.source}\n"]
    for c in chunks:
        parts.append(f"[chunk {c.idx}]\nUSER: {c.question}\nASSISTANT: {c.answer}\n")
    return "\n".join(parts)[:MAX_CHARS]


class Distiller:
    def __init__(self, embedder: Embedder, llm: LLM | None = None):
        self.embedder = embedder
        self.llm = llm

    def run(self, conv: Conversation) -> tuple[list[Chunk], Distillation]:
        chunks = chunk_conv(conv)
        if not chunks:
            return [], Distillation()
        raw = self.llm.extract(_transcript(conv, chunks)) if self.llm else heuristic.extract(chunks)
        d = self._build(conv, chunks, raw)
        self._embed(chunks, d)
        return chunks, d

    # -- internals -------------------------------------------------------
    def _build(self, conv: Conversation, chunks: list[Chunk], raw: dict[str, Any]) -> Distillation:
        by_idx = {c.idx: c for c in chunks}
        d = Distillation()
        concepts: dict[str, Concept] = {}

        def concept(name: str) -> Concept:
            key = name.strip().lower()
            if key not in concepts:
                concepts[key] = Concept(id=concept_id(key), name=key)
            return concepts[key]

        for a in raw.get("atoms", []):
            idxs = [i for i in a.get("chunk_idx", []) if i in by_idx] or [chunks[0].idx]
            atom = Atom(
                id=stable_id(conv.id, a["title"]),
                type=AtomType(a.get("type", "concept")) if a.get("type") in AtomType._value2member_map_ else AtomType.concept,
                title=a["title"], summary=a.get("summary", ""), eli5=a.get("eli5", ""),
                analogy=a.get("analogy", ""), hook=a.get("hook", ""),
                concepts=[c.strip().lower() for c in a.get("concepts", []) if c.strip()],
                chunk_ids=[by_idx[i].id for i in idxs], conv_id=conv.id,
                learned_at=by_idx[idxs[0]].ts or conv.created_at,
            )
            d.atoms.append(atom)
            for name in atom.concepts:
                concept(name).times_asked += 1
            for j, card in enumerate(a.get("cards", [])):
                d.cards.append(Card(id=stable_id(atom.id, str(j), card["front"]), atom_id=atom.id,
                                    kind=CardKind(card.get("kind", "qa")) if card.get("kind") in CardKind._value2member_map_ else CardKind.qa,
                                    front=card["front"], back=card["back"]))

        for r in raw.get("relations", []):
            if r["src"].strip().lower() == r["dst"].strip().lower():
                continue
            s, t = concept(r["src"]), concept(r["dst"])
            d.relations.append(Relation(src=s.id, dst=t.id, type=r.get("type", "related_to")))
        d.concepts = list(concepts.values())
        return d

    def _embed(self, chunks: list[Chunk], d: Distillation) -> None:
        texts = ([c.text for c in chunks] + [f"{a.title}. {a.summary}" for a in d.atoms] + [c.name for c in d.concepts])
        if not texts:
            return
        vecs = iter(self.embedder.embed(texts))
        for obj in [*chunks, *d.atoms, *d.concepts]:
            obj.embedding = next(vecs)
