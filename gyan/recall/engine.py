"""Recall engine: the part that makes knowledge *come back*.

- ask():        hybrid search (vector + full-text + graph hop), with provenance to your original chat
- daily():      today's due cards + one surprise resurfacing + "you keep asking about" + on-this-day
- review():     grade a card, reschedule with FSRS, update concept mastery, earn XP
- quiz():       connection questions generated from the knowledge graph
- wrapped():    weekly summary ("Gyan Wrapped")
"""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from ..agent.distiller import Distiller
from ..embed import Embedder
from ..ingest import load
from ..models import Atom, Card, CardKind, Concept, Review, now, stable_id
from ..store import Store
from . import fsrs, game

RRF_K = 60


@dataclass
class Hit:
    atom: Atom
    score: float
    why: list[str] = field(default_factory=list)  # "semantic", "keyword", "graph:<concept>"
    source: str = ""  # "ChatGPT · 'Title' · 2025-03-12"


@dataclass
class DailySession:
    due: list[Card]
    surprise: list[Atom]
    asked_again: list[Concept]
    on_this_day: list[Atom]


class RecallEngine:
    def __init__(self, store: Store, embedder: Embedder, distiller: Distiller):
        self.store, self.embedder, self.distiller = store, embedder, distiller

    # -- ingest ------------------------------------------------------------
    def ingest(self, path: str | Path, source: str | None = None) -> dict[str, int]:
        stats = Counter()
        for conv in load(path, source):
            if not conv.messages or self.store.has_conversation(conv.content_hash):
                stats["skipped"] += 1
                continue
            chunks, d = self.distiller.run(conv)
            self.store.save_conversation(conv)
            self.store.save_chunks(chunks)
            self.store.save_distillation(d)
            stats["conversations"] += 1
            stats["atoms"] += len(d.atoms)
            stats["cards"] += len(d.cards)
        return dict(stats)

    # -- ask -----------------------------------------------------------------
    def _provenance(self, a: Atom) -> str:
        conv = self.store.get_conversation(a.conv_id)
        if not conv:
            return ""
        when = (a.learned_at or conv.created_at)
        return f"{conv.source} · “{conv.title}”" + (f" · {when:%d %b %Y}" if when else "")

    def ask(self, query: str, k: int = 5) -> list[Hit]:
        vec = self.embedder.embed([query])[0]
        ranked: dict[str, Hit] = {}

        def add(results: list[tuple[Atom, float]], why: str) -> None:
            for rank, (atom, _) in enumerate(results):
                h = ranked.setdefault(atom.id, Hit(atom=atom, score=0.0))
                h.score += 1 / (RRF_K + rank)
                h.why.append(why)

        add(self.store.vector_search_atoms(vec, k * 2), "semantic")
        add(self.store.text_search_atoms(query, k * 2), "keyword")
        # Graph hop: atoms about concepts adjacent to the top concepts in the query.
        for word in {w.lower() for w in query.split()}:
            c = self.store.find_concept(word)
            if not c:
                continue
            add([(a, 0) for a in self.store.atoms_for_concept(c.id, k)], f"concept:{c.name}")
            for nb, _ in self.store.neighbors(c.id, depth=1)[:3]:
                add([(a, 0) for a in self.store.atoms_for_concept(nb.id, 2)], f"graph:{c.name}→{nb.name}")
        hits = sorted(ranked.values(), key=lambda h: -h.score)[:k]
        for h in hits:
            h.source = self._provenance(h.atom)
        return hits

    # -- daily ---------------------------------------------------------------
    def daily(self, n: int = 5, at: datetime | None = None) -> DailySession:
        at = at or now()
        asked_again = [c for c in self.store.concepts(50) if c.times_asked >= 3 and c.mastery < 0.75][:3]
        otd = []
        for delta in (7, 30, 365):
            day = at - timedelta(days=delta)
            start = day.replace(hour=0, minute=0, second=0, microsecond=0)
            otd += self.store.atoms_between(start, start + timedelta(days=1))
        return DailySession(due=self.store.due_cards(at, n),
                            surprise=self.store.stale_atoms(at - timedelta(days=30), 1),
                            asked_again=asked_again, on_this_day=otd[:3])

    # -- review ----------------------------------------------------------------
    def review(self, card: Card, grade: int, at: datetime | None = None) -> Card:
        at = at or now()
        updated = fsrs.schedule(card, grade, at)
        self.store.save_card(updated)
        self.store.add_review(Review(card_id=card.id, ts=at, grade=grade))
        atom = self.store.get_atom(card.atom_id)
        if atom:
            cards = self.store.cards_for_atom(atom.id)
            m = sum(fsrs.mastery(c, at) for c in cards) / max(len(cards), 1)
            for name in atom.concepts:
                c = self.store.find_concept(name)
                if c:
                    self.store.update_concept_mastery(c.id, max(c.mastery * 0.7 + m * 0.3, 0.0))
        return updated

    # -- quiz ---------------------------------------------------------------------
    def quiz(self, topic: str, n: int = 3) -> list[Card]:
        """Connection questions: 'How does X relate to Y?' — builds understanding, not trivia."""
        c = self.store.find_concept(topic)
        if not c:
            hits = self.ask(topic, 1)
            if not hits or not hits[0].atom.concepts:
                return []
            c = self.store.find_concept(hits[0].atom.concepts[0])
            if not c:
                return []
        nbs = self.store.neighbors(c.id, depth=2)
        random.shuffle(nbs)
        cards = []
        for nb, rel in nbs[:n]:
            evidence = self.store.atoms_for_concept(nb.id, 1)
            back = f"{c.name} —{rel.type.replace('_', ' ')}→ {nb.name}"
            if evidence:
                back += f". Remember: {evidence[0].summary}"
            cards.append(Card(id=stable_id("quiz", c.id, nb.id), atom_id=evidence[0].id if evidence else "",
                              kind=CardKind.connect, front=f"How is “{c.name}” connected to “{nb.name}”?", back=back))
        return cards

    # -- stats / wrapped ----------------------------------------------------------
    def stats(self) -> dict:
        reviews = self.store.reviews_since(now() - timedelta(days=3650))
        total = game.xp(reviews)
        return {**self.store.counts(), "xp": total, "level": game.player_level(total),
                "streak": game.streak(reviews)}

    def wrapped(self, days: int = 7) -> dict:
        since = now() - timedelta(days=days)
        atoms = self.store.atoms_between(since, now())
        topics = Counter(c for a in atoms for c in a.concepts).most_common(5)
        reviews = self.store.reviews_since(since)
        return {"period_days": days, "new_atoms": len(atoms), "top_topics": topics,
                "reviews": len(reviews), "xp": game.xp(reviews),
                "keep_asking": [c.name for c in self.store.concepts(20) if c.times_asked >= 3][:5]}
