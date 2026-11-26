"""Core data model shared by every layer.

Conversation -> Chunk -> Atom (knowledge unit) -> Card (recall prompt)
                         Atom -> Concept nodes + Relation edges (knowledge graph)
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


def now() -> datetime:
    return datetime.now(timezone.utc)


def stable_id(*parts: str) -> str:
    return hashlib.sha1("\x1f".join(parts).encode()).hexdigest()[:16]


# ---------- Ingestion ----------
class Message(BaseModel):
    role: str  # "user" | "assistant" | "system"
    text: str
    ts: Optional[datetime] = None


class Conversation(BaseModel):
    id: str
    source: str  # chatgpt | claude | gemini | markdown | ...
    title: str = "Untitled"
    created_at: Optional[datetime] = None
    messages: list[Message] = Field(default_factory=list)

    @property
    def content_hash(self) -> str:
        return stable_id(self.source, *(m.text for m in self.messages))


class Chunk(BaseModel):
    """One question -> answer exchange."""
    id: str
    conv_id: str
    idx: int
    question: str
    answer: str
    ts: Optional[datetime] = None
    embedding: Optional[list[float]] = None

    @property
    def text(self) -> str:
        return f"Q: {self.question}\nA: {self.answer}"


# ---------- Knowledge ----------
class AtomType(str, Enum):
    fact = "fact"
    concept = "concept"
    howto = "how-to"
    insight = "insight"
    decision = "decision"
    question = "question"


class Atom(BaseModel):
    """A small, self-contained unit of knowledge, plus memory aids."""
    id: str
    type: AtomType = AtomType.concept
    title: str
    summary: str  # one-line takeaway
    eli5: str = ""  # explain-like-I'm-5
    analogy: str = ""
    hook: str = ""  # mnemonic / memorable hook
    concepts: list[str] = Field(default_factory=list)
    chunk_ids: list[str] = Field(default_factory=list)
    conv_id: str = ""
    learned_at: Optional[datetime] = None
    embedding: Optional[list[float]] = None


class Concept(BaseModel):
    id: str
    name: str
    aliases: list[str] = Field(default_factory=list)
    mastery: float = 0.0  # 0..1, derived from card reviews
    times_asked: int = 0  # how often you asked AIs about it ("asked again" signal)
    embedding: Optional[list[float]] = None


class Relation(BaseModel):
    src: str  # concept id
    dst: str  # concept id
    type: str = "related_to"  # is_a | part_of | uses | contrasts_with | led_to | related_to
    weight: float = 1.0
    atom_id: str = ""  # evidence


class CardKind(str, Enum):
    qa = "qa"
    cloze = "cloze"
    why = "why"
    connect = "connect"  # "How is X related to Y?"


class Card(BaseModel):
    id: str
    atom_id: str
    kind: CardKind = CardKind.qa
    front: str
    back: str
    # FSRS scheduling state
    stability: float = 0.0
    difficulty: float = 0.0
    reps: int = 0
    lapses: int = 0
    due: datetime = Field(default_factory=now)
    last_review: Optional[datetime] = None


class Review(BaseModel):
    card_id: str
    ts: datetime = Field(default_factory=now)
    grade: int  # 1 Again, 2 Hard, 3 Good, 4 Easy


class Distillation(BaseModel):
    """What the distiller agent returns for one conversation."""
    atoms: list[Atom] = Field(default_factory=list)
    cards: list[Card] = Field(default_factory=list)
    concepts: list[Concept] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
