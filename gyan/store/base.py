"""Storage interface. MonkDBStore is the production backend; LocalStore (SQLite) is for dev/tests."""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from ..models import Atom, Card, Chunk, Concept, Conversation, Distillation, Relation, Review


class Store(ABC):
    # -- ingestion -----------------------------------------------------
    @abstractmethod
    def has_conversation(self, content_hash: str) -> bool: ...

    @abstractmethod
    def save_conversation(self, conv: Conversation) -> None: ...

    @abstractmethod
    def save_chunks(self, chunks: list[Chunk]) -> None: ...

    @abstractmethod
    def save_distillation(self, d: Distillation) -> None:
        """Upsert atoms/cards; merge concepts (increment times_asked); add relation weights."""

    # -- lookup / search -------------------------------------------------
    @abstractmethod
    def get_atom(self, atom_id: str) -> Atom | None: ...

    @abstractmethod
    def get_conversation(self, conv_id: str) -> Conversation | None: ...

    @abstractmethod
    def vector_search_atoms(self, vec: list[float], k: int = 10) -> list[tuple[Atom, float]]: ...

    @abstractmethod
    def text_search_atoms(self, query: str, k: int = 10) -> list[tuple[Atom, float]]: ...

    @abstractmethod
    def atoms_for_concept(self, concept_id: str, k: int = 20) -> list[Atom]: ...

    @abstractmethod
    def atoms_between(self, start: datetime, end: datetime) -> list[Atom]: ...

    # -- graph -----------------------------------------------------------
    @abstractmethod
    def find_concept(self, name: str) -> Concept | None: ...

    @abstractmethod
    def concepts(self, limit: int = 500) -> list[Concept]: ...

    @abstractmethod
    def relations(self, limit: int = 2000) -> list[Relation]: ...

    @abstractmethod
    def neighbors(self, concept_id: str, depth: int = 1) -> list[tuple[Concept, Relation]]: ...

    @abstractmethod
    def update_concept_mastery(self, concept_id: str, mastery: float) -> None: ...

    # -- recall ------------------------------------------------------------
    @abstractmethod
    def due_cards(self, at: datetime, limit: int = 20) -> list[Card]: ...

    @abstractmethod
    def cards_for_atom(self, atom_id: str) -> list[Card]: ...

    @abstractmethod
    def save_card(self, card: Card) -> None: ...

    @abstractmethod
    def add_review(self, r: Review) -> None: ...

    @abstractmethod
    def reviews_since(self, since: datetime) -> list[Review]: ...

    @abstractmethod
    def stale_atoms(self, before: datetime, k: int = 5) -> list[Atom]:
        """Atoms whose cards haven't been reviewed since `before` (for "surprise" resurfacing)."""

    @abstractmethod
    def counts(self) -> dict[str, int]: ...
