"""SQLite + numpy implementation of Store. Zero setup; brute-force vector search."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np

from ..models import Atom, Card, Chunk, Concept, Conversation, Distillation, Relation, Review
from .base import Store

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (id TEXT PRIMARY KEY, hash TEXT UNIQUE, source TEXT, title TEXT,
  created_at TEXT, doc TEXT);
CREATE TABLE IF NOT EXISTS chunks (id TEXT PRIMARY KEY, conv_id TEXT, idx INT, text TEXT, ts TEXT, embedding TEXT);
CREATE TABLE IF NOT EXISTS atoms (id TEXT PRIMARY KEY, conv_id TEXT, title TEXT, summary TEXT, learned_at TEXT,
  doc TEXT, embedding TEXT);
CREATE TABLE IF NOT EXISTS atom_concepts (atom_id TEXT, concept_id TEXT, PRIMARY KEY (atom_id, concept_id));
CREATE TABLE IF NOT EXISTS concepts (id TEXT PRIMARY KEY, name TEXT UNIQUE, aliases TEXT, mastery REAL,
  times_asked INT, embedding TEXT);
CREATE TABLE IF NOT EXISTS relations (src TEXT, dst TEXT, type TEXT, weight REAL, atom_id TEXT,
  PRIMARY KEY (src, dst, type));
CREATE TABLE IF NOT EXISTS cards (id TEXT PRIMARY KEY, atom_id TEXT, due TEXT, doc TEXT);
CREATE TABLE IF NOT EXISTS reviews (card_id TEXT, ts TEXT, grade INT);
CREATE VIRTUAL TABLE IF NOT EXISTS atoms_fts USING fts5(id UNINDEXED, title, summary, body);
"""


def _iso(d: datetime | None) -> str | None:
    return d.isoformat() if d else None


class LocalStore(Store):
    def __init__(self, path: str = ":memory:", dim: int = 768):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.dim = dim

    # -- ingestion -------------------------------------------------------
    def has_conversation(self, content_hash: str) -> bool:
        return self.db.execute("SELECT 1 FROM conversations WHERE hash=?", (content_hash,)).fetchone() is not None

    def save_conversation(self, conv: Conversation) -> None:
        self.db.execute("INSERT OR REPLACE INTO conversations VALUES (?,?,?,?,?,?)",
                        (conv.id, conv.content_hash, conv.source, conv.title, _iso(conv.created_at),
                         conv.model_dump_json()))
        self.db.commit()

    def save_chunks(self, chunks: list[Chunk]) -> None:
        self.db.executemany("INSERT OR REPLACE INTO chunks VALUES (?,?,?,?,?,?)",
                            [(c.id, c.conv_id, c.idx, c.text, _iso(c.ts), json.dumps(c.embedding)) for c in chunks])
        self.db.commit()

    def save_distillation(self, d: Distillation) -> None:
        cur = self.db
        for c in d.concepts:
            row = cur.execute("SELECT times_asked FROM concepts WHERE id=?", (c.id,)).fetchone()
            if row:
                cur.execute("UPDATE concepts SET times_asked = times_asked + ? WHERE id=?", (max(c.times_asked, 1), c.id))
            else:
                cur.execute("INSERT INTO concepts VALUES (?,?,?,?,?,?)",
                            (c.id, c.name, json.dumps(c.aliases), c.mastery, max(c.times_asked, 1), json.dumps(c.embedding)))
        for a in d.atoms:
            doc = a.model_dump_json(exclude={"embedding"})
            cur.execute("INSERT OR REPLACE INTO atoms VALUES (?,?,?,?,?,?,?)",
                        (a.id, a.conv_id, a.title, a.summary, _iso(a.learned_at), doc, json.dumps(a.embedding)))
            cur.execute("DELETE FROM atoms_fts WHERE id=?", (a.id,))
            cur.execute("INSERT INTO atoms_fts VALUES (?,?,?,?)",
                        (a.id, a.title, a.summary, " ".join([a.eli5, a.analogy, a.hook, *a.concepts])))
            for name in a.concepts:
                cid = next((c.id for c in d.concepts if c.name == name), None)
                if cid:
                    cur.execute("INSERT OR IGNORE INTO atom_concepts VALUES (?,?)", (a.id, cid))
        for r in d.relations:
            cur.execute("""INSERT INTO relations VALUES (?,?,?,?,?)
                           ON CONFLICT(src,dst,type) DO UPDATE SET weight = weight + excluded.weight""",
                        (r.src, r.dst, r.type, r.weight, r.atom_id))
        for card in d.cards:
            if not cur.execute("SELECT 1 FROM cards WHERE id=?", (card.id,)).fetchone():
                self._put_card(card)
        cur.commit()

    # -- lookup ------------------------------------------------------------
    def _atom(self, row) -> Atom:
        return Atom.model_validate_json(row["doc"])

    def get_atom(self, atom_id: str) -> Atom | None:
        row = self.db.execute("SELECT doc FROM atoms WHERE id=?", (atom_id,)).fetchone()
        return self._atom(row) if row else None

    def get_conversation(self, conv_id: str) -> Conversation | None:
        row = self.db.execute("SELECT doc FROM conversations WHERE id=?", (conv_id,)).fetchone()
        return Conversation.model_validate_json(row["doc"]) if row else None

    def vector_search_atoms(self, vec: list[float], k: int = 10) -> list[tuple[Atom, float]]:
        rows = self.db.execute("SELECT doc, embedding FROM atoms WHERE embedding != 'null'").fetchall()
        if not rows:
            return []
        m = np.array([json.loads(r["embedding"]) for r in rows], dtype=np.float32)
        q = np.asarray(vec, dtype=np.float32)
        sims = m @ q / (np.linalg.norm(m, axis=1) * (np.linalg.norm(q) or 1) + 1e-9)
        top = np.argsort(-sims)[:k]
        return [(self._atom(rows[i]), float(sims[i])) for i in top]

    def text_search_atoms(self, query: str, k: int = 10) -> list[tuple[Atom, float]]:
        terms = " OR ".join(f'"{t}"' for t in query.replace('"', " ").split() if t)
        if not terms:
            return []
        rows = self.db.execute("""SELECT a.doc, bm25(atoms_fts) AS s FROM atoms_fts JOIN atoms a ON a.id = atoms_fts.id
                                  WHERE atoms_fts MATCH ? ORDER BY s LIMIT ?""", (terms, k)).fetchall()
        return [(self._atom(r), -float(r["s"])) for r in rows]

    def atoms_for_concept(self, concept_id: str, k: int = 20) -> list[Atom]:
        rows = self.db.execute("""SELECT a.doc FROM atoms a JOIN atom_concepts ac ON ac.atom_id = a.id
                                  WHERE ac.concept_id = ? ORDER BY a.learned_at DESC LIMIT ?""", (concept_id, k)).fetchall()
        return [self._atom(r) for r in rows]

    def atoms_between(self, start: datetime, end: datetime) -> list[Atom]:
        rows = self.db.execute("SELECT doc FROM atoms WHERE learned_at >= ? AND learned_at < ?",
                               (start.isoformat(), end.isoformat())).fetchall()
        return [self._atom(r) for r in rows]

    # -- graph ---------------------------------------------------------------
    def _concept(self, r) -> Concept:
        return Concept(id=r["id"], name=r["name"], aliases=json.loads(r["aliases"] or "[]"), mastery=r["mastery"],
                       times_asked=r["times_asked"])

    def find_concept(self, name: str) -> Concept | None:
        r = self.db.execute("SELECT * FROM concepts WHERE name = ?", (name.strip().lower(),)).fetchone()
        return self._concept(r) if r else None

    def concepts(self, limit: int = 500) -> list[Concept]:
        rows = self.db.execute("SELECT * FROM concepts ORDER BY times_asked DESC LIMIT ?", (limit,)).fetchall()
        return [self._concept(r) for r in rows]

    def relations(self, limit: int = 2000) -> list[Relation]:
        rows = self.db.execute("SELECT * FROM relations ORDER BY weight DESC LIMIT ?", (limit,)).fetchall()
        return [Relation(**dict(r)) for r in rows]

    def neighbors(self, concept_id: str, depth: int = 1) -> list[tuple[Concept, Relation]]:
        seen, frontier, out = {concept_id}, [concept_id], []
        for _ in range(depth):
            nxt = []
            for cid in frontier:
                rows = self.db.execute("SELECT * FROM relations WHERE src=? OR dst=?", (cid, cid)).fetchall()
                for r in rows:
                    other = r["dst"] if r["src"] == cid else r["src"]
                    if other in seen:
                        continue
                    seen.add(other)
                    nxt.append(other)
                    c = self.db.execute("SELECT * FROM concepts WHERE id=?", (other,)).fetchone()
                    if c:
                        out.append((self._concept(c), Relation(**dict(r))))
            frontier = nxt
        return out

    def update_concept_mastery(self, concept_id: str, mastery: float) -> None:
        self.db.execute("UPDATE concepts SET mastery=? WHERE id=?", (mastery, concept_id))
        self.db.commit()

    # -- recall ----------------------------------------------------------------
    def _put_card(self, card: Card) -> None:
        self.db.execute("INSERT OR REPLACE INTO cards VALUES (?,?,?,?)",
                        (card.id, card.atom_id, card.due.isoformat(), card.model_dump_json()))

    def due_cards(self, at: datetime, limit: int = 20) -> list[Card]:
        rows = self.db.execute("SELECT doc FROM cards WHERE due <= ? ORDER BY due LIMIT ?",
                               (at.isoformat(), limit)).fetchall()
        return [Card.model_validate_json(r["doc"]) for r in rows]

    def cards_for_atom(self, atom_id: str) -> list[Card]:
        rows = self.db.execute("SELECT doc FROM cards WHERE atom_id=?", (atom_id,)).fetchall()
        return [Card.model_validate_json(r["doc"]) for r in rows]

    def save_card(self, card: Card) -> None:
        self._put_card(card)
        self.db.commit()

    def add_review(self, r: Review) -> None:
        self.db.execute("INSERT INTO reviews VALUES (?,?,?)", (r.card_id, r.ts.isoformat(), r.grade))
        self.db.commit()

    def reviews_since(self, since: datetime) -> list[Review]:
        rows = self.db.execute("SELECT * FROM reviews WHERE ts >= ? ORDER BY ts", (since.isoformat(),)).fetchall()
        return [Review(card_id=r["card_id"], ts=datetime.fromisoformat(r["ts"]), grade=r["grade"]) for r in rows]

    def stale_atoms(self, before: datetime, k: int = 5) -> list[Atom]:
        rows = self.db.execute("""SELECT a.doc FROM atoms a WHERE NOT EXISTS (
                                    SELECT 1 FROM cards c JOIN reviews r ON r.card_id = c.id
                                    WHERE c.atom_id = a.id AND r.ts >= ?)
                                  ORDER BY RANDOM() LIMIT ?""", (before.isoformat(), k)).fetchall()
        return [self._atom(r) for r in rows]

    def counts(self) -> dict[str, int]:
        return {t: self.db.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                for t in ("conversations", "chunks", "atoms", "concepts", "relations", "cards", "reviews")}
