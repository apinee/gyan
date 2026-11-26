"""MonkDB implementation of Store.

Uses MonkDB's SQL interface via the official `monkdb` Python client (DB-API 2.0):
  - FLOAT_VECTOR + KNN_MATCH  -> semantic search over chunks/atoms/concepts
  - INDEX ... USING FULLTEXT + MATCH -> keyword search
  - CREATE GRAPH / VERTEX / EDGE TABLE + traverse()/graph_neighbors() -> knowledge graph
    (falls back to plain SQL hops when MONKDB_NATIVE_GRAPH=0)

NOTE: written against the MonkDB v3 docs; verify against your server version
(`pip install gyan[monkdb]`, then `gyan doctor`).
"""
from __future__ import annotations

from datetime import datetime
from importlib import resources

from ..config import Settings
from ..models import Atom, Card, Chunk, Concept, Conversation, Distillation, Relation, Review
from .base import Store


class MonkDBStore(Store):
    def __init__(self, s: Settings):
        from monkdb import client  # optional dep

        self.s = s
        self.schema = s.monkdb_schema
        self.native_graph = s.monkdb_native_graph
        self.conn = client.connect(f"http://{s.monkdb_user}:{s.monkdb_password}@{s.monkdb_host}:{s.monkdb_port}",
                                   username=s.monkdb_user)
        self.migrate(s.embed_dim)

    # -- plumbing ----------------------------------------------------------
    def _x(self, sql: str, params: tuple | list = ()) -> list[tuple]:
        cur = self.conn.cursor()
        try:
            cur.execute(sql.replace("{schema}", self.schema), params)
            try:
                return cur.fetchall()
            except Exception:
                return []
        finally:
            cur.close()

    def _refresh(self, *tables: str) -> None:
        # MonkDB (like other Lucene-backed engines) is near-real-time; refresh for read-your-writes.
        for t in tables:
            self._x(f"REFRESH TABLE {{schema}}.{t}")

    def migrate(self, dim: int) -> None:
        pkg = resources.files("gyan.store")
        files = ["monkdb_schema.sql"] + (["monkdb_graph.sql"] if self.native_graph else [])
        for f in files:
            ddl = pkg.joinpath(f).read_text().replace("{dim}", str(dim))
            for stmt in ddl.split(";\n"):
                body = "\n".join(l for l in stmt.splitlines() if not l.strip().startswith("--")).strip()
                if body:
                    self._x(body)

    # -- ingestion -------------------------------------------------------------
    def has_conversation(self, content_hash: str) -> bool:
        return bool(self._x("SELECT 1 FROM {schema}.conversations WHERE hash = ? LIMIT 1", (content_hash,)))

    def save_conversation(self, conv: Conversation) -> None:
        self._x("""INSERT INTO {schema}.conversations (id, hash, source, title, created_at, doc) VALUES (?,?,?,?,?,?)
                   ON CONFLICT (id) DO UPDATE SET hash = excluded.hash, doc = excluded.doc""",
                (conv.id, conv.content_hash, conv.source, conv.title,
                 conv.created_at.isoformat() if conv.created_at else None, conv.model_dump_json()))
        self._refresh("conversations")

    def save_chunks(self, chunks: list[Chunk]) -> None:
        for c in chunks:
            self._x("""INSERT INTO {schema}.chunks (id, conv_id, idx, text, ts, embedding) VALUES (?,?,?,?,?,?)
                       ON CONFLICT (id) DO UPDATE SET text = excluded.text, embedding = excluded.embedding""",
                    (c.id, c.conv_id, c.idx, c.text, c.ts.isoformat() if c.ts else None, c.embedding))
        self._refresh("chunks")

    def save_distillation(self, d: Distillation) -> None:
        for c in d.concepts:
            self._x("""INSERT INTO {schema}.concepts (id, name, aliases, mastery, times_asked, embedding)
                       VALUES (?,?,?,?,?,?)
                       ON CONFLICT (id) DO UPDATE SET times_asked = times_asked + excluded.times_asked""",
                    (c.id, c.name, c.aliases, c.mastery, max(c.times_asked, 1), c.embedding))
        name_to_id = {c.name: c.id for c in d.concepts}
        for a in d.atoms:
            self._x("""INSERT INTO {schema}.atoms (id, conv_id, type, title, summary, body, learned_at, doc, embedding)
                       VALUES (?,?,?,?,?,?,?,?,?)
                       ON CONFLICT (id) DO UPDATE SET summary = excluded.summary, doc = excluded.doc,
                                                      embedding = excluded.embedding""",
                    (a.id, a.conv_id, a.type.value, a.title, a.summary,
                     " ".join([a.eli5, a.analogy, a.hook, *a.concepts]),
                     a.learned_at.isoformat() if a.learned_at else None,
                     a.model_dump_json(exclude={"embedding"}), a.embedding))
            for name in a.concepts:
                if name in name_to_id:
                    self._x("INSERT INTO {schema}.atom_concepts (atom_id, concept_id) VALUES (?,?) ON CONFLICT DO NOTHING",
                            (a.id, name_to_id[name]))
        for r in d.relations:
            self._x("""INSERT INTO {schema}.relations (id, src, dst, type, weight, atom_id) VALUES (?,?,?,?,?,?)
                       ON CONFLICT (id) DO UPDATE SET weight = weight + excluded.weight""",
                    (f"{r.src}|{r.dst}|{r.type}", r.src, r.dst, r.type, r.weight, r.atom_id))
        for card in d.cards:
            self._x("INSERT INTO {schema}.cards (id, atom_id, due, doc) VALUES (?,?,?,?) ON CONFLICT DO NOTHING",
                    (card.id, card.atom_id, card.due.isoformat(), card.model_dump_json()))
        self._refresh("concepts", "atoms", "atom_concepts", "relations", "cards")

    # -- lookup / search -------------------------------------------------------
    @staticmethod
    def _atom(doc: str) -> Atom:
        return Atom.model_validate_json(doc)

    def get_atom(self, atom_id: str) -> Atom | None:
        rows = self._x("SELECT doc FROM {schema}.atoms WHERE id = ?", (atom_id,))
        return self._atom(rows[0][0]) if rows else None

    def get_conversation(self, conv_id: str) -> Conversation | None:
        rows = self._x("SELECT doc FROM {schema}.conversations WHERE id = ?", (conv_id,))
        return Conversation.model_validate_json(rows[0][0]) if rows else None

    def vector_search_atoms(self, vec: list[float], k: int = 10) -> list[tuple[Atom, float]]:
        rows = self._x("""SELECT doc, _score FROM {schema}.atoms WHERE KNN_MATCH(embedding, ?, ?)
                          ORDER BY _score DESC LIMIT ?""", (vec, k, k))
        return [(self._atom(doc), float(score)) for doc, score in rows]

    def text_search_atoms(self, query: str, k: int = 10) -> list[tuple[Atom, float]]:
        rows = self._x("""SELECT doc, _score FROM {schema}.atoms WHERE MATCH(atom_ft, ?)
                          ORDER BY _score DESC LIMIT ?""", (query, k))
        return [(self._atom(doc), float(score)) for doc, score in rows]

    def atoms_for_concept(self, concept_id: str, k: int = 20) -> list[Atom]:
        rows = self._x("""SELECT a.doc FROM {schema}.atoms a JOIN {schema}.atom_concepts ac ON ac.atom_id = a.id
                          WHERE ac.concept_id = ? ORDER BY a.learned_at DESC LIMIT ?""", (concept_id, k))
        return [self._atom(r[0]) for r in rows]

    def atoms_between(self, start: datetime, end: datetime) -> list[Atom]:
        rows = self._x("SELECT doc FROM {schema}.atoms WHERE learned_at >= ? AND learned_at < ?",
                       (start.isoformat(), end.isoformat()))
        return [self._atom(r[0]) for r in rows]

    # -- graph -------------------------------------------------------------------
    _CCOLS = "id, name, aliases, mastery, times_asked"

    @staticmethod
    def _concept(r) -> Concept:
        return Concept(id=r[0], name=r[1], aliases=r[2] or [], mastery=r[3] or 0.0, times_asked=r[4] or 0)

    def find_concept(self, name: str) -> Concept | None:
        rows = self._x(f"SELECT {self._CCOLS} FROM {{schema}}.concepts WHERE name = ?", (name.strip().lower(),))
        return self._concept(rows[0]) if rows else None

    def concepts(self, limit: int = 500) -> list[Concept]:
        rows = self._x(f"SELECT {self._CCOLS} FROM {{schema}}.concepts ORDER BY times_asked DESC LIMIT ?", (limit,))
        return [self._concept(r) for r in rows]

    def relations(self, limit: int = 2000) -> list[Relation]:
        rows = self._x("SELECT src, dst, type, weight, atom_id FROM {schema}.relations ORDER BY weight DESC LIMIT ?",
                       (limit,))
        return [Relation(src=r[0], dst=r[1], type=r[2], weight=r[3], atom_id=r[4] or "") for r in rows]

    def neighbors(self, concept_id: str, depth: int = 1) -> list[tuple[Concept, Relation]]:
        if self.native_graph:
            ids = [r[0] for r in self._x("SELECT vertex FROM traverse('gyan_knowledge', ?, ?)", (concept_id, depth))
                   if r[0] != concept_id]
        else:
            ids, frontier, seen = [], [concept_id], {concept_id}
            for _ in range(depth):
                if not frontier:
                    break
                rows = self._x("SELECT src, dst FROM {schema}.relations WHERE src = ANY(?) OR dst = ANY(?)",
                               (frontier, frontier))
                frontier = [x for s, t in rows for x in (s, t) if x not in seen]
                seen.update(frontier)
                ids += frontier
        out = []
        for cid in dict.fromkeys(ids):
            c = self._x(f"SELECT {self._CCOLS} FROM {{schema}}.concepts WHERE id = ?", (cid,))
            r = self._x("""SELECT src, dst, type, weight, atom_id FROM {schema}.relations
                           WHERE (src = ? AND dst = ?) OR (src = ? AND dst = ?) LIMIT 1""",
                        (concept_id, cid, cid, concept_id))
            if c:
                rel = Relation(src=r[0][0], dst=r[0][1], type=r[0][2], weight=r[0][3]) if r else \
                    Relation(src=concept_id, dst=cid, type="related_to")
                out.append((self._concept(c[0]), rel))
        return out

    def update_concept_mastery(self, concept_id: str, mastery: float) -> None:
        self._x("UPDATE {schema}.concepts SET mastery = ? WHERE id = ?", (mastery, concept_id))

    # -- recall --------------------------------------------------------------------
    def due_cards(self, at: datetime, limit: int = 20) -> list[Card]:
        rows = self._x("SELECT doc FROM {schema}.cards WHERE due <= ? ORDER BY due LIMIT ?", (at.isoformat(), limit))
        return [Card.model_validate_json(r[0]) for r in rows]

    def cards_for_atom(self, atom_id: str) -> list[Card]:
        rows = self._x("SELECT doc FROM {schema}.cards WHERE atom_id = ?", (atom_id,))
        return [Card.model_validate_json(r[0]) for r in rows]

    def save_card(self, card: Card) -> None:
        self._x("""INSERT INTO {schema}.cards (id, atom_id, due, doc) VALUES (?,?,?,?)
                   ON CONFLICT (id) DO UPDATE SET due = excluded.due, doc = excluded.doc""",
                (card.id, card.atom_id, card.due.isoformat(), card.model_dump_json()))
        self._refresh("cards")

    def add_review(self, r: Review) -> None:
        self._x("INSERT INTO {schema}.reviews (card_id, ts, grade) VALUES (?,?,?)", (r.card_id, r.ts.isoformat(), r.grade))
        self._refresh("reviews")

    def reviews_since(self, since: datetime) -> list[Review]:
        rows = self._x("SELECT card_id, ts, grade FROM {schema}.reviews WHERE ts >= ? ORDER BY ts", (since.isoformat(),))
        return [Review(card_id=r[0], ts=r[1] if isinstance(r[1], datetime) else datetime.fromtimestamp(r[1] / 1000),
                       grade=r[2]) for r in rows]

    def stale_atoms(self, before: datetime, k: int = 5) -> list[Atom]:
        reviewed = {r[0] for r in self._x("""SELECT DISTINCT c.atom_id FROM {schema}.cards c
                                              JOIN {schema}.reviews r ON r.card_id = c.id WHERE r.ts >= ?""",
                                           (before.isoformat(),))}
        rows = self._x("SELECT id, doc FROM {schema}.atoms ORDER BY random() LIMIT ?", (k * 4,))
        return [self._atom(doc) for aid, doc in rows if aid not in reviewed][:k]

    def counts(self) -> dict[str, int]:
        return {t: self._x(f"SELECT COUNT(*) FROM {{schema}}.{t}")[0][0]
                for t in ("conversations", "chunks", "atoms", "concepts", "relations", "cards", "reviews")}
