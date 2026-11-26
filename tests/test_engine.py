from datetime import timedelta

from gyan.models import now


def test_end_to_end(engine, fixtures):
    for f in ("chatgpt.json", "claude.json", "gemini.json", "chat.md"):
        engine.ingest(fixtures / f)
    assert engine.ingest(fixtures / "claude.json") == {"skipped": 1}  # dedup

    hits = engine.ask("cosine similarity vector database")
    assert hits and "vector" in hits[0].atom.title.lower()
    assert "claude" in hits[0].source

    s = engine.daily(10)
    assert s.due
    card = s.due[0]
    updated = engine.review(card, 3)
    assert updated.due > now()
    assert card.id not in {c.id for c in engine.daily(10, at=now() + timedelta(minutes=1)).due}
    assert engine.stats()["xp"] == 10


def test_graph_quiz(engine, fixtures):
    engine.ingest(fixtures / "claude.json")
    c = engine.store.concepts(1)[0]
    assert engine.store.neighbors(c.id)
    assert engine.quiz(c.name)
