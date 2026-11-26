from datetime import timedelta

from gyan.models import Card, now
from gyan.recall import fsrs, game
from gyan.models import Review


def test_intervals_grow_with_good_grades():
    c, t, gaps = Card(id="x", atom_id="a", front="q", back="a"), now(), []
    for _ in range(4):
        c = fsrs.schedule(c, 3, t)
        gaps.append(c.due - t)
        t = c.due
    assert all(b > a for a, b in zip(gaps, gaps[1:]))


def test_again_comes_back_soon():
    c = fsrs.schedule(Card(id="x", atom_id="a", front="q", back="a"), 3)
    c2 = fsrs.schedule(c, 1, c.due)
    assert c2.due - c.due == timedelta(minutes=10) and c2.lapses == 1


def test_streak():
    t = now()
    rs = [Review(card_id="x", ts=t - timedelta(days=d), grade=3) for d in (0, 1, 2, 4)]
    assert game.streak(rs, t.date()) == 3
