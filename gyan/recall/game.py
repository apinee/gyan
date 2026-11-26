"""The fun layer: XP, streaks, levels. Pure functions over the review log."""
from __future__ import annotations

from datetime import date, timedelta

from ..models import Review

XP_PER_GRADE = {1: 2, 2: 5, 3: 10, 4: 12}

LEVELS = [  # mastery thresholds for a concept
    (0.0, "🌰 Seed"),
    (0.25, "🌱 Sprout"),
    (0.5, "🌿 Sapling"),
    (0.75, "🌳 Tree"),
    (0.9, "🌲 Forest"),
]


def xp(reviews: list[Review]) -> int:
    return sum(XP_PER_GRADE.get(r.grade, 0) for r in reviews)


def player_level(total_xp: int) -> int:
    # Level n needs 50 * n^2 XP — quick early wins, slower later.
    n = 0
    while 50 * (n + 1) ** 2 <= total_xp:
        n += 1
    return n + 1


def streak(reviews: list[Review], today: date | None = None) -> int:
    days = {r.ts.date() for r in reviews}
    today = today or date.today()
    d = today if today in days else today - timedelta(days=1)  # today not done yet is fine
    n = 0
    while d in days:
        n += 1
        d -= timedelta(days=1)
    return n


def concept_level(mastery: float) -> str:
    label = LEVELS[0][1]
    for threshold, name in LEVELS:
        if mastery >= threshold:
            label = name
    return label
