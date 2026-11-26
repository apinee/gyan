"""Minimal FSRS (Free Spaced Repetition Scheduler, v4.5 formulas, default weights).

Grades: 1 Again, 2 Hard, 3 Good, 4 Easy. Interval chosen so that predicted recall = target retention.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta

from ..models import Card, now

W = [0.4872, 1.4003, 3.7145, 13.8206, 5.1618, 1.2298, 0.8975, 0.031, 1.6474, 0.1367,
     1.0461, 2.1072, 0.0793, 0.3246, 1.587, 0.2272, 2.8755]
TARGET_RETENTION = 0.9
MAX_INTERVAL_DAYS = 365 * 3


def retrievability(elapsed_days: float, stability: float) -> float:
    return (1 + elapsed_days / (9 * stability)) ** -1 if stability > 0 else 0.0


def _init_difficulty(g: int) -> float:
    return min(max(W[4] - (g - 3) * W[5], 1.0), 10.0)


def _next_difficulty(d: float, g: int) -> float:
    d2 = d - W[6] * (g - 3)
    d2 = W[7] * _init_difficulty(4) + (1 - W[7]) * d2  # mean reversion
    return min(max(d2, 1.0), 10.0)


def _interval(stability: float) -> float:
    days = stability * 9 * (1 / TARGET_RETENTION - 1)
    return min(max(days, 1 / 1440 * 10), MAX_INTERVAL_DAYS)  # at least 10 minutes


def schedule(card: Card, grade: int, at: datetime | None = None) -> Card:
    """Return an updated copy of `card` after a review with `grade`."""
    if grade not in (1, 2, 3, 4):
        raise ValueError("grade must be 1..4")
    at = at or now()
    c = card.model_copy()
    if c.reps == 0 or c.stability <= 0:
        c.stability = W[grade - 1]
        c.difficulty = _init_difficulty(grade)
    else:
        elapsed = max((at - (c.last_review or at)).total_seconds() / 86400, 0.0)
        r = retrievability(elapsed, c.stability)
        if grade == 1:
            c.lapses += 1
            c.stability = W[11] * c.difficulty ** -W[12] * ((c.stability + 1) ** W[13] - 1) * math.exp(W[14] * (1 - r))
        else:
            hard = W[15] if grade == 2 else 1.0
            easy = W[16] if grade == 4 else 1.0
            c.stability = c.stability * (1 + math.exp(W[8]) * (11 - c.difficulty) * c.stability ** -W[9]
                                         * (math.exp(W[10] * (1 - r)) - 1) * hard * easy)
        c.difficulty = _next_difficulty(c.difficulty, grade)
    c.reps += 1
    c.last_review = at
    # "Again" comes back in 10 minutes in the same session.
    c.due = at + (timedelta(minutes=10) if grade == 1 else timedelta(days=_interval(c.stability)))
    return c


def mastery(card: Card, at: datetime | None = None) -> float:
    """0..1 — current predicted recall probability."""
    if not card.last_review:
        return 0.0
    at = at or now()
    return retrievability((at - card.last_review).total_seconds() / 86400, card.stability)
