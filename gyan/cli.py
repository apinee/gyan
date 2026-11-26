"""`gyan` command line.

  gyan import <file> [--source chatgpt|claude|gemini|markdown]
  gyan ask "how does attention work"
  gyan review            # today's spaced-repetition session
  gyan quiz <topic>      # connection questions from the knowledge graph
  gyan stats | wrapped | doctor
"""
from __future__ import annotations

import argparse
import sys

from .app import build
from .config import get_settings
from .recall import game

GRADES = {"1": 1, "2": 2, "3": 3, "4": 4, "a": 1, "h": 2, "g": 3, "e": 4}


def cmd_import(e, a):
    print(e.ingest(a.path, a.source))


def cmd_ask(e, a):
    hits = e.ask(" ".join(a.query), a.k)
    if not hits:
        print("Nothing yet — import some chats first.")
    for i, h in enumerate(hits, 1):
        at = h.atom
        print(f"\n{i}. {at.title}\n   💡 {at.summary}")
        if at.hook:
            print(f"   🪝 {at.hook}")
        if at.analogy:
            print(f"   🔗 like: {at.analogy}")
        print(f"   📍 {h.source}   [{', '.join(dict.fromkeys(h.why))}]")


def _drill(e, cards):
    for card in cards:
        print(f"\n❓ {card.front}")
        input("   (think, then press Enter) ")
        print(f"✅ {card.back}")
        g = None
        while g is None:
            g = GRADES.get(input("   How did you do? [a]gain [h]ard [g]ood [e]asy: ").strip().lower()[:1])
        if card.atom_id:
            e.review(card, g)


def cmd_review(e, a):
    s = e.daily(a.n)
    if s.asked_again:
        print("🔁 You keep asking AIs about: " + ", ".join(c.name for c in s.asked_again))
    for atom in s.on_this_day:
        print(f"📅 On this day you learned: {atom.title} — {atom.summary}")
    for atom in s.surprise:
        print(f"🎁 Surprise from the vault: {atom.title} — {atom.summary}")
    if not s.due:
        print("🎉 Nothing due. Come back tomorrow!")
        return
    _drill(e, s.due)
    st = e.stats()
    print(f"\n🔥 streak {st['streak']} days · ⭐ {st['xp']} XP · level {st['level']}")


def cmd_quiz(e, a):
    cards = e.quiz(" ".join(a.topic))
    if not cards:
        print("No connections found for that topic yet.")
    _drill(e, cards)


def cmd_stats(e, a):
    st = e.stats()
    print(st)
    for c in e.store.concepts(10):
        print(f"  {game.concept_level(c.mastery)}  {c.name}  (asked {c.times_asked}×)")


def cmd_wrapped(e, a):
    print(e.wrapped(a.days))


def cmd_doctor(e, a):
    s = get_settings()
    print(f"store={s.store} llm={s.llm} embedder={s.embedder} dim={s.embed_dim}")
    print(e.store.counts())


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="gyan", description="Recall what you learned from AI chats.")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("import"); s.add_argument("path"); s.add_argument("--source"); s.set_defaults(f=cmd_import)
    s = sub.add_parser("ask"); s.add_argument("query", nargs="+"); s.add_argument("-k", type=int, default=5); s.set_defaults(f=cmd_ask)
    s = sub.add_parser("review"); s.add_argument("-n", type=int, default=5); s.set_defaults(f=cmd_review)
    s = sub.add_parser("quiz"); s.add_argument("topic", nargs="+"); s.set_defaults(f=cmd_quiz)
    sub.add_parser("stats").set_defaults(f=cmd_stats)
    s = sub.add_parser("wrapped"); s.add_argument("--days", type=int, default=7); s.set_defaults(f=cmd_wrapped)
    sub.add_parser("doctor").set_defaults(f=cmd_doctor)
    a = p.parse_args(argv)
    try:
        a.f(build(), a)
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
