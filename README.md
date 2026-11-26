# ज्ञान Gyan

**Turn everything you've learned from ChatGPT, Claude and Gemini into knowledge you can actually recall.**

You've had thousands of great conversations with AI. You remember almost none of them.
Gyan reads your chat exports, distills them into small memorable *knowledge atoms*,
links them into a personal knowledge graph, and brings them back with spaced repetition,
before you forget.

```
chat exports ─► importers ─► distiller agent ─► MonkDB (vectors + full-text + graph + time series)
                                                     │
                       ask · daily review · quiz · wrapped ◄── recall engine
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full design.

## Quick start (zero setup, offline)

```bash
pip install -e ".[dev]"
gyan import ~/Downloads/chatgpt-export.zip      # or conversations.json / Claude export / Gemini MyActivity.json / .md
gyan ask "how does attention work"
gyan review                                     # your daily 5
gyan quiz embeddings                            # connection questions from your graph
gyan stats
```

By default Gyan uses SQLite plus a tiny offline embedder and a keyword-based distiller, so it runs with no services or keys.

## Real mode: MonkDB + Claude

```bash
pip install -e ".[all]"
cp .env.example .env   # set GYAN_STORE=monkdb, GYAN_LLM=anthropic, GYAN_EMBEDDER=sentence-transformers
docker compose up -d monkdb
gyan doctor
```

| Feature | MonkDB feature used |
|---|---|
| Semantic recall | `FLOAT_VECTOR(768)` + `KNN_MATCH` |
| Keyword recall | `INDEX … USING FULLTEXT` + `MATCH` |
| Knowledge graph | `CREATE GRAPH / VERTEX TABLE / EDGE TABLE`, `traverse()` (set `MONKDB_NATIVE_GRAPH=1`; otherwise SQL hops) |
| Streaks, forgetting curves | time-series `reviews` table |

## Where the recall value comes from

- **Memory aids for every idea:** a one-line takeaway, an explain-like-I'm-5 version, an analogy and a memory hook.
- **Spaced repetition (FSRS):** a review lands right before you'd forget.
- **"You keep asking about…":** topics you've asked AIs about 3+ times get top priority.
- **On this day and surprise resurfacing:** old ideas you haven't reviewed come back.
- **Connection quizzes:** "How is X related to Y?" builds understanding, not trivia.
- **Provenance:** every answer links to the original chat ("Claude · 'Vector databases' · 01 Feb 2025").
- **Game layer:** XP, streaks, levels and concept mastery (🌰 Seed → 🌲 Forest), plus a weekly *Gyan Wrapped*.

## Interfaces

- **CLI:** `gyan …`
- **Web/API:** `uvicorn gyan.api.app:app --reload`, then open http://localhost:8000
- **MCP:** `python -m gyan.mcp_server` lets any AI assistant query what *you* already know

## Layout

```
gyan/
  ingest/      chatgpt.py claude.py gemini.py markdown.py  → Conversation
  agent/       distiller.py (LLM tool-use) heuristic.py (offline) llm.py prompts.py
  store/       base.py (interface) local.py (SQLite) monkdb.py + monkdb_*.sql
  recall/      engine.py fsrs.py game.py
  api/         FastAPI app + static/index.html
  cli.py  mcp_server.py  embed.py  models.py  config.py
tests/         pytest with fixtures for every importer
```

See [docs/ROADMAP.md](docs/ROADMAP.md) for what's next.
