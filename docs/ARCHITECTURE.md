# Gyan architecture

## Pipeline

1. **Importers** (`gyan/ingest`) normalise every vendor export into `Conversation{messages}`.
   - ChatGPT: `conversations.json`, walking the `mapping` tree from `current_node`.
   - Claude: `conversations.json` with `chat_messages`.
   - Gemini: Takeout `MyActivity.json`, grouped by day.
   - Markdown: `User:` / `Assistant:` transcripts.
   - Dedup by content hash. `chunk()` splits a chat into question → answer pairs.
2. **Distiller agent** (`gyan/agent`): a forced tool call (`record_knowledge`) returns structured
   atoms, flashcards, concepts and typed relations. The system prompt and tool are prompt-cached for bulk imports.
   The heuristic distiller is the offline fallback.
3. **Embeddings** (`gyan/embed.py`) for chunks, atoms and concepts. `hash` is for dev; `sentence-transformers` is for real use.
4. **Store** (`gyan/store`): `Store` interface, `MonkDBStore` (production) and `LocalStore` (SQLite, for tests and dev).
5. **Recall engine** (`gyan/recall`):
   - `ask`: reciprocal-rank fusion of vector, full-text and graph-hop results, with provenance.
   - `daily`: due cards, plus asked-again concepts, on-this-day atoms and one surprise atom.
   - `review`: FSRS scheduling, then concept mastery, then XP and streak.
   - `quiz`: connection cards from 1–2 hop neighbours.
   - `wrapped`: weekly summary.

## Data model (MonkDB)

| Table | Notes |
|---|---|
| conversations | raw doc kept for provenance |
| chunks | `FLOAT_VECTOR(dim)` + fulltext index |
| atoms | knowledge units + memory aids; `FLOAT_VECTOR(dim)` + fulltext over title/summary/body |
| atom_concepts | atom ↔ concept join |
| concepts | graph vertices: mastery, times_asked, embedding |
| relations | graph edges: src, dst, type, weight (accumulates on repeat) |
| cards | FSRS state in `doc`, `due` indexed |
| reviews | time series of grades |

The graph runs natively (`CREATE GRAPH gyan_knowledge` + vertex/edge registration, `traverse()`) when
`MONKDB_NATIVE_GRAPH=1`, a licensed enterprise feature. Otherwise `neighbors()` does SQL hops over `relations`.

## Extending

- **New chat source:** add `ingest/<name>.py` with `parse(data) -> list[Conversation]` and register it in `PARSERS` and `detect()`.
- **New LLM:** implement `LLM.extract(transcript) -> dict` matching `prompts.TOOL["input_schema"]`.
- **New store:** subclass `Store`.

## Privacy

Local-first. Only distillation calls leave the machine, and only when `GYAN_LLM=anthropic`.
