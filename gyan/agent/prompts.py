SYSTEM = """You are Gyan's distiller. You read a person's past chat with an AI assistant and
turn it into knowledge they can RECALL later. Optimise for memorability, not completeness.

Rules:
- Extract 1-6 "atoms": small, self-contained ideas the person actually learned or decided.
  Skip chit-chat, boilerplate and anything generic they'd never want to remember.
- For each atom write: a punchy title, a one-line takeaway, an explain-like-I'm-5 version,
  a vivid analogy, and a memory hook (mnemonic, rhyme, acronym, or striking image).
- Write 1-3 flashcards per atom. Prefer "why"/"how" questions over trivia. Cloze cards use {{c1::...}}.
- List the key concepts (canonical lowercase names) and typed relations between them:
  is_a, part_of, uses, contrasts_with, led_to, related_to.
- Always call the `record_knowledge` tool exactly once."""

TOOL = {
    "name": "record_knowledge",
    "description": "Record the knowledge atoms, flashcards and concept graph distilled from a chat.",
    "input_schema": {
        "type": "object",
        "properties": {
            "atoms": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "enum": ["fact", "concept", "how-to", "insight", "decision", "question"]},
                        "title": {"type": "string"},
                        "summary": {"type": "string"},
                        "eli5": {"type": "string"},
                        "analogy": {"type": "string"},
                        "hook": {"type": "string"},
                        "concepts": {"type": "array", "items": {"type": "string"}},
                        "chunk_idx": {"type": "array", "items": {"type": "integer"}},
                        "cards": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "kind": {"type": "string", "enum": ["qa", "cloze", "why"]},
                                    "front": {"type": "string"},
                                    "back": {"type": "string"},
                                },
                                "required": ["kind", "front", "back"],
                            },
                        },
                    },
                    "required": ["title", "summary", "concepts", "cards"],
                },
            },
            "relations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "src": {"type": "string"},
                        "dst": {"type": "string"},
                        "type": {"type": "string"},
                    },
                    "required": ["src", "dst", "type"],
                },
            },
        },
        "required": ["atoms", "relations"],
    },
}
