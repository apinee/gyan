"""MCP server: lets Claude / ChatGPT / any MCP client query *your* knowledge.

Run: python -m gyan.mcp_server   (requires `pip install gyan[mcp]`)
"""
from __future__ import annotations

from dataclasses import asdict


def main() -> None:
    from mcp.server.fastmcp import FastMCP

    from .app import engine

    mcp = FastMCP("gyan")

    @mcp.tool()
    def recall(query: str, k: int = 5) -> list[dict]:
        """Search what the user has previously learned from AI chats. Returns atoms with source chat."""
        return [{"title": h.atom.title, "summary": h.atom.summary, "hook": h.atom.hook, "source": h.source}
                for h in engine().ask(query, k)]

    @mcp.tool()
    def related_concepts(concept: str) -> list[dict]:
        """Concepts connected to `concept` in the user's knowledge graph."""
        c = engine().store.find_concept(concept)
        return [] if not c else [{"concept": n.name, "relation": r.type, "mastery": n.mastery}
                                 for n, r in engine().store.neighbors(c.id, 1)]

    @mcp.tool()
    def due_reviews(n: int = 5) -> dict:
        """Today's spaced-repetition session."""
        return asdict(engine().daily(n))

    mcp.run()


if __name__ == "__main__":
    main()
