"""LLM providers for the distiller. Add OpenAI / Gemini / Ollama by implementing `LLM`."""
from __future__ import annotations

from typing import Any, Protocol

from ..config import Settings
from . import prompts


class LLM(Protocol):
    def extract(self, transcript: str) -> dict[str, Any]: ...


class AnthropicLLM:
    def __init__(self, model: str):
        import anthropic  # optional dep: pip install gyan[llm]

        self.client = anthropic.Anthropic()
        self.model = model

    def extract(self, transcript: str) -> dict[str, Any]:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            # Cache the (static) system prompt + tool schema across a bulk import.
            system=[{"type": "text", "text": prompts.SYSTEM, "cache_control": {"type": "ephemeral"}}],
            tools=[prompts.TOOL],
            tool_choice={"type": "tool", "name": prompts.TOOL["name"]},
            messages=[{"role": "user", "content": transcript}],
        )
        for block in resp.content:
            if block.type == "tool_use":
                return block.input
        return {"atoms": [], "relations": []}


def get_llm(s: Settings) -> LLM | None:
    if s.llm == "anthropic":
        return AnthropicLLM(s.model)
    return None  # -> heuristic distiller
