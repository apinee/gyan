from gyan.ingest import chunk, load


def test_chatgpt(fixtures):
    [c] = load(fixtures / "chatgpt.json")
    assert c.source == "chatgpt" and c.title.startswith("Attention")
    assert [m.role for m in c.messages] == ["user", "assistant", "user"]
    assert len(chunk(c)) == 1  # trailing "thanks" has no answer


def test_claude(fixtures):
    [c] = load(fixtures / "claude.json")
    assert c.source == "claude"
    assert len(chunk(c)) == 2
    assert "attention layers" in chunk(c)[1].answer


def test_gemini(fixtures):
    [c] = load(fixtures / "gemini.json")
    assert c.messages[0].text == "What is spaced repetition?"
    assert "growing intervals" in c.messages[1].text


def test_markdown(fixtures):
    [c] = load(fixtures / "chat.md")
    assert [m.role for m in c.messages] == ["user", "assistant"]
