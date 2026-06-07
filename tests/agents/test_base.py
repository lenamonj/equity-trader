import os
import pytest
from equity_trader.config import ModelSpec
from equity_trader.agents.base import build_agent, _client_for, shared_rules_block


def test_client_for_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    c = _client_for(ModelSpec("openai", "gpt-4o"))
    assert c.base_url is None or "openai.com" in str(c.base_url)


def test_client_for_groq(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "x")
    c = _client_for(ModelSpec("groq", "llama-3.3-70b-versatile"))
    assert "groq.com" in str(c.base_url)


def test_client_for_deepseek(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "x")
    c = _client_for(ModelSpec("deepseek", "deepseek-chat"))
    assert "deepseek.com" in str(c.base_url)


def test_client_for_gemini_raises_without_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        _client_for(ModelSpec("gemini", "gemini-2.5-pro"))


def test_build_agent_constructs(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "x")
    a = build_agent(
        name="test_agent",
        instructions="You are a test.",
        tools=[],
        spec=ModelSpec("groq", "llama-3.3-70b-versatile"),
        output_type=dict,
    )
    assert a.name == "test_agent"


def test_shared_rules_block_mentions_horizon_and_schema():
    text = shared_rules_block()
    assert "3-6 month" in text
    assert "data_cited" in text
    assert "thesis" in text
    assert "risks" in text


def test_shared_rules_block_enforces_exact_recommendation_literals():
    # Models occasionally return SOLD/BOUGHT/HELD instead of BUY/HOLD/SELL,
    # which Pydantic rejects and we lose the whole verdict.
    text = shared_rules_block()
    assert "BUY, HOLD, SELL" in text
    assert "SOLD" in text  # explicit anti-example


def test_shared_rules_block_caps_tool_calls():
    text = shared_rules_block()
    assert "AT MOST ONCE" in text
