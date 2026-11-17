import pytest
from agents import OpenAIChatCompletionsModel, OpenAIResponsesModel
from pydantic import BaseModel

from equity_trader.agents.base import _client_for, build_agent, shared_rules_block
from equity_trader.config import ModelSpec


class Answer(BaseModel):
    text: str


def test_client_for_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    c = _client_for(ModelSpec("openai", "gpt-6-luna"))
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
        spec=ModelSpec("groq", "llama-3.3-70b-versatile"),
        output_type=Answer,
    )
    assert a.name == "test_agent"
    assert isinstance(a.model, OpenAIChatCompletionsModel)
    # A provider that cannot enforce a schema has its output checked afterwards.
    assert not a.output_type.is_strict_json_schema()


def test_build_agent_uses_responses_api_for_openai():
    a = build_agent(name="t", instructions="x",
                    spec=ModelSpec("openai", "gpt-6-luna"), output_type=Answer)
    assert isinstance(a.model, OpenAIResponsesModel)
    assert a.output_type.is_strict_json_schema()


def test_shared_rules_block_mentions_horizon_and_schema():
    text = shared_rules_block()
    assert "3-6 month" in text
    assert "`evidence`" in text
    assert "thesis" in text
    assert "risks" in text


def test_shared_rules_block_says_the_data_is_all_there_is():
    text = shared_rules_block()
    assert "You have no tools and nothing else to fetch" in text
    assert "UNAVAILABLE" in text
