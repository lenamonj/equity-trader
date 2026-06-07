import os
from openai import AsyncOpenAI
from agents import Agent, OpenAIChatCompletionsModel

from equity_trader.config import ModelSpec


_BASE_URLS = {
    "groq": "https://api.groq.com/openai/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
}
_ENV_KEYS = {
    "openai": "OPENAI_API_KEY",
    "groq": "GROQ_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "gemini": "GOOGLE_API_KEY",
}


def _client_for(spec: ModelSpec) -> AsyncOpenAI:
    env_key = _ENV_KEYS[spec.provider]
    api_key = os.environ.get(env_key)
    if not api_key:
        raise RuntimeError(f"{env_key} not set")
    if spec.provider == "openai":
        return AsyncOpenAI(api_key=api_key)
    return AsyncOpenAI(api_key=api_key, base_url=_BASE_URLS[spec.provider])


def build_agent(*, name: str, instructions: str, tools: list,
                spec: ModelSpec, output_type):
    client = _client_for(spec)
    model = OpenAIChatCompletionsModel(model=spec.model, openai_client=client)
    return Agent(
        name=name,
        instructions=instructions,
        tools=tools,
        model=model,
        output_type=output_type,
    )


def shared_rules_block() -> str:
    return (
        "HARD RULES:\n"
        "1. Horizon is strictly 3-6 months. Reject theses pegged to longer windows.\n"
        "2. Cite specific data points in `data_cited` (e.g., '10-Q Q1 revenue',\n"
        "   'DGS10 5/29 close', 'RSI 28 on 2026-06-05').\n"
        "3. Provide 1-6 thesis bullets and 1-4 concrete risks.\n"
        "4. Conviction is 1-10 where 10 = bet-the-book confidence.\n"
        "5. Output MUST conform to the AgentVerdict schema.\n"
        "6. If you genuinely lack data, return HOLD with conviction 3 and explain in risks.\n"
    )
