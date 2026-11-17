import os

from agents import (
    Agent,
    AgentOutputSchema,
    OpenAIChatCompletionsModel,
    OpenAIResponsesModel,
)
from openai import AsyncOpenAI

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
    # OpenAI's current models only take function tools on the Responses API.
    # The other providers expose the Chat Completions API only.
    model_cls = (OpenAIResponsesModel if spec.provider == "openai"
                 else OpenAIChatCompletionsModel)
    model = model_cls(model=spec.model, openai_client=client)
    return Agent(
        name=name,
        instructions=instructions,
        tools=tools,
        model=model,
        output_type=AgentOutputSchema(output_type, strict_json_schema=False),
    )


def shared_rules_block() -> str:
    return (
        "HARD RULES:\n"
        "1. Horizon is strictly 3-6 months. Reject theses pegged to longer windows.\n"
        "2. Cite specific data points in `data_cited` (e.g., '10-Q Q1 revenue',\n"
        "   'DGS10 5/29 close', 'RSI 28 on 2026-06-05').\n"
        "3. ALWAYS provide thesis bullets (1-6) AND risks (1-4), regardless of\n"
        "   whether you call BUY, HOLD, or SELL. Thesis = your observations and\n"
        "   findings (the data points and inferences that led to your call).\n"
        "   Risks = what could invalidate your observations or hurt the position.\n"
        "   A HOLD with no thesis is useless - even 'rule X hit rate is 48% on\n"
        "   23 trades, no edge' is a thesis bullet.\n"
        "4. Conviction is 1-10 where 10 = bet-the-book confidence.\n"
        "5. `recommendation` MUST be exactly one of: BUY, HOLD, SELL (uppercase,\n"
        "   exact spelling). Never use SOLD, BOUGHT, HELD, STRONG BUY, REDUCE,\n"
        "   or any other variant. Pydantic will reject anything else and your\n"
        "   entire verdict will be discarded.\n"
        "6. Output MUST conform to the AgentVerdict schema. No prose outside it.\n"
        "7. If you genuinely lack data, return HOLD with conviction 3 and explain\n"
        "   in risks.\n"
        "8. Call each tool AT MOST ONCE with the same arguments. Do not re-call a\n"
        "   tool to verify it - one call gives you the data, then synthesize.\n"
        "   tool_get_fred_series may be called for up to three different series.\n"
        "9. Leave `error_note` null. It is reserved for system failures. Missing\n"
        "   or stale data belongs in `risks`.\n"
        "10. Call tool_get_company_snapshot FIRST. It gives today's date and the\n"
        "   current price. Never assume either. Anchor any price target to that\n"
        "   price.\n"
        "11. Tool outputs carry their own dates and sample sizes; cite them. A\n"
        "   null value means the data is unavailable: say so in `risks` rather\n"
        "   than estimating it. Do not do arithmetic a tool has already done.\n"
        "\n"
        "CONTEXT TOOLS EVERY DESK HAS:\n"
        "- tool_get_company_snapshot(ticker): date, price, sector, 52-week\n"
        "  range, next earnings date, short interest, ownership. Required.\n"
        "- tool_get_macro_dashboard(): rates, inflation, growth, labor, credit,\n"
        "  liquidity and market readings with their direction, plus scheduled\n"
        "  data releases. Call it once so your view sits in the macro backdrop.\n"
        "- tool_get_sector_macro(ticker): series that drive this sector.\n"
        "- tool_search_fred(query) and tool_get_fred_series(series_id): any\n"
        "  other FRED series. Search for the id; do not guess it.\n"
        "Context informs your call. Your verdict must still rest on your own\n"
        "desk's discipline.\n"
    )
