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


def build_agent(*, name: str, instructions: str, spec: ModelSpec, output_type):
    """A model with instructions and a schema it must answer in. It has no
    tools; its data arrives in the request."""
    client = _client_for(spec)
    # OpenAI is called on the Responses API. The other providers expose the
    # Chat Completions API only.
    model_cls = (OpenAIResponsesModel if spec.provider == "openai"
                 else OpenAIChatCompletionsModel)
    model = model_cls(model=spec.model, openai_client=client)
    return Agent(
        name=name,
        instructions=instructions,
        model=model,
        # OpenAI holds the model to the schema while it writes. The other
        # providers do not, so their output is validated after the fact.
        output_type=AgentOutputSchema(output_type,
                                      strict_json_schema=spec.provider == "openai"),
    )


def shared_rules_block() -> str:
    return (
        "HARD RULES:\n"
        "1. Horizon is strictly 3-6 months. Reject theses pegged to longer windows.\n"
        "2. `evidence` holds the readings your call rests on, five to twelve of\n"
        "   them, most important first. Each is ONE number copied from the\n"
        "   DATA, with the section it came from, the date it is for, and what\n"
        "   it signals. Copy the number as the DATA writes it: 0.29 stays 0.29\n"
        "   with unit `fraction`. Never compute a value for `evidence`, and never\n"
        "   cite a number that is not in the DATA. Code checks each one.\n"
        "3. ALWAYS provide thesis bullets (1-6) AND risks (1-4), regardless of\n"
        "   whether you call BUY, HOLD, or SELL. Thesis = your findings, built\n"
        "   on your evidence. Risks = what could invalidate them or hurt the\n"
        "   position. Keep each bullet to one or two sentences. A HOLD with no\n"
        "   thesis is useless - even 'rule X hit rate is 48% on 23 trades, no\n"
        "   edge' is a thesis bullet.\n"
        "4. Conviction is 1-10 where 10 = bet-the-book confidence.\n"
        "5. `recommendation` is exactly one of BUY, HOLD, SELL.\n"
        "6. Answer in the report schema only. No prose outside it.\n"
        "7. If you genuinely lack data, return HOLD with conviction 3 and explain\n"
        "   in risks.\n"
        "8. Everything you have is in the DATA block of the request, one\n"
        "   section per heading. You have no tools and nothing else to fetch.\n"
        "9. `company_snapshot` gives today's date and the current price.\n"
        "   Never assume either. Anchor any price target to that price.\n"
        "10. The data carries its own dates and sample sizes; cite them. A null\n"
        "   value, or a section marked UNAVAILABLE, means the data could not be\n"
        "   fetched: say so in `risks` rather than estimating it. Do not redo\n"
        "   arithmetic the data already contains.\n"
        "\n"
        "CONTEXT EVERY DESK IS GIVEN:\n"
        "- `company_snapshot`: date, price, sector, 52-week range, next\n"
        "  earnings date, short interest, ownership.\n"
        "- `macro_dashboard`: rates, inflation, growth, labor, credit, liquidity\n"
        "  and market readings with their direction, plus scheduled data\n"
        "  releases. Set your view in this backdrop.\n"
        "- `sector_macro`: series that drive this sector.\n"
        "Choose what bears on your call from what you are given.\n"
        "Context informs your call. Your verdict must still rest on your own\n"
        "desk's discipline.\n"
    )
