from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from app.config import settings


def get_llm() -> BaseChatModel:
    """Return a chat model chosen by the LLM_PROVIDER setting.

    The pipeline nodes depend only on the LangChain BaseChatModel interface
    (``.invoke([messages])``), so the backend is swappable without touching
    any node logic. Two backends are supported:

    - "ollama"    — a self-hosted model (default). Requires no API key and no
                    paid service; ``format="json"`` constrains the model to
                    emit valid JSON, which the nodes parse into typed objects.
    - "anthropic" — Claude via the Anthropic API. Requires ANTHROPIC_API_KEY.

    Providers are imported lazily so that running with one backend does not
    require the other's package or credentials to be present.
    """
    provider = settings.llm_provider.lower()

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            num_predict=settings.llm_max_tokens,
            temperature=0.2,
            format="json",
        )

    if provider == "anthropic":
        if not settings.anthropic_api_key:
            raise RuntimeError(
                "LLM_PROVIDER=anthropic requires ANTHROPIC_API_KEY to be set."
            )
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=settings.llm_model,
            api_key=settings.anthropic_api_key,
            max_tokens=settings.llm_max_tokens,
        )

    raise RuntimeError(
        f"Unknown LLM_PROVIDER {settings.llm_provider!r}; expected 'ollama' or 'anthropic'."
    )
