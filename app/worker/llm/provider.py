from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from app.config import settings


def get_llm() -> BaseChatModel:
    """Return a chat model chosen by LLM_PROVIDER.

    - "ollama"    — self-hosted, no key (default); format="json" forces valid JSON.
    - "anthropic" — Claude; requires ANTHROPIC_API_KEY.

    Backends are imported lazily so only the selected one needs to be installed.
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
