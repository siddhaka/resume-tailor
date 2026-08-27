import functools

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # LLM provider selection: "ollama" (self-hosted, no key) or "anthropic".
    # Ollama is the default so the pipeline runs end-to-end with no paid API.
    llm_provider: str = "ollama"
    llm_max_tokens: int = 4096

    # Ollama (used when llm_provider == "ollama")
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5-coder:7b"

    # Anthropic (used when llm_provider == "anthropic")
    anthropic_api_key: str | None = None
    llm_model: str = "claude-haiku-4-5-20251001"

    # Security
    api_key_salt: str = "dev-salt-change-me"
    secret_key: str = "dev-secret-change-me"

    # Rate limiting
    rate_limit_per_minute: int = 60

    # Runtime
    environment: str = "development"
    log_level: str = "INFO"

    # Database (SQLite for API key storage)
    database_url: str = "sqlite:///./resume_tailor.db"


@functools.lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
