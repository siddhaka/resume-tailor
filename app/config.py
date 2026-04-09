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

    # Anthropic / LLM
    anthropic_api_key: str
    llm_model: str = "claude-haiku-4-5-20251001"
    llm_max_tokens: int = 4096

    # Security
    api_key_salt: str
    secret_key: str

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
