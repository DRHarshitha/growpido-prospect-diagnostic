"""Application configuration loaded from environment variables."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Keys remain optional until their integration phases run."""

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    openai_api_key: str | None = Field(default=None, repr=False)
    tavily_api_key: str | None = Field(default=None, repr=False)
    gemini_api_key: str | None = Field(default=None, repr=False)

    llm_provider: str = "ollama"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:0.5b"

    openai_model: str = "gpt-4.1-mini"
    gemini_model: str = "gemini-2.5-flash"

    search_provider: str = "tavily"
    runs_directory: str = "data/runs"


@lru_cache
def get_settings() -> Settings:
    return Settings()