"""Application configuration.

All configuration comes from environment variables (12-factor). Secrets are
NEVER hard-coded, committed, or logged. See .env.example for the contract.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- App ---
    app_name: str = "Crypto Intelligence Engine"
    environment: str = Field(default="development")  # development | test | production
    debug: bool = Field(default=False)
    api_prefix: str = "/api/v1"

    # Hard safety flag. Phase 1 is paper-trading only; there is no real
    # execution path in the codebase at all. This flag is a second guard rail.
    live_trading_enabled: bool = Field(default=False)

    # --- Frontend ---
    # Empty => auto-resolve to the repo's frontend/. Set FRONTEND_DIR to override,
    # or leave empty in an API-only deployment where the frontend is absent.
    frontend_dir: str = Field(default="")

    # --- CORS ---
    cors_origins: List[str] = Field(default_factory=lambda: ["http://localhost:3000"])

    # --- Database (PostgreSQL, async) ---
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "cie"
    postgres_password: str = "cie"
    postgres_db: str = "cie"

    # --- Redis ---
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0

    # --- Freshness budgets (seconds) — used by the data-quality layer ---
    ticker_max_age_seconds: float = 10.0
    candle_max_age_seconds: float = 120.0

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def redis_url(self) -> str:
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()
