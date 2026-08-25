"""Crypto Intelligence Engine — single-file backend (Phase 1 + frontend serving).

Consolidated into one module for a minimal file layout. Behaviour is identical
to the previous modular version:

* configuration from environment variables (paper-only; no live-trading path);
* data-quality primitives (VALID/STALE/INVALID/MISSING/CONFLICTED/UNKNOWN);
* health aggregation + REAL Postgres (SELECT 1) and Redis (PING) probes;
* endpoints /health, /health/live, /health/ready, /api/v1/;
* serves the single-file frontend/index.html at GET /.

No fabricated data. No real order/withdrawal capability exists anywhere.
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import AsyncIterator, Dict, List, Optional

import redis.asyncio as aioredis
from fastapi import APIRouter, FastAPI, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


# ============================ logging ============================
def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())


log = logging.getLogger("app")


# ============================ configuration ============================
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    app_name: str = "Crypto Intelligence Engine"
    environment: str = Field(default="development")  # development | test | production
    debug: bool = Field(default=False)
    api_prefix: str = "/api/v1"

    # Hard safety flag. There is NO live-execution path in this codebase; this
    # is a second guard rail on top of that.
    live_trading_enabled: bool = Field(default=False)

    # Empty => auto-resolve to the repo's frontend/. Set FRONTEND_DIR to override.
    frontend_dir: str = Field(default="")

    cors_origins: List[str] = Field(default_factory=lambda: ["http://localhost:3000"])

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "cie"
    postgres_password: str = "cie"
    postgres_db: str = "cie"

    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0

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


settings = get_settings()
configure_logging("DEBUG" if settings.debug else "INFO")


# ============================ data quality ============================
class DataQualityStatus(str, Enum):
    VALID = "VALID"
    STALE = "STALE"
    INVALID = "INVALID"
    MISSING = "MISSING"
    CONFLICTED = "CONFLICTED"
    UNKNOWN = "UNKNOWN"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def compute_age_seconds(timestamp: datetime, now: Optional[datetime] = None) -> float:
    if timestamp.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware (UTC)")
    reference = now or utcnow()
    if reference.tzinfo is None:
        raise ValueError("reference time must be timezone-aware (UTC)")
    return (reference - timestamp).total_seconds()


def classify_freshness(
    timestamp: Optional[datetime], max_age_seconds: float, now: Optional[datetime] = None
) -> DataQualityStatus:
    if timestamp is None:
        return DataQualityStatus.MISSING
    try:
        age = compute_age_seconds(timestamp, now=now)
    except ValueError:
        return DataQualityStatus.INVALID
    if age < 0:
        return DataQualityStatus.INVALID
    if age <= max_age_seconds:
        return DataQualityStatus.VALID
    return DataQualityStatus.STALE


@dataclass(frozen=True)
class QualifiedValue:
    value: Optional[float]
    source: str
    timestamp: Optional[datetime]
    status: DataQualityStatus

    @property
    def is_usable(self) -> bool:
        return self.status == DataQualityStatus.VALID and self.value is not None

    def age_seconds(self, now: Optional[datetime] = None) -> Optional[float]:
        if self.timestamp is None:
            return None
        try:
            return compute_age_seconds(self.timestamp, now=now)
        except ValueError:
            return None


# ============================ health model ============================
class HealthState(str, Enum):
    UP = "UP"
    DOWN = "DOWN"
    DEGRADED = "DEGRADED"
    UNKNOWN = "UNKNOWN"


_SEVERITY = {
    HealthState.DOWN: 3,
    HealthState.DEGRADED: 2,
    HealthState.UNKNOWN: 1,
    HealthState.UP: 0,
}


@dataclass
class ComponentHealth:
    name: str
    state: HealthState = HealthState.UNKNOWN
    detail: Optional[str] = None
    latency_ms: Optional[float] = None
    checked_at: datetime = field(default_factory=utcnow)

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "state": self.state.value,
            "detail": self.detail,
            "latency_ms": self.latency_ms,
            "checked_at": self.checked_at.isoformat(),
        }


def aggregate_health(components: List[ComponentHealth]) -> HealthState:
    if not components:
        return HealthState.UNKNOWN
    worst = max(components, key=lambda c: _SEVERITY[c.state]).state
    if worst == HealthState.DOWN:
        return HealthState.DOWN
    if worst in (HealthState.DEGRADED, HealthState.UNKNOWN):
        return HealthState.DEGRADED
    return HealthState.UP


@dataclass
class HealthReport:
    overall: HealthState
    components: List[ComponentHealth]
    generated_at: datetime = field(default_factory=utcnow)

    @classmethod
    def from_components(cls, components: List[ComponentHealth]) -> "HealthReport":
        return cls(overall=aggregate_health(components), components=components)

    def to_dict(self) -> Dict[str, object]:
        return {
            "overall": self.overall.value,
            "generated_at": self.generated_at.isoformat(),
            "components": [c.to_dict() for c in self.components],
        }


# ============================ db + redis + probes ============================
engine = create_async_engine(settings.database_url, pool_pre_ping=True, echo=False, future=True)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
redis_client = aioredis.from_url(settings.redis_url, decode_responses=True)


@asynccontextmanager
async def get_session() -> AsyncIterator[AsyncSession]:
    async with SessionLocal() as session:
        yield session


async def check_database() -> ComponentHealth:
    """Real probe: SELECT 1. Returns DOWN + reason on failure, never a fake UP."""
    start = time.perf_counter()
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        latency = (time.perf_counter() - start) * 1000
        return ComponentHealth("postgres", HealthState.UP, latency_ms=round(latency, 2))
    except Exception as exc:  # noqa: BLE001 - surface the reason, don't swallow it
        return ComponentHealth("postgres", HealthState.DOWN, detail=str(exc))


async def check_redis() -> ComponentHealth:
    """Real probe: PING. Returns DOWN + reason on failure, never a fake UP."""
    start = time.perf_counter()
    try:
        pong = await redis_client.ping()
        if pong is not True:
            return ComponentHealth("redis", HealthState.DEGRADED, detail="unexpected PING reply")
        latency = (time.perf_counter() - start) * 1000
        return ComponentHealth("redis", HealthState.UP, latency_ms=round(latency, 2))
    except Exception as exc:  # noqa: BLE001
        return ComponentHealth("redis", HealthState.DOWN, detail=str(exc))


# ============================ routers ============================
health_router = APIRouter(tags=["health"])


@health_router.get("/health/live")
async def liveness() -> dict:
    return {"status": "alive"}


@health_router.get("/health")
async def health() -> dict:
    components = [await check_database(), await check_redis()]
    return HealthReport.from_components(components).to_dict()


@health_router.get("/health/ready")
async def readiness(response: Response) -> dict:
    components = [await check_database(), await check_redis()]
    report = HealthReport.from_components(components)
    if report.overall != HealthState.UP:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return report.to_dict()


api_router = APIRouter()


@api_router.get("/")
async def api_root() -> dict:
    return {
        "service": "crypto-intelligence-engine",
        "phase": 1,
        "status": "operational",
        "note": "Paper trading only. No real execution. No withdrawals.",
    }


# ============================ frontend serving ============================
def _frontend_dir(cfg: Settings) -> Path:
    """Resolve the frontend directory. FRONTEND_DIR overrides; otherwise the
    repo's frontend/ (main.py lives at backend/main.py -> parents[1] = repo)."""
    if cfg.frontend_dir:
        return Path(cfg.frontend_dir)
    return Path(__file__).resolve().parents[1] / "frontend"


def _mount_frontend(app: FastAPI, cfg: Settings) -> None:
    """Serve the single-file frontend at GET /. No-op (API-only) if absent,
    so the Docker image / API-only deployments keep working unchanged."""
    frontend = _frontend_dir(cfg)
    index = frontend / "index.html"
    if not index.exists():
        log.info("Frontend not found at %s - serving API only.", frontend)
        return
    for sub in ("css", "js", "assets"):
        directory = frontend / sub
        if directory.is_dir():
            app.mount(f"/{sub}", StaticFiles(directory=str(directory)), name=f"static-{sub}")

    @app.get("/", include_in_schema=False)
    async def serve_index() -> FileResponse:
        return FileResponse(str(index), media_type="text/html")

    log.info("Serving frontend from %s", frontend)


# ============================ app factory ============================
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting %s (env=%s)", settings.app_name, settings.environment)
    if settings.is_production and settings.live_trading_enabled:
        raise RuntimeError(
            "live_trading_enabled=True but this build supports paper trading only."
        )
    yield
    log.info("Shutting down %s", settings.app_name)


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version="0.2.0",
        description="Real-time crypto intelligence & paper-trading platform (single-file build).",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health_router)
    app.include_router(api_router, prefix=settings.api_prefix)
    _mount_frontend(app, settings)
    return app


app = create_app()
