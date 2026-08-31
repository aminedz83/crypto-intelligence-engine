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

import asyncio
import json
import logging
import re
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from decimal import Decimal, InvalidOperation
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Set

import httpx
import redis.asyncio as aioredis
from fastapi import APIRouter, FastAPI, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    MetaData,
    Numeric,
    String,
    Table,
    text,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

try:
    import websockets
except ImportError:  # pragma: no cover - only needed when the WS feed is used
    websockets = None  # type: ignore[assignment]


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

    # Application-level guard rail (NOT a Coinbase limit): max provider windows
    # (sequential REST calls) a single /history request may fan out to.
    history_max_windows: int = 20

    # Persistence (Postgres). Batch size = technical statement size inside ONE
    # atomic transaction (not a separate commit per batch). Margin (seconds) after
    # a bucket's own end before we consider it time-closed (0 = fully elapsed).
    persist_batch_size: int = 500
    candle_finalization_margin_seconds: float = 0.0
    db_read_max_rows: int = 1000

    # Massive (ex-Polygon) Forex REST. api_key empty by default: no calls happen
    # until an officially-verified symbol mapping is registered (never deduced).
    massive_api_key: str = ""
    massive_rest_url: str = "https://api.massive.com"
    massive_request_timeout_seconds: float = 10.0

    # Twelve Data REST (Gold XAU/USD spot). Key server-side only, header auth; no
    # call is made without a key. Availability on the account's plan is determined
    # at runtime from the provider response, never assumed.
    twelvedata_api_key: str = ""
    twelvedata_rest_url: str = "https://api.twelvedata.com"
    twelvedata_request_timeout_seconds: float = 10.0

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
    report = HealthReport.from_components(components).to_dict()
    report["persistence"] = persistence_state.to_dict()
    return report


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


# ============================ market data (Phase 2) ============================
# Provider: Coinbase Advanced Trade PUBLIC market data. Endpoints verified against
# the official docs (docs.cdp.coinbase.com):
#   REST base  : https://api.coinbase.com/api/v3/brokerage   (public market data,
#                no authentication)
#   WS (public): wss://advanced-trade-ws.coinbase.com        (market channels work
#                without auth; subscribe within 5s; heartbeats keep it open)
# No API key, no orders, no withdrawals. No fabricated data: absent/late/invalid
# data is surfaced as MISSING/STALE/INVALID/UNKNOWN, never invented.

COINBASE_REST_URL = "https://api.coinbase.com/api/v3/brokerage"
COINBASE_WS_URL = "wss://advanced-trade-ws.coinbase.com"
WS_INITIAL_BACKOFF = 1.0
WS_MAX_BACKOFF = 60.0
WS_ALLOWED_CHANNELS = {
    "ticker", "ticker_batch", "candles", "market_trades", "level2", "status", "heartbeats",
}

# Candles (defined before CoinbaseProvider: used as a default arg in get_candles).
# friendly -> (Coinbase enum, bucket duration in seconds)
GRANULARITIES: Dict[str, tuple] = {
    "1m": ("ONE_MINUTE", 60),
    "5m": ("FIVE_MINUTE", 300),
    "15m": ("FIFTEEN_MINUTE", 900),
    "30m": ("THIRTY_MINUTE", 1800),
    "1h": ("ONE_HOUR", 3600),
    "2h": ("TWO_HOUR", 7200),
    "4h": ("FOUR_HOUR", 14400),
    "6h": ("SIX_HOUR", 21600),
    "1d": ("ONE_DAY", 86400),
}
CANDLE_MAX_LIMIT = 350
# Coinbase caps a request at CANDLE_MAX_LIMIT candles. `end` inclusivity is not
# guaranteed by the docs, so a paginated window must never request more than
# CANDLE_MAX_LIMIT candidate starts: width = (CANDLE_MAX_LIMIT - 1) * bucket
# keeps it <= CANDLE_MAX_LIMIT even if `end` is inclusive. Never hardcode 349.
PROVIDER_SAFE_BUCKETS = CANDLE_MAX_LIMIT - 1


def parse_iso8601(value: Any) -> Optional[datetime]:
    """Parse an ISO-8601 string to aware UTC. Returns None if absent/invalid/naive
    (never invents a timestamp)."""
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


@dataclass(frozen=True)
class MarketDatum:
    source: str
    symbol: str
    value: Optional[float]
    timestamp: Optional[datetime]
    status: DataQualityStatus

    def to_dict(self) -> Dict[str, object]:
        age: Optional[float] = None
        if self.timestamp is not None:
            age = max(0.0, (utcnow() - self.timestamp).total_seconds())
        return {
            "source": self.source,
            "symbol": self.symbol,
            "value": self.value,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "freshness_seconds": age,
            "quality": self.status.value,
        }


def ticker_datum_from_payload(
    symbol: str, payload: Any, now: Optional[datetime] = None
) -> MarketDatum:
    """Build a qualified MarketDatum from a Coinbase ticker payload. Pure, no network.
    Coinbase returns recent trades; we use the latest trade's price + time."""
    src = "coinbase"
    trades = payload.get("trades") if isinstance(payload, dict) else None
    if not isinstance(trades, list) or not trades or not isinstance(trades[0], dict):
        return MarketDatum(src, symbol, None, None, DataQualityStatus.MISSING)
    latest = trades[0]
    price = _to_float(latest.get("price"))
    if price is None or price <= 0:
        return MarketDatum(src, symbol, None, None, DataQualityStatus.INVALID)
    ts = parse_iso8601(latest.get("time"))
    quality = classify_freshness(ts, settings.ticker_max_age_seconds, now=now)
    return MarketDatum(src, symbol, price, ts, quality)


class CoinbaseProvider:
    """Public REST access to Coinbase Advanced Trade market data. No API key."""

    SOURCE = "coinbase"

    def __init__(self, rest_url: str = COINBASE_REST_URL) -> None:
        self.rest_url = rest_url.rstrip("/")
        self.client: Optional[httpx.AsyncClient] = None

    async def connect(self) -> None:
        if self.client is None:
            self.client = httpx.AsyncClient(
                base_url=self.rest_url, timeout=10.0, headers={"Accept": "application/json"}
            )

    async def disconnect(self) -> None:
        if self.client is not None:
            await self.client.aclose()
            self.client = None

    async def _get(self, path: str, params: Optional[dict] = None) -> dict:
        if self.client is None:
            await self.connect()
        assert self.client is not None
        resp = await self.client.get(path, params=params)
        resp.raise_for_status()
        payload = resp.json()
        if not isinstance(payload, dict):
            raise ValueError("Coinbase returned a non-object JSON response")
        return payload

    async def get_ticker(self, symbol: str) -> MarketDatum:
        symbol = symbol.upper()
        payload = await self._get(f"/market/products/{symbol}/ticker")
        return ticker_datum_from_payload(symbol, payload)

    async def get_candles(self, symbol: str, granularity: str, limit: int = CANDLE_MAX_LIMIT):
        """Fetch qualified candles from Coinbase Advanced Trade (public, no auth).
        Verified params: granularity string enum + start/end UNIX seconds, max 350."""
        if granularity not in GRANULARITIES:
            raise ValueError(f"Unsupported granularity: {granularity}")
        enum_value, bucket_seconds = GRANULARITIES[granularity]
        limit = max(1, min(int(limit), CANDLE_MAX_LIMIT))
        end = int(utcnow().timestamp())
        start = end - limit * bucket_seconds
        symbol = symbol.upper()
        payload = await self._get(
            f"/market/products/{symbol}/candles",
            params={
                "start": str(start),
                "end": str(end),
                "granularity": enum_value,
                "limit": limit,
            },
        )
        # A candle is "recent enough" within ~2 buckets of its own timeframe.
        return candles_from_payload(payload, max_age_seconds=bucket_seconds * 2)

    async def get_candles_range(self, symbol: str, granularity: str, start: int, end: int):
        """Fetch candles for an EXPLICIT [start, end] window (UNIX seconds). Used by
        the paginated history layer. Reuses the validated candles_from_payload parser.
        Historical candles are qualified on data validity only (freshness is not a
        meaningful axis for an explicit past range), so max_age is effectively off."""
        if granularity not in GRANULARITIES:
            raise ValueError(f"Unsupported granularity: {granularity}")
        enum_value, _bucket = GRANULARITIES[granularity]
        symbol = symbol.upper()
        payload = await self._get(
            f"/market/products/{symbol}/candles",
            params={
                "start": str(int(start)),
                "end": str(int(end)),
                "granularity": enum_value,
                "limit": CANDLE_MAX_LIMIT,
            },
        )
        return candles_from_payload(payload, max_age_seconds=float("inf"))

    async def health_check(self) -> bool:
        try:
            payload = await self._get("/market/products/BTC-USD")
            return isinstance(payload, dict)
        except Exception:  # noqa: BLE001 - health probe must not raise
            log.warning("Coinbase REST health check failed", exc_info=True)
            return False


def ws_backoff(attempt: int) -> float:
    """Capped exponential backoff in seconds. attempt starts at 1."""
    return min(WS_MAX_BACKOFF, WS_INITIAL_BACKOFF * 2 ** max(0, attempt - 1))


class MarketWsManager:
    """Coinbase PUBLIC market-data WebSocket manager. Idle until start() is called.
    Builds real subscribe messages (verified format) and never fabricates data."""

    SOURCE = "coinbase"

    def __init__(self, url: str = COINBASE_WS_URL) -> None:
        self.url = url
        self.running = False
        self.websocket: Any = None
        self.subscriptions: Dict[str, set] = {}
        self.last_message_at: Optional[datetime] = None
        self.attempt = 0
        self._task: Optional[asyncio.Task] = None

    @staticmethod
    def build_subscribe(channel: str, products: List[str]) -> Dict[str, object]:
        return {
            "type": "subscribe",
            "channel": channel,
            "product_ids": [p.upper() for p in products],
        }

    async def start(self) -> None:
        if websockets is None:
            raise RuntimeError("websockets dependency is not installed")
        if self.running:
            return
        self.running = True
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        self.running = False
        if self.websocket is not None:
            try:
                await self.websocket.close()
            except Exception:  # noqa: BLE001
                pass
            self.websocket = None
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def subscribe(self, channel: str, products: List[str]) -> None:
        prods = {p.upper() for p in products}
        self.subscriptions.setdefault(channel, set()).update(prods)
        if self.websocket is not None:
            await self.websocket.send(json.dumps(self.build_subscribe(channel, sorted(prods))))

    async def _run_loop(self) -> None:  # pragma: no cover - needs a live socket
        while self.running:
            try:
                async with websockets.connect(
                    self.url, ping_interval=20, ping_timeout=20, close_timeout=5
                ) as ws:
                    self.websocket = ws
                    self.attempt = 0
                    await market_store.reset_transport()
                    for channel, prods in list(self.subscriptions.items()):
                        await ws.send(json.dumps(self.build_subscribe(channel, sorted(prods))))
                    # Heartbeats keep sparse subscriptions open (Coinbase docs).
                    await ws.send(json.dumps({"type": "subscribe", "channel": "heartbeats"}))
                    async for raw in ws:
                        self.last_message_at = utcnow()
                        await self._handle(raw)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                self.websocket = None
                if not self.running:
                    break
                self.attempt += 1
                delay = ws_backoff(self.attempt)
                log.warning("Coinbase WS disconnected: %s - reconnecting in %.1fs", exc, delay)
                await asyncio.sleep(delay)

    async def _handle(self, raw: str | bytes) -> None:
        received_at = utcnow()
        msg = parse_ws_message(raw)
        if msg is None:
            return  # malformed -> no state change, no fabricated data
        channel = msg.get("channel")
        await market_store.check_sequence(msg.get("sequence_num"))
        if channel in WS_TICKER_TYPES:
            for datum in extract_ticker_data(msg, received_at):
                await market_store.apply_ticker(datum)
                await market_bus.publish(datum)
        elif channel == "candles":
            for datum in extract_candle_data(msg, received_at):
                await market_store.apply_candle(datum)
                await market_bus.publish(datum)
        elif channel == "heartbeats":
            events = msg.get("events")
            counter = None
            if isinstance(events, list) and events and isinstance(events[0], dict):
                counter = events[0].get("heartbeat_counter")
            await market_store.record_heartbeat(
                counter if isinstance(counter, int) else None, received_at
            )
        # unknown channel -> ignored (no fabricated data)

    async def health_check(self) -> Dict[str, object]:
        connected = self.websocket is not None
        age: Optional[float] = None
        if self.last_message_at is not None:
            age = (utcnow() - self.last_message_at).total_seconds()
        return {
            "connected": connected,
            "last_message_at": self.last_message_at.isoformat() if self.last_message_at else None,
            "last_message_age_seconds": age,
            "quality": (
                DataQualityStatus.VALID.value if connected else DataQualityStatus.UNKNOWN.value
            ),
        }


market_provider = CoinbaseProvider()
market_ws = MarketWsManager()


class WsSubscribeRequest(BaseModel):
    channel: str
    products: List[str]


@api_router.get("/market/ticker/{symbol}")
async def market_ticker(symbol: str) -> dict:
    try:
        datum = await market_provider.get_ticker(symbol)
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=503, detail={"status": "UNAVAILABLE", "reason": str(exc)}
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500, detail={"status": "UNKNOWN", "reason": str(exc)}
        ) from exc
    return datum.to_dict()


@api_router.post("/market/websocket/start")
async def market_ws_start() -> dict:
    await market_ws.start()
    return {"status": "started", "source": "coinbase", "endpoint": COINBASE_WS_URL}


@api_router.post("/market/websocket/stop")
async def market_ws_stop() -> dict:
    await market_ws.stop()
    return {"status": "stopped", "source": "coinbase"}


@api_router.post("/market/websocket/subscribe")
async def market_ws_subscribe(req: WsSubscribeRequest) -> dict:
    if req.channel not in WS_ALLOWED_CHANNELS:
        raise HTTPException(
            status_code=400, detail={"status": "INVALID", "reason": "Unsupported public channel"}
        )
    if not req.products:
        raise HTTPException(
            status_code=400, detail={"status": "INVALID", "reason": "At least one product required"}
        )
    await market_ws.subscribe(req.channel, req.products)
    return {
        "status": "subscribed",
        "channel": req.channel,
        "products": [p.upper() for p in req.products],
    }


@api_router.get("/market/websocket/health")
async def market_ws_health() -> dict:
    connection = await market_ws.health_check()
    transport = await market_store.health()
    return {"connection": connection, "transport": transport}


# ---- candles (Coinbase Advanced Trade, verified OpenAPI) --------------------
# GET /api/v3/brokerage/market/products/{product_id}/candles (public, no auth).
# Query: start, end (UNIX seconds, required), granularity (string enum, required),
# limit (max 350). Response: {"candles":[{start,low,high,open,close,volume}]} where
# every field is a STRING and `start` is a UNIX timestamp in seconds.
# NOT the old Exchange API (which used integer-second granularities 60/300/...).


@dataclass(frozen=True)
class Candle:
    start: Optional[datetime]
    low: Optional[float]
    high: Optional[float]
    open: Optional[float]
    close: Optional[float]
    volume: Optional[float]
    status: DataQualityStatus

    def to_dict(self) -> Dict[str, object]:
        return {
            "start": self.start.isoformat() if self.start else None,
            "low": self.low,
            "high": self.high,
            "open": self.open,
            "close": self.close,
            "volume": self.volume,
            "quality": self.status.value,
        }


def _to_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_decimal(value: Any) -> Optional[Decimal]:
    """Parse a financial value (a string from the provider) into an exact Decimal.
    Never routes through float. Returns None on missing/non-numeric/non-finite."""
    if value is None:
        return None
    try:
        parsed = Decimal(str(value).strip())
    except (InvalidOperation, ValueError, TypeError):
        return None
    return parsed if parsed.is_finite() else None


def _unix_seconds_to_dt(value: Any) -> Optional[datetime]:
    """Coinbase candle `start` is a UNIX timestamp in seconds, as a string.
    Returns aware UTC, or None if unparseable (never invents a timestamp)."""
    try:
        seconds = int(str(value))
    except (TypeError, ValueError):
        return None
    if seconds <= 0:
        return None
    try:
        return datetime.fromtimestamp(seconds, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None


def candle_from_payload(
    item: Any, max_age_seconds: float, now: Optional[datetime] = None
) -> Candle:
    """Convert one Coinbase candle object into a qualified Candle. Pure, no network.
    Invalid timestamp or unparseable OHLCV -> INVALID; never fabricated."""
    if not isinstance(item, dict):
        return Candle(None, None, None, None, None, None, DataQualityStatus.INVALID)
    start = _unix_seconds_to_dt(item.get("start"))
    low = _to_float(item.get("low"))
    high = _to_float(item.get("high"))
    open_ = _to_float(item.get("open"))
    close = _to_float(item.get("close"))
    volume = _to_float(item.get("volume"))
    if (
        start is None
        or low is None
        or high is None
        or open_ is None
        or close is None
        or volume is None
    ):
        return Candle(start, low, high, open_, close, volume, DataQualityStatus.INVALID)
    if low < 0 or high < 0 or open_ < 0 or close < 0 or volume < 0:
        return Candle(start, low, high, open_, close, volume, DataQualityStatus.INVALID)
    status = classify_freshness(start, max_age_seconds, now=now)
    return Candle(start, low, high, open_, close, volume, status)


def candles_from_payload(
    payload: Any, max_age_seconds: float, now: Optional[datetime] = None
) -> tuple:
    """Parse a Coinbase candles response into (list[Candle], overall_status).
    Empty/malformed -> ([], MISSING)."""
    raw = payload.get("candles") if isinstance(payload, dict) else None
    if not isinstance(raw, list) or not raw:
        return [], DataQualityStatus.MISSING
    candles = [candle_from_payload(item, max_age_seconds, now=now) for item in raw]
    if all(c.status == DataQualityStatus.INVALID for c in candles):
        return candles, DataQualityStatus.INVALID
    return candles, DataQualityStatus.VALID


def _latest_quality(candles: List[Candle]) -> str:
    """Quality of the MOST RECENT candle actually received (freshness of the last
    bar), or MISSING if none. A set of candles is only as 'live' as its newest bar
    — a successful HTTP call never implies LIVE."""
    dated = [c for c in candles if c.start is not None]
    if not dated:
        return DataQualityStatus.MISSING.value
    newest = max(dated, key=lambda c: c.start.timestamp())  # type: ignore[union-attr]
    return newest.status.value


@api_router.get("/market/candles/{symbol}")
async def market_candles(
    symbol: str, granularity: str = "1m", limit: int = CANDLE_MAX_LIMIT
) -> dict:
    if granularity not in GRANULARITIES:
        raise HTTPException(
            status_code=400,
            detail={
                "status": "INVALID",
                "reason": "Unsupported granularity",
                "allowed": sorted(GRANULARITIES),
            },
        )
    try:
        candles, status = await market_provider.get_candles(symbol, granularity, limit)
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=503, detail={"status": "UNAVAILABLE", "reason": str(exc)}
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500, detail={"status": "UNKNOWN", "reason": str(exc)}
        ) from exc
    # Coinbase returns candles newest-first; sort ascending deterministically so
    # consumers can reliably take the most-recent slice. Undated (INVALID) go first.
    ordered = sorted(candles, key=lambda c: c.start.timestamp() if c.start else float("-inf"))
    return {
        "source": "coinbase",
        "symbol": symbol.upper(),
        "granularity": granularity,
        "count": len(ordered),
        "quality": status.value,               # overall (any non-invalid) — unchanged
        "latest_quality": _latest_quality(ordered),  # freshness of the NEWEST bar
        "candles": [c.to_dict() for c in ordered],
    }


# ============================ realtime pipeline (increment 3) =================
# Turn REAL Coinbase Advanced Trade WS messages into qualified internal data.
# Verified envelope (docs.cdp.coinbase.com): {channel, timestamp (server send
# time, ISO8601), sequence_num (PER-CONNECTION), events:[{type: snapshot|update,
# ...}]}.  ticker -> events[].tickers[] (price, product_id); market/server time =
# envelope timestamp. candles -> events[].candles[] (start, OHLCV, product_id);
# WS candles are 5-minute buckets refreshed every second (same `start` UPDATES
# the bucket, it is NOT a duplicate). heartbeats -> connection health only.
#
# Two-layer integrity: (1) per-connection sequence_num for transport gap/dup/
# out-of-order diagnostics; (2) per-product ordering by real timestamp so an
# older update never overwrites a newer state. No fabricated data or sequence.

WS_TICKER_TYPES = {"ticker", "ticker_batch"}
WS_CANDLE_MAX_AGE_SECONDS = 330.0  # 5-min bucket + buffer


@dataclass(frozen=True)
class RealtimeDatum:
    source: str
    product_id: str
    data_type: str  # "ticker" | "candle"
    value: Optional[float]  # ticker price, or candle close
    source_timestamp: Optional[datetime]  # ticker: server send time; candle: bucket start
    received_at: datetime
    status: DataQualityStatus
    sequence_num: Optional[int]
    ohlcv: Optional[Dict[str, float]] = None  # candles only

    def to_dict(self) -> Dict[str, object]:
        return {
            "source": self.source,
            "product_id": self.product_id,
            "data_type": self.data_type,
            "value": self.value,
            "source_timestamp": (
                self.source_timestamp.isoformat() if self.source_timestamp else None
            ),
            "received_at": self.received_at.isoformat(),
            "quality": self.status.value,
            "sequence_num": self.sequence_num,
            "ohlcv": self.ohlcv,
        }


def parse_ws_message(raw: Any) -> Optional[dict]:
    """json.loads a raw WS frame -> dict envelope, or None if malformed / not a
    JSON object. Never fabricates a message."""
    if isinstance(raw, (bytes, bytearray)):
        try:
            raw = raw.decode("utf-8")
        except (UnicodeDecodeError, AttributeError):
            return None
    if not isinstance(raw, str):
        return None
    try:
        msg = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return msg if isinstance(msg, dict) else None


def extract_ticker_data(
    msg: dict, received_at: datetime, now: Optional[datetime] = None
) -> List[RealtimeDatum]:
    """Pure: qualified ticker data from a ticker/ticker_batch envelope. Invalid
    entries are skipped, never fabricated."""
    out: List[RealtimeDatum] = []
    seq = msg.get("sequence_num")
    seq_num = seq if isinstance(seq, int) else None
    server_ts = parse_iso8601(msg.get("timestamp"))
    events = msg.get("events")
    if not isinstance(events, list):
        return out
    for event in events:
        if not isinstance(event, dict):
            continue
        tickers = event.get("tickers")
        if not isinstance(tickers, list):
            continue
        for tick in tickers:
            if not isinstance(tick, dict):
                continue
            product_id = tick.get("product_id")
            if not isinstance(product_id, str) or not product_id:
                continue
            price = _to_float(tick.get("price"))
            if price is None or price <= 0:
                continue
            status = classify_freshness(server_ts, settings.ticker_max_age_seconds, now=now)
            out.append(
                RealtimeDatum(
                    source="coinbase",
                    product_id=product_id,
                    data_type="ticker",
                    value=price,
                    source_timestamp=server_ts,
                    received_at=received_at,
                    status=status,
                    sequence_num=seq_num,
                )
            )
    return out


def extract_candle_data(
    msg: dict, received_at: datetime, now: Optional[datetime] = None
) -> List[RealtimeDatum]:
    """Pure: qualified candle data from a candles envelope. WS candles are 5-min
    buckets refreshed every second; `start` identifies the bucket."""
    out: List[RealtimeDatum] = []
    seq = msg.get("sequence_num")
    seq_num = seq if isinstance(seq, int) else None
    events = msg.get("events")
    if not isinstance(events, list):
        return out
    for event in events:
        if not isinstance(event, dict):
            continue
        candles = event.get("candles")
        if not isinstance(candles, list):
            continue
        for item in candles:
            if not isinstance(item, dict):
                continue
            product_id = item.get("product_id")
            if not isinstance(product_id, str) or not product_id:
                continue
            candle = candle_from_payload(item, WS_CANDLE_MAX_AGE_SECONDS, now=now)
            ohlcv: Optional[Dict[str, float]] = None
            value: Optional[float] = None
            if (
                candle.status != DataQualityStatus.INVALID
                and candle.open is not None
                and candle.high is not None
                and candle.low is not None
                and candle.close is not None
                and candle.volume is not None
            ):
                value = candle.close
                ohlcv = {
                    "open": candle.open,
                    "high": candle.high,
                    "low": candle.low,
                    "close": candle.close,
                    "volume": candle.volume,
                }
            out.append(
                RealtimeDatum(
                    source="coinbase",
                    product_id=product_id,
                    data_type="candle",
                    value=value,
                    source_timestamp=candle.start,
                    received_at=received_at,
                    status=candle.status,
                    sequence_num=seq_num,
                    ohlcv=ohlcv,
                )
            )
    return out


class MarketBus:
    """Fan-out of qualified realtime data to registered consumers. Future Chart /
    Signal engines and persistence subscribe here without touching the WS manager."""

    def __init__(self) -> None:
        self._consumers: List[Callable[[RealtimeDatum], Any]] = []

    def subscribe(self, consumer: Callable[[RealtimeDatum], Any]) -> None:
        self._consumers.append(consumer)

    async def publish(self, datum: RealtimeDatum) -> None:
        for consumer in list(self._consumers):
            result = consumer(datum)
            if asyncio.iscoroutine(result):
                await result


class MarketStateStore:
    """Single in-memory source of truth for realtime data, guarded by one lock.

    Transport integrity uses the per-connection sequence_num; per-product state
    ordering uses each datum's real timestamp. Malformed / invalid data never
    overwrites a valid last state, and no value or sequence is fabricated."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._tickers: Dict[str, RealtimeDatum] = {}
        self._candles: Dict[str, RealtimeDatum] = {}
        self._last_sequence: Optional[int] = None
        self._gaps = 0
        self._duplicates = 0
        self._out_of_order = 0
        self._messages = 0
        self._heartbeat_counter: Optional[int] = None
        self._last_heartbeat_at: Optional[datetime] = None

    async def reset_transport(self) -> None:
        """Called on every new/reconnected socket: never compare a new socket's
        first sequence with the previous connection's last one."""
        async with self._lock:
            self._last_sequence = None

    async def check_sequence(self, seq: Optional[int]) -> str:
        async with self._lock:
            self._messages += 1
            if not isinstance(seq, int):
                return "unknown"
            if self._last_sequence is None:
                self._last_sequence = seq
                return "first"
            if seq == self._last_sequence + 1:
                self._last_sequence = seq
                return "ok"
            if seq > self._last_sequence + 1:
                self._gaps += seq - self._last_sequence - 1
                self._last_sequence = seq
                return "gap"
            if seq == self._last_sequence:
                self._duplicates += 1
                return "duplicate"
            self._out_of_order += 1
            return "out_of_order"

    async def apply_ticker(self, datum: RealtimeDatum) -> bool:
        if datum.status in (DataQualityStatus.INVALID, DataQualityStatus.MISSING):
            return False
        if datum.value is None:
            return False
        async with self._lock:
            prev = self._tickers.get(datum.product_id)
            if (
                prev is not None
                and prev.source_timestamp is not None
                and datum.source_timestamp is not None
                and datum.source_timestamp < prev.source_timestamp
            ):
                return False  # strictly older than stored -> keep the newer state
            self._tickers[datum.product_id] = datum
            return True

    async def apply_candle(self, datum: RealtimeDatum) -> bool:
        if datum.status == DataQualityStatus.INVALID:
            return False
        async with self._lock:
            prev = self._candles.get(datum.product_id)
            if (
                prev is not None
                and prev.source_timestamp is not None
                and datum.source_timestamp is not None
                and datum.source_timestamp < prev.source_timestamp
            ):
                return False  # earlier bucket than stored -> do not overwrite
            # same `start` is allowed: the live 5-min bucket updates in place.
            self._candles[datum.product_id] = datum
            return True

    async def record_heartbeat(self, counter: Optional[int], at: datetime) -> None:
        async with self._lock:
            if isinstance(counter, int):
                self._heartbeat_counter = counter
            self._last_heartbeat_at = at

    async def get_ticker(self, product_id: str) -> Optional[RealtimeDatum]:
        async with self._lock:
            return self._tickers.get(product_id)

    async def get_candle(self, product_id: str) -> Optional[RealtimeDatum]:
        async with self._lock:
            return self._candles.get(product_id)

    async def get_realtime(self, product_id: str) -> Dict[str, object]:
        async with self._lock:
            ticker = self._tickers.get(product_id)
            candle = self._candles.get(product_id)
        if ticker is None and candle is None:
            return {
                "product_id": product_id,
                "status": DataQualityStatus.MISSING.value,
                "ticker": None,
                "candle": None,
            }
        return {
            "product_id": product_id,
            "status": DataQualityStatus.VALID.value,
            "ticker": ticker.to_dict() if ticker else None,
            "candle": candle.to_dict() if candle else None,
        }

    async def health(self) -> Dict[str, object]:
        async with self._lock:
            products = sorted(set(self._tickers) | set(self._candles))
            return {
                "messages": self._messages,
                "last_sequence_num": self._last_sequence,
                "gaps_detected": self._gaps,
                "duplicates_detected": self._duplicates,
                "out_of_order_detected": self._out_of_order,
                "heartbeat_counter": self._heartbeat_counter,
                "last_heartbeat_at": (
                    self._last_heartbeat_at.isoformat() if self._last_heartbeat_at else None
                ),
                "products_tracked": products,
            }


market_store = MarketStateStore()
market_bus = MarketBus()


@api_router.get("/market/realtime/{symbol}")
async def market_realtime(symbol: str) -> dict:
    return await market_store.get_realtime(symbol.upper())


# ============================ paginated history (increment 4) =================
# Fetch a candle history longer than one Coinbase request (max CANDLE_MAX_LIMIT)
# by fanning out to sequential windows, then merge/dedup/sort. Robust to unknown
# `end` inclusivity: each provider window requests at most PROVIDER_SAFE_BUCKETS
# (= CANDLE_MAX_LIMIT - 1) buckets, so even an inclusive `end` yields <= 350
# candidate starts; a repeated boundary candle is removed by dedup on
# (product_id, granularity, start). The internal contract is a half-open range
# [start, end): start < end and both aligned to the granularity. No candle is
# ever fabricated; absences are diagnosed, never filled.


def max_history_span(granularity: str) -> int:
    """Max span (seconds) a single /history request may cover, from the safe
    window width and the application guard rail. Granularity-dependent."""
    bucket = GRANULARITIES[granularity][1]
    return PROVIDER_SAFE_BUCKETS * bucket * settings.history_max_windows


def plan_candle_windows(granularity: str, start: int, end: int) -> List[tuple]:
    """Pure, deterministic window planner (no network). Windows are [w_start,
    w_end] in UNIX seconds, each <= PROVIDER_SAFE_BUCKETS * bucket wide, stepping
    by the same amount (1-bucket overlap at each boundary, removed later by dedup)."""
    if granularity not in GRANULARITIES:
        raise ValueError(f"unsupported granularity: {granularity}")
    bucket = GRANULARITIES[granularity][1]
    if start % bucket != 0 or end % bucket != 0:
        raise ValueError("start and end must be aligned to the granularity (seconds)")
    if start >= end:
        raise ValueError("start must be strictly before end")
    step = PROVIDER_SAFE_BUCKETS * bucket
    windows: List[tuple] = []
    w_start = start
    while w_start < end:
        w_end = min(end, w_start + step)
        windows.append((w_start, w_end))
        w_start += step
    return windows


def _missing_buckets_24_7(starts: List[int], bucket: int) -> List[Dict[str, int]]:
    """Gap diagnostic on a 24/7 grid: within the data span, which buckets are
    absent. Kept ABSTRACT so a market-calendar-aware version (Forex/Gold sessions,
    weekends, holidays) can replace it later without touching the pipeline."""
    gaps: List[Dict[str, int]] = []
    for i in range(1, len(starts)):
        step = starts[i] - starts[i - 1]
        if step > bucket:
            gaps.append({"after_start": starts[i - 1], "missing_buckets": step // bucket - 1})
    return gaps


async def fetch_candle_history(symbol: str, granularity: str, start: int, end: int) -> dict:
    """Assemble a paginated candle history. Raises ValueError on invalid request
    (unknown granularity, misaligned/reversed range, range too large)."""
    if granularity in GRANULARITIES and end - start > max_history_span(granularity):
        raise ValueError(
            f"requested range too large (max {settings.history_max_windows} windows)"
        )
    windows = plan_candle_windows(granularity, start, end)  # validates gran/align/order
    bucket = GRANULARITIES[granularity][1]

    collected: Dict[int, Candle] = {}
    invalid_count = 0
    failed = 0
    succeeded = 0
    for w_start, w_end in windows:
        try:
            candles, _status = await market_provider.get_candles_range(
                symbol, granularity, w_start, w_end
            )
            succeeded += 1
        except httpx.HTTPError:
            failed += 1
            continue
        for candle in candles:
            if candle.start is None or candle.status == DataQualityStatus.INVALID:
                invalid_count += 1
                continue
            key = int(candle.start.timestamp())
            if key < start or key >= end:  # enforce internal [start, end) contract
                continue
            collected[key] = candle  # dedup by start (identity: product+gran+start)

    starts_sorted = sorted(collected)
    kept = [collected[k] for k in starts_sorted]
    # Coinbase is 24/7: route gap detection through the ALWAYS_OPEN_24_7 calendar.
    # Always24_7Calendar.analyze_gaps delegates to _missing_buckets_24_7, so the
    # result is byte-for-byte identical to the pre-6A behaviour (no regression).
    gaps = _COINBASE_CALENDAR.analyze_gaps(starts_sorted, bucket).missing

    transport_complete = failed == 0
    if failed > 0:
        status = "PARTIAL"
    elif not kept:
        status = "EMPTY"
    elif invalid_count > 0 or gaps:
        status = "PARTIAL"
    else:
        status = "COMPLETE"
    data_complete = status == "COMPLETE"

    return {
        "source": "coinbase",
        "symbol": symbol.upper(),
        "granularity": granularity,
        "status": status,
        "requested_range": {"start": start, "end": end},
        "provider_windows": {
            "planned": len(windows),
            "succeeded": succeeded,
            "failed": failed,
        },
        "first_candle_start": starts_sorted[0] if starts_sorted else None,
        "last_candle_start": starts_sorted[-1] if starts_sorted else None,
        "count": len(kept),
        "invalid_candles_count": invalid_count,
        "gaps": gaps,
        "transport_complete": transport_complete,
        "data_complete": data_complete,
        "complete": data_complete,
        "candles": [c.to_dict() for c in kept],
    }


@api_router.get("/market/candles/{symbol}/history")
async def market_candles_history(
    symbol: str, start: int, end: int, granularity: str = "1m"
) -> dict:
    try:
        return await fetch_candle_history(symbol, granularity, start, end)
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail={"status": "INVALID", "reason": str(exc)}
        ) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=503, detail={"status": "UNAVAILABLE", "reason": str(exc)}
        ) from exc


# ============================ persistence (increment 5) ======================
# Postgres = durable HISTORICAL truth (MarketStateStore stays the in-memory
# REALTIME truth). One pipeline: WS/REST -> parse/quality -> MarketBus ->
# PersistenceConsumer. Identity/PK = (source, product_id, granularity,
# bucket_start). OHLCV = NUMERIC(38,18) (exact decimal, never float).
#
# Time semantics kept distinct:
#   observed_at      = OUR pipeline receive/observe clock (homogeneous REST/WS) —
#                      the ONLY field used to arbitrate freshness on upsert.
#   source_timestamp = provider time when present (audit/diagnostic only).
#   updated_at       = OUR DB write time (audit only; never a freshness proof).
# is_closed = bucket time-closed by OUR clock+margin; NEVER "provider-certified
# final", so a time-closed row may still receive a newer admissible correction.

metadata = MetaData()

candles_table = Table(
    "candles",
    metadata,
    Column("source", String, primary_key=True),
    Column("product_id", String, primary_key=True),
    Column("granularity", String, primary_key=True),
    Column("bucket_start", DateTime(timezone=True), primary_key=True),
    Column("open", Numeric(38, 18), nullable=False),
    Column("high", Numeric(38, 18), nullable=False),
    Column("low", Numeric(38, 18), nullable=False),
    Column("close", Numeric(38, 18), nullable=False),
    Column("volume", Numeric(38, 18), nullable=False),
    Column("quality", String, nullable=False),
    Column("is_closed", Boolean, nullable=False),
    Column("origin", String, nullable=False),  # "ws" | "rest"
    Column("source_timestamp", DateTime(timezone=True), nullable=True),
    Column("observed_at", DateTime(timezone=True), nullable=False),
    Column("received_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)


class PersistenceStatus(str, Enum):
    READY = "READY"
    DEGRADED = "DEGRADED"
    UNAVAILABLE = "UNAVAILABLE"


class PersistenceState:
    """Explicit, observable persistence health. Never a silent false success."""

    def __init__(self) -> None:
        self.ready = False
        self.status = PersistenceStatus.UNAVAILABLE
        self.errors = 0
        self.last_error: Optional[str] = None

    def mark_ready(self) -> None:
        self.ready = True
        self.status = PersistenceStatus.READY

    def mark_init_failed(self, detail: str) -> None:
        self.ready = False
        self.status = PersistenceStatus.UNAVAILABLE
        self.errors += 1
        self.last_error = detail

    def mark_runtime_error(self, detail: str) -> None:
        # Transient runtime failure after a successful init: degrade, don't reset
        # ready=False permanently; the counter stays cumulative.
        self.errors += 1
        self.last_error = detail
        if self.ready:
            self.status = PersistenceStatus.DEGRADED

    def mark_write_ok(self) -> None:
        # Deterministic recovery: a later successful write clears a transient
        # DEGRADED back to READY. Never clears the cumulative error counter.
        if self.ready and self.status == PersistenceStatus.DEGRADED:
            self.status = PersistenceStatus.READY

    def to_dict(self) -> Dict[str, object]:
        return {
            "persistence_ready": self.ready,
            "persistence_status": self.status.value,
            "persistence_errors": self.errors,
            "persistence_last_error": self.last_error,
        }


persistence_state = PersistenceState()


def is_candle_closed(
    bucket_start: datetime, bucket_seconds: int, now: Optional[datetime] = None
) -> bool:
    """True if the bucket is time-closed by OUR clock + margin. Not provider-final."""
    reference = now or utcnow()
    margin = settings.candle_finalization_margin_seconds
    end = bucket_start + timedelta(seconds=bucket_seconds + margin)
    return reference >= end


async def init_candle_schema() -> None:
    """Create the candles table if absent (idempotent). Raises on real DDL failure
    so it is NEVER swallowed into a silent false success."""
    async with engine.begin() as conn:
        await conn.run_sync(metadata.create_all)


@dataclass(frozen=True)
class CandleRow:
    source: str
    product_id: str
    granularity: str
    bucket_start: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    quality: DataQualityStatus
    origin: str  # "ws" | "rest"
    source_timestamp: Optional[datetime]
    observed_at: datetime


def _bucket_seconds_for(granularity: str) -> int:
    if granularity in GRANULARITIES:
        return GRANULARITIES[granularity][1]
    return 300  # WS candles are 5-minute buckets


def candle_row_from_realtime(datum: RealtimeDatum) -> Optional[CandleRow]:
    """Build a persistable CandleRow from a WS candle RealtimeDatum. Returns None
    for non-candle / invalid / incomplete data (never fabricates)."""
    if datum.data_type != "candle" or datum.status == DataQualityStatus.INVALID:
        return None
    if datum.source_timestamp is None or datum.ohlcv is None:
        return None
    o = datum.ohlcv
    return CandleRow(
        source=datum.source,
        product_id=datum.product_id,
        granularity="5m",  # Coinbase WS candles are 5-minute buckets
        bucket_start=datum.source_timestamp,
        open=o["open"], high=o["high"], low=o["low"], close=o["close"], volume=o["volume"],
        quality=datum.status,
        origin="ws",
        source_timestamp=datum.source_timestamp,
        observed_at=datum.received_at,
    )


def _row_to_values(row: CandleRow, now: datetime) -> Dict[str, object]:
    bucket_seconds = _bucket_seconds_for(row.granularity)
    return {
        "source": row.source,
        "product_id": row.product_id,
        "granularity": row.granularity,
        "bucket_start": row.bucket_start,
        "open": Decimal(str(row.open)),
        "high": Decimal(str(row.high)),
        "low": Decimal(str(row.low)),
        "close": Decimal(str(row.close)),
        "volume": Decimal(str(row.volume)),
        "quality": row.quality.value,
        "is_closed": is_candle_closed(row.bucket_start, bucket_seconds, now=now),
        "origin": row.origin,
        "source_timestamp": row.source_timestamp,
        "observed_at": row.observed_at,
        "received_at": now,
        "updated_at": now,
    }


async def persist_candles(rows: List[CandleRow]) -> int:
    """Idempotent, ATOMIC batch upsert of candle rows. One persist_candles call =
    ONE transaction (multiple SQL batches inside, no intermediate commit). Any
    batch failure rolls back the WHOLE operation. Returns the number of rows sent.
    Upsert accepts a row only if it is not strictly older than the stored one
    (observed_at), and INVALID rows are never sent. Raises on DB error."""
    rows = [r for r in rows if r.quality != DataQualityStatus.INVALID]
    if not rows:
        return 0
    now = utcnow()
    batch_size = max(1, settings.persist_batch_size)
    try:
        async with engine.begin() as conn:  # BEGIN ... COMMIT (or ROLLBACK on error)
            for i in range(0, len(rows), batch_size):
                chunk = rows[i:i + batch_size]
                values = [_row_to_values(r, now) for r in chunk]
                stmt = pg_insert(candles_table).values(values)
                update_cols = {
                    c: stmt.excluded[c]
                    for c in (
                        "open", "high", "low", "close", "volume", "quality",
                        "is_closed", "origin", "source_timestamp", "observed_at",
                        "updated_at",
                    )
                }
                stmt = stmt.on_conflict_do_update(
                    index_elements=["source", "product_id", "granularity", "bucket_start"],
                    set_=update_cols,
                    # accept only a non-older observation; never degrade with stale data
                    where=stmt.excluded["observed_at"] >= candles_table.c.observed_at,
                )
                await conn.execute(stmt)
    except Exception as exc:  # noqa: BLE001 - surface, never a false success
        persistence_state.mark_runtime_error(str(exc))
        raise
    persistence_state.mark_write_ok()
    return len(rows)


class PersistenceConsumer:
    """MarketBus consumer that persists WS candle events. A DB error never kills
    the bus/WS loop: it is logged, counted, and flips persistence to DEGRADED."""

    async def __call__(self, datum: RealtimeDatum) -> None:
        if not persistence_state.ready:
            return
        row = candle_row_from_realtime(datum)
        if row is None:
            return
        try:
            await persist_candles([row])
        except Exception as exc:  # noqa: BLE001
            log.warning("Persistence consumer error: %s", exc, exc_info=True)


persistence_consumer = PersistenceConsumer()


async def persist_history_result(result: Dict[str, object]) -> Dict[str, object]:
    """Persist the VALID candles of a fetch_candle_history result. Persists real
    VALID candles even when the fetch was PARTIAL, but NEVER asserts range
    completeness in the DB (completeness is recomputed on read)."""
    source = str(result.get("source", "coinbase"))
    product_id = str(result["symbol"])
    granularity = str(result["granularity"])
    rows = _history_dicts_to_rows(source, product_id, granularity, result.get("candles"), utcnow())
    written = await persist_candles(rows)
    return {
        "persisted": written,
        "fetch_status": result.get("status"),
        "data_complete": result.get("data_complete"),
    }


def _history_dicts_to_rows(
    source: str, product_id: str, granularity: str, raw: object, observed_at: datetime
) -> List[CandleRow]:
    rows: List[CandleRow] = []
    if not isinstance(raw, list):
        return rows
    for item in raw:
        if not isinstance(item, dict):
            continue
        if item.get("quality") == DataQualityStatus.INVALID.value:
            continue
        start = parse_iso8601(item.get("start"))
        o = _to_float(item.get("open"))
        h = _to_float(item.get("high"))
        low = _to_float(item.get("low"))
        c = _to_float(item.get("close"))
        v = _to_float(item.get("volume"))
        if start is None or o is None or h is None or low is None or c is None or v is None:
            continue
        rows.append(
            CandleRow(
                source=source, product_id=product_id.upper(), granularity=granularity,
                bucket_start=start,
                open=o, high=h, low=low, close=c, volume=v,
                quality=DataQualityStatus.VALID, origin="rest",
                source_timestamp=None, observed_at=observed_at,
            )
        )
    return rows


async def read_stored_candles(
    symbol: str, granularity: str, start: int, end: int, limit: int
) -> List[Dict[str, object]]:
    """Read persisted candles, chronological ascending, half-open [start, end).
    Parameterised query; raises on DB error (caller maps to 503)."""
    bucket_seconds = _bucket_seconds_for(granularity)
    start_dt = datetime.fromtimestamp(int(start), tz=timezone.utc)
    end_dt = datetime.fromtimestamp(int(end), tz=timezone.utc)
    now = utcnow()
    t = candles_table
    stmt = (
        t.select()
        .where(t.c.source == "coinbase")
        .where(t.c.product_id == symbol.upper())
        .where(t.c.granularity == granularity)
        .where(t.c.bucket_start >= start_dt)
        .where(t.c.bucket_start < end_dt)
        .order_by(t.c.bucket_start.asc())
        .limit(max(1, min(int(limit), settings.db_read_max_rows)))
    )
    out: List[Dict[str, object]] = []
    async with engine.connect() as conn:
        result = await conn.execute(stmt)
        for r in result.mappings():
            out.append(
                {
                    "source": r["source"],
                    "product_id": r["product_id"],
                    "granularity": r["granularity"],
                    "start": int(r["bucket_start"].timestamp()),
                    "open": str(r["open"]),
                    "high": str(r["high"]),
                    "low": str(r["low"]),
                    "close": str(r["close"]),
                    "volume": str(r["volume"]),
                    "quality": r["quality"],
                    "is_closed": is_candle_closed(r["bucket_start"], bucket_seconds, now=now),
                    "origin": r["origin"],
                }
            )
    return out


@api_router.get("/market/candles/{symbol}/stored")
async def market_candles_stored(
    symbol: str, start: int, end: int, granularity: str = "1m", limit: int = 1000
) -> dict:
    if not persistence_state.ready:
        raise HTTPException(
            status_code=503,
            detail={"status": "UNAVAILABLE", "reason": "persistence not ready"},
        )
    if granularity not in GRANULARITIES:
        raise HTTPException(
            status_code=400, detail={"status": "INVALID", "reason": "unsupported granularity"}
        )
    if start >= end:
        raise HTTPException(
            status_code=400, detail={"status": "INVALID", "reason": "start must be before end"}
        )
    try:
        candles = await read_stored_candles(symbol, granularity, start, end, limit)
    except Exception as exc:  # noqa: BLE001
        persistence_state.mark_runtime_error(str(exc))
        raise HTTPException(
            status_code=503, detail={"status": "UNAVAILABLE", "reason": str(exc)}
        ) from exc
    return {
        "source": "coinbase",
        "symbol": symbol.upper(),
        "granularity": granularity,
        "status": "EMPTY" if not candles else "OK",
        "count": len(candles),
        "candles": candles,
    }


# ============================ multi-asset foundation (increment 6A) ===========
# Canonical, provider-agnostic model so Chart/Strategy/Signal/History/DB never
# need to know Coinbase (or a future provider) specifics. 6A is a PURE abstraction:
# no external connector, no invented market calendar/hours, no invented instrument
# metadata. Coinbase stays functionally identical; its 24/7 behaviour is the
# ALWAYS_OPEN_24_7 policy, and gap detection is routed through it unchanged.


class AssetClass(str, Enum):
    CRYPTO = "CRYPTO"
    FOREX = "FOREX"
    METAL = "METAL"
    INDEX = "INDEX"


class Capability(str, Enum):
    TICKER_REST = "TICKER_REST"
    TICKER_WS = "TICKER_WS"
    CANDLES_REST = "CANDLES_REST"
    CANDLES_WS = "CANDLES_WS"
    HISTORY_INTRADAY = "HISTORY_INTRADAY"
    HISTORY_DAILY = "HISTORY_DAILY"
    VOLUME = "VOLUME"
    BID_ASK = "BID_ASK"
    TRADES = "TRADES"
    ORDER_BOOK = "ORDER_BOOK"


class VolumeSemantics(str, Enum):
    BASE_ASSET_VOLUME = "BASE_ASSET_VOLUME"
    QUOTE_VOLUME = "QUOTE_VOLUME"
    TICK_VOLUME = "TICK_VOLUME"
    CONTRACT_VOLUME = "CONTRACT_VOLUME"
    NOT_AVAILABLE = "NOT_AVAILABLE"
    UNKNOWN = "UNKNOWN"


class MarketCalendarPolicy(str, Enum):
    ALWAYS_OPEN_24_7 = "ALWAYS_OPEN_24_7"
    NOT_CONFIGURED = "NOT_CONFIGURED"
    # Declared intent only; NOT implemented in 6A (no invented hours). Until real
    # official hours are added, calendar_for() maps these to NotConfigured -> UNKNOWN.
    FOREX_WEEK = "FOREX_WEEK"
    US_EQUITY_RTH = "US_EQUITY_RTH"


class OpenState(str, Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    UNKNOWN = "UNKNOWN"


class MarketAvailability(str, Enum):
    # Orthogonal to DataQualityStatus (which is untouched). A normal market close
    # is NOT a provider outage is NOT missing data — three distinct axes.
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"
    NOT_SUPPORTED = "NOT_SUPPORTED"
    CALENDAR_UNKNOWN = "CALENDAR_UNKNOWN"


@dataclass(frozen=True)
class Instrument:
    canonical_symbol: str  # the ONLY global identity used across the app
    asset_class: AssetClass
    base_asset: Optional[str]
    quote_asset: Optional[str]
    display_name: str
    timezone: str  # IANA market timezone (stored; no session math in 6A)
    market_calendar: MarketCalendarPolicy
    volume_semantics: VolumeSemantics
    price_precision: Optional[int] = None  # None until verified (future risk calc)
    tick_size: Optional[Decimal] = None  # None until verified (future risk calc)


class InstrumentRegistry:
    """Canonical instrument identities. In-memory in 6A (no Postgres table)."""

    def __init__(self) -> None:
        self._by_canonical: Dict[str, Instrument] = {}

    def register(self, instrument: Instrument) -> None:
        self._by_canonical[instrument.canonical_symbol] = instrument

    def get(self, canonical_symbol: str) -> Optional[Instrument]:
        return self._by_canonical.get(canonical_symbol)

    def all(self) -> List[Instrument]:
        return list(self._by_canonical.values())


class ProviderSymbolMap:
    """Bidirectional (provider, provider_symbol) <-> canonical_symbol mapping.
    Never assumes canonical == provider symbol. Unmapped -> None (NOT_MAPPED)."""

    def __init__(self) -> None:
        self._to_provider: Dict[tuple, str] = {}
        self._to_canonical: Dict[tuple, str] = {}

    def add(self, provider: str, canonical: str, provider_symbol: str) -> None:
        self._to_provider[(provider, canonical)] = provider_symbol
        self._to_canonical[(provider, provider_symbol)] = canonical

    def to_provider(self, provider: str, canonical: str) -> Optional[str]:
        return self._to_provider.get((provider, canonical))

    def to_canonical(self, provider: str, provider_symbol: str) -> Optional[str]:
        return self._to_canonical.get((provider, provider_symbol))


@dataclass
class ProviderProfile:
    """Explicit provider capabilities, PER asset class (a provider may offer FX
    intraday but a metal only daily). Missing capability -> NOT_SUPPORTED, never a
    silent fallback."""

    name: str
    capabilities_by_asset_class: Dict[AssetClass, Set[Capability]]
    granularities_by_asset_class: Dict[AssetClass, Set[str]]

    def supports(self, capability: Capability, asset_class: AssetClass) -> bool:
        return capability in self.capabilities_by_asset_class.get(asset_class, set())

    def supports_granularity(self, granularity: str, asset_class: AssetClass) -> bool:
        return granularity in self.granularities_by_asset_class.get(asset_class, set())


class MarketCalendar:
    """Interface. is_market_expected_open / expected_bucket_starts / analyze_gaps.
    Never concludes OPEN or CLOSED by assumption when not configured."""

    policy: MarketCalendarPolicy = MarketCalendarPolicy.NOT_CONFIGURED

    def is_market_expected_open(self, ts_unix: int) -> OpenState:
        raise NotImplementedError

    def expected_bucket_starts(self, granularity: str, start: int, end: int) -> Optional[List[int]]:
        raise NotImplementedError

    def analyze_gaps(self, starts: List[int], bucket: int) -> "GapReport":
        raise NotImplementedError


@dataclass(frozen=True)
class GapReport:
    status: str  # "ANALYZED" (missing meaningful) | "UNKNOWN" (no conclusion)
    missing: List[Dict[str, int]]


class Always24_7Calendar(MarketCalendar):
    """Crypto 24/7. Reproduces the exact pre-6A gap logic (delegates to
    _missing_buckets_24_7) so Coinbase behaviour is unchanged."""

    policy = MarketCalendarPolicy.ALWAYS_OPEN_24_7

    def is_market_expected_open(self, ts_unix: int) -> OpenState:
        return OpenState.OPEN

    def expected_bucket_starts(self, granularity: str, start: int, end: int) -> Optional[List[int]]:
        if granularity not in GRANULARITIES:
            return None
        bucket = GRANULARITIES[granularity][1]
        return list(range(start, end, bucket))

    def analyze_gaps(self, starts: List[int], bucket: int) -> GapReport:
        return GapReport("ANALYZED", _missing_buckets_24_7(starts, bucket))


class NotConfiguredCalendar(MarketCalendar):
    """No verified hours -> everything UNKNOWN. Never invents OPEN/CLOSED/gaps."""

    policy = MarketCalendarPolicy.NOT_CONFIGURED

    def is_market_expected_open(self, ts_unix: int) -> OpenState:
        return OpenState.UNKNOWN

    def expected_bucket_starts(self, granularity: str, start: int, end: int) -> Optional[List[int]]:
        return None

    def analyze_gaps(self, starts: List[int], bucket: int) -> GapReport:
        return GapReport("UNKNOWN", [])


# Verified weekly Forex hours: opens Sunday 17:00 and closes Friday 17:00 in
# America/New_York local time (DST handled by IANA -> 22:00 UTC in winter, 21:00
# UTC in summer). NEVER a fixed UTC offset. Source: widely corroborated retail
# spot-forex week (FOREX.com, City Index, TMGM, babypips, ...).
FOREX_ANCHOR_TZ = "America/New_York"
FOREX_WEEK_OPEN_HOUR = 17   # Sunday 17:00 New York
FOREX_WEEK_CLOSE_HOUR = 17  # Friday 17:00 New York

# INDICATIVE financial-center session hours (local business hours via IANA, DST
# automatic). These are indicative CENTER hours, NOT a specific broker's hours,
# and are NOT used to authorise/deny trading. Tokyo does not observe DST.
FOREX_SESSIONS = (
    ("Sydney", "Australia/Sydney", 8, 17),
    ("Tokyo", "Asia/Tokyo", 9, 18),
    ("London", "Europe/London", 8, 17),
    ("New York", "America/New_York", 8, 17),
)


def _zone(name: str) -> Optional[ZoneInfo]:
    """Return a ZoneInfo or None (fail-safe) if IANA data is unavailable."""
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, KeyError, ValueError):
        return None


def _forex_week_bounds(now_utc: datetime) -> Optional[Dict[str, object]]:
    """Compute Forex weekly OPEN/CLOSED anchored on America/New_York 17:00 Sun->Fri.
    Returns dict{is_open, weekend, next_open_utc, next_close_utc} or None if the
    timezone database is unavailable (caller maps None -> UNKNOWN)."""
    ny = _zone(FOREX_ANCHOR_TZ)
    if ny is None:
        return None
    now_ny = now_utc.astimezone(ny)
    # Monday=0 .. Sunday=6
    wd = now_ny.weekday()

    def at_hour(day_offset: int, hour: int) -> datetime:
        base = (now_ny + timedelta(days=day_offset)).replace(
            hour=hour, minute=0, second=0, microsecond=0
        )
        return base.astimezone(timezone.utc)

    # Previous Sunday 17:00 and this Friday 17:00 in NY local terms.
    days_since_sunday = (wd + 1) % 7  # Sunday -> 0, Monday -> 1, ... Saturday -> 6
    sunday_open = at_hour(-days_since_sunday, FOREX_WEEK_OPEN_HOUR)
    friday_close = at_hour(-days_since_sunday + 5, FOREX_WEEK_CLOSE_HOUR)
    is_open = sunday_open <= now_utc < friday_close
    weekend = not is_open
    if is_open:
        next_close_utc: Optional[datetime] = friday_close
        next_open_utc: Optional[datetime] = None
    else:
        # Next Sunday 17:00 NY (this week's if still ahead, else next week's).
        candidate = sunday_open if now_utc < sunday_open else at_hour(-days_since_sunday + 7,
                                                                      FOREX_WEEK_OPEN_HOUR)
        next_open_utc = candidate
        next_close_utc = None
    return {
        "is_open": is_open,
        "weekend": weekend,
        "next_open_utc": next_open_utc,
        "next_close_utc": next_close_utc,
    }


class ForexWeekCalendar(MarketCalendar):
    """Forex weekly calendar: OPEN/CLOSED anchored on America/New_York 17:00 Sun->Fri
    (DST via IANA). Gap analysis stays UNKNOWN: a missing bar is never a gap because
    Massive emits no bar without a new quote. Holidays are NOT modelled (UNKNOWN)."""

    policy = MarketCalendarPolicy.FOREX_WEEK

    def is_market_expected_open(self, ts_unix: int) -> OpenState:
        try:
            now = datetime.fromtimestamp(int(ts_unix), tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return OpenState.UNKNOWN
        bounds = _forex_week_bounds(now)
        if bounds is None:
            return OpenState.UNKNOWN
        return OpenState.OPEN if bounds["is_open"] else OpenState.CLOSED

    def expected_bucket_starts(self, granularity: str, start: int, end: int) -> Optional[List[int]]:
        return None  # never fabricate a forex grid

    def analyze_gaps(self, starts: List[int], bucket: int) -> GapReport:
        return GapReport("UNKNOWN", [])  # no-quote != gap -> no fabricated gaps


_ALWAYS_24_7 = Always24_7Calendar()
_NOT_CONFIGURED = NotConfiguredCalendar()
_FOREX_WEEK = ForexWeekCalendar()
_COINBASE_CALENDAR = _ALWAYS_24_7


def calendar_for(policy: MarketCalendarPolicy) -> MarketCalendar:
    """ALWAYS_OPEN_24_7 -> crypto; FOREX_WEEK -> Forex weekly (NY-anchored). Every
    other policy resolves to NotConfigured -> UNKNOWN (no invented hours)."""
    if policy == MarketCalendarPolicy.ALWAYS_OPEN_24_7:
        return _ALWAYS_24_7
    if policy == MarketCalendarPolicy.FOREX_WEEK:
        return _FOREX_WEEK
    return _NOT_CONFIGURED


def forex_active_sessions(now_utc: datetime) -> List[Dict[str, object]]:
    """INDICATIVE financial-center sessions (local business hours via IANA). Marked
    indicative; NOT broker hours; NOT a trading authorisation. A session is active
    only on a local weekday within its local business hours. tz missing -> active
    UNKNOWN (None) for that center, never fabricated."""
    out: List[Dict[str, object]] = []
    for name, tz_name, open_h, close_h in FOREX_SESSIONS:
        tz = _zone(tz_name)
        if tz is None:
            out.append({"name": name, "tz": tz_name, "active": None, "indicative": True})
            continue
        local = now_utc.astimezone(tz)
        weekday = local.weekday() < 5  # Mon-Fri local
        active = bool(weekday and open_h <= local.hour < close_h)
        out.append({
            "name": name, "tz": tz_name, "active": active, "indicative": True,
            "local_open_hour": open_h, "local_close_hour": close_h,
        })
    return out


def forex_market_state(now_utc: Optional[datetime] = None) -> Dict[str, object]:
    """Full Forex market-state payload. Market truth = OPEN/CLOSED/CLOSED_WEEKEND/
    UNKNOWN (NY-anchored). Sessions are indicative only. Never OPEN by default;
    unprovable fields stay null/UNKNOWN. Market state is independent from data
    quality (OPEN != LIVE; CLOSED != provider down)."""
    now = now_utc or utcnow()
    bounds = _forex_week_bounds(now)
    sessions = forex_active_sessions(now)
    if bounds is None:
        market_state = "UNKNOWN"
        reason = "timezone database unavailable"
        next_open = next_close = None
        current = None
    elif bounds["is_open"]:
        market_state = "OPEN"
        reason = "within the Forex trading week (Sun 17:00 -> Fri 17:00 New York)"
        next_open, next_close = None, bounds["next_close_utc"]
        active_names = [s["name"] for s in sessions if s.get("active") is True]
        current = active_names[0] if active_names else None
    else:
        market_state = "CLOSED_WEEKEND"
        reason = "weekend close (Fri 17:00 -> Sun 17:00 New York)"
        next_open, next_close = bounds["next_open_utc"], None
        current = None

    def iso(dt: object) -> Optional[str]:
        return dt.isoformat() if isinstance(dt, datetime) else None

    return {
        "asset_class": "FOREX",
        "market_state": market_state,
        "reason": reason,
        "current_session": current,       # indicative; may be null
        "sessions": sessions,             # indicative center hours (not broker hours)
        "next_open": iso(next_open),
        "next_close": iso(next_close),
        "timezone_internal": "UTC",
        "display_timezone": "America/Toronto",
        "holidays": "NOT_IMPLEMENTED",    # no robust verified holiday rule yet
        "source": "retail spot-forex week: Sun 17:00 -> Fri 17:00 America/New_York",
        "as_of": now.isoformat(),
    }


instrument_registry = InstrumentRegistry()
provider_symbol_map = ProviderSymbolMap()


def _register_coinbase_instruments() -> None:
    """The only PROVEN real case in 6A. Coinbase crypto: 24/7, base-asset volume.
    Unverified financial metadata (precision/tick) stays None (never invented)."""
    for canon, base, quote, name in (
        ("BTC-USD", "BTC", "USD", "Bitcoin / US Dollar"),
        ("ETH-USD", "ETH", "USD", "Ethereum / US Dollar"),
    ):
        instrument_registry.register(
            Instrument(
                canonical_symbol=canon,
                asset_class=AssetClass.CRYPTO,
                base_asset=base,
                quote_asset=quote,
                display_name=name,
                timezone="UTC",
                market_calendar=MarketCalendarPolicy.ALWAYS_OPEN_24_7,
                volume_semantics=VolumeSemantics.BASE_ASSET_VOLUME,
                price_precision=None,
                tick_size=None,
            )
        )
        provider_symbol_map.add("coinbase", canon, canon)  # Coinbase symbol == canonical here


COINBASE_PROFILE = ProviderProfile(
    name="coinbase",
    capabilities_by_asset_class={
        AssetClass.CRYPTO: {
            Capability.TICKER_REST,
            Capability.TICKER_WS,
            Capability.CANDLES_REST,
            Capability.CANDLES_WS,
            Capability.HISTORY_INTRADAY,
            Capability.VOLUME,
        },
    },
    granularities_by_asset_class={AssetClass.CRYPTO: set(GRANULARITIES)},
)

_register_coinbase_instruments()


# ============================ Massive Forex REST (increment 6B-1) =============
# First real Multi-Asset connector: Massive (ex-Polygon) Forex REST aggregates.
# Reuses the 6A canonical model + the existing Candle/persistence bricks (NO
# parallel architecture). Officially verified (massive.com/docs/rest/forex):
#   GET /v2/aggs/ticker/{forexTicker}/range/{multiplier}/{timespan}/{from}/{to}
#   response {results:[{o,h,l,c,v,t}]} with t = Unix MILLISECONDS, bars aligned in
#   Eastern Time; aggregates are derived from bid/ask QUOTES, not executed trades,
#   and no bar is emitted when no quote arrives (absence != gap).
# Rules honoured: internal time = UTC; volume semantics = UNKNOWN (never invented);
# no invented Forex calendar (NOT_CONFIGURED -> gaps UNKNOWN); NO provider symbol
# registered by deduction -> unverified canonical stays NOT_MAPPED.

MASSIVE_FOREX_GRANULARITIES: Dict[str, tuple] = {
    "1m": (1, "minute"),
    "5m": (5, "minute"),
    "15m": (15, "minute"),
    "30m": (30, "minute"),
    "1h": (1, "hour"),
    "2h": (2, "hour"),
    "4h": (4, "hour"),
    "6h": (6, "hour"),
    "1d": (1, "day"),
}


def _unix_ms_to_dt(value: Any) -> Optional[datetime]:
    """Massive aggregate `t` is a UNIX timestamp in MILLISECONDS. Returns aware UTC,
    or None if unparseable (never invents a timestamp)."""
    try:
        ms = int(value)
    except (TypeError, ValueError):
        return None
    if ms <= 0:
        return None
    try:
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None


def massive_agg_to_candle(
    item: Any, max_age_seconds: float, now: Optional[datetime] = None
) -> Candle:
    """Convert one Massive forex aggregate into a qualified Candle. Pure, no network.
    Missing/unparseable OHLCV or timestamp -> INVALID; never fabricated."""
    if not isinstance(item, dict):
        return Candle(None, None, None, None, None, None, DataQualityStatus.INVALID)
    start = _unix_ms_to_dt(item.get("t"))
    open_ = _to_float(item.get("o"))
    high = _to_float(item.get("h"))
    low = _to_float(item.get("l"))
    close = _to_float(item.get("c"))
    volume = _to_float(item.get("v"))
    if (
        start is None or low is None or high is None
        or open_ is None or close is None or volume is None
    ):
        return Candle(start, low, high, open_, close, volume, DataQualityStatus.INVALID)
    if low < 0 or high < 0 or open_ < 0 or close < 0 or volume < 0:
        return Candle(start, low, high, open_, close, volume, DataQualityStatus.INVALID)
    return Candle(start, low, high, open_, close, volume,
                  classify_freshness(start, max_age_seconds, now=now))


def massive_aggs_to_candles(
    payload: Any, max_age_seconds: float, now: Optional[datetime] = None
) -> tuple:
    """Parse a Massive forex aggregates response into (list[Candle], status).
    Empty/malformed -> ([], MISSING)."""
    raw = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(raw, list) or not raw:
        return [], DataQualityStatus.MISSING
    candles = [massive_agg_to_candle(x, max_age_seconds, now=now) for x in raw]
    has_valid = any(c.status != DataQualityStatus.INVALID for c in candles)
    return candles, (DataQualityStatus.VALID if has_valid else DataQualityStatus.INVALID)


class MassiveForexProvider:
    """Massive Forex REST aggregates adapter. No WebSocket, no failover (6B-1).
    Resolves the provider ticker via the verified ProviderSymbolMap only."""

    SOURCE = "massive"

    def __init__(self, rest_url: Optional[str] = None, api_key: Optional[str] = None) -> None:
        self.rest_url = (rest_url or settings.massive_rest_url).rstrip("/")
        self.api_key = api_key if api_key is not None else settings.massive_api_key
        self.client: Optional[httpx.AsyncClient] = None

    async def connect(self) -> None:
        if self.client is None:
            headers = {"Accept": "application/json"}
            if self.api_key:
                # Header auth keeps the key OUT of the URL, so it can never leak via
                # an httpx exception/request-URL. (Redaction below is defence in depth.)
                headers["Authorization"] = f"Bearer {self.api_key}"
            self.client = httpx.AsyncClient(
                base_url=self.rest_url,
                timeout=settings.massive_request_timeout_seconds,
                headers=headers,
            )

    async def disconnect(self) -> None:
        if self.client is not None:
            await self.client.aclose()
            self.client = None

    async def _get(self, path: str, params: Optional[dict] = None) -> dict:
        if self.client is None:
            await self.connect()
        assert self.client is not None
        resp = await self.client.get(path, params=params)
        resp.raise_for_status()
        payload = resp.json()
        if not isinstance(payload, dict):
            raise ValueError("Massive returned a non-object JSON response")
        return payload

    async def get_candles_range(
        self, canonical_symbol: str, granularity: str, start: int, end: int
    ):
        """Fetch aggregates for a canonical forex symbol over [start, end] UNIX
        seconds. NOT_MAPPED / NOT_SUPPORTED raise ValueError (fail-safe, no fake)."""
        ticker = provider_symbol_map.to_provider(self.SOURCE, canonical_symbol)
        if ticker is None:
            raise ValueError(f"NOT_MAPPED: no verified Massive symbol for {canonical_symbol}")
        if granularity not in MASSIVE_FOREX_GRANULARITIES:
            raise ValueError(f"NOT_SUPPORTED granularity for Massive forex: {granularity}")
        multiplier, timespan = MASSIVE_FOREX_GRANULARITIES[granularity]
        start_ms = int(start) * 1000
        end_ms = int(end) * 1000
        payload = await self._get(
            f"/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{start_ms}/{end_ms}",
            params={"adjusted": "true", "sort": "asc", "limit": 50000},
        )
        return massive_aggs_to_candles(payload, max_age_seconds=float("inf"))

    async def list_forex_tickers(self) -> List[str]:
        """Fetch the REAL set of Massive forex ticker symbols (for verified mapping
        activation). Header auth; a single page of up to 1000 (covers the majors)."""
        payload = await self._get(
            "/v3/reference/tickers",
            params={"market": "fx", "active": "true", "limit": 1000},
        )
        results = payload.get("results")
        out: List[str] = []
        if isinstance(results, list):
            for row in results:
                if isinstance(row, dict) and isinstance(row.get("ticker"), str):
                    out.append(row["ticker"])
        return out


def _register_massive_forex_instruments() -> None:
    """Register the 8 CANONICAL forex identities (ours, not provider deductions).
    Calendar NOT_CONFIGURED (no invented hours); volume UNKNOWN; precision/tick None.
    NO provider_symbol mapping is added here: Massive symbols stay NOT_MAPPED until
    officially verified via /v3/reference/tickers (register_massive_forex_symbol)."""
    pairs = (
        ("EUR-USD", "EUR", "USD", "Euro / US Dollar"),
        ("GBP-USD", "GBP", "USD", "British Pound / US Dollar"),
        ("USD-JPY", "USD", "JPY", "US Dollar / Japanese Yen"),
        ("USD-CHF", "USD", "CHF", "US Dollar / Swiss Franc"),
        ("AUD-USD", "AUD", "USD", "Australian Dollar / US Dollar"),
        ("USD-CAD", "USD", "CAD", "US Dollar / Canadian Dollar"),
        ("NZD-USD", "NZD", "USD", "New Zealand Dollar / US Dollar"),
        ("EUR-CAD", "EUR", "CAD", "Euro / Canadian Dollar"),
    )
    for canon, base, quote, name in pairs:
        instrument_registry.register(
            Instrument(
                canonical_symbol=canon,
                asset_class=AssetClass.FOREX,
                base_asset=base,
                quote_asset=quote,
                display_name=name,
                timezone="UTC",
                market_calendar=MarketCalendarPolicy.FOREX_WEEK,
                volume_semantics=VolumeSemantics.UNKNOWN,
                price_precision=None,
                tick_size=None,
            )
        )


MASSIVE_FOREX_PROFILE = ProviderProfile(
    name="massive",
    capabilities_by_asset_class={
        AssetClass.FOREX: {
            Capability.CANDLES_REST,
            Capability.HISTORY_INTRADAY,
            Capability.HISTORY_DAILY,
            Capability.BID_ASK,
        },
    },
    granularities_by_asset_class={AssetClass.FOREX: set(MASSIVE_FOREX_GRANULARITIES)},
)


def register_massive_forex_symbol(canonical: str, provider_ticker: str) -> None:
    """Register a Massive forex mapping ONLY after official verification via
    /v3/reference/tickers. Never called at import (unverified -> NOT_MAPPED)."""
    provider_symbol_map.add("massive", canonical, provider_ticker)


# Candidate provider tickers per canonical (naming convention only). A mapping is
# activated ONLY if the exact ticker is really present in the official Massive
# response (activate_massive_forex_mappings) -> never a deduction.
EXPECTED_MASSIVE_FOREX: Dict[str, str] = {
    "EUR-USD": "C:EURUSD",
    "GBP-USD": "C:GBPUSD",
    "USD-JPY": "C:USDJPY",
    "USD-CHF": "C:USDCHF",
    "AUD-USD": "C:AUDUSD",
    "USD-CAD": "C:USDCAD",
    "NZD-USD": "C:NZDUSD",
    "EUR-CAD": "C:EURCAD",
}
MASSIVE_XAU_TICKER = "C:XAUUSD"  # only DETECTED/reported; never auto-integrated in 6B-1A

_APIKEY_RE = re.compile(r"(apikey=)[^&\s]+", re.IGNORECASE)


def _redact_secret(text: str) -> str:
    """Mask an apiKey=... query value in any string before logging (defence in
    depth; header auth already keeps the key out of URLs)."""
    return _APIKEY_RE.sub(r"\1REDACTED", text)


class ForexMappingActivation:
    """Explicit, key-free diagnostic of the runtime mapping activation."""

    def __init__(self) -> None:
        self.attempted = False
        self.activated = False
        self.reason: Optional[str] = None
        self.confirmed: Dict[str, str] = {}
        self.xau: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "attempted": self.attempted,
            "activated": self.activated,
            "reason": self.reason,
            "confirmed_count": len(self.confirmed),
            "xau": self.xau,  # {"ticker": "C:XAUUSD"} or None (reported, NOT integrated)
        }


massive_forex_activation = ForexMappingActivation()


async def activate_massive_forex_mappings() -> Dict[str, object]:
    """Runtime activation. No MASSIVE_API_KEY -> no call, mappings stay NOT_MAPPED,
    backend stays healthy. With a key -> query the official ticker reference and
    register ONLY the symbols really returned. Any failure (network/401/403/429/
    timeout/invalid) is caught: explicit diagnostic, no mapping, never a crash. The
    key is never logged (header auth + redaction)."""
    state = massive_forex_activation
    state.attempted = True
    if not settings.massive_api_key:
        state.reason = "MASSIVE_API_KEY not set; Massive forex mappings remain NOT_MAPPED"
        return state.to_dict()
    try:
        tickers = await massive_forex_provider.list_forex_tickers()
    except Exception as exc:  # noqa: BLE001 - must never fail the whole backend
        state.reason = _redact_secret(str(exc))[:200] or "activation failed"
        log.warning("Massive forex mapping activation failed: %s", _redact_secret(str(exc)))
        return state.to_dict()
    ticker_set = set(tickers)
    for canonical, expected in EXPECTED_MASSIVE_FOREX.items():
        if expected in ticker_set:  # verified present in the OFFICIAL response
            register_massive_forex_symbol(canonical, expected)
            state.confirmed[canonical] = expected
    if MASSIVE_XAU_TICKER in ticker_set:
        state.xau = {"ticker": MASSIVE_XAU_TICKER}  # reported only; NOT wired to Metal
    state.activated = True
    state.reason = (
        f"activated {len(state.confirmed)}/{len(EXPECTED_MASSIVE_FOREX)} forex mappings"
    )
    return state.to_dict()


@api_router.get("/market/forex/mappings")
async def market_forex_mappings() -> dict:
    mappings = []
    for canonical in EXPECTED_MASSIVE_FOREX:
        provider_symbol = provider_symbol_map.to_provider("massive", canonical)
        mappings.append(
            {
                "canonical": canonical,
                "provider_symbol": provider_symbol,
                "status": "MAPPED" if provider_symbol else "NOT_MAPPED",
            }
        )
    return {
        "provider": "massive",
        "asset_class": "FOREX",
        "mappings": mappings,
        "activation": massive_forex_activation.to_dict(),
    }


@api_router.get("/market/forex/market-state")
async def market_forex_state() -> dict:
    """Forex market open/closed + indicative sessions. Market state (NY-anchored
    weekly hours) is independent from data quality: OPEN never implies LIVE, and a
    normal CLOSED never implies a provider outage."""
    return forex_market_state()


# ==================== Twelve Data REST — Gold XAU/USD (sub-increment 1/3) ======
# First metal connector foundation: Twelve Data /time_series for XAU/USD (SPOT,
# officially catalogued as "Gold Spot / Precious Metal"). This sub-increment adds
# ONLY the REST provider + strict Decimal parsing. No METAL instrument, no public
# endpoint, no persistence, no UI yet (later sub-increments).
#
# Verified officially (twelvedata.com/docs): symbol "XAU/USD"; /time_series returns
# {values:[{datetime, open, high, low, close, volume}]} as STRINGS; intraday
# datetime honours timezone=UTC; header auth "Authorization: apikey <key>"; errors
# may arrive as HTTP 4xx/5xx OR as HTTP 200 with body {"status":"error","code":...}.
# OHLC parsed strictly to Decimal (never float). Volume optional for spot metal.

TWELVEDATA_REST_URL = "https://api.twelvedata.com"
# Only officially-verified intraday intervals for this sub-increment. 6h and 1d are
# intentionally excluded: 6h is NOT_SUPPORTED by the provider; 1d has a different
# timezone semantic (daily ignores timezone=UTC) and is left NOT_IMPLEMENTED here.
TWELVEDATA_GRANULARITIES: Dict[str, str] = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
}


@dataclass(frozen=True)
class TwelveDataBar:
    """A parsed Twelve Data time-series bar. OHLC are exact Decimals (never float);
    volume is Optional (spot metal may omit it)."""
    datetime_utc: Optional[datetime]
    open: Optional[Decimal]
    high: Optional[Decimal]
    low: Optional[Decimal]
    close: Optional[Decimal]
    volume: Optional[Decimal]
    status: DataQualityStatus


@dataclass(frozen=True)
class TwelveDataResult:
    """Explicit outcome of a Twelve Data request. status is a business state; bars
    are only meaningful for OK. reason is a CLEAN message (no key, no raw URL)."""
    status: str  # OK | EMPTY | NOT_MAPPED | NOT_SUPPORTED | NO_KEY |
    #              ACCESS_DENIED | RATE_LIMITED | UNAVAILABLE
    bars: List[TwelveDataBar]
    reason: Optional[str] = None


def _td_parse_dt(value: Any) -> Optional[datetime]:
    """Parse an intraday Twelve Data `datetime` string requested with timezone=UTC.
    Returns an aware UTC datetime, or None. Date-only (daily) is intentionally NOT
    parsed here (daily is NOT_IMPLEMENTED in this sub-increment)."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def twelvedata_bar_from_value(
    item: Any, max_age_seconds: float, now: Optional[datetime] = None
) -> TwelveDataBar:
    """Parse one time-series value into a qualified TwelveDataBar. OHLC via Decimal
    only; missing/non-numeric/negative OHLC or timestamp -> INVALID (never faked)."""
    if not isinstance(item, dict):
        return TwelveDataBar(None, None, None, None, None, None, DataQualityStatus.INVALID)
    dt = _td_parse_dt(item.get("datetime"))
    open_ = _to_decimal(item.get("open"))
    high = _to_decimal(item.get("high"))
    low = _to_decimal(item.get("low"))
    close = _to_decimal(item.get("close"))
    volume = _to_decimal(item.get("volume"))  # optional for spot metal
    if dt is None or open_ is None or high is None or low is None or close is None:
        return TwelveDataBar(dt, open_, high, low, close, volume, DataQualityStatus.INVALID)
    if open_ < 0 or high < 0 or low < 0 or close < 0 or (volume is not None and volume < 0):
        return TwelveDataBar(dt, open_, high, low, close, volume, DataQualityStatus.INVALID)
    status = classify_freshness(dt, max_age_seconds, now=now)
    return TwelveDataBar(dt, open_, high, low, close, volume, status)


def parse_twelvedata_time_series(
    payload: Any, max_age_seconds: float, now: Optional[datetime] = None
) -> TwelveDataResult:
    """Parse a /time_series response body. CRITICAL: Twelve Data may return HTTP 200
    with {"status":"error", ...}; that is a provider error, never a success."""
    if not isinstance(payload, dict):
        return TwelveDataResult("UNAVAILABLE", [], "malformed provider response")
    if payload.get("status") == "error":
        code = payload.get("code")
        if code == 429:
            return TwelveDataResult("RATE_LIMITED", [], "rate limited by provider (429)")
        if code in (401, 403):
            return TwelveDataResult("ACCESS_DENIED", [], f"access denied by provider ({code})")
        return TwelveDataResult("UNAVAILABLE", [], "provider returned an error status")
    values = payload.get("values")
    if not isinstance(values, list) or not values:
        return TwelveDataResult("EMPTY", [], None)
    bars = [twelvedata_bar_from_value(v, max_age_seconds, now=now) for v in values]
    return TwelveDataResult("OK", bars, None)


class TwelveDataProvider:
    """Twelve Data REST adapter (Gold XAU/USD spot). Header auth; the key is never
    placed in the URL/query, never logged, never returned. No network without a key.
    Availability on the plan is decided by the provider response, never assumed."""

    SOURCE = "twelvedata"

    def __init__(self, rest_url: Optional[str] = None, api_key: Optional[str] = None) -> None:
        self.rest_url = (rest_url or settings.twelvedata_rest_url).rstrip("/")
        self.api_key = api_key if api_key is not None else settings.twelvedata_api_key
        self.client: Optional[httpx.AsyncClient] = None

    async def connect(self) -> None:
        if self.client is None:
            headers = {"Accept": "application/json"}
            if self.api_key:
                # Header auth keeps the key OUT of the URL (no leak via exceptions).
                headers["Authorization"] = f"apikey {self.api_key}"
            self.client = httpx.AsyncClient(
                base_url=self.rest_url,
                timeout=settings.twelvedata_request_timeout_seconds,
                headers=headers,
            )

    async def disconnect(self) -> None:
        if self.client is not None:
            await self.client.aclose()
            self.client = None

    async def get_time_series(
        self, canonical_symbol: str, granularity: str, outputsize: int = 30,
        start: Optional[str] = None, end: Optional[str] = None,
    ) -> TwelveDataResult:
        """Fetch XAU/USD (or any mapped twelvedata symbol) intraday bars. Returns an
        explicit TwelveDataResult; never raises for provider/HTTP errors."""
        symbol = provider_symbol_map.to_provider(self.SOURCE, canonical_symbol)
        if symbol is None:
            return TwelveDataResult(
                "NOT_MAPPED", [], f"no verified twelvedata symbol for {canonical_symbol}"
            )
        if granularity not in TWELVEDATA_GRANULARITIES:
            return TwelveDataResult(
                "NOT_SUPPORTED", [], f"granularity {granularity} not supported (twelvedata)"
            )
        if not self.api_key:
            return TwelveDataResult("NO_KEY", [], "TWELVEDATA_API_KEY not set")
        if self.client is None:
            await self.connect()
        assert self.client is not None
        params: Dict[str, Any] = {
            "symbol": symbol,
            "interval": TWELVEDATA_GRANULARITIES[granularity],
            "timezone": "UTC",       # intraday honours UTC (verified)
            "order": "asc",
            "outputsize": outputsize,
        }
        if start is not None:
            params["start_date"] = start
        if end is not None:
            params["end_date"] = end
        try:
            resp = await self.client.get("/time_series", params=params)
        except httpx.TimeoutException:
            return TwelveDataResult("UNAVAILABLE", [], "provider timeout")
        except httpx.HTTPError:
            return TwelveDataResult("UNAVAILABLE", [], "provider unreachable")
        code = resp.status_code
        if code in (401, 403):
            return TwelveDataResult("ACCESS_DENIED", [], f"access denied by provider ({code})")
        if code == 429:
            return TwelveDataResult("RATE_LIMITED", [], "rate limited by provider (429)")
        if code >= 500:
            return TwelveDataResult("UNAVAILABLE", [], f"provider server error ({code})")
        try:
            payload = resp.json()
        except (ValueError, json.JSONDecodeError):
            return TwelveDataResult("UNAVAILABLE", [], "malformed provider response")
        # max_age off for explicit history; freshness re-derived by callers later.
        return parse_twelvedata_time_series(payload, max_age_seconds=float("inf"))


# Officially-catalogued Twelve Data symbol for gold spot -> verified provider
# mapping (a mapping is not an entitlement: MAPPED can coexist with NOT_ENTITLED).
provider_symbol_map.add("twelvedata", "XAU-USD", "XAU/USD")

twelvedata_provider = TwelveDataProvider()


massive_forex_provider = MassiveForexProvider()
_register_massive_forex_instruments()


async def fetch_forex_history(
    canonical_symbol: str, granularity: str, start: int, end: int
) -> dict:
    """Assemble Massive forex history for a CANONICAL symbol. Reuses the 6A calendar
    for gap semantics: forex is NOT_CONFIGURED -> gaps UNKNOWN, and a missing bar is
    NEVER turned into a gap/MISSING (Massive emits no bar without a new quote)."""
    inst = instrument_registry.get(canonical_symbol)
    if inst is None or inst.asset_class != AssetClass.FOREX:
        raise ValueError(f"unknown forex instrument: {canonical_symbol}")
    provider_ticker = provider_symbol_map.to_provider("massive", canonical_symbol)
    if provider_ticker is None:
        return {
            "source": "massive", "canonical_symbol": canonical_symbol, "provider_symbol": None,
            "granularity": granularity, "status": "NOT_MAPPED",
            "reason": "Massive symbol not officially verified/mapped",
            "count": 0, "candles": [],
        }
    if granularity not in MASSIVE_FOREX_GRANULARITIES:
        raise ValueError(f"NOT_SUPPORTED granularity for Massive forex: {granularity}")
    if start >= end:
        raise ValueError("start must be strictly before end")
    bucket = GRANULARITIES[granularity][1] if granularity in GRANULARITIES else 60
    try:
        candles, _status = await massive_forex_provider.get_candles_range(
            canonical_symbol, granularity, start, end
        )
    except httpx.HTTPStatusError as exc:
        # Map the provider HTTP status to a CLEAN status/reason. Never expose the
        # raw exception, request URL, MDN link or the API key. Mapping is kept.
        code = exc.response.status_code if exc.response is not None else 0
        if code in (401, 403):
            status_val, reason = "ACCESS_DENIED", f"access denied by provider ({code})"
        elif code == 429:
            status_val, reason = "RATE_LIMITED", "provider rate limit reached (429)"
        else:
            status_val, reason = "UNAVAILABLE", f"provider error ({code})"
        return {
            "source": "massive", "canonical_symbol": canonical_symbol,
            "provider_symbol": provider_ticker, "granularity": granularity,
            "status": status_val, "reason": reason, "http_status": code,
            "count": 0, "candles": [],
        }
    except httpx.HTTPError:
        return {
            "source": "massive", "canonical_symbol": canonical_symbol,
            "provider_symbol": provider_ticker, "granularity": granularity,
            "status": "UNAVAILABLE", "reason": "provider unreachable",
            "count": 0, "candles": [],
        }
    collected: Dict[int, Candle] = {}
    invalid = 0
    for candle in candles:
        if candle.start is None or candle.status == DataQualityStatus.INVALID:
            invalid += 1
            continue
        key = int(candle.start.timestamp())
        if key < start or key >= end:
            continue
        collected[key] = candle
    starts = sorted(collected)
    kept = [collected[k] for k in starts]
    # Forex calendar NOT_CONFIGURED -> gaps UNKNOWN (absence of a bar is legitimate).
    report = calendar_for(inst.market_calendar).analyze_gaps(starts, bucket)
    return {
        "source": "massive",
        "canonical_symbol": canonical_symbol,
        "provider_symbol": provider_ticker,
        "granularity": granularity,
        "status": "EMPTY" if not kept else "OK",
        "requested_range": {"start": start, "end": end},
        "count": len(kept),
        "invalid_candles_count": invalid,
        "latest_quality": _latest_quality(kept),
        "gaps_status": report.status,   # "UNKNOWN" for forex (no verified calendar)
        "gaps": report.missing,          # [] when UNKNOWN
        "data_complete": False,          # gap analysis unavailable -> never assert complete
        "timezone": inst.timezone,       # UTC internal; ET bar-alignment is a provider detail
        "volume_semantics": inst.volume_semantics.value,  # UNKNOWN (never invented)
        "candles": [c.to_dict() for c in kept],
    }


async def persist_forex_result(result: Dict[str, object]) -> int:
    """Persist a fetch_forex_history result's VALID candles under source='massive',
    product_id=provider_symbol. Reuses the generic history->rows + persist_candles."""
    if result.get("status") != "OK" or not result.get("provider_symbol"):
        return 0
    rows = _history_dicts_to_rows(
        "massive", str(result["provider_symbol"]), str(result["granularity"]),
        result.get("candles"), utcnow(),
    )
    return await persist_candles(rows)


@api_router.get("/market/forex/{symbol}/history")
async def market_forex_history(
    symbol: str, start: int, end: int, granularity: str = "1h"
) -> dict:
    try:
        result = await fetch_forex_history(symbol.upper(), granularity, start, end)
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail={"status": "INVALID", "reason": str(exc)}
        ) from exc
    status = result.get("status")
    if status == "NOT_MAPPED":
        raise HTTPException(
            status_code=409,
            detail={"status": "NOT_MAPPED", "reason": result.get("reason")},
        )
    if status == "ACCESS_DENIED":
        raise HTTPException(
            status_code=403,
            detail={"status": "ACCESS_DENIED", "reason": result.get("reason")},
        )
    if status == "RATE_LIMITED":
        raise HTTPException(
            status_code=429,
            detail={"status": "RATE_LIMITED", "reason": result.get("reason")},
        )
    if status == "UNAVAILABLE":
        raise HTTPException(
            status_code=503,
            detail={"status": "UNAVAILABLE", "reason": result.get("reason")},
        )
    return result


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
    await market_provider.connect()
    await massive_forex_provider.connect()
    activation = await activate_massive_forex_mappings()
    log.info("Massive forex mapping activation: %s", activation)
    try:
        await init_candle_schema()
        persistence_state.mark_ready()
        market_bus.subscribe(persistence_consumer)
    except Exception as exc:  # noqa: BLE001 - explicit, never a silent false success
        persistence_state.mark_init_failed(str(exc))
        log.error("Candle schema init failed; persistence UNAVAILABLE: %s", exc)
    try:
        yield
    finally:
        await market_ws.stop()
        await market_provider.disconnect()
        await massive_forex_provider.disconnect()
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