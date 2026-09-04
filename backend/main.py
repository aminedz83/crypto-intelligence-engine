# CI395 recovery marker: cumulative V16-M5B25B-FIX2 backend baseline.
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
import hashlib
import json
import logging
import re
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from decimal import Decimal, InvalidOperation, ROUND_DOWN
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Set, Tuple
from urllib.parse import quote as url_quote

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
    Integer,
    MetaData,
    Numeric,
    String,
    Table,
    Text,
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

    # Paper monitor. One second matches the backend-to-frontend realtime cadence.
    paper_monitor_interval_seconds: float = 1.0

    # Massive (ex-Polygon) Forex REST. api_key empty by default: no calls happen
    # until an officially-verified symbol mapping is registered (never deduced).
    massive_api_key: str = ""
    massive_rest_url: str = "https://api.massive.com"
    massive_request_timeout_seconds: float = 10.0
    massive_forex_ws_url: str = "wss://socket.massive.com/forex"
    # Massive Indices Starter WebSocket documented as 15-minute delayed. Never label LIVE.
    massive_indices_ws_url: str = "wss://delayed.massive.com/indices"

    # Twelve Data REST (Gold XAU/USD spot). Key server-side only, header auth; no
    # call is made without a key. Availability on the account's plan is determined
    # at runtime from the provider response, never assumed.
    twelvedata_api_key: str = ""
    twelvedata_rest_url: str = "https://api.twelvedata.com"
    twelvedata_request_timeout_seconds: float = 10.0
    twelvedata_ws_url: str = "wss://ws.twelvedata.com/v1/quotes/price"

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

    async def get_product_specs(self, symbol: str) -> dict:
        symbol = symbol.upper()
        return await self._get(f"/market/products/{symbol}")

    async def list_public_spot_products(self, limit: int = 250) -> dict:
        """List public SPOT products, explicitly ranked by 24h quote volume."""
        limit = max(100, min(int(limit), 1000))
        return await self._get(
            "/market/products",
            params={
                "limit": limit,
                "product_type": "SPOT",
                "products_sort_order": "PRODUCTS_SORT_ORDER_VOLUME_24H_DESCENDING",
            },
        )

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

PAPER_ACCOUNT_ID = "default"
PAPER_INITIAL_CAPITAL = Decimal("1000")
PAPER_ACCOUNT_CURRENCY = "USD"

paper_account_table = Table(
    "paper_account",
    metadata,
    Column("account_id", String, primary_key=True),
    Column("currency", String, nullable=False),
    Column("initial_capital", Numeric(38, 18), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)

signal_decision_history_table = Table(
    "signal_decision_history",
    metadata,
    Column("decision_id", String, primary_key=True),
    Column("timestamp", DateTime(timezone=True), nullable=False),
    Column("symbol", String, nullable=False),
    Column("state", String, nullable=False),
    Column("reason", String, nullable=False),
    Column("setup_state", String, nullable=True),
    Column("latest_closed_timestamp", String, nullable=True),
    Column("detector_context", Text, nullable=True),
    Column("paper_only", Boolean, nullable=False),
    Column("execution", Boolean, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

paper_positions_table = Table(
    "paper_positions",
    metadata,
    Column("position_id", String, primary_key=True),
    Column("symbol", String, nullable=False),
    Column("side", String, nullable=False),
    Column("status", String, nullable=False),
    Column("entry", Numeric(38, 18), nullable=False),
    Column("stop_loss", Numeric(38, 18), nullable=False),
    Column("take_profit", Numeric(38, 18), nullable=False),
    Column("size", Numeric(38, 18), nullable=False),
    Column("size_unit", String, nullable=False),
    Column("risk_money", Numeric(38, 18), nullable=False),
    Column("risk_percent", Numeric(18, 8), nullable=False),
    Column("capital_before", Numeric(38, 18), nullable=False),
    Column("source", String, nullable=False),
    Column("source_timestamp", DateTime(timezone=True), nullable=False),
    Column("opened_at", DateTime(timezone=True), nullable=False),
    Column("close_reason", String, nullable=True),
    Column("close_price", Numeric(38, 18), nullable=True),
    Column("closed_at", DateTime(timezone=True), nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)


paper_fx_conversion_snapshots_table = Table(
    "paper_fx_conversion_snapshots",
    metadata,
    Column("position_id", String, primary_key=True),
    Column("phase", String, primary_key=True),
    Column("quote_currency", String, nullable=False),
    Column("quote_to_usd", Numeric(38, 18), nullable=False),
    Column("conversion_symbol", String, nullable=False),
    Column("conversion_price", Numeric(38, 18), nullable=True),
    Column("inverse", Boolean, nullable=False),
    Column("source", String, nullable=False),
    Column("source_timestamp", DateTime(timezone=True), nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
)


# V16-M5B30B — immutable strategy context captured at paper entry time.
# Kept in a dedicated table so existing paper_positions schema remains backward compatible.
paper_position_context_table = Table(
    "paper_position_context",
    metadata,
    Column("position_id", String, primary_key=True),
    Column("strategy_id", String, nullable=True),
    Column("strategy_version", String, nullable=True),
    Column("timeframe", String, nullable=True),
    Column("session", String, nullable=True),
    Column("market_regime", String, nullable=True),
    Column("setup_context", Text, nullable=True),
    Column("captured_at", DateTime(timezone=True), nullable=False),
)


# V16-M5B30E — audit log for explicit shadow-gate evaluations.
paper_adaptive_edge_shadow_table = Table(
    "paper_adaptive_edge_shadow",
    metadata,
    Column("decision_id", String, primary_key=True),
    Column("symbol", String, nullable=False),
    Column("strategy_id", String, nullable=False),
    Column("timeframe", String, nullable=False),
    Column("session", String, nullable=False),
    Column("market_regime", String, nullable=False),
    Column("period", String, nullable=False),
    Column("action", String, nullable=False),
    Column("reason", String, nullable=False),
    Column("evidence", String, nullable=True),
    Column("rank", Integer, nullable=True),
    Column("score", Numeric(10, 2), nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

# V16-M5B30F — immutable audit of active PAPER-only adaptive gate decisions.
paper_adaptive_edge_active_table = Table(
    "paper_adaptive_edge_active",
    metadata,
    Column("decision_id", String, primary_key=True),
    Column("symbol", String, nullable=False),
    Column("strategy_id", String, nullable=False),
    Column("timeframe", String, nullable=True),
    Column("session", String, nullable=True),
    Column("market_regime", String, nullable=True),
    Column("period", String, nullable=False),
    Column("action", String, nullable=False),
    Column("reason", String, nullable=False),
    Column("evidence", String, nullable=True),
    Column("rank", Integer, nullable=True),
    Column("score", Numeric(10, 2), nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
)


# V16-M5B30G — exact adaptive-edge evidence snapshot captured with each active gate.
paper_adaptive_edge_explainability_table = Table(
    "paper_adaptive_edge_explainability",
    metadata,
    Column("decision_id", String, primary_key=True),
    Column("components_json", Text, nullable=False),
    Column("metrics_json", Text, nullable=False),
    Column("policy_version", String, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)


# V16-M5B30H — links executed paper positions to the exact active edge gate.
paper_adaptive_edge_trade_link_table = Table(
    "paper_adaptive_edge_trade_link",
    metadata,
    Column("position_id", String, primary_key=True),
    Column("decision_id", String, nullable=False),
    Column("gate_action", String, nullable=False),
    Column("linked_at", DateTime(timezone=True), nullable=False),
)


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


class ServerSignalRequest(BaseModel):
    symbol: str = Field(min_length=1, max_length=64)
    setup_state: str
    direction: Optional[str] = None
    entry: Optional[Decimal] = None
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    risk_reward: Optional[Decimal] = None
    structure_confirmed: bool = False
    displacement_confirmed: bool = False
    order_block_confirmed: bool = False
    source_timestamp: datetime


class VerifiedAutoPaperEntryRequest(ServerSignalRequest):
    risk_percent: Decimal = Decimal("1")
    performance_timeframe: Optional[str] = None
    performance_session: Optional[str] = None
    performance_market_regime: Optional[str] = None
    performance_setup_context: Optional[str] = None


class AutoEntryCandidateRequest(VerifiedAutoPaperEntryRequest):
    candidate_id: str = Field(min_length=1, max_length=128)


@dataclass
class AutoEntryCandidateState:
    request: VerifiedAutoPaperEntryRequest
    candidate_id: str
    registered_at: datetime
    last_status: str = "PENDING"
    last_reason: Optional[str] = None
    attempts: int = 0


class PaperAutoEntryGateRequest(BaseModel):
    symbol: str = Field(min_length=1, max_length=64)
    signal_decision: str
    entry: Optional[Decimal] = None
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    risk_reward: Optional[Decimal] = None


class PaperPositionCreate(BaseModel):
    position_id: str = Field(min_length=1, max_length=128)
    symbol: str = Field(min_length=1, max_length=64)
    side: str
    entry: Decimal
    stop_loss: Decimal
    take_profit: Decimal
    size: Decimal
    size_unit: str = Field(min_length=1, max_length=32)
    risk_money: Decimal
    risk_percent: Decimal
    capital_before: Decimal
    source: str = Field(min_length=1, max_length=64)
    source_timestamp: datetime
    opened_at: datetime
    fx_quote_currency: Optional[str] = None
    fx_quote_to_usd: Optional[Decimal] = None
    fx_conversion_symbol: Optional[str] = None
    fx_conversion_price: Optional[Decimal] = None
    fx_conversion_inverse: Optional[bool] = None
    fx_conversion_source: Optional[str] = None
    fx_conversion_source_timestamp: Optional[datetime] = None
    performance_strategy_id: Optional[str] = None
    performance_strategy_version: Optional[str] = None
    performance_timeframe: Optional[str] = None
    performance_session: Optional[str] = None
    performance_market_regime: Optional[str] = None
    performance_setup_context: Optional[str] = None


def validate_paper_position_create(req: PaperPositionCreate) -> None:
    if req.side not in {"LONG", "SHORT"}:
        raise HTTPException(
            status_code=400, detail={"status": "INVALID", "reason": "SIDE_INVALID"}
        )
    positive = (req.entry, req.stop_loss, req.take_profit, req.size, req.risk_money)
    if any(value <= Decimal("0") for value in positive):
        raise HTTPException(
            status_code=400, detail={"status": "INVALID", "reason": "VALUE_INVALID"}
        )
    if req.risk_percent <= Decimal("0") or req.risk_percent > Decimal("100"):
        raise HTTPException(
            status_code=400, detail={"status": "INVALID", "reason": "RISK_INVALID"}
        )
    if req.capital_before <= Decimal("0"):
        raise HTTPException(
            status_code=400, detail={"status": "INVALID", "reason": "CAPITAL_INVALID"}
        )
    fx_fields = (
        req.fx_quote_currency,
        req.fx_quote_to_usd,
        req.fx_conversion_symbol,
        req.fx_conversion_inverse,
        req.fx_conversion_source,
    )
    if any(value is not None for value in fx_fields):
        if any(value is None for value in fx_fields):
            raise HTTPException(
                status_code=400,
                detail={"status": "INVALID", "reason": "FX_SNAPSHOT_INCOMPLETE"},
            )
        if req.fx_quote_to_usd is None or req.fx_quote_to_usd <= Decimal("0"):
            raise HTTPException(
                status_code=400,
                detail={"status": "INVALID", "reason": "FX_CONVERSION_INVALID"},
            )
    if req.side == "LONG" and not (req.stop_loss < req.entry < req.take_profit):
        raise HTTPException(
            status_code=400,
            detail={"status": "INVALID", "reason": "LONG_LEVELS_INVALID"},
        )
    if req.side == "SHORT" and not (req.take_profit < req.entry < req.stop_loss):
        raise HTTPException(
            status_code=400,
            detail={"status": "INVALID", "reason": "SHORT_LEVELS_INVALID"},
        )


def paper_position_to_dict(row: Any) -> Dict[str, object]:
    data = dict(row._mapping)
    raw_entry = data.get("entry")
    raw_size = data.get("size")
    raw_close = data.get("close_price")
    raw_side = data.get("side")
    realized_pnl: Optional[Decimal] = None
    if (
        data.get("status") == "CLOSED"
        and isinstance(raw_entry, Decimal)
        and isinstance(raw_size, Decimal)
        and isinstance(raw_close, Decimal)
        and isinstance(raw_side, str)
        and raw_side in {"LONG", "SHORT"}
    ):
        realized_pnl = calculate_paper_pnl(raw_side, raw_entry, raw_close, raw_size)
    for key in (
        "entry",
        "stop_loss",
        "take_profit",
        "size",
        "risk_money",
        "risk_percent",
        "capital_before",
        "close_price",
    ):
        if data.get(key) is not None:
            data[key] = str(data[key])
    for key in ("source_timestamp", "opened_at", "closed_at", "created_at", "updated_at"):
        if data.get(key) is not None:
            data[key] = data[key].isoformat()
    data["realized_pnl"] = str(realized_pnl) if realized_pnl is not None else None
    data["paper_only"] = True
    data["execution"] = False
    return data


def parse_coinbase_spot_specs(symbol: str, payload: dict) -> Dict[str, object]:
    required = (
        "base_increment",
        "quote_increment",
        "base_min_size",
        "base_max_size",
        "quote_min_size",
        "quote_max_size",
    )
    values: Dict[str, Decimal] = {}
    try:
        for key in required:
            value = Decimal(str(payload[key]))
            if value <= 0:
                raise ValueError(key)
            values[key] = value
    except (KeyError, ValueError, ArithmeticError):
        return {
            "status": "INVALID",
            "symbol": symbol,
            "source": "coinbase_public_product",
            "source_timestamp": None,
        }

    return {
        "status": "VALID",
        "symbol": symbol,
        "asset_class": "CRYPTO",
        "sizing_mode": "BASE_UNITS",
        **{key: str(value) for key, value in values.items()},
        "source": "coinbase_public_product",
        "source_timestamp": None,
    }


@api_router.get("/paper/instrument-specs/{symbol}")
async def get_paper_instrument_specs(symbol: str) -> Dict[str, object]:
    canonical = symbol.upper().replace("/", "-")
    instrument = instrument_registry.get(canonical)
    observed_at = utcnow()
    if instrument is None:
        return {
            "status": "UNAVAILABLE",
            "symbol": canonical,
            "reason": "INSTRUMENT_NOT_REGISTERED",
            "observed_at": observed_at.isoformat(),
        }
    if instrument.asset_class != AssetClass.CRYPTO:
        return {
            "status": "NOT_SUPPORTED",
            "symbol": canonical,
            "reason": "VERIFIED_SIZING_SOURCE_NOT_IMPLEMENTED",
            "observed_at": observed_at.isoformat(),
        }
    provider_symbol = provider_symbol_map.to_provider("coinbase", canonical)
    if provider_symbol is None:
        return {
            "status": "UNAVAILABLE",
            "symbol": canonical,
            "reason": "PROVIDER_SYMBOL_NOT_MAPPED",
            "observed_at": observed_at.isoformat(),
        }
    try:
        payload = await market_provider.get_product_specs(provider_symbol)
    except (httpx.HTTPError, ValueError):
        return {
            "status": "UNAVAILABLE",
            "symbol": canonical,
            "reason": "PROVIDER_UNAVAILABLE",
            "observed_at": observed_at.isoformat(),
        }
    specs = parse_coinbase_spot_specs(canonical, payload)
    specs["observed_at"] = observed_at.isoformat()
    return specs



# V16-M5B29A — Multi-Asset Verified Paper Sizing Foundation (non-executing)
# This increment intentionally DOES NOT authorize Forex/Metal/Index entries.
# It establishes explicit sizing/readiness semantics without inventing broker
# contract metadata. USD-quoted spot instruments can be preview-sized in native
# units because P&L is mathematically quote-currency units per price move; all
# broker-dependent contracts and non-USD quote conversions stay fail-closed.
MULTI_ASSET_SIZING_VERSION = "SERVER_MULTI_ASSET_SIZING_FOUNDATION_V1"


class MultiAssetSizingPreviewRequest(BaseModel):
    symbol: str = Field(min_length=3, max_length=32)
    capital: Decimal = Field(gt=Decimal("0"))
    risk_percent: Decimal = Field(gt=Decimal("0"), le=Decimal("100"))
    entry: Decimal = Field(gt=Decimal("0"))
    stop_loss: Decimal = Field(gt=Decimal("0"))


def multi_asset_sizing_readiness(symbol: str) -> Dict[str, object]:
    """Describe what is objectively safe to size without broker assumptions.

    CRYPTO keeps its existing Coinbase verified-specs path and is reported as
    delegated. Forex/metal/index readiness is deliberately conservative:
    - USD-quoted spot FX: native BASE_UNITS preview can be calculated in USD.
    - XAU-USD spot: native XAU_UNITS preview can be calculated in USD.
    - non-USD quoted FX requires a real FX conversion source before sizing.
    - cash indices require a verified contract/tick-value model before sizing.
    This function never opens or queues a paper position.
    """
    canonical = symbol.upper().replace("/", "-")
    instrument = instrument_registry.get(canonical)
    observed_at = utcnow()
    base: Dict[str, object] = {
        "validation": MULTI_ASSET_SIZING_VERSION,
        "symbol": canonical,
        "observed_at": observed_at.isoformat(),
        "paper_only": True,
        "execution": False,
        "auto_entry_authorized": False,
    }
    if instrument is None:
        return {**base, "status": "UNAVAILABLE", "reason": "INSTRUMENT_NOT_REGISTERED"}

    base.update(
        {
            "asset_class": instrument.asset_class.value,
            "base_asset": instrument.base_asset,
            "quote_asset": instrument.quote_asset,
            "market_calendar": instrument.market_calendar.value,
        }
    )

    if instrument.asset_class == AssetClass.CRYPTO:
        return {
            **base,
            "status": "DELEGATED",
            "reason": "USE_EXISTING_VERIFIED_CRYPTO_SPECS",
            "sizing_mode": "BASE_UNITS",
            "specs_endpoint": f"/api/v1/paper/instrument-specs/{canonical}",
        }

    # We require a configured calendar before any future multi-asset execution.
    if instrument.market_calendar == MarketCalendarPolicy.NOT_CONFIGURED:
        return {
            **base,
            "status": "BLOCKED",
            "reason": "MARKET_CALENDAR_NOT_CONFIGURED",
            "sizing_mode": None,
        }

    if instrument.asset_class == AssetClass.FOREX:
        if instrument.quote_asset != "USD":
            return {
                **base,
                "status": "BLOCKED",
                "reason": "ACCOUNT_CURRENCY_CONVERSION_NOT_IMPLEMENTED",
                "sizing_mode": None,
            }
        return {
            **base,
            "status": "PREVIEW_READY",
            "reason": "USD_QUOTED_SPOT_UNIT_PNL",
            "sizing_mode": "BASE_UNITS",
            "pnl_currency": "USD",
            "requires_broker_contract_specs_for_live_lots": True,
        }

    if instrument.asset_class == AssetClass.METAL:
        if canonical != "XAU-USD" or instrument.quote_asset != "USD":
            return {
                **base,
                "status": "BLOCKED",
                "reason": "METAL_SIZING_MODEL_NOT_VERIFIED",
                "sizing_mode": None,
            }
        # Currently XAU-USD is registered with NOT_CONFIGURED calendar, so the
        # calendar guard above intentionally blocks it until the session model is
        # independently implemented/validated.
        return {
            **base,
            "status": "PREVIEW_READY",
            "reason": "USD_QUOTED_SPOT_UNIT_PNL",
            "sizing_mode": "XAU_UNITS",
            "pnl_currency": "USD",
            "requires_broker_contract_specs_for_live_lots": True,
        }

    if instrument.asset_class == AssetClass.INDEX:
        return {
            **base,
            "status": "BLOCKED",
            "reason": "VERIFIED_INDEX_CONTRACT_VALUE_NOT_IMPLEMENTED",
            "sizing_mode": None,
        }

    return {**base, "status": "BLOCKED", "reason": "ASSET_CLASS_NOT_SUPPORTED"}


def calculate_multi_asset_unit_size(
    capital: Decimal,
    risk_percent: Decimal,
    entry: Decimal,
    stop_loss: Decimal,
    readiness: Dict[str, object],
) -> Dict[str, object]:
    """Preview unit sizing for objectively USD-quoted spot instruments only."""
    if readiness.get("status") != "PREVIEW_READY":
        return {
            "status": "BLOCKED",
            "reason": str(readiness.get("reason") or "SIZING_NOT_READY"),
            "execution": False,
        }
    if capital <= 0 or risk_percent <= 0 or risk_percent > Decimal("100"):
        return {"status": "BLOCKED", "reason": "RISK_INVALID", "execution": False}
    distance = abs(entry - stop_loss)
    if distance <= 0 or entry <= 0:
        return {"status": "BLOCKED", "reason": "STOP_DISTANCE_INVALID", "execution": False}

    risk_money = capital * risk_percent / Decimal("100")
    raw_units = risk_money / distance
    if raw_units <= 0:
        return {"status": "BLOCKED", "reason": "SIZE_INVALID", "execution": False}
    return {
        "status": "VALID",
        "validation": MULTI_ASSET_SIZING_VERSION,
        "size": raw_units,
        "size_unit": readiness.get("sizing_mode"),
        "risk_money": risk_money,
        "risk_percent": risk_percent,
        "stop_distance": distance,
        "pnl_currency": readiness.get("pnl_currency"),
        "broker_contract_size_applied": False,
        "paper_only": True,
        "execution": False,
    }


@api_router.get("/paper/multi-asset-sizing/readiness/{symbol}")
async def get_multi_asset_sizing_readiness(symbol: str) -> Dict[str, object]:
    return multi_asset_sizing_readiness(symbol)


@api_router.post("/paper/multi-asset-sizing/preview")
async def preview_multi_asset_sizing(
    req: MultiAssetSizingPreviewRequest,
) -> Dict[str, object]:
    readiness = multi_asset_sizing_readiness(req.symbol)
    sizing = calculate_multi_asset_unit_size(
        req.capital, req.risk_percent, req.entry, req.stop_loss, readiness
    )
    return {
        "status": sizing.get("status", "BLOCKED"),
        "symbol": req.symbol.upper().replace("/", "-"),
        "readiness": readiness,
        "sizing": sizing,
        "paper_only": True,
        "execution": False,
        "auto_entry_authorized": False,
    }

def evaluate_server_signal(req: ServerSignalRequest) -> Dict[str, object]:
    canonical = req.symbol.upper().replace("/", "-")
    reasons: List[str] = []
    decision = "WAIT"

    if req.setup_state != "ENTRY_NOW":
        reasons.append("SETUP_NOT_ENTRY_NOW")
    if req.direction not in {"BULLISH", "BEARISH"}:
        reasons.append("DIRECTION_INVALID")
    if not req.structure_confirmed:
        reasons.append("STRUCTURE_NOT_CONFIRMED")
    if not req.displacement_confirmed:
        reasons.append("DISPLACEMENT_NOT_CONFIRMED")
    if not req.order_block_confirmed:
        reasons.append("ORDER_BLOCK_NOT_CONFIRMED")

    levels = (req.entry, req.stop_loss, req.take_profit, req.risk_reward)
    if any(value is None for value in levels):
        reasons.append("TRADE_PLAN_INCOMPLETE")
    elif req.risk_reward is not None and req.risk_reward <= Decimal("0"):
        reasons.append("RR_INVALID")
    elif req.entry is not None and req.stop_loss is not None and req.take_profit is not None:
        if req.direction == "BULLISH" and not req.stop_loss < req.entry < req.take_profit:
            reasons.append("LONG_LEVELS_INVALID")
        if req.direction == "BEARISH" and not req.take_profit < req.entry < req.stop_loss:
            reasons.append("SHORT_LEVELS_INVALID")

    quality = classify_freshness(
        req.source_timestamp,
        settings.ticker_max_age_seconds,
        now=utcnow(),
    )
    if quality != DataQualityStatus.VALID:
        reasons.append("SOURCE_NOT_VALID")

    if not reasons:
        decision = "LONG" if req.direction == "BULLISH" else "SHORT"

    return {
        "status": "READY" if decision in {"LONG", "SHORT"} else "WAIT",
        "symbol": canonical,
        "decision": decision,
        "reasons": reasons,
        "source_timestamp": req.source_timestamp.isoformat(),
        "quality": quality.value,
        "authoritative": True,
        "execution": False,
    }


@api_router.post("/paper/signal/evaluate")
async def evaluate_paper_server_signal(req: ServerSignalRequest) -> Dict[str, object]:
    return evaluate_server_signal(req)


def floor_to_increment(value: Decimal, increment: Decimal) -> Decimal:
    if value <= 0 or increment <= 0:
        return Decimal("0")
    units = (value / increment).to_integral_value(rounding=ROUND_DOWN)
    return units * increment


def build_auto_paper_position_id(
    symbol: str, decision: str, source_timestamp: datetime
) -> str:
    raw = f"{symbol}|{decision}|{source_timestamp.isoformat()}".encode()
    digest = hashlib.sha256(raw).hexdigest()[:24]
    return f"auto-{symbol.lower()}-{digest}"


def calculate_verified_crypto_size(
    capital: Decimal,
    risk_percent: Decimal,
    entry: Decimal,
    stop_loss: Decimal,
    specs: Dict[str, object],
) -> Dict[str, object]:
    if capital <= 0 or risk_percent <= 0 or risk_percent > Decimal("100"):
        return {"status": "BLOCKED", "reason": "RISK_INVALID"}
    distance = abs(entry - stop_loss)
    if distance <= 0 or entry <= 0:
        return {"status": "BLOCKED", "reason": "STOP_DISTANCE_INVALID"}
    try:
        increment = Decimal(str(specs["base_increment"]))
        base_min = Decimal(str(specs["base_min_size"]))
        base_max = Decimal(str(specs["base_max_size"]))
        quote_min = Decimal(str(specs["quote_min_size"]))
        quote_max = Decimal(str(specs["quote_max_size"]))
    except (KeyError, ValueError, InvalidOperation):
        return {"status": "BLOCKED", "reason": "SPECS_INVALID"}

    risk_money = capital * risk_percent / Decimal("100")
    risk_size = risk_money / distance
    cash_size = capital / entry
    raw_size = min(risk_size, cash_size, base_max)
    size = floor_to_increment(raw_size, increment)
    notional = size * entry
    if size < base_min or size <= 0:
        return {"status": "BLOCKED", "reason": "SIZE_BELOW_MIN"}
    if size > base_max:
        return {"status": "BLOCKED", "reason": "SIZE_ABOVE_MAX"}
    if notional < quote_min:
        return {"status": "BLOCKED", "reason": "NOTIONAL_BELOW_MIN"}
    if notional > quote_max:
        return {"status": "BLOCKED", "reason": "NOTIONAL_ABOVE_MAX"}
    actual_risk = distance * size
    return {
        "status": "VALID",
        "size": size,
        "size_unit": "BASE_UNITS",
        "risk_money": actual_risk,
        "risk_percent": actual_risk / capital * Decimal("100"),
        "notional": notional,
    }


PAPER_MAX_OPEN_POSITIONS = 5
PAPER_MAX_TOTAL_OPEN_RISK_PERCENT = Decimal("5")
auto_paper_portfolio_lock = asyncio.Lock()


def evaluate_paper_portfolio_risk_guard(
    open_positions: List[Dict[str, object]],
    symbol: str,
    candidate_risk_money: Decimal,
    capital: Decimal,
) -> Dict[str, object]:
    canonical = symbol.upper().replace("/", "-")
    if capital <= 0 or candidate_risk_money <= 0:
        return {"status": "BLOCKED", "reason": "PORTFOLIO_RISK_INPUT_INVALID"}
    if len(open_positions) >= PAPER_MAX_OPEN_POSITIONS:
        return {"status": "BLOCKED", "reason": "MAX_OPEN_POSITIONS_REACHED"}
    if any(str(item.get("symbol", "")).upper() == canonical for item in open_positions):
        return {"status": "BLOCKED", "reason": "SYMBOL_ALREADY_OPEN"}
    try:
        open_risk = sum(
            (Decimal(str(item["risk_money"])) for item in open_positions),
            Decimal("0"),
        )
    except (KeyError, ValueError, InvalidOperation):
        return {"status": "BLOCKED", "reason": "OPEN_RISK_INVALID"}
    max_risk = capital * PAPER_MAX_TOTAL_OPEN_RISK_PERCENT / Decimal("100")
    projected_risk = open_risk + candidate_risk_money
    if projected_risk > max_risk:
        return {"status": "BLOCKED", "reason": "PORTFOLIO_RISK_LIMIT_REACHED"}
    return {
        "status": "VALID",
        "open_positions": len(open_positions),
        "open_risk_money": open_risk,
        "projected_risk_money": projected_risk,
        "max_risk_money": max_risk,
    }


async def get_open_paper_risk_snapshot() -> List[Dict[str, object]]:
    async with engine.connect() as conn:
        result = await conn.execute(
            text(
                "SELECT symbol, risk_money FROM paper_positions "
                "WHERE status='OPEN' ORDER BY opened_at ASC, position_id ASC"
            )
        )
        return [dict(row._mapping) for row in result.fetchall()]


async def _verified_auto_paper_entry_unlocked(
    req: VerifiedAutoPaperEntryRequest,
) -> Dict[str, object]:
    if not persistence_state.ready:
        raise HTTPException(
            status_code=503,
            detail={"status": "UNAVAILABLE", "reason": "persistence not ready"},
        )

    signal = evaluate_server_signal(req)
    if signal["status"] != "READY":
        return {
            "status": "BLOCKED",
            "reason": "SERVER_SIGNAL_NOT_READY",
            "signal": signal,
            "paper_only": True,
            "execution": False,
        }

    edge_gate = await evaluate_adaptive_edge_active_gate(
        req.symbol, "SMC_LIQUIDITY_REVERSAL", req.performance_timeframe,
        req.performance_session, req.performance_market_regime,
    )
    if edge_gate.get("action") == "BLOCKED":
        return {
            "status": "BLOCKED", "reason": "ADAPTIVE_EDGE_WEAK",
            "adaptive_edge_gate": edge_gate, "paper_only": True, "execution": False,
        }
    specs = await get_paper_instrument_specs(req.symbol)
    if specs.get("status") != "VALID":
        return {
            "status": "BLOCKED",
            "reason": "SERVER_INSTRUMENT_SPECS_NOT_VALID",
            "specs": specs,
            "paper_only": True,
            "execution": False,
        }

    account = await get_paper_account()
    capital = Decimal(str(account["current_capital"]))
    if req.entry is None or req.stop_loss is None or req.take_profit is None:
        return {
            "status": "BLOCKED",
            "reason": "TRADE_PLAN_INCOMPLETE",
            "paper_only": True,
            "execution": False,
        }
    sizing = calculate_verified_crypto_size(
        capital, req.risk_percent, req.entry, req.stop_loss, specs
    )
    if sizing["status"] != "VALID":
        return {
            "status": "BLOCKED",
            "reason": sizing["reason"],
            "sizing": sizing,
            "paper_only": True,
            "execution": False,
        }

    open_positions = await get_open_paper_risk_snapshot()
    portfolio_guard = evaluate_paper_portfolio_risk_guard(
        open_positions,
        req.symbol,
        Decimal(str(sizing["risk_money"])),
        capital,
    )
    if portfolio_guard["status"] != "VALID":
        return {
            "status": "BLOCKED",
            "reason": portfolio_guard["reason"],
            "portfolio_guard": portfolio_guard,
            "paper_only": True,
            "execution": False,
        }

    canonical = req.symbol.upper().replace("/", "-")
    decision = str(signal["decision"])
    position_id = build_auto_paper_position_id(
        canonical, decision, req.source_timestamp
    )
    position = PaperPositionCreate(
        position_id=position_id,
        symbol=canonical,
        side=decision,
        entry=req.entry,
        stop_loss=req.stop_loss,
        take_profit=req.take_profit,
        size=Decimal(str(sizing["size"])),
        size_unit="BASE_UNITS",
        risk_money=Decimal(str(sizing["risk_money"])),
        risk_percent=Decimal(str(sizing["risk_percent"])),
        capital_before=capital,
        source="server_signal+coinbase_public_product",
        source_timestamp=req.source_timestamp,
        opened_at=utcnow(),
        performance_strategy_id="SMC_LIQUIDITY_REVERSAL",
        performance_strategy_version="1.0",
        performance_timeframe=req.performance_timeframe,
        performance_session=req.performance_session,
        performance_market_regime=req.performance_market_regime,
        performance_setup_context=req.performance_setup_context,
    )
    created = await create_paper_position(position)
    await persist_adaptive_edge_trade_link(position_id, edge_gate)
    return {
        "status": "OPENED",
        "position": created,
        "specs_source": specs["source"],
        "sizing": {key: str(value) for key, value in sizing.items()},
        "paper_only": True,
        "execution": False,
    }


@api_router.post("/paper/auto-entry/verified", status_code=201)
async def verified_auto_paper_entry(
    req: VerifiedAutoPaperEntryRequest,
) -> Dict[str, object]:
    # Serializes portfolio check + create within one application process.
    # Existing verified path remains delegated below: evaluate_server_signal(req),
    # get_paper_instrument_specs(req.symbol), get_paper_account(),
    # create_paper_position(position), "paper_only": True, "execution": False.
    # Database uniqueness remains the final duplicate-position guard.
    async with auto_paper_portfolio_lock:
        return await _verified_auto_paper_entry_unlocked(req)


SERVER_SETUP_GRANULARITY = "5m"
SERVER_SETUP_CANDLE_LIMIT = 120
SERVER_HTF_GRANULARITY = "1h"
SERVER_HTF_CANDLE_LIMIT = 120
SERVER_SWING_STRENGTH = 2
SERVER_DISPLACEMENT_LOOKBACK = 20
SERVER_DISPLACEMENT_BODY_MULTIPLIER = 1.5
SERVER_DISPLACEMENT_MIN_BODY_RANGE_RATIO = 0.70
SERVER_DISPLACEMENT_CLOSE_EXTREME_FRACTION = 0.20
SERVER_ORDER_BLOCK_LOOKBACK = 5
SERVER_FVG_MIN_GAP = 0.0


def closed_valid_candles(candles: List[Candle], now: datetime) -> List[Candle]:
    """Return structurally valid closed candles for historical analysis.

    ``STALE`` is a freshness state, not malformed OHLC data. Older candles in a
    live Coinbase window are therefore usable for SMC history. Endpoint-level
    latest-candle freshness is checked separately before analysis.
    """
    bucket_seconds = GRANULARITIES[SERVER_SETUP_GRANULARITY][1]
    result: List[Candle] = []
    usable_statuses = {DataQualityStatus.VALID, DataQualityStatus.STALE}
    for candle in candles:
        if (
            candle.start is None
            or candle.status not in usable_statuses
            or candle.open is None
            or candle.high is None
            or candle.low is None
            or candle.close is None
        ):
            continue
        if candle.start + timedelta(seconds=bucket_seconds) > now:
            continue
        result.append(candle)
    return sorted(result, key=lambda item: item.start or datetime.min.replace(tzinfo=timezone.utc))


SERVER_REGIME_LOOKBACK = 20
SERVER_REGIME_ATR_WINDOW = 10
SERVER_REGIME_TREND_EFFICIENCY_MIN = 0.55
SERVER_REGIME_RANGE_EFFICIENCY_MAX = 0.35
SERVER_REGIME_EXPANSION_RATIO = 1.25
SERVER_REGIME_COMPRESSION_RATIO = 0.80


def classify_server_market_regime(candles: List[Candle], now: datetime) -> Dict[str, object]:
    """Classify trend/range and volatility from closed VALID candles only."""
    closed = closed_valid_candles(candles, now)
    required = max(SERVER_REGIME_LOOKBACK + 1, SERVER_REGIME_ATR_WINDOW * 2 + 1)
    if len(closed) < required:
        return {"status": "WAIT", "reason": "INSUFFICIENT_CLOSED_CANDLES"}
    sample = closed[-required:]
    closes = [float(item.close) for item in sample if item.close is not None]
    if len(closes) != required or any(value <= 0 for value in closes):
        return {"status": "WAIT", "reason": "INVALID_CLOSE_SERIES"}

    trend_closes = closes[-(SERVER_REGIME_LOOKBACK + 1):]
    net_change = trend_closes[-1] - trend_closes[0]
    path = sum(
        abs(trend_closes[index] - trend_closes[index - 1])
        for index in range(1, len(trend_closes))
    )
    efficiency = abs(net_change) / path if path > 0 else 0.0
    if efficiency >= SERVER_REGIME_TREND_EFFICIENCY_MIN and net_change != 0:
        structure = "TREND"
        direction = "BULLISH" if net_change > 0 else "BEARISH"
    elif efficiency <= SERVER_REGIME_RANGE_EFFICIENCY_MAX:
        structure = "RANGE"
        direction = None
    else:
        structure = "TRANSITION"
        direction = "BULLISH" if net_change > 0 else "BEARISH" if net_change < 0 else None

    true_ranges: List[float] = []
    for index in range(1, len(sample)):
        current = sample[index]
        previous_close = sample[index - 1].close
        if current.high is None or current.low is None or previous_close is None:
            return {"status": "WAIT", "reason": "INVALID_RANGE_SERIES"}
        true_ranges.append(
            max(
                current.high - current.low,
                abs(current.high - previous_close),
                abs(current.low - previous_close),
            )
        )
    prior = true_ranges[-(SERVER_REGIME_ATR_WINDOW * 2):-SERVER_REGIME_ATR_WINDOW]
    recent = true_ranges[-SERVER_REGIME_ATR_WINDOW:]
    prior_atr = sum(prior) / len(prior)
    recent_atr = sum(recent) / len(recent)
    volatility_ratio = recent_atr / prior_atr if prior_atr > 0 else None
    if volatility_ratio is None:
        volatility = "UNKNOWN"
    elif volatility_ratio >= SERVER_REGIME_EXPANSION_RATIO:
        volatility = "EXPANSION"
    elif volatility_ratio <= SERVER_REGIME_COMPRESSION_RATIO:
        volatility = "COMPRESSION"
    else:
        volatility = "NORMAL"

    latest = sample[-1]
    return {
        "status": "READY",
        "regime": structure,
        "direction": direction,
        "volatility": volatility,
        "efficiency_ratio": round(efficiency, 6),
        "volatility_ratio": round(volatility_ratio, 6) if volatility_ratio is not None else None,
        "closed_candles": len(closed),
        "latest_closed_timestamp": latest.start.isoformat() if latest.start else None,
        "marker": "SERVER_MARKET_REGIME_V1",
        "signal": False,
        "auto_queue": False,
    }


def confirmed_swing_indexes(
    candles: List[Candle], strength: int = SERVER_SWING_STRENGTH
) -> Tuple[List[int], List[int]]:
    highs: List[int] = []
    lows: List[int] = []
    if strength < 1:
        return highs, lows
    for index in range(strength, len(candles) - strength):
        high = candles[index].high
        low = candles[index].low
        if high is None or low is None:
            continue
        neighbors = range(index - strength, index + strength + 1)
        is_swing_high = True
        is_swing_low = True
        for offset in neighbors:
            if offset == index:
                continue
            neighbor_high = candles[offset].high
            neighbor_low = candles[offset].low
            if neighbor_high is None or high <= neighbor_high:
                is_swing_high = False
            if neighbor_low is None or low >= neighbor_low:
                is_swing_low = False
        if is_swing_high:
            highs.append(index)
        if is_swing_low:
            lows.append(index)
    return highs, lows


def latest_confirmed_break(
    candles: List[Candle],
    swing_highs: List[int],
    swing_lows: List[int],
    strength: int = SERVER_SWING_STRENGTH,
) -> Optional[Dict[str, object]]:
    """Return the latest close-confirmed break with no swing look-ahead.

    A swing at index ``i`` with strength ``n`` only becomes knowable once candle
    ``i + n`` is closed. A structural break is therefore eligible strictly after
    that confirmation candle, starting at ``i + n + 1``.
    """
    if not candles or strength < 1:
        return None
    events: List[Dict[str, object]] = []
    for swing_index in swing_highs:
        if swing_index < 0 or swing_index >= len(candles):
            continue
        level = candles[swing_index].high
        if level is None:
            continue
        first_eligible_break = swing_index + strength + 1
        for index in range(first_eligible_break, len(candles)):
            close = candles[index].close
            if close is not None and close > level:
                events.append(
                    {
                        "direction": "BULLISH",
                        "swing_index": swing_index,
                        "confirmation_index": swing_index + strength,
                        "break_index": index,
                        "level": level,
                        "close": close,
                    }
                )
                break
    for swing_index in swing_lows:
        if swing_index < 0 or swing_index >= len(candles):
            continue
        level = candles[swing_index].low
        if level is None:
            continue
        first_eligible_break = swing_index + strength + 1
        for index in range(first_eligible_break, len(candles)):
            close = candles[index].close
            if close is not None and close < level:
                events.append(
                    {
                        "direction": "BEARISH",
                        "swing_index": swing_index,
                        "confirmation_index": swing_index + strength,
                        "break_index": index,
                        "level": level,
                        "close": close,
                    }
                )
                break
    if not events:
        return None

    def break_index_value(event: Dict[str, object]) -> int:
        value = event.get("break_index")
        return value if isinstance(value, int) else -1

    return max(events, key=break_index_value)


def latest_confirmed_liquidity_sweep(
    candles: List[Candle],
    swing_highs: List[int],
    swing_lows: List[int],
    strength: int = SERVER_SWING_STRENGTH,
) -> Optional[Dict[str, object]]:
    """Return the latest strict liquidity sweep without look-ahead.

    BSL: after a Swing High is confirmed, a later closed candle must trade
    strictly above the level and close strictly back below it.
    SSL: after a Swing Low is confirmed, a later closed candle must trade
    strictly below the level and close strictly back above it.
    The confirmation candle itself is never eligible.
    """
    if not candles or strength < 1:
        return None
    events: List[Dict[str, object]] = []

    for swing_index in swing_highs:
        if swing_index < 0 or swing_index >= len(candles):
            continue
        level = candles[swing_index].high
        if level is None:
            continue
        first_eligible = swing_index + strength + 1
        for index in range(first_eligible, len(candles)):
            high = candles[index].high
            close = candles[index].close
            if high is not None and close is not None and high > level and close < level:
                events.append({
                    "type": "BSL_SWEEP",
                    "direction": "BEARISH",
                    "swing_index": swing_index,
                    "confirmation_index": swing_index + strength,
                    "sweep_index": index,
                    "level": level,
                    "extreme": high,
                    "close": close,
                })
                break

    for swing_index in swing_lows:
        if swing_index < 0 or swing_index >= len(candles):
            continue
        level = candles[swing_index].low
        if level is None:
            continue
        first_eligible = swing_index + strength + 1
        for index in range(first_eligible, len(candles)):
            low = candles[index].low
            close = candles[index].close
            if low is not None and close is not None and low < level and close > level:
                events.append({
                    "type": "SSL_SWEEP",
                    "direction": "BULLISH",
                    "swing_index": swing_index,
                    "confirmation_index": swing_index + strength,
                    "sweep_index": index,
                    "level": level,
                    "extreme": low,
                    "close": close,
                })
                break

    if not events:
        return None

    def sweep_index_value(event: Dict[str, object]) -> int:
        value = event.get("sweep_index")
        return value if isinstance(value, int) else -1

    return max(events, key=sweep_index_value)



def latest_confirmed_displacement(
    candles: List[Candle],
    lookback: int = SERVER_DISPLACEMENT_LOOKBACK,
) -> Optional[Dict[str, object]]:
    """Return the latest objective displacement candle from closed VALID input.

    The candidate body must be >= 1.5x the mean body of the previous 20
    candles, body/range >= 70%, and its close must finish inside the final
    20% of the candle range in the displacement direction. No future candle
    is consulted, so the detector is non-repainting once the candle is closed.
    """
    if lookback < 1 or len(candles) <= lookback:
        return None
    events: List[Dict[str, object]] = []
    for index in range(lookback, len(candles)):
        candidate = candles[index]
        if (candidate.open is None or candidate.high is None
                or candidate.low is None or candidate.close is None):
            continue
        prior = candles[index - lookback:index]
        prior_bodies: List[float] = []
        for candle in prior:
            if candle.open is None or candle.close is None:
                prior_bodies = []
                break
            prior_bodies.append(abs(candle.close - candle.open))
        if len(prior_bodies) != lookback:
            continue
        average_body = sum(prior_bodies) / lookback
        if average_body <= 0:
            continue
        body = abs(candidate.close - candidate.open)
        candle_range = candidate.high - candidate.low
        if candle_range <= 0:
            continue
        if body < average_body * SERVER_DISPLACEMENT_BODY_MULTIPLIER:
            continue
        if body / candle_range < SERVER_DISPLACEMENT_MIN_BODY_RANGE_RATIO:
            continue
        if candidate.close > candidate.open:
            extreme_threshold = candidate.high - (
                candle_range * SERVER_DISPLACEMENT_CLOSE_EXTREME_FRACTION
            )
            if candidate.close < extreme_threshold:
                continue
            direction = "BULLISH"
        elif candidate.close < candidate.open:
            extreme_threshold = candidate.low + (
                candle_range * SERVER_DISPLACEMENT_CLOSE_EXTREME_FRACTION
            )
            if candidate.close > extreme_threshold:
                continue
            direction = "BEARISH"
        else:
            continue
        events.append({
            "event": "DISPLACEMENT",
            "direction": direction,
            "candle_index": index,
            "body": body,
            "range": candle_range,
            "average_reference_body": average_body,
            "body_multiple": body / average_body,
            "body_range_ratio": body / candle_range,
        })
    return events[-1] if events else None


def latest_confirmed_fvg(
    candles: List[Candle],
) -> Optional[Dict[str, object]]:
    """Return the latest strict 3-candle Fair Value Gap from closed VALID input.

    Bullish FVG: candle[i-2].high < candle[i].low.
    Bearish FVG: candle[i-2].low > candle[i].high.
    The gap is confirmed only when candle i is closed. Later closed candles can
    move the zone state OPEN -> PARTIALLY_MITIGATED -> MITIGATED. A FVG alone
    never authorizes an entry.
    """
    if len(candles) < 3:
        return None
    events: List[Dict[str, object]] = []
    for index in range(2, len(candles)):
        first = candles[index - 2]
        third = candles[index]
        if (first.high is None or first.low is None
                or third.high is None or third.low is None):
            continue

        direction: Optional[str] = None
        zone_low: Optional[float] = None
        zone_high: Optional[float] = None
        if third.low - first.high > SERVER_FVG_MIN_GAP:
            direction = "BULLISH"
            zone_low = first.high
            zone_high = third.low
        elif first.low - third.high > SERVER_FVG_MIN_GAP:
            direction = "BEARISH"
            zone_low = third.high
            zone_high = first.low
        if direction is None or zone_low is None or zone_high is None:
            continue

        state = "OPEN"
        mitigation_index: Optional[int] = None
        for later_index in range(index + 1, len(candles)):
            later = candles[later_index]
            if direction == "BULLISH":
                if later.low is None:
                    continue
                if later.low <= zone_low:
                    state = "MITIGATED"
                    mitigation_index = later_index
                    break
                if later.low < zone_high:
                    state = "PARTIALLY_MITIGATED"
                    mitigation_index = later_index
            else:
                if later.high is None:
                    continue
                if later.high >= zone_high:
                    state = "MITIGATED"
                    mitigation_index = later_index
                    break
                if later.high > zone_low:
                    state = "PARTIALLY_MITIGATED"
                    mitigation_index = later_index

        events.append({
            "event": "FVG",
            "direction": direction,
            "formation_index": index,
            "first_index": index - 2,
            "middle_index": index - 1,
            "zone_low": zone_low,
            "zone_high": zone_high,
            "gap_size": zone_high - zone_low,
            "state": state,
            "mitigation_index": mitigation_index,
        })
    return events[-1] if events else None


def latest_confirmed_order_block(
    candles: List[Candle],
    displacement: Optional[Dict[str, object]] = None,
    lookback: int = SERVER_ORDER_BLOCK_LOOKBACK,
) -> Optional[Dict[str, object]]:
    """Return the last opposite candle before a confirmed displacement.

    A bullish OB is the latest bearish candle before bullish displacement; a
    bearish OB is the latest bullish candle before bearish displacement. The
    search is bounded to the preceding five closed VALID candles. The full
    candle low/high defines the zone. Later closed candles move FRESH to
    RETESTED on overlap, or INVALIDATED when close crosses beyond the far edge.
    An Order Block alone never authorizes an entry.
    """
    if lookback < 1 or not candles:
        return None
    event = displacement or latest_confirmed_displacement(candles)
    if event is None:
        return None
    raw_index = event.get("candle_index")
    direction = event.get("direction")
    if not isinstance(raw_index, int) or raw_index <= 0 or raw_index >= len(candles):
        return None
    if direction not in {"BULLISH", "BEARISH"}:
        return None

    start = max(0, raw_index - lookback)
    ob_index: Optional[int] = None
    for index in range(raw_index - 1, start - 1, -1):
        candle = candles[index]
        if candle.open is None or candle.close is None:
            continue
        is_opposite = (
            direction == "BULLISH" and candle.close < candle.open
        ) or (direction == "BEARISH" and candle.close > candle.open)
        if is_opposite:
            ob_index = index
            break
    if ob_index is None:
        return None

    ob = candles[ob_index]
    if ob.low is None or ob.high is None or ob.low >= ob.high:
        return None
    zone_low = ob.low
    zone_high = ob.high
    state = "FRESH"
    retest_index: Optional[int] = None
    invalidation_index: Optional[int] = None
    for index in range(raw_index + 1, len(candles)):
        candle = candles[index]
        if candle.low is None or candle.high is None or candle.close is None:
            continue
        if direction == "BULLISH" and candle.close < zone_low:
            state = "INVALIDATED"
            invalidation_index = index
            break
        if direction == "BEARISH" and candle.close > zone_high:
            state = "INVALIDATED"
            invalidation_index = index
            break
        overlaps = candle.low <= zone_high and candle.high >= zone_low
        if overlaps and state == "FRESH":
            state = "RETESTED"
            retest_index = index

    return {
        "event": "ORDER_BLOCK",
        "direction": direction,
        "order_block_index": ob_index,
        "displacement_index": raw_index,
        "zone_low": zone_low,
        "zone_high": zone_high,
        "state": state,
        "retest_index": retest_index,
        "invalidation_index": invalidation_index,
    }


def latest_confirmed_revalidation(
    candles: List[Candle],
    order_block: Optional[Dict[str, object]],
    fvg: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Revalidate a confirmed Order Block after its first closed-candle retest.

    Revalidation is deliberately conservative: the OB must already be RETESTED,
    the retest candle must still be inside the available closed VALID sequence,
    and its close must reject through the OB midpoint in the expected direction.
    A same-direction, non-mitigated FVG is exposed as confluence but is not
    mandatory. This function never authorizes or queues an entry.
    """
    base: Dict[str, object] = {
        "event": "REVALIDATION",
        "state": "WAITING_RETEST",
        "direction": None,
        "retest_index": None,
        "revalidation_index": None,
        "fvg_confluence": False,
    }
    if not order_block or order_block.get("event") != "ORDER_BLOCK":
        return base

    direction = order_block.get("direction")
    base["direction"] = direction
    if direction not in {"BULLISH", "BEARISH"}:
        base["state"] = "INVALIDATED"
        return base
    if order_block.get("state") == "INVALIDATED":
        base["state"] = "INVALIDATED"
        return base
    if order_block.get("state") != "RETESTED":
        return base

    raw_retest = order_block.get("retest_index")
    zone_low = order_block.get("zone_low")
    zone_high = order_block.get("zone_high")
    if (
        not isinstance(raw_retest, int)
        or raw_retest < 0
        or raw_retest >= len(candles)
        or not isinstance(zone_low, (int, float))
        or not isinstance(zone_high, (int, float))
        or zone_low >= zone_high
    ):
        base["state"] = "INVALIDATED"
        return base

    base["retest_index"] = raw_retest
    retest = candles[raw_retest]
    if retest.close is None or retest.low is None or retest.high is None:
        return base
    midpoint = (float(zone_low) + float(zone_high)) / 2.0
    overlaps = retest.low <= float(zone_high) and retest.high >= float(zone_low)
    rejected = (
        direction == "BULLISH" and retest.close > midpoint
    ) or (direction == "BEARISH" and retest.close < midpoint)
    if not overlaps or not rejected:
        base["state"] = "RETESTED_UNCONFIRMED"
        return base

    if fvg and fvg.get("event") == "FVG":
        same_direction = fvg.get("direction") == direction
        active_fvg = fvg.get("state") in {"OPEN", "PARTIALLY_MITIGATED"}
        base["fvg_confluence"] = same_direction and active_fvg
    base["state"] = "REVALIDATED"
    base["revalidation_index"] = raw_retest
    return base



def build_server_trade_plan(
    candles: List[Candle],
    swing_highs: List[int],
    swing_lows: List[int],
    structure_event: Dict[str, object],
    liquidity_sweep: Optional[Dict[str, object]],
    displacement: Optional[Dict[str, object]],
    fvg: Optional[Dict[str, object]],
    order_block: Optional[Dict[str, object]],
    revalidation: Dict[str, object],
) -> Dict[str, object]:
    """Build a paper-only trade-plan candidate from the complete SMC chain.

    Every confirmation must agree on direction. Entry is the confirmed Order
    Block zone, the reference entry is its midpoint, the structural stop is the
    far OB boundary, and the target is the nearest confirmed opposite-liquidity
    swing beyond entry. The result never queues or executes an order.
    """
    waiting: Dict[str, object] = {
        "event": "TRADE_PLAN",
        "state": "WAIT",
        "direction": None,
        "entry_zone_low": None,
        "entry_zone_high": None,
        "entry_reference": None,
        "stop_loss": None,
        "take_profit": None,
        "risk_reward": None,
        "auto_queue": False,
    }
    if revalidation.get("state") != "REVALIDATED":
        return waiting
    direction = revalidation.get("direction")
    if direction not in {"BULLISH", "BEARISH"}:
        return waiting

    confirmations = (
        structure_event,
        liquidity_sweep,
        displacement,
        fvg,
        order_block,
    )
    if any(not item for item in confirmations):
        return waiting
    if any(item.get("direction") != direction for item in confirmations if item):
        return waiting
    if order_block is None or order_block.get("state") == "INVALIDATED":
        return waiting

    zone_low = order_block.get("zone_low")
    zone_high = order_block.get("zone_high")
    if (
        not isinstance(zone_low, (int, float))
        or not isinstance(zone_high, (int, float))
        or zone_low >= zone_high
    ):
        return waiting
    entry = (float(zone_low) + float(zone_high)) / 2.0

    target_candidates: List[float] = []
    indexes = swing_highs if direction == "BULLISH" else swing_lows
    for index in indexes:
        if index < 0 or index >= len(candles):
            continue
        value = candles[index].high if direction == "BULLISH" else candles[index].low
        if value is None:
            continue
        numeric = float(value)
        if direction == "BULLISH" and numeric > entry:
            target_candidates.append(numeric)
        elif direction == "BEARISH" and numeric < entry:
            target_candidates.append(numeric)
    if not target_candidates:
        return waiting

    stop = float(zone_low) if direction == "BULLISH" else float(zone_high)
    target = min(target_candidates) if direction == "BULLISH" else max(target_candidates)
    risk = abs(entry - stop)
    reward = abs(target - entry)
    if risk <= 0.0 or reward <= 0.0:
        return waiting

    return {
        "event": "TRADE_PLAN",
        "state": "CANDIDATE_READY",
        "direction": direction,
        "entry_zone_low": float(zone_low),
        "entry_zone_high": float(zone_high),
        "entry_reference": entry,
        "stop_loss": stop,
        "take_profit": target,
        "risk_reward": reward / risk,
        "auto_queue": False,
    }

def evaluate_server_entry_now_gate(
    candles: List[Candle],
    trade_plan: Dict[str, object],
    revalidation: Dict[str, object],
    order_block: Optional[Dict[str, object]],
) -> Dict[str, object]:
    """Return the paper-only lifecycle state for the current closed candle.

    ENTRY_NOW is deliberately ephemeral: it is emitted only while the latest
    closed VALID candle is the same candle that produced REVALIDATED. A setup
    becomes EXPIRED on the next closed candle if it was not consumed. An
    invalidated Order Block always wins. This gate never queues or executes.
    """
    base: Dict[str, object] = {
        "event": "ENTRY_GATE",
        "state": "WAIT",
        "direction": None,
        "revalidation_index": None,
        "latest_closed_index": len(candles) - 1 if candles else None,
        "reason": "SETUP_INCOMPLETE",
        "auto_queue": False,
    }
    if order_block and order_block.get("state") == "INVALIDATED":
        base["state"] = "INVALIDATED"
        base["reason"] = "ORDER_BLOCK_INVALIDATED"
        return base
    if revalidation.get("state") == "INVALIDATED":
        base["state"] = "INVALIDATED"
        base["reason"] = "REVALIDATION_INVALIDATED"
        return base
    if trade_plan.get("state") != "CANDIDATE_READY":
        return base
    if revalidation.get("state") != "REVALIDATED":
        return base

    direction = trade_plan.get("direction")
    if direction not in {"BULLISH", "BEARISH"}:
        return base
    if revalidation.get("direction") != direction:
        base["reason"] = "DIRECTION_MISMATCH"
        return base
    base["direction"] = direction

    entry = trade_plan.get("entry_reference")
    stop = trade_plan.get("stop_loss")
    target = trade_plan.get("take_profit")
    zone_low = trade_plan.get("entry_zone_low")
    zone_high = trade_plan.get("entry_zone_high")
    if not (
        isinstance(entry, (int, float))
        and isinstance(stop, (int, float))
        and isinstance(target, (int, float))
        and isinstance(zone_low, (int, float))
        and isinstance(zone_high, (int, float))
    ):
        base["reason"] = "TRADE_LEVELS_INVALID"
        return base
    entry_f = float(entry)
    stop_f = float(stop)
    target_f = float(target)
    low_f = float(zone_low)
    high_f = float(zone_high)
    if low_f >= high_f or not low_f <= entry_f <= high_f:
        base["reason"] = "ENTRY_ZONE_INVALID"
        return base
    levels_valid = (
        direction == "BULLISH" and stop_f < entry_f < target_f
    ) or (direction == "BEARISH" and target_f < entry_f < stop_f)
    if not levels_valid:
        base["reason"] = "TRADE_LEVELS_INVALID"
        return base

    raw_index = revalidation.get("revalidation_index")
    if not isinstance(raw_index, int) or raw_index < 0:
        base["reason"] = "REVALIDATION_INDEX_INVALID"
        return base
    base["revalidation_index"] = raw_index
    if not candles or raw_index >= len(candles):
        base["reason"] = "REVALIDATION_INDEX_INVALID"
        return base

    latest_index = len(candles) - 1
    if raw_index < latest_index:
        base["state"] = "EXPIRED"
        base["reason"] = "ENTRY_WINDOW_CLOSED"
        return base
    if raw_index > latest_index:
        base["reason"] = "REVALIDATION_INDEX_INVALID"
        return base

    base["state"] = "ENTRY_NOW"
    base["reason"] = "CURRENT_CLOSED_CANDLE_REVALIDATED"
    return base


def structure_from_confirmed_swings(
    candles: List[Candle],
    swing_highs: List[int],
    swing_lows: List[int],
) -> str:
    """Classify structure from already-confirmed swing indexes only."""
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return "RANGE"
    high_a = candles[swing_highs[-2]].high
    high_b = candles[swing_highs[-1]].high
    low_a = candles[swing_lows[-2]].low
    low_b = candles[swing_lows[-1]].low
    if high_a is None or high_b is None or low_a is None or low_b is None:
        return "RANGE"
    if high_b > high_a and low_b > low_a:
        return "BULLISH"
    if high_b < high_a and low_b < low_a:
        return "BEARISH"
    return "RANGE"


def structure_before_break(
    candles: List[Candle],
    swing_highs: List[int],
    swing_lows: List[int],
    break_index: int,
    strength: int = SERVER_SWING_STRENGTH,
) -> str:
    """Return structure using only swings confirmed before the break candle.

    A swing can contribute only when ``swing_index + strength < break_index``.
    This prevents a later-confirmed swing from reclassifying an older event.
    """
    if break_index <= 0 or strength < 1:
        return "RANGE"
    eligible_highs = [
        index for index in swing_highs if index + strength < break_index
    ]
    eligible_lows = [
        index for index in swing_lows if index + strength < break_index
    ]
    return structure_from_confirmed_swings(candles, eligible_highs, eligible_lows)


def classify_bos_choch(
    prior_structure: str,
    break_event: Optional[Dict[str, object]],
) -> Dict[str, object]:
    if break_event is None:
        return {"event": "NONE", "direction": None}
    direction = str(break_event["direction"])
    if prior_structure == "BULLISH":
        event = "BOS" if direction == "BULLISH" else "CHOCH_MSS"
    elif prior_structure == "BEARISH":
        event = "BOS" if direction == "BEARISH" else "CHOCH_MSS"
    else:
        event = "BOS"
    return {
        "event": event,
        "direction": direction,
        "level": break_event["level"],
        "break_index": break_event["break_index"],
    }


def detect_server_market_structure(candles: List[Candle], now: datetime) -> Dict[str, object]:
    closed = closed_valid_candles(candles, now)
    if len(closed) < SERVER_SWING_STRENGTH * 2 + 3:
        return {"status": "WAIT", "reason": "INSUFFICIENT_CLOSED_CANDLES"}
    highs, lows = confirmed_swing_indexes(closed)
    if len(highs) < 2 or len(lows) < 2:
        return {"status": "WAIT", "reason": "INSUFFICIENT_CONFIRMED_SWINGS"}

    high_a = closed[highs[-2]].high
    high_b = closed[highs[-1]].high
    low_a = closed[lows[-2]].low
    low_b = closed[lows[-1]].low
    if high_a is None or high_b is None or low_a is None or low_b is None:
        return {"status": "WAIT", "reason": "SWING_VALUE_MISSING"}

    if high_b > high_a and low_b > low_a:
        structure = "BULLISH"
    elif high_b < high_a and low_b < low_a:
        structure = "BEARISH"
    else:
        structure = "RANGE"

    break_event = latest_confirmed_break(closed, highs, lows)
    if break_event is None:
        prior_structure = "RANGE"
    else:
        raw_break_index = break_event.get("break_index")
        break_index = raw_break_index if isinstance(raw_break_index, int) else -1
        prior_structure = structure_before_break(
            closed, highs, lows, break_index
        )
    structure_event = classify_bos_choch(prior_structure, break_event)
    structure_event["prior_structure"] = prior_structure
    liquidity_sweep = latest_confirmed_liquidity_sweep(closed, highs, lows)
    displacement = latest_confirmed_displacement(closed)
    fvg = latest_confirmed_fvg(closed)
    order_block = latest_confirmed_order_block(closed, displacement)
    revalidation = latest_confirmed_revalidation(closed, order_block, fvg)
    trade_plan = build_server_trade_plan(
        closed,
        highs,
        lows,
        structure_event,
        liquidity_sweep,
        displacement,
        fvg,
        order_block,
        revalidation,
    )
    entry_gate = evaluate_server_entry_now_gate(
        closed, trade_plan, revalidation, order_block
    )
    latest = closed[-1]
    return {
        "status": "READY",
        "structure": structure,
        "structure_event": structure_event,
        "closed_candles": len(closed),
        "confirmed_swing_highs": len(highs),
        "confirmed_swing_lows": len(lows),
        "latest_closed_timestamp": latest.start.isoformat() if latest.start else None,
        "setup_state": entry_gate["state"],
        "auto_queue": False,
        "smc_confirmation": (
            "STRUCTURE_EVENTS_LIQUIDITY_SWEEP_DISPLACEMENT_"
            "FVG_OB_RETEST_PLAN_GATE_V1"
        ),
        "liquidity_sweep": liquidity_sweep or {"event": "NONE", "direction": None},
        "displacement": displacement or {"event": "NONE", "direction": None},
        "fvg": fvg or {"event": "NONE", "direction": None},
        "order_block": order_block or {"event": "NONE", "direction": None},
        "revalidation": revalidation,
        "trade_plan": trade_plan,
        "entry_gate": entry_gate,
    }



def classify_server_htf_context(candles: List[Candle], now: datetime) -> Dict[str, object]:
    """Objective HTF structure context from confirmed closed candles only.

    This foundation is intentionally non-executing: it does not alter the 5m
    ENTRY_NOW gate yet. Confirmed swings inherit the existing no-look-ahead rule.
    """
    closed = closed_valid_candles(candles, now)
    if len(closed) < SERVER_SWING_STRENGTH * 2 + 3:
        return {"status": "WAIT", "reason": "INSUFFICIENT_HTF_CLOSED_CANDLES"}
    highs, lows = confirmed_swing_indexes(closed)
    if len(highs) < 2 or len(lows) < 2:
        return {"status": "WAIT", "reason": "INSUFFICIENT_HTF_CONFIRMED_SWINGS"}

    high_a = closed[highs[-2]].high
    high_b = closed[highs[-1]].high
    low_a = closed[lows[-2]].low
    low_b = closed[lows[-1]].low
    if high_a is None or high_b is None or low_a is None or low_b is None:
        return {"status": "WAIT", "reason": "HTF_SWING_VALUE_MISSING"}

    if high_b > high_a and low_b > low_a:
        structure = "BULLISH"
    elif high_b < high_a and low_b < low_a:
        structure = "BEARISH"
    else:
        structure = "RANGE"

    range_high = float(high_b)
    range_low = float(low_b)
    if range_high <= range_low:
        range_high = max(float(high_a), float(high_b))
        range_low = min(float(low_a), float(low_b))
    midpoint = (range_high + range_low) / 2.0
    latest_close = closed[-1].close
    if latest_close is None:
        location = "UNKNOWN"
    elif latest_close > midpoint:
        location = "PREMIUM"
    elif latest_close < midpoint:
        location = "DISCOUNT"
    else:
        location = "EQUILIBRIUM"

    return {
        "status": "READY",
        "structure": structure,
        "location": location,
        "range_high": range_high,
        "range_low": range_low,
        "equilibrium": midpoint,
        "closed_candles": len(closed),
        "confirmed_swing_highs": len(highs),
        "confirmed_swing_lows": len(lows),
        "latest_closed_timestamp": (
            closed[-1].start.isoformat() if closed[-1].start else None
        ),
        "validation": "SERVER_HTF_CONTEXT_V1",
        "no_lookahead": True,
        "execution": False,
    }


@api_router.get("/market/htf-context/{symbol}")
async def get_server_htf_context(symbol: str) -> Dict[str, object]:
    canonical = symbol.upper().replace("/", "-")
    instrument = instrument_registry.get(canonical)
    if instrument is None:
        return {"status": "UNAVAILABLE", "symbol": canonical, "reason": "INSTRUMENT_NOT_REGISTERED"}
    if instrument.asset_class != AssetClass.CRYPTO:
        return {
            "status": "NOT_SUPPORTED",
            "symbol": canonical,
            "reason": "HTF_CONTEXT_CRYPTO_ONLY_V1",
        }
    provider_symbol = provider_symbol_map.to_provider("coinbase", canonical)
    if provider_symbol is None:
        return {
            "status": "UNAVAILABLE",
            "symbol": canonical,
            "reason": "PROVIDER_SYMBOL_NOT_MAPPED",
        }
    try:
        candles, quality = await market_provider.get_candles(
            provider_symbol, SERVER_HTF_GRANULARITY, SERVER_HTF_CANDLE_LIMIT
        )
    except (httpx.HTTPError, ValueError):
        return {"status": "UNAVAILABLE", "symbol": canonical, "reason": "HTF_CANDLES_UNAVAILABLE"}
    if quality != DataQualityStatus.VALID:
        return {
            "status": "WAIT",
            "symbol": canonical,
            "reason": "HTF_CANDLES_NOT_VALID",
            "quality": quality.value,
        }
    latest_quality = _latest_quality(candles)
    if latest_quality != DataQualityStatus.VALID.value:
        return {
            "status": "WAIT",
            "symbol": canonical,
            "reason": "HTF_LATEST_CANDLE_NOT_FRESH",
            "quality": latest_quality,
        }
    result = classify_server_htf_context(candles, utcnow())
    return {
        **result,
        "symbol": canonical,
        "source": "coinbase",
        "granularity": SERVER_HTF_GRANULARITY,
        "quality": quality.value,
        "paper_only": True,
    }


@api_router.get("/market/regime/{symbol}")
async def get_server_market_regime(symbol: str) -> Dict[str, object]:
    canonical = symbol.upper().replace("/", "-")
    instrument = instrument_registry.get(canonical)
    if instrument is None:
        return {"status": "UNAVAILABLE", "symbol": canonical, "reason": "INSTRUMENT_NOT_REGISTERED"}
    if instrument.asset_class != AssetClass.CRYPTO:
        return {"status": "NOT_SUPPORTED", "symbol": canonical, "reason": "REGIME_CRYPTO_ONLY_V1"}
    provider_symbol = provider_symbol_map.to_provider("coinbase", canonical)
    if provider_symbol is None:
        return {
            "status": "UNAVAILABLE",
            "symbol": canonical,
            "reason": "PROVIDER_SYMBOL_NOT_MAPPED",
        }
    try:
        candles, quality = await market_provider.get_candles(
            provider_symbol, SERVER_SETUP_GRANULARITY, SERVER_SETUP_CANDLE_LIMIT
        )
    except (httpx.HTTPError, ValueError):
        return {"status": "UNAVAILABLE", "symbol": canonical, "reason": "CANDLES_UNAVAILABLE"}
    if quality != DataQualityStatus.VALID:
        return {
            "status": "WAIT",
            "symbol": canonical,
            "reason": "CANDLES_NOT_VALID",
            "quality": quality.value,
        }
    latest_quality = _latest_quality(candles)
    if latest_quality != DataQualityStatus.VALID.value:
        return {
            "status": "WAIT",
            "symbol": canonical,
            "reason": "LATEST_CANDLE_NOT_FRESH",
            "quality": latest_quality,
        }
    result = classify_server_market_regime(candles, utcnow())
    return {
        **result,
        "symbol": canonical,
        "source": "coinbase",
        "granularity": SERVER_SETUP_GRANULARITY,
        "quality": quality.value,
        "paper_only": True,
        "execution": False,
    }


@api_router.get("/paper/auto-entry/detector/{symbol}")
def apply_htf_context_to_ltf_setup(
    ltf: Dict[str, object], htf: Dict[str, object]
) -> Dict[str, object]:
    """Gate an LTF ENTRY_NOW setup with objective 1h structure/location."""
    result = dict(ltf)
    result["htf_context"] = htf
    result["htf_granularity"] = SERVER_HTF_GRANULARITY
    result["validation"] = "SERVER_HTF_LTF_INTEGRATION_V1"
    result["paper_only"] = True
    if result.get("setup_state") != "ENTRY_NOW":
        return result
    if htf.get("status") != "READY":
        result["setup_state"] = "WAIT"
        result["reason"] = str(htf.get("reason") or "HTF_CONTEXT_NOT_READY")
        return result
    gate = result.get("entry_gate")
    if not isinstance(gate, dict):
        result["setup_state"] = "WAIT"
        result["reason"] = "LTF_ENTRY_GATE_MISSING"
        return result
    direction = gate.get("direction")
    structure = htf.get("structure")
    location = htf.get("location")
    if direction == "BULLISH":
        if structure != "BULLISH":
            result["setup_state"] = "WAIT"
            result["reason"] = "HTF_STRUCTURE_NOT_BULLISH"
        elif location == "PREMIUM":
            result["setup_state"] = "WAIT"
            result["reason"] = "HTF_BULLISH_ENTRY_IN_PREMIUM"
    elif direction == "BEARISH":
        if structure != "BEARISH":
            result["setup_state"] = "WAIT"
            result["reason"] = "HTF_STRUCTURE_NOT_BEARISH"
        elif location == "DISCOUNT":
            result["setup_state"] = "WAIT"
            result["reason"] = "HTF_BEARISH_ENTRY_IN_DISCOUNT"
    else:
        result["setup_state"] = "WAIT"
        result["reason"] = "LTF_ENTRY_DIRECTION_INVALID"
    return result


async def get_server_market_setup_detector(symbol: str) -> Dict[str, object]:
    canonical = symbol.upper().replace("/", "-")
    instrument = instrument_registry.get(canonical)
    if instrument is None:
        return {
            "status": "UNAVAILABLE",
            "symbol": canonical,
            "reason": "INSTRUMENT_NOT_REGISTERED",
            "auto_queue": False,
        }
    if instrument.asset_class != AssetClass.CRYPTO:
        return {
            "status": "NOT_SUPPORTED",
            "symbol": canonical,
            "reason": "SERVER_CANDLE_DETECTOR_CRYPTO_ONLY",
            "auto_queue": False,
        }
    provider_symbol = provider_symbol_map.to_provider("coinbase", canonical)
    if provider_symbol is None:
        return {
            "status": "UNAVAILABLE",
            "symbol": canonical,
            "reason": "PROVIDER_SYMBOL_NOT_MAPPED",
            "auto_queue": False,
        }
    try:
        candles, quality = await market_provider.get_candles(
            provider_symbol,
            SERVER_SETUP_GRANULARITY,
            SERVER_SETUP_CANDLE_LIMIT,
        )
    except (httpx.HTTPError, ValueError):
        return {
            "status": "UNAVAILABLE",
            "symbol": canonical,
            "reason": "CANDLES_UNAVAILABLE",
            "auto_queue": False,
        }
    if quality != DataQualityStatus.VALID:
        return {
            "status": "WAIT",
            "symbol": canonical,
            "reason": "CANDLES_NOT_VALID",
            "quality": quality.value,
            "auto_queue": False,
        }
    latest_quality = _latest_quality(candles)
    if latest_quality != DataQualityStatus.VALID.value:
        return {
            "status": "WAIT",
            "symbol": canonical,
            "reason": "LATEST_CANDLE_NOT_FRESH",
            "quality": latest_quality,
            "auto_queue": False,
        }
    result = detect_server_market_structure(candles, utcnow())
    ltf_result: Dict[str, object] = {
        **result,
        "symbol": canonical,
        "source": "coinbase",
        "granularity": SERVER_SETUP_GRANULARITY,
        "quality": quality.value,
    }
    if ltf_result.get("setup_state") != "ENTRY_NOW":
        return ltf_result
    try:
        htf_candles, htf_quality = await market_provider.get_candles(
            provider_symbol, SERVER_HTF_GRANULARITY, SERVER_HTF_CANDLE_LIMIT
        )
    except (httpx.HTTPError, ValueError):
        return apply_htf_context_to_ltf_setup(
            ltf_result, {"status": "WAIT", "reason": "HTF_CANDLES_UNAVAILABLE"}
        )
    if htf_quality != DataQualityStatus.VALID:
        return apply_htf_context_to_ltf_setup(
            ltf_result,
            {
                "status": "WAIT",
                "reason": "HTF_CANDLES_NOT_VALID",
                "quality": htf_quality.value,
            },
        )
    htf_latest_quality = _latest_quality(htf_candles)
    if htf_latest_quality != DataQualityStatus.VALID.value:
        return apply_htf_context_to_ltf_setup(
            ltf_result,
            {
                "status": "WAIT",
                "reason": "HTF_LATEST_CANDLE_NOT_FRESH",
                "quality": htf_latest_quality,
            },
        )
    htf_context = classify_server_htf_context(htf_candles, utcnow())
    return apply_htf_context_to_ltf_setup(ltf_result, htf_context)


AUTO_ENTRY_ORCHESTRATOR_INTERVAL_SECONDS = 5.0
auto_entry_candidates: Dict[str, AutoEntryCandidateState] = {}
auto_entry_orchestrator_task: Optional[asyncio.Task] = None
AUTO_DECISION_TRACE_MAX = 200
auto_decision_trace: List[Dict[str, object]] = []

auto_scan_runtime: Dict[str, object] = {
    "iterations": 0,
    "last_started_at": None,
    "last_completed_at": None,
    "last_generation": None,
    "last_queue": None,
    "last_error": None,
}


def reset_auto_scan_runtime() -> None:
    """Reset observable runtime counters without changing trading state."""
    auto_scan_runtime.update(
        {
            "iterations": 0,
            "last_started_at": None,
            "last_completed_at": None,
            "last_generation": None,
            "last_queue": None,
            "last_error": None,
        }
    )
    auto_decision_trace.clear()


def _decision_reason_from_detector(detector: Dict[str, object]) -> str:
    """Return the first objective server blocker for the current SMC setup."""
    if detector.get("status") != "READY":
        raw_reason = detector.get("reason")
        return str(raw_reason) if raw_reason else "DETECTOR_NOT_READY"
    setup_state = str(detector.get("setup_state", "WAIT"))
    if setup_state == "ENTRY_NOW":
        return "ALL_CONFIRMATIONS_VALID"
    if setup_state == "INVALIDATED":
        return "ENTRY_GATE_INVALIDATED"
    if setup_state == "EXPIRED":
        return "ENTRY_WINDOW_EXPIRED"
    checks = (
        ("structure_event", "STRUCTURE_NOT_CONFIRMED"),
        ("liquidity_sweep", "LIQUIDITY_SWEEP_NOT_CONFIRMED"),
        ("displacement", "DISPLACEMENT_NOT_CONFIRMED"),
        ("fvg", "FVG_NOT_CONFIRMED"),
        ("order_block", "ORDER_BLOCK_NOT_CONFIRMED"),
    )
    for key, reason in checks:
        if not isinstance(detector.get(key), dict):
            return reason
    revalidation = detector.get("revalidation")
    if not isinstance(revalidation, dict):
        return "RETEST_REVALIDATION_NOT_CONFIRMED"
    if revalidation.get("state") != "REVALIDATED":
        return str(revalidation.get("state") or "RETEST_REVALIDATION_NOT_CONFIRMED")
    trade_plan = detector.get("trade_plan")
    if not isinstance(trade_plan, dict) or trade_plan.get("state") != "CANDIDATE_READY":
        return "TRADE_PLAN_NOT_READY"
    gate = detector.get("entry_gate")
    if isinstance(gate, dict) and gate.get("state"):
        return f'ENTRY_GATE_{gate["state"]}'
    return "WAITING_FOR_VALID_ENTRY"


def record_auto_decision_trace(
    symbol: str, state: str, reason: str, detector: Optional[Dict[str, object]] = None
) -> Dict[str, object]:
    """Store a bounded, paper-only runtime explanation for one scan decision."""
    entry: Dict[str, object] = {
        "timestamp": utcnow().isoformat(),
        "symbol": symbol.upper().replace("/", "-"),
        "state": state,
        "reason": reason,
        "paper_only": True,
        "execution": False,
    }
    if detector is not None:
        entry["setup_state"] = detector.get("setup_state", "WAIT")
        entry["latest_closed_timestamp"] = detector.get("latest_closed_timestamp")
    auto_decision_trace.append(entry)
    if len(auto_decision_trace) > AUTO_DECISION_TRACE_MAX:
        del auto_decision_trace[:-AUTO_DECISION_TRACE_MAX]
    return entry


async def persist_auto_decision_trace(
    entry: Dict[str, object], detector: Optional[Dict[str, object]] = None
) -> bool:
    """Persist one scanner decision without ever inventing market information."""
    if not persistence_state.ready:
        return False
    timestamp_raw = str(entry.get("timestamp", ""))
    try:
        timestamp = datetime.fromisoformat(timestamp_raw)
    except ValueError:
        return False
    if timestamp.tzinfo is None:
        return False
    symbol = str(entry.get("symbol", ""))
    state = str(entry.get("state", ""))
    reason = str(entry.get("reason", ""))
    identity = f"{timestamp_raw}|{symbol}|{state}|{reason}"
    decision_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    context = json.dumps(detector, sort_keys=True, default=str) if detector else None
    values = {
        "decision_id": decision_id,
        "timestamp": timestamp,
        "symbol": symbol,
        "state": state,
        "reason": reason,
        "setup_state": str(entry.get("setup_state", "")) or None,
        "latest_closed_timestamp": (
            str(entry.get("latest_closed_timestamp"))
            if entry.get("latest_closed_timestamp") is not None
            else None
        ),
        "detector_context": context,
        "paper_only": True,
        "execution": False,
        "created_at": utcnow(),
    }
    try:
        async with engine.begin() as conn:
            stmt = pg_insert(signal_decision_history_table).values(values)
            stmt = stmt.on_conflict_do_nothing(index_elements=["decision_id"])
            await conn.execute(stmt)
    except Exception as exc:  # noqa: BLE001
        log.error("Decision history persistence failed: %s", exc)
        return False
    return True


async def record_and_persist_auto_decision_trace(
    symbol: str,
    state: str,
    reason: str,
    detector: Optional[Dict[str, object]] = None,
) -> None:
    entry = record_auto_decision_trace(symbol, state, reason, detector)
    await persist_auto_decision_trace(entry, detector)


async def signal_decision_history(
    limit: int = 100, symbol: Optional[str] = None, state: Optional[str] = None
) -> Dict[str, object]:
    """Read durable scanner decisions newest-first from PostgreSQL."""
    if not persistence_state.ready:
        raise HTTPException(
            status_code=503,
            detail={"status": "UNAVAILABLE", "reason": "PERSISTENCE_NOT_READY"},
        )
    safe_limit = max(1, min(limit, 500))
    clauses: List[str] = []
    params: Dict[str, object] = {"limit": safe_limit}
    if symbol:
        clauses.append("symbol = :symbol")
        params["symbol"] = symbol.upper().replace("/", "-")
    if state:
        clauses.append("state = :state")
        params["state"] = state.upper()
    sql = "SELECT * FROM signal_decision_history"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY timestamp DESC, decision_id DESC LIMIT :limit"
    try:
        async with engine.connect() as conn:
            result = await conn.execute(text(sql), params)
            items = [dict(row._mapping) for row in result.fetchall()]
    except Exception as exc:
        persistence_state.mark_runtime_error(str(exc))
        raise HTTPException(
            status_code=503,
            detail={"status": "UNAVAILABLE", "reason": "decision history failed"},
        ) from exc
    for item in items:
        for key in ("timestamp", "created_at"):
            if isinstance(item.get(key), datetime):
                item[key] = item[key].isoformat()
        raw_context = item.get("detector_context")
        if isinstance(raw_context, str):
            try:
                item["detector_context"] = json.loads(raw_context)
            except json.JSONDecodeError:
                item["detector_context"] = None
    return {
        "status": "OK",
        "validation": "SERVER_SIGNAL_DECISION_HISTORY_V1",
        "count": len(items),
        "limit": safe_limit,
        "items": items,
        "paper_only": True,
        "broker_execution": False,
        "live_trading_enabled": False,
    }


def auto_decision_trace_status(limit: int = 50) -> Dict[str, object]:
    safe_limit = max(1, min(limit, AUTO_DECISION_TRACE_MAX))
    return {
        "validation": "SERVER_DECISION_TRACE_V1",
        "count": len(auto_decision_trace),
        "limit": safe_limit,
        "items": list(reversed(auto_decision_trace[-safe_limit:])),
        "paper_only": True,
        "broker_execution": False,
        "live_trading_enabled": False,
    }


AUTO_SCAN_WATCHDOG_STALE_AFTER_SECONDS = max(
    15.0, AUTO_ENTRY_ORCHESTRATOR_INTERVAL_SECONDS * 3
)


def evaluate_auto_scan_watchdog(
    runtime: Dict[str, object],
    task_running: bool,
    now: Optional[datetime] = None,
) -> Dict[str, object]:
    """Evaluate scanner liveness from its real heartbeat without fabricating activity."""
    current = now or utcnow()
    if current.tzinfo is None:
        raise ValueError("watchdog now must be timezone-aware")
    completed_raw = runtime.get("last_completed_at")
    last_error = runtime.get("last_error")
    age_seconds: Optional[float] = None
    heartbeat_valid = False
    if isinstance(completed_raw, str) and completed_raw:
        try:
            completed = datetime.fromisoformat(completed_raw)
        except ValueError:
            completed = None
        if completed is not None and completed.tzinfo is not None:
            age_seconds = max(0.0, (current - completed).total_seconds())
            heartbeat_valid = True
    if not task_running:
        state = "STOPPED"
        reason = "ORCHESTRATOR_TASK_NOT_RUNNING"
    elif last_error:
        state = "ERROR"
        reason = f"LAST_SCAN_ERROR:{last_error}"
    elif not heartbeat_valid:
        state = "STARTING"
        reason = "WAITING_FOR_FIRST_COMPLETED_SCAN"
    elif age_seconds is not None and age_seconds > AUTO_SCAN_WATCHDOG_STALE_AFTER_SECONDS:
        state = "STALE"
        reason = "SCAN_HEARTBEAT_STALE"
    else:
        state = "HEALTHY"
        reason = "SCAN_HEARTBEAT_FRESH"
    return {
        "status": state,
        "reason": reason,
        "validation": "SERVER_AUTO_SCAN_WATCHDOG_V1",
        "heartbeat_age_seconds": age_seconds,
        "stale_after_seconds": AUTO_SCAN_WATCHDOG_STALE_AFTER_SECONDS,
        "iterations": runtime.get("iterations", 0),
        "last_completed_at": completed_raw,
        "paper_only": True,
        "broker_execution": False,
        "live_trading_enabled": False,
    }


def auto_scan_watchdog_status() -> Dict[str, object]:
    task = auto_entry_orchestrator_task
    task_running = task is not None and not task.done()
    return evaluate_auto_scan_watchdog(auto_scan_runtime, task_running)


def auto_scan_runtime_status() -> Dict[str, object]:
    """Return a snapshot of the continuous paper-only scanner heartbeat."""
    task = auto_entry_orchestrator_task
    return {
        "status": "RUNNING" if task is not None and not task.done() else "STOPPED",
        "validation": "SERVER_CONTINUOUS_AUTO_SCAN_V1",
        "interval_seconds": AUTO_ENTRY_ORCHESTRATOR_INTERVAL_SECONDS,
        **auto_scan_runtime,
        "decision_trace_count": len(auto_decision_trace),
        "latest_decision": auto_decision_trace[-1] if auto_decision_trace else None,
        "paper_only": True,
        "broker_execution": False,
        "live_trading_enabled": False,
    }


def register_auto_entry_candidate(
    req: AutoEntryCandidateRequest,
) -> Dict[str, object]:
    signal = evaluate_server_signal(req)
    if signal["status"] != "READY":
        return {
            "status": "BLOCKED",
            "candidate_id": req.candidate_id,
            "reason": "SERVER_SIGNAL_NOT_READY",
            "paper_only": True,
            "execution": False,
        }
    state = AutoEntryCandidateState(
        request=VerifiedAutoPaperEntryRequest(**req.model_dump(exclude={"candidate_id"})),
        candidate_id=req.candidate_id,
        registered_at=utcnow(),
    )
    auto_entry_candidates[req.candidate_id] = state
    return {
        "status": "QUEUED",
        "candidate_id": req.candidate_id,
        "paper_only": True,
        "execution": False,
    }


def build_verified_auto_entry_request_from_detector(
    symbol: str, detector: Dict[str, object]
) -> Optional[VerifiedAutoPaperEntryRequest]:
    """Convert a complete server ENTRY_NOW setup into a paper-only request.

    The browser is not trusted. Every required SMC component must already be
    server-confirmed and directionally coherent. This function only builds the
    request; verified_auto_paper_entry remains authoritative for specs, sizing,
    capital, persistence, and idempotent position creation.
    """
    if detector.get("status") != "READY" or detector.get("setup_state") != "ENTRY_NOW":
        return None
    gate = detector.get("entry_gate")
    plan = detector.get("trade_plan")
    if not isinstance(gate, dict) or gate.get("state") != "ENTRY_NOW":
        return None
    if not isinstance(plan, dict) or plan.get("state") != "CANDIDATE_READY":
        return None
    direction = gate.get("direction")
    if direction not in {"BULLISH", "BEARISH"} or plan.get("direction") != direction:
        return None

    required = (
        ("structure_event", {"BOS", "CHOCH_MSS"}),
        ("liquidity_sweep", {"BSL_SWEEP", "SSL_SWEEP"}),
        ("displacement", {"DISPLACEMENT"}),
        ("fvg", {"FVG"}),
        ("order_block", {"ORDER_BLOCK"}),
    )
    for key, events in required:
        item = detector.get(key)
        if not isinstance(item, dict):
            return None
        if item.get("event") not in events or item.get("direction") != direction:
            return None
    order_block = detector["order_block"]
    if isinstance(order_block, dict) and order_block.get("state") == "INVALIDATED":
        return None

    raw_timestamp = detector.get("latest_closed_timestamp")
    if not isinstance(raw_timestamp, str):
        return None
    try:
        source_timestamp = datetime.fromisoformat(raw_timestamp)
    except ValueError:
        return None
    if source_timestamp.tzinfo is None:
        return None

    try:
        entry = Decimal(str(plan["entry_reference"]))
        stop_loss = Decimal(str(plan["stop_loss"]))
        take_profit = Decimal(str(plan["take_profit"]))
        risk_reward = Decimal(str(plan["risk_reward"]))
    except (KeyError, ValueError, InvalidOperation):
        return None

    return VerifiedAutoPaperEntryRequest(
        symbol=symbol.upper().replace("/", "-"),
        setup_state="ENTRY_NOW",
        direction=str(direction),
        entry=entry,
        stop_loss=stop_loss,
        take_profit=take_profit,
        risk_reward=risk_reward,
        structure_confirmed=True,
        displacement_confirmed=True,
        order_block_confirmed=True,
        source_timestamp=source_timestamp,
        risk_percent=Decimal("1"),
    )


def apply_realtime_market_fill_to_auto_request(
    request: VerifiedAutoPaperEntryRequest,
    detector: Dict[str, object],
    ticker: MarketDatum,
) -> Optional[VerifiedAutoPaperEntryRequest]:
    """Use a fresh real Coinbase ticker as the paper MARKET fill reference."""
    if (
        ticker.status != DataQualityStatus.VALID
        or ticker.value is None
        or ticker.timestamp is None
        or ticker.timestamp.tzinfo is None
    ):
        return None
    plan = detector.get("trade_plan")
    if not isinstance(plan, dict):
        return None
    zone_low = plan.get("entry_zone_low")
    zone_high = plan.get("entry_zone_high")
    if not isinstance(zone_low, (int, float)) or not isinstance(zone_high, (int, float)):
        return None
    price = Decimal(str(ticker.value))
    if not Decimal(str(zone_low)) <= price <= Decimal(str(zone_high)):
        return None
    if request.stop_loss is None or request.take_profit is None:
        return None
    if request.direction == "BULLISH":
        if not request.stop_loss < price < request.take_profit:
            return None
    elif request.direction == "BEARISH":
        if not request.take_profit < price < request.stop_loss:
            return None
    else:
        return None
    risk = abs(price - request.stop_loss)
    reward = abs(request.take_profit - price)
    if risk <= 0 or reward <= 0:
        return None
    return request.model_copy(
        update={
            "entry": price,
            "risk_reward": reward / risk,
            "source_timestamp": ticker.timestamp,
        }
    )


def auto_entry_request_rejection_reason(detector: Dict[str, object]) -> str:
    """Explain why an ENTRY_NOW detector could not become a verified request."""
    if detector.get("status") != "READY":
        return "DETECTOR_NOT_READY"
    if detector.get("setup_state") != "ENTRY_NOW":
        return "SETUP_NOT_ENTRY_NOW"
    gate = detector.get("entry_gate")
    if not isinstance(gate, dict):
        return "ENTRY_GATE_MISSING"
    if gate.get("state") != "ENTRY_NOW":
        return "ENTRY_GATE_NOT_ENTRY_NOW"
    plan = detector.get("trade_plan")
    if not isinstance(plan, dict):
        return "TRADE_PLAN_MISSING"
    if plan.get("state") != "CANDIDATE_READY":
        return "TRADE_PLAN_NOT_CANDIDATE_READY"
    direction = gate.get("direction")
    if direction not in {"BULLISH", "BEARISH"}:
        return "ENTRY_DIRECTION_INVALID"
    if plan.get("direction") != direction:
        return "TRADE_PLAN_DIRECTION_MISMATCH"
    required = (
        ("structure_event", {"BOS", "CHOCH_MSS"}),
        ("liquidity_sweep", {"BSL_SWEEP", "SSL_SWEEP"}),
        ("displacement", {"DISPLACEMENT"}),
        ("fvg", {"FVG"}),
        ("order_block", {"ORDER_BLOCK"}),
    )
    for key, events in required:
        item = detector.get(key)
        if not isinstance(item, dict):
            return f"{key.upper()}_MISSING"
        if item.get("event") not in events:
            return f"{key.upper()}_EVENT_INVALID"
        if item.get("direction") != direction:
            return f"{key.upper()}_DIRECTION_MISMATCH"
    order_block = detector.get("order_block")
    if isinstance(order_block, dict) and order_block.get("state") == "INVALIDATED":
        return "ORDER_BLOCK_INVALIDATED"
    raw_timestamp = detector.get("latest_closed_timestamp")
    if not isinstance(raw_timestamp, str):
        return "SOURCE_TIMESTAMP_MISSING"
    try:
        source_timestamp = datetime.fromisoformat(raw_timestamp)
    except ValueError:
        return "SOURCE_TIMESTAMP_INVALID"
    if source_timestamp.tzinfo is None:
        return "SOURCE_TIMESTAMP_NAIVE"
    for key in ("entry_reference", "stop_loss", "take_profit", "risk_reward"):
        try:
            Decimal(str(plan[key]))
        except (KeyError, ValueError, InvalidOperation):
            return f"TRADE_PLAN_{key.upper()}_INVALID"
    return "AUTO_ENTRY_REQUEST_BUILDABLE"


def realtime_fill_rejection_reason(
    request: VerifiedAutoPaperEntryRequest,
    detector: Dict[str, object],
    ticker: MarketDatum,
) -> str:
    """Explain why a real Coinbase ticker cannot be used as the paper fill."""
    if ticker.status != DataQualityStatus.VALID:
        return f"REALTIME_TICKER_{ticker.status.value}"
    if ticker.value is None:
        return "REALTIME_TICKER_PRICE_MISSING"
    if ticker.timestamp is None:
        return "REALTIME_TICKER_TIMESTAMP_MISSING"
    if ticker.timestamp.tzinfo is None:
        return "REALTIME_TICKER_TIMESTAMP_NAIVE"
    plan = detector.get("trade_plan")
    if not isinstance(plan, dict):
        return "REALTIME_TRADE_PLAN_MISSING"
    zone_low = plan.get("entry_zone_low")
    zone_high = plan.get("entry_zone_high")
    if not isinstance(zone_low, (int, float)) or not isinstance(zone_high, (int, float)):
        return "REALTIME_ENTRY_ZONE_INVALID"
    price = Decimal(str(ticker.value))
    if not Decimal(str(zone_low)) <= price <= Decimal(str(zone_high)):
        return "REALTIME_PRICE_OUTSIDE_ENTRY_ZONE"
    if request.stop_loss is None or request.take_profit is None:
        return "REALTIME_SL_TP_MISSING"
    if request.direction == "BULLISH":
        if not request.stop_loss < price < request.take_profit:
            return "REALTIME_PRICE_INVALID_FOR_BULLISH_PLAN"
    elif request.direction == "BEARISH":
        if not request.take_profit < price < request.stop_loss:
            return "REALTIME_PRICE_INVALID_FOR_BEARISH_PLAN"
    else:
        return "REALTIME_DIRECTION_INVALID"
    risk = abs(price - request.stop_loss)
    reward = abs(request.take_profit - price)
    if risk <= 0:
        return "REALTIME_RISK_NOT_POSITIVE"
    if reward <= 0:
        return "REALTIME_REWARD_NOT_POSITIVE"
    return "REALTIME_FILL_ELIGIBLE"


async def run_server_auto_paper_generation_once() -> Dict[str, int]:
    """Scan crypto instruments and record every server decision observably."""
    stats = {
        "checked": 0,
        "entry_now": 0,
        "opened": 0,
        "already_consumed": 0,
        "blocked": 0,
    }
    if not persistence_state.ready:
        return stats
    for instrument in instrument_registry.all():
        if instrument.asset_class != AssetClass.CRYPTO:
            continue
        symbol = instrument.canonical_symbol
        stats["checked"] += 1
        detector: Optional[Dict[str, object]] = None
        try:
            detector = await get_server_market_setup_detector(symbol)
            request = build_verified_auto_entry_request_from_detector(symbol, detector)
            if request is None:
                setup_state = str(detector.get("setup_state", "WAIT"))
                if setup_state == "ENTRY_NOW":
                    reason = auto_entry_request_rejection_reason(detector)
                    await record_and_persist_auto_decision_trace(
                        symbol, "BLOCKED", reason, detector
                    )
                else:
                    await record_and_persist_auto_decision_trace(
                        symbol,
                        setup_state,
                        _decision_reason_from_detector(detector),
                        detector,
                    )
                continue
            provider_symbol = provider_symbol_map.to_provider("coinbase", symbol)
            if provider_symbol is None:
                stats["blocked"] += 1
                await record_and_persist_auto_decision_trace(
                    symbol, "BLOCKED", "PROVIDER_SYMBOL_UNAVAILABLE", detector
                )
                continue
            ticker = await market_provider.get_ticker(provider_symbol)
            fill_request = apply_realtime_market_fill_to_auto_request(
                request, detector, ticker
            )
            if fill_request is None:
                stats["blocked"] += 1
                reason = realtime_fill_rejection_reason(request, detector, ticker)
                await record_and_persist_auto_decision_trace(
                    symbol, "BLOCKED", reason, detector
                )
                continue
            request = fill_request
            request.performance_timeframe = SERVER_SETUP_GRANULARITY
            session_snapshot = market_session_context(symbol, utcnow())
            session_value = session_snapshot.get("current_session")
            request.performance_session = (
                str(session_value) if session_value is not None else None
            )
            regime_snapshot = await get_server_market_regime(symbol)
            regime_value = regime_snapshot.get("regime")
            request.performance_market_regime = (
                str(regime_value) if regime_value is not None else None
            )
            request.performance_setup_context = json.dumps(
                {
                    "setup_state": detector.get("setup_state"),
                    "direction": detector.get("direction"),
                    "latest_closed_timestamp": detector.get("latest_closed_timestamp"),
                },
                default=str,
                sort_keys=True,
            )
            stats["entry_now"] += 1
            result = await verified_auto_paper_entry(request)
        except HTTPException as exc:
            if exc.status_code == 409:
                stats["already_consumed"] += 1
                reason = "DUPLICATE_OR_PORTFOLIO_CONFLICT"
            else:
                stats["blocked"] += 1
                reason = f"HTTP_{exc.status_code}"
            await record_and_persist_auto_decision_trace(symbol, "BLOCKED", reason, detector)
            continue
        except Exception as exc:  # noqa: BLE001
            log.error("Server auto-paper generation failed: %s", exc)
            stats["blocked"] += 1
            await record_and_persist_auto_decision_trace(
                symbol, "BLOCKED", f"RUNTIME_{type(exc).__name__}", detector
            )
            continue
        if result.get("status") == "OPENED":
            stats["opened"] += 1
            await record_and_persist_auto_decision_trace(
                symbol, "ENTRY_NOW", "PAPER_POSITION_OPENED", detector
            )
        else:
            stats["blocked"] += 1
            raw_reason = result.get("reason")
            reason = str(raw_reason) if raw_reason else "VERIFIED_ENTRY_BLOCKED"
            await record_and_persist_auto_decision_trace(symbol, "BLOCKED", reason, detector)
    return stats


async def run_auto_entry_orchestrator_once() -> Dict[str, int]:
    opened = 0
    blocked = 0
    conflicts = 0
    for candidate_id, state in list(auto_entry_candidates.items()):
        state.attempts += 1
        try:
            result = await verified_auto_paper_entry(state.request)
        except HTTPException as exc:
            if exc.status_code == 409:
                state.last_status = "CONFLICT"
                state.last_reason = "DUPLICATE_POSITION"
                conflicts += 1
                auto_entry_candidates.pop(candidate_id, None)
                continue
            state.last_status = "BLOCKED"
            state.last_reason = "HTTP_ERROR"
            blocked += 1
            continue
        status_value = str(result.get("status", "BLOCKED"))
        state.last_status = status_value
        state.last_reason = (
            str(result.get("reason")) if result.get("reason") is not None else None
        )
        if status_value == "OPENED":
            opened += 1
            auto_entry_candidates.pop(candidate_id, None)
        else:
            blocked += 1
    return {
        "checked": opened + blocked + conflicts,
        "opened": opened,
        "blocked": blocked,
        "conflicts": conflicts,
    }


async def auto_entry_orchestrator_loop() -> None:
    while True:
        auto_scan_runtime["last_started_at"] = utcnow().isoformat()
        auto_scan_runtime["last_error"] = None
        try:
            generation = await run_server_auto_paper_generation_once()
            trend_generation = await run_trend_pullback_paper_generation_once()
            breakout_generation = await run_breakout_expansion_paper_generation_once()
            multi_asset_analysis = await run_multi_asset_strategy_analysis_once()
            multi_asset_generation = await run_multi_asset_paper_generation_once()
            queue = await run_auto_entry_orchestrator_once()
            auto_scan_runtime["last_generation"] = generation
            auto_scan_runtime["last_trend_pullback_generation"] = trend_generation
            auto_scan_runtime["last_breakout_expansion_generation"] = breakout_generation
            auto_scan_runtime["last_multi_asset_analysis"] = multi_asset_analysis
            auto_scan_runtime["last_multi_asset_generation"] = multi_asset_generation
            auto_scan_runtime["last_queue"] = queue
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            auto_scan_runtime["last_error"] = type(exc).__name__
            log.error("Auto-entry orchestrator iteration failed: %s", exc)
        finally:
            iterations = auto_scan_runtime.get("iterations", 0)
            if not isinstance(iterations, int):
                iterations = 0
            auto_scan_runtime["iterations"] = iterations + 1
            auto_scan_runtime["last_completed_at"] = utcnow().isoformat()
        await asyncio.sleep(AUTO_ENTRY_ORCHESTRATOR_INTERVAL_SECONDS)


def auto_paper_e2e_readiness() -> Dict[str, object]:
    """Expose the server-side wiring required for paper auto-trading E2E.

    This is a readiness/contract audit, not proof that a market setup exists.
    A real position still requires real VALID market data and ENTRY_NOW.
    """
    task = auto_entry_orchestrator_task
    orchestrator_running = task is not None and not task.done()
    ready = persistence_state.ready and orchestrator_running
    return {
        "status": "READY" if ready else "NOT_READY",
        "validation": "SERVER_AUTO_PAPER_E2E_READINESS_V1",
        "persistence_ready": persistence_state.ready,
        "orchestrator_running": orchestrator_running,
        "paper_monitor_interval_seconds": max(
            settings.paper_monitor_interval_seconds, 1.0
        ),
        "pipeline": [
            "REAL_MARKET_DATA",
            "SERVER_SMC_SETUP",
            "ENTRY_NOW_GATE",
            "PORTFOLIO_RISK_GUARD",
            "PAPER_POSITION_CREATE",
            "REALTIME_MARK",
            "SL_TP_CLOSE",
            "PAPER_HISTORY",
        ],
        "paper_only": True,
        "broker_execution": False,
        "live_trading_enabled": False,
    }


@api_router.get("/paper/auto-entry/e2e-readiness")
async def get_auto_paper_e2e_readiness() -> Dict[str, object]:
    return auto_paper_e2e_readiness()


@api_router.get("/paper/auto-entry/runtime-status")
async def get_auto_scan_runtime_status() -> Dict[str, object]:
    return auto_scan_runtime_status()


@api_router.get("/paper/auto-entry/watchdog-status")
async def get_auto_scan_watchdog_status() -> Dict[str, object]:
    return auto_scan_watchdog_status()


@api_router.get("/paper/auto-entry/decision-trace")
async def get_auto_decision_trace(limit: int = 50) -> Dict[str, object]:
    return auto_decision_trace_status(limit)


@api_router.get("/paper/auto-entry/decision-history")
async def get_signal_decision_history(
    limit: int = 100, symbol: Optional[str] = None, state: Optional[str] = None
) -> Dict[str, object]:
    return await signal_decision_history(limit=limit, symbol=symbol, state=state)


@api_router.post("/paper/auto-entry/candidates")
async def queue_auto_entry_candidate(
    req: AutoEntryCandidateRequest,
) -> Dict[str, object]:
    return register_auto_entry_candidate(req)


@api_router.get("/paper/auto-entry/orchestrator/status")
async def get_auto_entry_orchestrator_status() -> Dict[str, object]:
    task = auto_entry_orchestrator_task
    return {
        "status": "RUNNING" if task is not None and not task.done() else "STOPPED",
        "queued_candidates": len(auto_entry_candidates),
        "interval_seconds": AUTO_ENTRY_ORCHESTRATOR_INTERVAL_SECONDS,
        "market_setup_detection": "STRUCTURE_BOS_CHOCH_V1",
        "liquidity_sweep_detection": "SERVER_LIQUIDITY_SWEEP_V1",
        "displacement_detection": "SERVER_DISPLACEMENT_V1",
        "fvg_detection": "SERVER_FVG_V1",
        "order_block_detection": "SERVER_ORDER_BLOCK_V1",
        "retest_revalidation": "SERVER_RETEST_REVALIDATION_V1",
        "trade_plan_builder": "SERVER_TRADE_PLAN_V1",
        "entry_now_gate": "SERVER_ENTRY_NOW_GATE_V1",
        "smc_auto_candidate_generation": "SERVER_AUTO_PAPER_POSITION_V1",
        "auto_position_creation": "SERVER_AUTO_PAPER_POSITION_V1",
        "decision_trace": "SERVER_DECISION_TRACE_V1",
        "decision_history": "SERVER_SIGNAL_DECISION_HISTORY_V1",
        "paper_only": True,
        "execution": False,
    }


@api_router.post("/paper/auto-entry/gate")
async def evaluate_paper_auto_entry_gate(
    req: PaperAutoEntryGateRequest,
) -> Dict[str, object]:
    canonical = req.symbol.upper().replace("/", "-")
    instrument = instrument_registry.get(canonical)
    blockers: List[str] = []

    if req.signal_decision not in {"LONG", "SHORT", "WAIT"}:
        blockers.append("SIGNAL_DECISION_INVALID")
    elif req.signal_decision == "WAIT":
        blockers.append("SIGNAL_WAIT")

    if instrument is None:
        blockers.append("INSTRUMENT_NOT_REGISTERED")

    levels = (req.entry, req.stop_loss, req.take_profit, req.risk_reward)
    if any(value is None for value in levels):
        blockers.append("TRADE_PLAN_INCOMPLETE")
    elif req.risk_reward is not None and req.risk_reward <= Decimal("0"):
        blockers.append("RR_INVALID")
    elif req.entry is not None and req.stop_loss is not None and req.take_profit is not None:
        if req.signal_decision == "LONG" and not (
            req.stop_loss < req.entry < req.take_profit
        ):
            blockers.append("LONG_LEVELS_INVALID")
        if req.signal_decision == "SHORT" and not (
            req.take_profit < req.entry < req.stop_loss
        ):
            blockers.append("SHORT_LEVELS_INVALID")

    # V16-M2 provides a server-authoritative signal evaluator. This gate request still
    # carries a decision field for compatibility; unattended creation must call the
    # server evaluator first and use its derived decision, never trust browser voting.

    # The backend registry intentionally contains no invented broker sizing rules.
    # Auto entry stays blocked until source/timestamp + volume/tick/contract rules
    # are represented and verified server-side for the selected instrument.
    blockers.append("SERVER_INSTRUMENT_SPECS_REQUIRED")

    return {
        "status": "BLOCKED",
        "symbol": canonical,
        "signal_decision": req.signal_decision,
        "blockers": list(dict.fromkeys(blockers)),
        "auto_create_position": False,
        "paper_only": True,
        "execution": False,
    }


@api_router.post("/paper/positions", status_code=201)
async def create_paper_position(req: PaperPositionCreate) -> Dict[str, object]:
    if not persistence_state.ready:
        raise HTTPException(
            status_code=503, detail={"status": "UNAVAILABLE", "reason": "persistence not ready"}
        )
    validate_paper_position_create(req)
    now = utcnow()
    values = {
        "position_id": req.position_id, "symbol": req.symbol.upper(), "side": req.side,
        "status": "OPEN", "entry": req.entry, "stop_loss": req.stop_loss,
        "take_profit": req.take_profit, "size": req.size, "size_unit": req.size_unit,
        "risk_money": req.risk_money, "risk_percent": req.risk_percent,
        "capital_before": req.capital_before, "source": req.source,
        "source_timestamp": req.source_timestamp, "opened_at": req.opened_at,
        "close_reason": None, "close_price": None, "closed_at": None,
        "created_at": now, "updated_at": now,
    }
    try:
        async with engine.begin() as conn:
            stmt = pg_insert(paper_positions_table).values(values)
            stmt = stmt.on_conflict_do_nothing(index_elements=["position_id"])
            result = await conn.execute(stmt)
            if result.rowcount != 1:
                raise HTTPException(
                    status_code=409, detail={"status": "CONFLICT", "reason": "POSITION_ID_EXISTS"}
                )
            if req.fx_quote_to_usd is not None:
                fx_values = {
                    "position_id": req.position_id,
                    "phase": "OPEN",
                    "quote_currency": str(req.fx_quote_currency),
                    "quote_to_usd": req.fx_quote_to_usd,
                    "conversion_symbol": str(req.fx_conversion_symbol),
                    "conversion_price": req.fx_conversion_price,
                    "inverse": bool(req.fx_conversion_inverse),
                    "source": str(req.fx_conversion_source),
                    "source_timestamp": req.fx_conversion_source_timestamp,
                    "created_at": now,
                }
                await conn.execute(
                    pg_insert(paper_fx_conversion_snapshots_table).values(fx_values)
                )
            context_values = (
                req.performance_strategy_id,
                req.performance_strategy_version,
                req.performance_timeframe,
                req.performance_session,
                req.performance_market_regime,
                req.performance_setup_context,
            )
            if any(value is not None for value in context_values):
                await conn.execute(
                    pg_insert(paper_position_context_table).values(
                        position_id=req.position_id,
                        strategy_id=req.performance_strategy_id,
                        strategy_version=req.performance_strategy_version,
                        timeframe=req.performance_timeframe,
                        session=req.performance_session,
                        market_regime=req.performance_market_regime,
                        setup_context=req.performance_setup_context,
                        captured_at=now,
                    ).on_conflict_do_nothing(index_elements=["position_id"])
                )
    except HTTPException:
        raise
    except Exception as exc:
        persistence_state.mark_runtime_error(str(exc))
        raise HTTPException(
            status_code=503, detail={"status": "UNAVAILABLE", "reason": "paper persistence failed"}
        ) from exc
    persistence_state.mark_write_ok()
    return {**values, "paper_only": True, "execution": False}


class PaperPositionMark(BaseModel):
    current_price: Decimal
    observed_at: datetime
    source: str = Field(min_length=1, max_length=64)
    source_timestamp: datetime


async def paper_mark_from_coinbase_rest(
    canonical: str, provider_symbol: str
) -> Optional[PaperPositionMark]:
    """Fail-safe real REST mark for crypto when the WS store has no fresh price."""
    try:
        datum = await market_provider.get_ticker(provider_symbol)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - monitoring fallback must fail closed
        log.warning("Coinbase REST paper mark unavailable for %s: %s", canonical, exc)
        return None
    if datum.status != DataQualityStatus.VALID:
        return None
    if datum.value is None or datum.value <= 0 or datum.timestamp is None:
        return None
    return PaperPositionMark(
        current_price=Decimal(str(datum.value)),
        observed_at=utcnow(),
        source="coinbase_rest_fallback",
        source_timestamp=datum.timestamp,
    )


async def paper_mark_from_realtime(symbol: str) -> Optional[PaperPositionMark]:
    canonical = symbol.upper().replace("/", "-")
    instrument = instrument_registry.get(canonical)
    if instrument is None:
        return None

    price: Optional[Decimal] = None
    received_at: Optional[datetime] = None
    source_timestamp: Optional[datetime] = None
    source = ""

    if instrument.asset_class == AssetClass.CRYPTO:
        provider_symbol = provider_symbol_map.to_provider("coinbase", canonical)
        if provider_symbol is None:
            return None
        datum = await market_store.get_ticker(provider_symbol)
        if (
            datum is None
            or datum.status != DataQualityStatus.VALID
            or datum.value is None
            or datum.value <= 0
            or datum.source_timestamp is None
        ):
            return await paper_mark_from_coinbase_rest(canonical, provider_symbol)
        price = Decimal(str(datum.value))
        received_at = datum.received_at
        source_timestamp = datum.source_timestamp
        source = datum.source

    elif instrument.asset_class == AssetClass.FOREX:
        quote = massive_forex_ws.quotes.get(canonical)
        if quote is None or quote.quality != DataQualityStatus.VALID:
            return None
        if quote.bid <= 0 or quote.ask <= 0 or quote.bid > quote.ask:
            return None
        price = (quote.bid + quote.ask) / Decimal("2")
        received_at = quote.received_at
        source_timestamp = quote.source_timestamp
        source = "massive"

    elif instrument.asset_class == AssetClass.METAL:
        if canonical != "XAU-USD":
            return None
        gold = twelvedata_gold_ws.last_price
        if gold is None or gold.quality != DataQualityStatus.VALID:
            return None
        if gold.price <= 0:
            return None
        price = gold.price
        received_at = gold.received_at
        source_timestamp = gold.source_timestamp
        source = "twelvedata"

    elif instrument.asset_class == AssetClass.INDEX:
        value = massive_indices_ws.values.get(canonical)
        if value is None or value.quality != DataQualityStatus.VALID:
            return None
        if value.value <= 0:
            return None
        price = value.value
        received_at = value.received_at
        source_timestamp = value.source_timestamp
        source = "massive"

    if price is None or received_at is None or source_timestamp is None or not source:
        return None
    return PaperPositionMark(
        current_price=price,
        observed_at=received_at,
        source=source,
        source_timestamp=source_timestamp,
    )


async def monitor_open_paper_positions_once() -> Dict[str, int]:
    if not persistence_state.ready:
        return {"checked": 0, "marked": 0, "unavailable": 0, "errors": 0}
    async with engine.connect() as conn:
        result = await conn.execute(
            text("SELECT position_id, symbol FROM paper_positions WHERE status='OPEN'")
        )
        rows = result.fetchall()
    marked = 0
    unavailable = 0
    errors = 0
    for row in rows:
        try:
            mark = await paper_mark_from_realtime(row._mapping["symbol"])
            if mark is None:
                unavailable += 1
                continue
            await mark_paper_position(row._mapping["position_id"], mark)
            marked += 1
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - isolate one position from the batch
            errors += 1
            log.error(
                "Paper position monitor failed for %s: %s",
                row._mapping["position_id"],
                exc,
            )
    return {
        "checked": len(rows),
        "marked": marked,
        "unavailable": unavailable,
        "errors": errors,
    }


async def paper_monitor_loop(stop_event: asyncio.Event) -> None:
    interval = max(settings.paper_monitor_interval_seconds, 1.0)
    while not stop_event.is_set():
        try:
            await monitor_open_paper_positions_once()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - loop must fail safe and keep serving
            log.error("Paper monitor iteration failed: %s", exc)
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass


def paper_mark_temporally_valid(
    mark: PaperPositionMark, opened_at: datetime
) -> bool:
    timestamps = (mark.observed_at, mark.source_timestamp, opened_at)
    if any(value.tzinfo is None or value.utcoffset() is None for value in timestamps):
        return False
    if mark.source_timestamp > mark.observed_at:
        return False
    if mark.source_timestamp < opened_at:
        return False
    return True


def evaluate_paper_close(side: str, price: Decimal, stop_loss: Decimal,
                         take_profit: Decimal) -> Optional[Tuple[str, Decimal]]:
    if side == "LONG":
        if price <= stop_loss:
            return ("STOP_LOSS", stop_loss)
        if price >= take_profit:
            return ("TAKE_PROFIT", take_profit)
    elif side == "SHORT":
        if price >= stop_loss:
            return ("STOP_LOSS", stop_loss)
        if price <= take_profit:
            return ("TAKE_PROFIT", take_profit)
    return None


def calculate_paper_pnl(side: str, entry: Decimal, exit_price: Decimal,
                        size: Decimal) -> Decimal:
    delta = exit_price - entry if side == "LONG" else entry - exit_price
    return delta * size


def paper_position_quote_to_usd_required(symbol: str) -> bool:
    instrument = instrument_registry.get(symbol.upper().replace("/", "-"))
    return bool(
        instrument is not None
        and instrument.asset_class == AssetClass.FOREX
        and instrument.quote_asset != "USD"
    )


def calculate_paper_pnl_usd(
    side: str,
    entry: Decimal,
    exit_price: Decimal,
    size: Decimal,
    quote_to_usd: Decimal = Decimal("1"),
) -> Decimal:
    if quote_to_usd <= 0:
        raise ValueError("quote_to_usd must be positive")
    return calculate_paper_pnl(side, entry, exit_price, size) * quote_to_usd


async def persist_paper_fx_conversion_snapshot(
    conn: Any, position_id: str, phase: str, conversion: Dict[str, object]
) -> None:
    rate = conversion.get("quote_to_usd")
    if not isinstance(rate, Decimal) or rate <= 0:
        raise ValueError("valid quote_to_usd required")
    values = {
        "position_id": position_id,
        "phase": phase,
        "quote_currency": str(conversion.get("quote_currency") or ""),
        "quote_to_usd": rate,
        "conversion_symbol": str(conversion.get("conversion_symbol") or ""),
        "conversion_price": conversion.get("conversion_price"),
        "inverse": bool(conversion.get("inverse", False)),
        "source": str(conversion.get("source") or "identity"),
        "source_timestamp": conversion.get("source_timestamp"),
        "created_at": utcnow(),
    }
    stmt = pg_insert(paper_fx_conversion_snapshots_table).values(values)
    stmt = stmt.on_conflict_do_update(
        index_elements=["position_id", "phase"],
        set_={
            "quote_currency": stmt.excluded.quote_currency,
            "quote_to_usd": stmt.excluded.quote_to_usd,
            "conversion_symbol": stmt.excluded.conversion_symbol,
            "conversion_price": stmt.excluded.conversion_price,
            "inverse": stmt.excluded.inverse,
            "source": stmt.excluded.source,
            "source_timestamp": stmt.excluded.source_timestamp,
            "created_at": stmt.excluded.created_at,
        },
    )
    await conn.execute(stmt)


@api_router.post("/paper/positions/{position_id}/mark")
async def mark_paper_position(position_id: str, req: PaperPositionMark) -> Dict[str, object]:
    if not persistence_state.ready:
        raise HTTPException(
            status_code=503, detail={"status": "UNAVAILABLE", "reason": "persistence not ready"}
        )
    if req.current_price <= Decimal("0"):
        raise HTTPException(
            status_code=400, detail={"status": "INVALID", "reason": "PRICE_INVALID"}
        )
    try:
        async with engine.begin() as conn:
            result = await conn.execute(
                text("SELECT * FROM paper_positions WHERE position_id = :position_id FOR UPDATE"),
                {"position_id": position_id},
            )
            row = result.fetchone()
            if row is None:
                raise HTTPException(
                    status_code=404, detail={"status": "NOT_FOUND", "reason": "POSITION_NOT_FOUND"}
                )
            data = dict(row._mapping)
            if data["status"] != "OPEN":
                return paper_position_to_dict(row)
            fx_conversion: Optional[Dict[str, object]] = None
            quote_to_usd = Decimal("1")
            if paper_position_quote_to_usd_required(str(data["symbol"])):
                fx_conversion = await realtime_fx_quote_to_usd(str(data["symbol"]))
                if fx_conversion.get("status") != "VALID":
                    raise HTTPException(
                        status_code=409,
                        detail={
                            "status": "CONFLICT",
                            "reason": "FX_CONVERSION_UNAVAILABLE",
                        },
                    )
                conversion_rate = fx_conversion.get("quote_to_usd")
                if not isinstance(conversion_rate, Decimal) or conversion_rate <= 0:
                    raise HTTPException(
                        status_code=409,
                        detail={
                            "status": "CONFLICT",
                            "reason": "FX_CONVERSION_INVALID",
                        },
                    )
                quote_to_usd = conversion_rate
            if not paper_mark_temporally_valid(req, data["opened_at"]):
                raise HTTPException(
                    status_code=409,
                    detail={"status": "CONFLICT", "reason": "STALE_OR_INVALID_MARK"},
                )
            outcome = evaluate_paper_close(
                data["side"], req.current_price, data["stop_loss"], data["take_profit"]
            )
            if outcome is None:
                payload = paper_position_to_dict(row)
                payload["mark_price"] = str(req.current_price)
                payload["mark_observed_at"] = req.observed_at.isoformat()
                payload["mark_source"] = req.source
                payload["mark_source_timestamp"] = req.source_timestamp.isoformat()
                raw_pnl = calculate_paper_pnl(
                    data["side"], data["entry"], req.current_price, data["size"]
                )
                payload["unrealized_pnl"] = str(raw_pnl * quote_to_usd)
                payload["unrealized_pnl_currency"] = "USD"
                if fx_conversion is not None:
                    payload["fx_quote_to_usd"] = str(quote_to_usd)
                    payload["fx_conversion_symbol"] = fx_conversion.get(
                        "conversion_symbol"
                    )
                return payload
            reason, close_price = outcome
            raw_pnl = calculate_paper_pnl(
                data["side"], data["entry"], close_price, data["size"]
            )
            pnl = raw_pnl * quote_to_usd
            now = utcnow()
            if fx_conversion is not None:
                await persist_paper_fx_conversion_snapshot(
                    conn, position_id, "CLOSE", fx_conversion
                )
            await conn.execute(
                text(
                    "UPDATE paper_positions SET status='CLOSED', close_reason=:reason, "
                    "close_price=:close_price, closed_at=:closed_at, updated_at=:updated_at "
                    "WHERE position_id=:position_id AND status='OPEN'"
                ),
                {
                    "reason": reason, "close_price": close_price, "closed_at": req.observed_at,
                    "updated_at": now, "position_id": position_id,
                },
            )
            data.update(
                status="CLOSED", close_reason=reason, close_price=close_price,
                closed_at=req.observed_at, updated_at=now,
            )
            payload = paper_position_to_dict(type("Row", (), {"_mapping": data})())
            payload["realized_pnl"] = str(pnl)
            payload["realized_pnl_currency"] = "USD"
            if fx_conversion is not None:
                payload["fx_quote_to_usd_close"] = str(quote_to_usd)
                payload["fx_conversion_symbol_close"] = fx_conversion.get(
                    "conversion_symbol"
                )
            payload["close_source"] = req.source
            payload["close_source_timestamp"] = req.source_timestamp.isoformat()
            return payload
    except HTTPException:
        raise
    except Exception as exc:
        persistence_state.mark_runtime_error(str(exc))
        raise HTTPException(
            status_code=503, detail={"status": "UNAVAILABLE", "reason": "paper mark failed"}
        ) from exc


@api_router.get("/paper/positions/live")
async def get_live_paper_positions() -> Dict[str, object]:
    if not persistence_state.ready:
        raise HTTPException(
            status_code=503, detail={"status": "UNAVAILABLE", "reason": "persistence not ready"}
        )
    try:
        async with engine.connect() as conn:
            result = await conn.execute(
                text(
                    "SELECT * FROM paper_positions WHERE status='OPEN' "
                    "ORDER BY opened_at DESC, position_id DESC"
                )
            )
            rows = result.fetchall()
    except Exception as exc:
        persistence_state.mark_runtime_error(str(exc))
        raise HTTPException(
            status_code=503, detail={"status": "UNAVAILABLE", "reason": "paper persistence failed"}
        ) from exc

    positions: List[Dict[str, object]] = []
    for row in rows:
        payload = paper_position_to_dict(row)
        mark = await paper_mark_from_realtime(str(row._mapping["symbol"]))
        payload["mark_status"] = "UNAVAILABLE"
        payload["mark_price"] = None
        payload["mark_source"] = None
        payload["mark_source_timestamp"] = None
        payload["unrealized_pnl"] = None
        if mark is not None:
            payload["mark_status"] = "VALID"
            payload["mark_price"] = str(mark.current_price)
            payload["mark_source"] = mark.source
            payload["mark_source_timestamp"] = mark.source_timestamp.isoformat()
            raw_pnl = calculate_paper_pnl(
                row._mapping["side"],
                row._mapping["entry"],
                mark.current_price,
                row._mapping["size"],
            )
            if paper_position_quote_to_usd_required(str(row._mapping["symbol"])):
                conversion = await realtime_fx_quote_to_usd(str(row._mapping["symbol"]))
                rate = conversion.get("quote_to_usd")
                if conversion.get("status") == "VALID" and isinstance(rate, Decimal):
                    payload["unrealized_pnl"] = str(raw_pnl * rate)
                    payload["fx_quote_to_usd"] = str(rate)
                    payload["fx_conversion_status"] = "VALID"
                else:
                    payload["unrealized_pnl"] = None
                    payload["fx_conversion_status"] = str(
                        conversion.get("reason") or "UNAVAILABLE"
                    )
            else:
                payload["unrealized_pnl"] = str(raw_pnl)
        positions.append(payload)
    return {
        "status": "OK",
        "positions": positions,
        "count": len(positions),
        "paper_only": True,
        "execution": False,
    }


@api_router.get("/paper/account/live")
async def get_live_paper_account() -> Dict[str, object]:
    account = await get_paper_account()
    if not persistence_state.ready:
        raise HTTPException(
            status_code=503, detail={"status": "UNAVAILABLE", "reason": "persistence not ready"}
        )
    try:
        async with engine.connect() as conn:
            result = await conn.execute(
                text(
                    "SELECT position_id, symbol, side, entry, size "
                    "FROM paper_positions WHERE status='OPEN' "
                    "ORDER BY opened_at ASC, position_id ASC"
                )
            )
            rows = result.fetchall()
    except Exception as exc:
        persistence_state.mark_runtime_error(str(exc))
        raise HTTPException(
            status_code=503, detail={"status": "UNAVAILABLE", "reason": "paper persistence failed"}
        ) from exc

    unrealized = Decimal("0")
    marked = 0
    unavailable = 0
    for row in rows:
        data = row._mapping
        mark = await paper_mark_from_realtime(str(data["symbol"]))
        if mark is None:
            unavailable += 1
            continue
        raw_pnl = calculate_paper_pnl(
            data["side"], data["entry"], mark.current_price, data["size"]
        )
        if paper_position_quote_to_usd_required(str(data["symbol"])):
            conversion = await realtime_fx_quote_to_usd(str(data["symbol"]))
            rate = conversion.get("quote_to_usd")
            if conversion.get("status") != "VALID" or not isinstance(rate, Decimal):
                unavailable += 1
                continue
            unrealized += raw_pnl * rate
        else:
            unrealized += raw_pnl
        marked += 1

    current_capital = Decimal(str(account["current_capital"]))
    live_equity = current_capital + unrealized
    complete = unavailable == 0
    return {
        **account,
        "unrealized_pnl": str(unrealized) if complete else None,
        "live_equity": str(live_equity) if complete else None,
        "marked_open_positions": marked,
        "unavailable_open_positions": unavailable,
        "live_equity_status": "VALID" if complete else "PARTIAL",
        "paper_only": True,
        "execution": False,
    }


@api_router.get("/paper/account")
async def get_paper_account() -> Dict[str, object]:
    if not persistence_state.ready:
        raise HTTPException(
            status_code=503, detail={"status": "UNAVAILABLE", "reason": "persistence not ready"}
        )
    try:
        async with engine.connect() as conn:
            account_result = await conn.execute(
                text(
                    "SELECT currency, initial_capital FROM paper_account "
                    "WHERE account_id=:account_id"
                ),
                {"account_id": PAPER_ACCOUNT_ID},
            )
            account = account_result.fetchone()
            result = await conn.execute(
                text(
                    "SELECT p.position_id, p.symbol, p.status, p.side, p.entry, p.size, "
                    "p.close_price, fx.quote_to_usd "
                    "FROM paper_positions p LEFT JOIN paper_fx_conversion_snapshots fx "
                    "ON fx.position_id=p.position_id AND fx.phase='CLOSE' "
                    "ORDER BY p.opened_at ASC, p.position_id ASC"
                )
            )
            rows = result.fetchall()
    except Exception as exc:
        persistence_state.mark_runtime_error(str(exc))
        raise HTTPException(
            status_code=503, detail={"status": "UNAVAILABLE", "reason": "paper persistence failed"}
        ) from exc
    if account is None:
        raise HTTPException(
            status_code=503,
            detail={"status": "UNAVAILABLE", "reason": "paper account not initialized"},
        )

    realized = Decimal("0")
    open_count = 0
    closed_count = 0
    for row in rows:
        data = row._mapping
        if data["status"] == "OPEN":
            open_count += 1
        elif data["status"] == "CLOSED" and data["close_price"] is not None:
            closed_count += 1
            raw_pnl = calculate_paper_pnl(
                data["side"], data["entry"], data["close_price"], data["size"]
            )
            if paper_position_quote_to_usd_required(str(data["symbol"])):
                if data["quote_to_usd"] is None:
                    raise HTTPException(
                        status_code=503,
                        detail={
                            "status": "UNAVAILABLE",
                            "reason": "CLOSE_FX_SNAPSHOT_MISSING",
                        },
                    )
                realized += raw_pnl * Decimal(str(data["quote_to_usd"]))
            else:
                realized += raw_pnl
    initial_capital = account._mapping["initial_capital"]
    current_capital = initial_capital + realized
    return {
        "status": "OK",
        "currency": account._mapping["currency"],
        "initial_capital": str(initial_capital),
        "current_capital": str(current_capital),
        "realized_pnl": str(realized),
        "open_positions": open_count,
        "closed_positions": closed_count,
        "paper_only": True,
        "execution": False,
    }


def calculate_paper_performance_metrics(
    closed_positions: List[Dict[str, object]], initial_capital: Decimal
) -> Dict[str, object]:
    """Calculate deterministic paper-only performance metrics from closed trades."""
    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive")
    wins = losses = breakeven = 0
    gross_profit = Decimal("0")
    gross_loss = Decimal("0")
    net_pnl = Decimal("0")
    rr_sum = Decimal("0")
    rr_count = 0
    equity = initial_capital
    peak = initial_capital
    max_drawdown = Decimal("0")
    max_drawdown_percent = Decimal("0")

    for position in closed_positions:
        side = str(position["side"])
        entry = Decimal(str(position["entry"]))
        close_price = Decimal(str(position["close_price"]))
        size = Decimal(str(position["size"]))
        risk_money = Decimal(str(position["risk_money"]))
        raw_pnl = calculate_paper_pnl(side, entry, close_price, size)
        symbol = str(position.get("symbol") or "")
        stored_rate = position.get("quote_to_usd")
        if paper_position_quote_to_usd_required(symbol):
            if stored_rate is None:
                raise ValueError("closed non-USD FX trade missing conversion snapshot")
            pnl = raw_pnl * Decimal(str(stored_rate))
        else:
            pnl = raw_pnl
        net_pnl += pnl
        if pnl > 0:
            wins += 1
            gross_profit += pnl
        elif pnl < 0:
            losses += 1
            gross_loss += -pnl
        else:
            breakeven += 1
        if risk_money > 0:
            rr_sum += pnl / risk_money
            rr_count += 1
        equity += pnl
        if equity > peak:
            peak = equity
        drawdown = peak - equity
        if drawdown > max_drawdown:
            max_drawdown = drawdown
        if peak > 0:
            drawdown_percent = drawdown / peak * Decimal("100")
            if drawdown_percent > max_drawdown_percent:
                max_drawdown_percent = drawdown_percent

    trades = len(closed_positions)
    win_rate = Decimal(wins) / Decimal(trades) * Decimal("100") if trades else None
    expectancy = net_pnl / Decimal(trades) if trades else None
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else None
    average_realized_rr = rr_sum / Decimal(rr_count) if rr_count else None
    return {
        "closed_trades": trades,
        "wins": wins,
        "losses": losses,
        "breakeven": breakeven,
        "win_rate_percent": str(win_rate) if win_rate is not None else None,
        "gross_profit": str(gross_profit),
        "gross_loss": str(gross_loss),
        "net_pnl": str(net_pnl),
        "profit_factor": str(profit_factor) if profit_factor is not None else None,
        "expectancy": str(expectancy) if expectancy is not None else None,
        "average_realized_rr": (
            str(average_realized_rr) if average_realized_rr is not None else None
        ),
        "max_drawdown": str(max_drawdown),
        "max_drawdown_percent": str(max_drawdown_percent),
    }


PAPER_PERFORMANCE_PERIODS = {"ALL", "DAY", "WEEK", "MONTH", "YEAR"}


def paper_performance_period_start(
    period: str, now: Optional[datetime] = None
) -> Optional[datetime]:
    """Return the UTC start boundary for an objective performance window."""
    normalized = period.upper()
    if normalized not in PAPER_PERFORMANCE_PERIODS:
        raise ValueError("unsupported paper performance period")
    if normalized == "ALL":
        return None
    current = now or utcnow()
    if current.tzinfo is None:
        raise ValueError("performance period now must be timezone-aware")
    current = current.astimezone(timezone.utc)
    if normalized == "DAY":
        return current.replace(hour=0, minute=0, second=0, microsecond=0)
    if normalized == "WEEK":
        day_start = current.replace(hour=0, minute=0, second=0, microsecond=0)
        return day_start - timedelta(days=day_start.weekday())
    if normalized == "MONTH":
        return current.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    return current.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)


@api_router.get("/paper/performance")
async def get_paper_performance(
    symbol: Optional[str] = None, period: str = "ALL"
) -> Dict[str, object]:
    """Return performance analytics derived only from persisted CLOSED paper trades."""
    if not persistence_state.ready:
        raise HTTPException(
            status_code=503, detail={"status": "UNAVAILABLE", "reason": "persistence not ready"}
        )
    canonical = symbol.upper().replace("/", "-") if symbol else None
    normalized_period = period.upper()
    try:
        period_start = paper_performance_period_start(normalized_period)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={"status": "INVALID", "reason": "unsupported performance period"},
        ) from exc
    try:
        async with engine.connect() as conn:
            account_result = await conn.execute(
                text(
                    "SELECT initial_capital FROM paper_account "
                    "WHERE account_id=:account_id"
                ),
                {"account_id": PAPER_ACCOUNT_ID},
            )
            account = account_result.fetchone()
            if account is None:
                raise HTTPException(
                    status_code=503,
                    detail={"status": "UNAVAILABLE", "reason": "paper account not initialized"},
                )
            sql = (
                "SELECT p.position_id, p.symbol, p.side, p.entry, p.close_price, p.size, "
                "p.risk_money, p.closed_at, fx.quote_to_usd "
                "FROM paper_positions p LEFT JOIN paper_fx_conversion_snapshots fx "
                "ON fx.position_id=p.position_id AND fx.phase='CLOSE' "
                "WHERE status='CLOSED' AND p.close_price IS NOT NULL"
            )
            params: Dict[str, object] = {}
            if canonical is not None:
                sql += " AND p.symbol=:symbol"
                params["symbol"] = canonical
            if period_start is not None:
                sql += " AND p.closed_at>=:period_start"
                params["period_start"] = period_start
            sql += " ORDER BY p.closed_at ASC, p.position_id ASC"
            result = await conn.execute(text(sql), params)
            rows = [dict(row._mapping) for row in result.fetchall()]
    except HTTPException:
        raise
    except Exception as exc:
        persistence_state.mark_runtime_error(str(exc))
        raise HTTPException(
            status_code=503,
            detail={"status": "UNAVAILABLE", "reason": "paper persistence failed"},
        ) from exc

    initial_capital = Decimal(str(account._mapping["initial_capital"]))
    metrics = calculate_paper_performance_metrics(rows, initial_capital)
    return {
        "status": "OK",
        "validation": "SERVER_PAPER_PERFORMANCE_ANALYTICS_V1",
        "period_validation": "SERVER_PAPER_PERFORMANCE_PERIODS_V1",
        "symbol": canonical,
        "period": normalized_period,
        "period_start": period_start.isoformat() if period_start is not None else None,
        "metrics": metrics,
        "paper_only": True,
        "execution": False,
    }


PAPER_PERFORMANCE_BREAKDOWN_GROUPS = {"SYMBOL", "SIDE"}


def build_paper_performance_breakdown(
    closed_positions: List[Dict[str, object]],
    initial_capital: Decimal,
    group_by: str,
) -> List[Dict[str, object]]:
    """Group persisted closed paper trades by an actually stored dimension."""
    normalized = group_by.upper()
    if normalized not in PAPER_PERFORMANCE_BREAKDOWN_GROUPS:
        raise ValueError("unsupported paper performance breakdown")
    grouped: Dict[str, List[Dict[str, object]]] = {}
    field = "symbol" if normalized == "SYMBOL" else "side"
    for position in closed_positions:
        raw_value = position.get(field)
        if raw_value is None or not str(raw_value).strip():
            raise ValueError(f"closed paper trade missing {field}")
        key = str(raw_value).upper()
        grouped.setdefault(key, []).append(position)
    return [
        {
            "group": key,
            "metrics": calculate_paper_performance_metrics(grouped[key], initial_capital),
        }
        for key in sorted(grouped)
    ]


# V16-M5B30B — full persisted paper performance dimensions.
# New positions persist entry-time TIMEFRAME / SESSION / REGIME in a dedicated
# immutable context row. Historical positions remain explicit UNKNOWN rather than
# being reconstructed from current market conditions.
PAPER_PERFORMANCE_MATRIX_VERSION = "SERVER_PAPER_PERFORMANCE_MATRIX_V2"
PAPER_PERFORMANCE_MATRIX_MIN_SAMPLE = 20


def paper_strategy_from_source(source: object) -> Dict[str, Optional[str]]:
    """Parse legacy strategy attribution without inventing missing metadata."""
    raw = str(source or "").strip()
    if not raw.startswith("strategy:"):
        return {"strategy_id": None, "strategy_version": None}
    attribution = raw[len("strategy:"):]
    if "@" not in attribution:
        return {"strategy_id": attribution or None, "strategy_version": None}
    strategy_id, strategy_version = attribution.split("@", 1)
    return {
        "strategy_id": strategy_id or None,
        "strategy_version": strategy_version or None,
    }


def paper_performance_matrix_label(metrics: Dict[str, object]) -> str:
    """Conservative evidence label; small samples never receive an edge claim."""
    trades = int(str(metrics.get("closed_trades") or 0))
    if trades < PAPER_PERFORMANCE_MATRIX_MIN_SAMPLE:
        return "INSUFFICIENT_SAMPLE"
    expectancy_raw = metrics.get("expectancy")
    pf_raw = metrics.get("profit_factor")
    expectancy = Decimal(str(expectancy_raw)) if expectancy_raw is not None else Decimal("0")
    profit_factor = Decimal(str(pf_raw)) if pf_raw is not None else None
    if expectancy <= 0 or (profit_factor is not None and profit_factor < Decimal("1")):
        return "WEAK"
    if trades >= 50 and profit_factor is not None and profit_factor >= Decimal("1.25"):
        return "ROBUST"
    return "PROMISING"


def build_paper_performance_matrix(
    closed_positions: List[Dict[str, object]], initial_capital: Decimal
) -> List[Dict[str, object]]:
    """Build SYMBOL x STRATEGY x TF x SESSION x REGIME from persisted context."""
    grouped: Dict[
        tuple[str, str, str, str, str, str], List[Dict[str, object]]
    ] = {}
    for position in closed_positions:
        symbol = str(position.get("symbol") or "").upper()
        legacy = paper_strategy_from_source(position.get("source"))
        strategy_id = str(position.get("strategy_id") or legacy["strategy_id"] or "")
        strategy_version = str(
            position.get("strategy_version") or legacy["strategy_version"] or "UNKNOWN"
        )
        if not symbol or not strategy_id:
            continue
        timeframe = str(position.get("timeframe") or "UNKNOWN")
        session = str(position.get("session") or "UNKNOWN")
        regime = str(position.get("market_regime") or "UNKNOWN")
        key: tuple[str, str, str, str, str, str] = (
            symbol, strategy_id, strategy_version, timeframe, session, regime
        )
        grouped.setdefault(key, []).append(position)

    cells: List[Dict[str, object]] = []
    for key in sorted(grouped):
        symbol, strategy_id, strategy_version, timeframe, session, regime = key
        metrics = calculate_paper_performance_metrics(grouped[key], initial_capital)
        cells.append(
            {
                "symbol": symbol,
                "strategy_id": strategy_id,
                "strategy_version": strategy_version,
                "timeframe": None if timeframe == "UNKNOWN" else timeframe,
                "session": None if session == "UNKNOWN" else session,
                "regime": None if regime == "UNKNOWN" else regime,
                "context_complete": all(
                    value != "UNKNOWN" for value in (timeframe, session, regime)
                ),
                "metrics": metrics,
                "evidence": paper_performance_matrix_label(metrics),
            }
        )
    return cells


def paper_performance_context_coverage(rows: List[Dict[str, object]]) -> Dict[str, int]:
    """Expose historical coverage without pretending old rows have entry context."""
    total = len(rows)
    complete = sum(
        1
        for row in rows
        if row.get("timeframe") is not None
        and row.get("session") is not None
        and row.get("market_regime") is not None
    )
    return {
        "closed_positions": total,
        "context_complete": complete,
        "legacy_missing": total - complete,
    }


@api_router.get("/paper/performance/matrix")
async def get_paper_performance_matrix(period: str = "ALL") -> Dict[str, object]:
    """Return an evidence-only matrix from persisted CLOSED paper trades."""
    if not persistence_state.ready:
        raise HTTPException(
            status_code=503,
            detail={"status": "UNAVAILABLE", "reason": "persistence not ready"},
        )
    normalized_period = period.upper()
    try:
        period_start = paper_performance_period_start(normalized_period)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={"status": "INVALID", "reason": "unsupported performance period"},
        ) from exc
    try:
        async with engine.connect() as conn:
            account_result = await conn.execute(
                text(
                    "SELECT initial_capital FROM paper_account "
                    "WHERE account_id=:account_id"
                ),
                {"account_id": PAPER_ACCOUNT_ID},
            )
            account = account_result.fetchone()
            if account is None:
                raise HTTPException(
                    status_code=503,
                    detail={"status": "UNAVAILABLE", "reason": "paper account not initialized"},
                )
            sql = (
                "SELECT p.position_id, p.symbol, p.side, p.entry, p.close_price, p.size, "
                "p.risk_money, p.closed_at, p.source, fx.quote_to_usd, "
                "ctx.strategy_id, ctx.strategy_version, ctx.timeframe, ctx.session, "
                "ctx.market_regime "
                "FROM paper_positions p LEFT JOIN paper_fx_conversion_snapshots fx "
                "ON fx.position_id=p.position_id AND fx.phase='CLOSE' "
                "LEFT JOIN paper_position_context ctx ON ctx.position_id=p.position_id "
                "WHERE status='CLOSED' AND p.close_price IS NOT NULL"
            )
            params: Dict[str, object] = {}
            if period_start is not None:
                sql += " AND p.closed_at>=:period_start"
                params["period_start"] = period_start
            sql += " ORDER BY p.closed_at ASC, p.position_id ASC"
            result = await conn.execute(text(sql), params)
            rows = [dict(row._mapping) for row in result.fetchall()]
    except HTTPException:
        raise
    except Exception as exc:
        persistence_state.mark_runtime_error(str(exc))
        raise HTTPException(
            status_code=503,
            detail={"status": "UNAVAILABLE", "reason": "paper persistence failed"},
        ) from exc

    initial_capital = Decimal(str(account._mapping["initial_capital"]))
    cells = build_paper_performance_matrix(rows, initial_capital)
    coverage = paper_performance_context_coverage(rows)
    return {
        "status": "OK",
        "validation": PAPER_PERFORMANCE_MATRIX_VERSION,
        "period": normalized_period,
        "period_start": period_start.isoformat() if period_start is not None else None,
        "minimum_sample": PAPER_PERFORMANCE_MATRIX_MIN_SAMPLE,
        "dimensions_available": [
            "SYMBOL", "STRATEGY", "TIMEFRAME", "SESSION", "REGIME"
        ],
        "dimensions_deferred": [],
        "context_coverage": coverage,
        "cells": cells,
        "paper_only": True,
        "execution": False,
    }


# V16-M5B30D — Adaptive Edge Ranking Engine V1.
# Observation-only: ranks persisted paper evidence but never changes signal,
# sizing, risk, order generation, or execution behavior.
ADAPTIVE_EDGE_RANKING_VERSION = "SERVER_ADAPTIVE_EDGE_RANKING_V1"
ADAPTIVE_EDGE_MIN_SAMPLE = PAPER_PERFORMANCE_MATRIX_MIN_SAMPLE


def _edge_decimal(value: object, default: Decimal = Decimal("0")) -> Decimal:
    if value is None:
        return default
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return default


def _edge_clamp(value: Decimal, low: Decimal, high: Decimal) -> Decimal:
    return max(low, min(high, value))


def adaptive_edge_score(metrics: Dict[str, object]) -> Dict[str, object]:
    """Return a transparent heuristic score from persisted paper metrics only."""
    trades = int(str(metrics.get("closed_trades") or 0))
    if trades < ADAPTIVE_EDGE_MIN_SAMPLE:
        return {
            "eligible": False,
            "score": None,
            "reason": "INSUFFICIENT_SAMPLE",
            "components": {},
        }

    profit_factor = _edge_decimal(metrics.get("profit_factor"))
    win_rate = _edge_decimal(metrics.get("win_rate_percent"))
    average_rr = _edge_decimal(metrics.get("average_realized_rr"))
    drawdown = _edge_decimal(metrics.get("max_drawdown_percent"))

    # 20 points: evidence depth. Full credit at 50 persisted closed trades.
    sample_points = _edge_clamp(
        Decimal(trades - ADAPTIVE_EDGE_MIN_SAMPLE)
        / Decimal(50 - ADAPTIVE_EDGE_MIN_SAMPLE)
        * Decimal("20"),
        Decimal("0"),
        Decimal("20"),
    )
    # 25 points: profit factor. PF 1.0 starts contributing; PF 2.0 is capped.
    pf_points = _edge_clamp(
        (profit_factor - Decimal("1")) * Decimal("25"),
        Decimal("0"),
        Decimal("25"),
    )
    # 25 points: realized expectancy in R, using the existing average RR metric.
    rr_points = _edge_clamp(
        average_rr * Decimal("25"), Decimal("0"), Decimal("25")
    )
    # 10 points: win-rate support. It is deliberately a smaller component.
    win_points = _edge_clamp(
        win_rate / Decimal("100") * Decimal("10"),
        Decimal("0"),
        Decimal("10"),
    )
    # 20 points: drawdown discipline. Zero DD gets full credit; >=20% gets zero.
    drawdown_points = _edge_clamp(
        Decimal("20") - drawdown, Decimal("0"), Decimal("20")
    )
    total = sample_points + pf_points + rr_points + win_points + drawdown_points
    score = total.quantize(Decimal("0.01"))
    return {
        "eligible": True,
        "score": str(score),
        "reason": "EVIDENCE_RANKABLE",
        "components": {
            "sample": str(sample_points.quantize(Decimal("0.01"))),
            "profit_factor": str(pf_points.quantize(Decimal("0.01"))),
            "realized_rr": str(rr_points.quantize(Decimal("0.01"))),
            "win_rate": str(win_points.quantize(Decimal("0.01"))),
            "drawdown": str(drawdown_points.quantize(Decimal("0.01"))),
        },
    }


def build_adaptive_edge_ranking(
    cells: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    """Rank strategies only against peers in the same persisted market context."""
    groups: Dict[tuple[str, str, str, str], List[Dict[str, object]]] = {}
    for cell in cells:
        if not bool(cell.get("context_complete")):
            continue
        symbol = str(cell.get("symbol") or "")
        timeframe = str(cell.get("timeframe") or "")
        session = str(cell.get("session") or "")
        regime = str(cell.get("regime") or "")
        if not all((symbol, timeframe, session, regime)):
            continue
        metrics_raw = cell.get("metrics")
        metrics = metrics_raw if isinstance(metrics_raw, dict) else {}
        scored = adaptive_edge_score(metrics)
        item = dict(cell)
        item["edge"] = scored
        groups.setdefault((symbol, timeframe, session, regime), []).append(item)

    rankings: List[Dict[str, object]] = []
    for context in sorted(groups):
        candidates = groups[context]
        rankable: List[Dict[str, object]] = []
        for item in candidates:
            edge = item.get("edge")
            if isinstance(edge, dict) and bool(edge.get("eligible")):
                rankable.append(item)

        def ranking_key(item: Dict[str, object]) -> tuple[Decimal, str, str]:
            edge = item.get("edge")
            score = edge.get("score") if isinstance(edge, dict) else None
            return (
                -_edge_decimal(score),
                str(item.get("strategy_id") or ""),
                str(item.get("strategy_version") or ""),
            )

        rankable.sort(key=ranking_key)
        ranked_entries: List[Dict[str, object]] = []
        for rank, item in enumerate(rankable, start=1):
            ranked = dict(item)
            ranked["rank"] = rank
            ranked_entries.append(ranked)
        unranked = [item for item in candidates if item not in rankable]
        unranked.sort(key=lambda item: str(item.get("strategy_id") or ""))
        for item in unranked:
            pending = dict(item)
            pending["rank"] = None
            ranked_entries.append(pending)
        rankings.append(
            {
                "symbol": context[0],
                "timeframe": context[1],
                "session": context[2],
                "regime": context[3],
                "ranked_strategies": ranked_entries,
            }
        )
    return rankings


@api_router.get("/paper/adaptive-edge/ranking")
async def get_adaptive_edge_ranking(period: str = "ALL") -> Dict[str, object]:
    """Expose observation-only adaptive ranking from the validated matrix."""
    matrix = await get_paper_performance_matrix(period)
    cells_raw = matrix.get("cells")
    cells: List[Dict[str, object]] = []
    if isinstance(cells_raw, list):
        cells = [item for item in cells_raw if isinstance(item, dict)]
    rankings = build_adaptive_edge_ranking(cells)
    return {
        "status": "OK",
        "validation": ADAPTIVE_EDGE_RANKING_VERSION,
        "period": matrix["period"],
        "period_start": matrix["period_start"],
        "minimum_sample": ADAPTIVE_EDGE_MIN_SAMPLE,
        "ranking_scope": "SYMBOL_X_TIMEFRAME_X_SESSION_X_REGIME",
        "score_scale": "0_TO_100_HEURISTIC",
        "score_components_max": {
            "sample": 20,
            "profit_factor": 25,
            "realized_rr": 25,
            "win_rate": 10,
            "drawdown": 20,
        },
        "context_coverage": matrix["context_coverage"],
        "rankings": rankings,
        "paper_only": True,
        "observation_only": True,
        "execution": False,
    }


# V16-M5B30E — Adaptive Edge Gating Shadow Mode.
# Advisory only: this evaluates what an evidence gate WOULD do. It never changes
# signal validity, sizing, risk guards, order generation, or paper execution.
ADAPTIVE_EDGE_SHADOW_VERSION = "SERVER_ADAPTIVE_EDGE_GATING_SHADOW_V1"


def adaptive_edge_shadow_decision(
    rankings: List[Dict[str, object]],
    symbol: str,
    strategy_id: str,
    timeframe: str,
    session: str,
    regime: str,
) -> Dict[str, object]:
    """Return a fail-neutral hypothetical gate from persisted ranking evidence."""
    canonical = symbol.upper().replace("/", "-")
    target_context = (canonical, timeframe, session, regime)
    for ranking in rankings:
        context = (
            str(ranking.get("symbol") or "").upper(),
            str(ranking.get("timeframe") or ""),
            str(ranking.get("session") or ""),
            str(ranking.get("regime") or ""),
        )
        if context != target_context:
            continue
        entries_raw = ranking.get("ranked_strategies")
        entries = entries_raw if isinstance(entries_raw, list) else []
        for raw_entry in entries:
            if not isinstance(raw_entry, dict):
                continue
            if str(raw_entry.get("strategy_id") or "") != strategy_id:
                continue
            evidence = str(raw_entry.get("evidence") or "INSUFFICIENT_SAMPLE")
            edge_raw = raw_entry.get("edge")
            edge = edge_raw if isinstance(edge_raw, dict) else {}
            rank_raw = raw_entry.get("rank")
            rank = int(str(rank_raw)) if rank_raw is not None else None
            if evidence == "WEAK":
                action = "WOULD_BLOCK"
                reason = "WEAK_PERSISTED_EDGE"
            elif evidence in {"PROMISING", "ROBUST"} and rank == 1:
                action = "WOULD_FAVOR"
                reason = "TOP_RANKED_POSITIVE_EDGE"
            elif evidence == "INSUFFICIENT_SAMPLE":
                action = "NEUTRAL"
                reason = "INSUFFICIENT_SAMPLE"
            else:
                action = "NEUTRAL"
                reason = "EVIDENCE_NOT_DECISIVE"
            components_raw = edge.get("components")
            components = components_raw if isinstance(components_raw, dict) else {}
            metrics_raw = raw_entry.get("metrics")
            metrics = metrics_raw if isinstance(metrics_raw, dict) else {}
            return {
                "action": action,
                "reason": reason,
                "evidence": evidence,
                "rank": rank,
                "score": edge.get("score"),
                "eligible": bool(edge.get("eligible")),
                "components": components,
                "metrics": metrics,
            }
    return {
        "action": "NEUTRAL",
        "reason": "NO_MATCHING_PERSISTED_CONTEXT",
        "evidence": None,
        "rank": None,
        "score": None,
        "eligible": False,
        "components": {},
        "metrics": {},
    }


@api_router.get("/paper/adaptive-edge/shadow")
async def get_adaptive_edge_shadow(
    symbol: str,
    strategy_id: str,
    timeframe: str,
    session: str,
    regime: str,
    period: str = "ALL",
) -> Dict[str, object]:
    """Expose the hypothetical adaptive gate without changing execution behavior."""
    ranking_payload = await get_adaptive_edge_ranking(period)
    rankings_raw = ranking_payload.get("rankings")
    rankings: List[Dict[str, object]] = []
    if isinstance(rankings_raw, list):
        rankings = [item for item in rankings_raw if isinstance(item, dict)]
    decision = adaptive_edge_shadow_decision(
        rankings, symbol, strategy_id, timeframe, session, regime
    )
    canonical = symbol.upper().replace("/", "-")
    created_at = utcnow()
    decision_seed = (
        f"{canonical}|{strategy_id}|{timeframe}|{session}|{regime}|"
        f"{ranking_payload.get('period')}|{created_at.isoformat()}"
    )
    decision_id = "edge-shadow-" + hashlib.sha256(decision_seed.encode()).hexdigest()[:24]
    score_raw = decision.get("score")
    score = Decimal(str(score_raw)) if score_raw is not None else None
    try:
        async with engine.begin() as conn:
            await conn.execute(
                pg_insert(paper_adaptive_edge_shadow_table).values(
                    decision_id=decision_id,
                    symbol=canonical,
                    strategy_id=strategy_id,
                    timeframe=timeframe,
                    session=session,
                    market_regime=regime,
                    period=str(ranking_payload.get("period") or period.upper()),
                    action=str(decision.get("action") or "NEUTRAL"),
                    reason=str(decision.get("reason") or "UNKNOWN"),
                    evidence=(
                        str(decision.get("evidence"))
                        if decision.get("evidence") is not None
                        else None
                    ),
                    rank=(
                        int(str(decision.get("rank")))
                        if decision.get("rank") is not None
                        else None
                    ),
                    score=score,
                    created_at=created_at,
                )
            )
    except Exception as exc:
        persistence_state.mark_runtime_error(str(exc))
        raise HTTPException(
            status_code=503,
            detail={"status": "UNAVAILABLE", "reason": "shadow audit persistence failed"},
        ) from exc
    return {
        "status": "OK",
        "validation": ADAPTIVE_EDGE_SHADOW_VERSION,
        "decision_id": decision_id,
        "period": ranking_payload.get("period"),
        "context": {
            "symbol": symbol.upper().replace("/", "-"),
            "strategy_id": strategy_id,
            "timeframe": timeframe,
            "session": session,
            "regime": regime,
        },
        "shadow_gate": decision,
        "policy": {
            "WEAK": "WOULD_BLOCK",
            "TOP_RANKED_PROMISING_OR_ROBUST": "WOULD_FAVOR",
            "INSUFFICIENT_SAMPLE": "NEUTRAL",
            "NO_MATCHING_CONTEXT": "NEUTRAL",
        },
        "paper_only": True,
        "shadow_mode": True,
        "changes_execution": False,
        "execution": False,
    }


# V16-M5B30F — Active Adaptive Edge Gate for PAPER execution only.
ADAPTIVE_EDGE_ACTIVE_VERSION = "SERVER_ADAPTIVE_EDGE_ACTIVE_PAPER_V1"


async def evaluate_adaptive_edge_active_gate(
    symbol: str, strategy_id: str, timeframe: object, session: object,
    regime: object, period: str = "ALL",
) -> Dict[str, object]:
    canonical = symbol.upper().replace("/", "-")
    context_values = (timeframe, session, regime)
    if any(value is None or not str(value).strip() for value in context_values):
        return {
            "action": "ALLOWED", "reason": "CONTEXT_NOT_PERSISTED_FAIL_NEUTRAL",
            "evidence": None, "rank": None, "score": None, "active": True,
        }
    tf, sess, reg = str(timeframe), str(session), str(regime)
    try:
        payload = await get_adaptive_edge_ranking(period)
        raw = payload.get("rankings")
        rankings: List[Dict[str, object]] = []
        if isinstance(raw, list):
            rankings = [item for item in raw if isinstance(item, dict)]
        advisory = adaptive_edge_shadow_decision(
            rankings, canonical, strategy_id, tf, sess, reg
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("Adaptive edge active gate fail-neutral: %s", type(exc).__name__)
        return {
            "action": "ALLOWED", "reason": "RANKING_UNAVAILABLE_FAIL_NEUTRAL",
            "evidence": None, "rank": None, "score": None, "active": True,
        }
    shadow_action = str(advisory.get("action") or "NEUTRAL")
    if shadow_action == "WOULD_BLOCK":
        action, reason = "BLOCKED", "WEAK_PERSISTED_EDGE"
    elif shadow_action == "WOULD_FAVOR":
        action, reason = "FAVORED", "TOP_RANKED_POSITIVE_EDGE"
    else:
        action = "ALLOWED"
        reason = str(advisory.get("reason") or "EVIDENCE_NOT_DECISIVE")
    created_at = utcnow()
    seed = f"{canonical}|{strategy_id}|{tf}|{sess}|{reg}|{period}|{created_at.isoformat()}"
    decision_id = "edge-active-" + hashlib.sha256(seed.encode()).hexdigest()[:24]
    score_raw = advisory.get("score")
    score = Decimal(str(score_raw)) if score_raw is not None else None
    components_raw = advisory.get("components")
    components = components_raw if isinstance(components_raw, dict) else {}
    metrics_raw = advisory.get("metrics")
    metrics = metrics_raw if isinstance(metrics_raw, dict) else {}
    try:
        async with engine.begin() as conn:
            await conn.execute(pg_insert(paper_adaptive_edge_active_table).values(
                decision_id=decision_id, symbol=canonical, strategy_id=strategy_id,
                timeframe=tf, session=sess, market_regime=reg, period=period.upper(),
                action=action, reason=reason,
                evidence=(str(advisory.get("evidence"))
                          if advisory.get("evidence") is not None else None),
                rank=(int(str(advisory.get("rank")))
                      if advisory.get("rank") is not None else None),
                score=score, created_at=created_at,
            ))
            await conn.execute(
                pg_insert(paper_adaptive_edge_explainability_table).values(
                    decision_id=decision_id,
                    components_json=json.dumps(components, sort_keys=True, default=str),
                    metrics_json=json.dumps(metrics, sort_keys=True, default=str),
                    policy_version=ADAPTIVE_EDGE_ACTIVE_VERSION,
                    created_at=created_at,
                )
            )
    except Exception as exc:  # noqa: BLE001
        log.warning("Adaptive edge active audit unavailable: %s", type(exc).__name__)
    return {**advisory, "decision_id": decision_id, "action": action,
            "reason": reason, "active": True, "paper_only": True}


@api_router.get("/paper/adaptive-edge/active")
async def get_adaptive_edge_active(
    symbol: str, strategy_id: str, timeframe: str, session: str, regime: str,
    period: str = "ALL",
) -> Dict[str, object]:
    gate = await evaluate_adaptive_edge_active_gate(
        symbol, strategy_id, timeframe, session, regime, period
    )
    return {
        "status": "OK", "validation": ADAPTIVE_EDGE_ACTIVE_VERSION, "gate": gate,
        "policy": {
            "WEAK": "BLOCKED",
            "TOP_RANKED_PROMISING_OR_ROBUST": "FAVORED",
            "INSUFFICIENT_SAMPLE": "ALLOWED_NEUTRAL",
            "MISSING_CONTEXT_OR_RANKING": "ALLOWED_FAIL_NEUTRAL",
        },
        "paper_only": True, "live_trading": False,
    }


ADAPTIVE_EDGE_REPLAY_VERSION = "SERVER_ADAPTIVE_EDGE_DECISION_REPLAY_V1"

ADAPTIVE_EDGE_IMPACT_VERSION = "SERVER_ADAPTIVE_EDGE_IMPACT_VALIDATION_V1"


async def persist_adaptive_edge_trade_link(
    position_id: str, gate: Dict[str, object]
) -> None:
    """Audit-only link from an executed paper position to its active edge decision."""
    decision_id = gate.get("decision_id")
    action = str(gate.get("action") or "ALLOWED").upper()
    if not isinstance(decision_id, str) or not decision_id:
        return
    if action not in {"ALLOWED", "FAVORED"}:
        return
    try:
        async with engine.begin() as conn:
            await conn.execute(
                pg_insert(paper_adaptive_edge_trade_link_table).values(
                    position_id=position_id,
                    decision_id=decision_id,
                    gate_action=action,
                    linked_at=utcnow(),
                ).on_conflict_do_nothing(index_elements=["position_id"])
            )
    except Exception as exc:  # noqa: BLE001
        log.warning("Adaptive edge trade link unavailable: %s", type(exc).__name__)




def _impact_metrics_or_empty(
    positions: List[Dict[str, object]], initial_capital: Decimal
) -> Dict[str, object]:
    if not positions:
        return {
            "closed_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate_percent": "0",
            "profit_factor": None,
            "expectancy": "0",
            "average_realized_rr": "0",
            "net_pnl": "0",
            "max_drawdown_percent": "0",
        }
    return calculate_paper_performance_metrics(positions, initial_capital)


@api_router.get("/paper/adaptive-edge/impact-validation")
async def get_adaptive_edge_impact_validation(period: str = "ALL") -> Dict[str, object]:
    """Observed paper impact only; blocked-trade counterfactual P&L is never invented."""
    if not persistence_state.ready:
        raise HTTPException(
            status_code=503,
            detail={"status": "UNAVAILABLE", "reason": "persistence not ready"},
        )
    normalized_period = period.upper()
    try:
        period_start = paper_performance_period_start(normalized_period, utcnow())
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={"status": "INVALID", "reason": "unsupported performance period"},
        ) from exc
    params: Dict[str, object] = {}
    date_filter = ""
    if period_start is not None:
        date_filter = " AND p.closed_at>=:period_start"
        params["period_start"] = period_start
    try:
        async with engine.connect() as conn:
            account_result = await conn.execute(
                text(
                    "SELECT initial_capital FROM paper_account "
                    "WHERE account_id=:account_id"
                ),
                {"account_id": PAPER_ACCOUNT_ID},
            )
            account = account_result.fetchone()
            if account is None:
                raise HTTPException(
                    status_code=503,
                    detail={"status": "UNAVAILABLE", "reason": "paper account not initialized"},
                )
            sql = (
                "SELECT p.position_id, p.symbol, p.side, p.entry, p.close_price, p.size, "
                "p.risk_money, p.closed_at, p.source, fx.quote_to_usd, "
                "l.gate_action, l.decision_id "
                "FROM paper_positions p "
                "LEFT JOIN paper_fx_conversion_snapshots fx "
                "ON fx.position_id=p.position_id AND fx.phase='CLOSE' "
                "LEFT JOIN paper_adaptive_edge_trade_link l "
                "ON l.position_id=p.position_id "
                "WHERE status='CLOSED' AND p.close_price IS NOT NULL"
                + date_filter
                + " ORDER BY p.closed_at ASC, p.position_id ASC"
            )
            result = await conn.execute(text(sql), params)
            rows = [dict(row._mapping) for row in result.fetchall()]
            blocked_sql = (
                "SELECT COUNT(*) AS count FROM paper_adaptive_edge_active "
                "WHERE action='BLOCKED'"
            )
            blocked_params: Dict[str, object] = {}
            if period_start is not None:
                blocked_sql += " AND created_at>=:period_start"
                blocked_params["period_start"] = period_start
            blocked_result = await conn.execute(text(blocked_sql), blocked_params)
            blocked_row = blocked_result.fetchone()
    except HTTPException:
        raise
    except Exception as exc:
        persistence_state.mark_runtime_error(str(exc))
        raise HTTPException(
            status_code=503,
            detail={"status": "UNAVAILABLE", "reason": "impact validation failed"},
        ) from exc

    initial_capital = Decimal(str(account._mapping["initial_capital"]))
    allowed = [row for row in rows if str(row.get("gate_action") or "") == "ALLOWED"]
    favored = [row for row in rows if str(row.get("gate_action") or "") == "FAVORED"]
    linked = allowed + favored
    historical = [row for row in rows if row.get("decision_id") is None]
    blocked_count = int(blocked_row._mapping["count"]) if blocked_row is not None else 0

    return {
        "status": "OK",
        "validation": ADAPTIVE_EDGE_IMPACT_VERSION,
        "period": normalized_period,
        "period_start": period_start.isoformat() if period_start is not None else None,
        "observed": {
            "linked_executed": _impact_metrics_or_empty(linked, initial_capital),
            "allowed": _impact_metrics_or_empty(allowed, initial_capital),
            "favored": _impact_metrics_or_empty(favored, initial_capital),
            "historical_unlinked": _impact_metrics_or_empty(historical, initial_capital),
        },
        "coverage": {
            "closed_positions": len(rows),
            "linked_closed_positions": len(linked),
            "historical_unlinked_closed_positions": len(historical),
            "blocked_gate_decisions": blocked_count,
        },
        "interpretation": {
            "historical_baseline_is_causal_control": False,
            "blocked_trade_outcomes_measured": False,
            "blocked_counterfactual_status": "NOT_MEASURED",
            "reason": (
                "Blocked entries create no paper position, so hypothetical P&L is not "
                "fabricated. This endpoint validates observed executed cohorts only."
            ),
        },
        "paper_only": True,
        "live_trading": False,
        "changes_execution": False,
    }


@api_router.get("/paper/adaptive-edge/decision-replay")
async def get_adaptive_edge_decision_replay(
    limit: int = 50,
    symbol: Optional[str] = None,
    action: Optional[str] = None,
) -> Dict[str, object]:
    """Read persisted active gate decisions with exact evidence snapshots."""
    if not persistence_state.ready:
        raise HTTPException(
            status_code=503,
            detail={"status": "UNAVAILABLE", "reason": "persistence not ready"},
        )
    safe_limit = max(1, min(limit, 200))
    clauses: List[str] = []
    params: Dict[str, object] = {"limit": safe_limit}
    if symbol:
        clauses.append("a.symbol = :symbol")
        params["symbol"] = symbol.upper().replace("/", "-")
    if action:
        normalized_action = action.upper()
        if normalized_action not in {"ALLOWED", "BLOCKED", "FAVORED"}:
            raise HTTPException(
                status_code=400,
                detail={"status": "INVALID", "reason": "unsupported adaptive edge action"},
            )
        clauses.append("a.action = :action")
        params["action"] = normalized_action
    sql = (
        "SELECT a.*, e.components_json, e.metrics_json, e.policy_version "
        "FROM paper_adaptive_edge_active a "
        "LEFT JOIN paper_adaptive_edge_explainability e "
        "ON e.decision_id = a.decision_id"
    )
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY a.created_at DESC, a.decision_id DESC LIMIT :limit"
    try:
        async with engine.connect() as conn:
            result = await conn.execute(text(sql), params)
            items = [dict(row._mapping) for row in result.fetchall()]
    except Exception as exc:
        persistence_state.mark_runtime_error(str(exc))
        raise HTTPException(
            status_code=503,
            detail={"status": "UNAVAILABLE", "reason": "adaptive edge replay failed"},
        ) from exc
    for item in items:
        created_at = item.get("created_at")
        if isinstance(created_at, datetime):
            item["created_at"] = created_at.isoformat()
        score_value = item.get("score")
        if score_value is not None:
            item["score"] = str(score_value)
        for source_key, target_key in (
            ("components_json", "components"),
            ("metrics_json", "metrics"),
        ):
            raw = item.pop(source_key, None)
            parsed: Dict[str, object] = {}
            if isinstance(raw, str):
                try:
                    loaded = json.loads(raw)
                    if isinstance(loaded, dict):
                        parsed = loaded
                except json.JSONDecodeError:
                    parsed = {}
            item[target_key] = parsed
        item["snapshot_available"] = bool(
            item.get("components") or item.get("metrics") or item.get("policy_version")
        )
    return {
        "status": "OK",
        "validation": ADAPTIVE_EDGE_REPLAY_VERSION,
        "count": len(items),
        "limit": safe_limit,
        "items": items,
        "paper_only": True,
        "live_trading": False,
        "read_only": True,
    }


@api_router.get("/paper/performance/breakdown")
async def get_paper_performance_breakdown(
    group_by: str = "SYMBOL",
    period: str = "ALL",
) -> Dict[str, object]:
    """Break down persisted CLOSED paper performance by stored symbol or side."""
    if not persistence_state.ready:
        raise HTTPException(
            status_code=503,
            detail={"status": "UNAVAILABLE", "reason": "persistence not ready"},
        )
    normalized_group = group_by.upper()
    if normalized_group not in PAPER_PERFORMANCE_BREAKDOWN_GROUPS:
        raise HTTPException(
            status_code=400,
            detail={"status": "INVALID", "reason": "unsupported performance breakdown"},
        )
    normalized_period = period.upper()
    try:
        period_start = paper_performance_period_start(normalized_period)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={"status": "INVALID", "reason": "unsupported performance period"},
        ) from exc
    try:
        async with engine.connect() as conn:
            account_result = await conn.execute(
                text(
                    "SELECT initial_capital FROM paper_account "
                    "WHERE account_id=:account_id"
                ),
                {"account_id": PAPER_ACCOUNT_ID},
            )
            account = account_result.fetchone()
            if account is None:
                raise HTTPException(
                    status_code=503,
                    detail={
                        "status": "UNAVAILABLE",
                        "reason": "paper account not initialized",
                    },
                )
            sql = (
                "SELECT p.position_id, p.symbol, p.side, p.entry, p.close_price, p.size, "
                "p.risk_money, p.closed_at, fx.quote_to_usd "
                "FROM paper_positions p LEFT JOIN paper_fx_conversion_snapshots fx "
                "ON fx.position_id=p.position_id AND fx.phase='CLOSE' "
                "WHERE status='CLOSED' AND p.close_price IS NOT NULL"
            )
            params: Dict[str, object] = {}
            if period_start is not None:
                sql += " AND p.closed_at>=:period_start"
                params["period_start"] = period_start
            sql += " ORDER BY p.closed_at ASC, p.position_id ASC"
            result = await conn.execute(text(sql), params)
            rows = [dict(row._mapping) for row in result.fetchall()]
    except HTTPException:
        raise
    except Exception as exc:
        persistence_state.mark_runtime_error(str(exc))
        raise HTTPException(
            status_code=503,
            detail={"status": "UNAVAILABLE", "reason": "paper persistence failed"},
        ) from exc

    initial_capital = Decimal(str(account._mapping["initial_capital"]))
    groups = build_paper_performance_breakdown(rows, initial_capital, normalized_group)
    return {
        "status": "OK",
        "validation": "SERVER_PAPER_PERFORMANCE_BREAKDOWN_V1",
        "group_by": normalized_group,
        "period": normalized_period,
        "period_start": period_start.isoformat() if period_start is not None else None,
        "groups": groups,
        "dimensions_available": ["SYMBOL", "SIDE"],
        "dimensions_deferred": ["TIMEFRAME", "STRATEGY", "SESSION", "REGIME"],
        "paper_only": True,
        "execution": False,
    }


@api_router.get("/paper/positions")
async def list_paper_positions(status_filter: Optional[str] = None) -> Dict[str, object]:
    if not persistence_state.ready:
        raise HTTPException(
            status_code=503, detail={"status": "UNAVAILABLE", "reason": "persistence not ready"}
        )
    sql = (
        "SELECT p.*, fx.quote_currency AS fx_quote_currency, "
        "fx.quote_to_usd AS fx_quote_to_usd_close, "
        "fx.conversion_symbol AS fx_conversion_symbol_close, "
        "fx.conversion_price AS fx_conversion_price_close, "
        "fx.inverse AS fx_conversion_inverse_close, "
        "fx.source AS fx_conversion_source_close, "
        "fx.source_timestamp AS fx_conversion_source_timestamp_close "
        "FROM paper_positions p LEFT JOIN paper_fx_conversion_snapshots fx "
        "ON fx.position_id=p.position_id AND fx.phase='CLOSE'"
    )
    params: Dict[str, object] = {}
    if status_filter is not None:
        if status_filter not in {"OPEN", "CLOSED", "CONFLICT"}:
            raise HTTPException(
                status_code=400, detail={"status": "INVALID", "reason": "STATUS_INVALID"}
            )
        sql += " WHERE p.status = :status"
        params["status"] = status_filter
    sql += " ORDER BY p.opened_at DESC, p.position_id DESC"
    try:
        async with engine.connect() as conn:
            result = await conn.execute(text(sql), params)
            rows = []
            for row in result.fetchall():
                payload = paper_position_to_dict(row)
                data = row._mapping
                if (
                    data.get("status") == "CLOSED"
                    and data.get("close_price") is not None
                    and paper_position_quote_to_usd_required(str(data.get("symbol") or ""))
                ):
                    rate = data.get("fx_quote_to_usd_close")
                    if rate is None:
                        payload["realized_pnl"] = None
                        payload["fx_conversion_status"] = "CLOSE_SNAPSHOT_MISSING"
                    else:
                        pnl = calculate_paper_pnl_usd(
                            str(data["side"]),
                            data["entry"],
                            data["close_price"],
                            data["size"],
                            Decimal(str(rate)),
                        )
                        payload["realized_pnl"] = str(pnl)
                        payload["realized_pnl_currency"] = "USD"
                        payload["fx_quote_to_usd_close"] = str(rate)
                        for key in (
                            "fx_conversion_price_close",
                            "fx_conversion_source_timestamp_close",
                        ):
                            value = data.get(key)
                            if isinstance(value, Decimal):
                                payload[key] = str(value)
                            elif isinstance(value, datetime):
                                payload[key] = value.isoformat()
                rows.append(payload)
    except Exception as exc:
        persistence_state.mark_runtime_error(str(exc))
        raise HTTPException(
            status_code=503, detail={"status": "UNAVAILABLE", "reason": "paper persistence failed"}
        ) from exc
    return {"status": "OK", "positions": rows, "count": len(rows), "paper_only": True}


async def _paper_ui_section(call: Any) -> Dict[str, object]:
    try:
        data = await call()
        return {"status": "OK", "data": data}
    except HTTPException as exc:
        detail = exc.detail if isinstance(exc.detail, dict) else {"reason": str(exc.detail)}
        return {
            "status": "UNAVAILABLE",
            "http_status": exc.status_code,
            "reason": detail.get("reason", "HTTP_ERROR"),
            "data": None,
        }
    except Exception as exc:  # noqa: BLE001
        log.error("Paper UI snapshot section failed: %s", exc)
        return {
            "status": "UNAVAILABLE",
            "http_status": 500,
            "reason": type(exc).__name__,
            "data": None,
        }


@api_router.get("/paper/ui-snapshot")
async def get_paper_ui_snapshot(
    symbol: Optional[str] = None, state: Optional[str] = None
) -> Dict[str, object]:
    """Return one server-stamped, fail-safe snapshot for the paper UI."""
    snapshot_at = utcnow()
    account = await _paper_ui_section(get_live_paper_account)
    positions = await _paper_ui_section(list_paper_positions)
    live_positions = await _paper_ui_section(get_live_paper_positions)
    performance = await _paper_ui_section(get_paper_performance)

    async def decisions() -> Dict[str, object]:
        return await signal_decision_history(limit=50, symbol=symbol, state=state)

    decision_history = await _paper_ui_section(decisions)
    watchdog: Dict[str, object] = {
        "status": "OK",
        "data": auto_scan_watchdog_status(),
    }
    runtime: Dict[str, object] = {
        "status": "OK",
        "data": auto_scan_runtime_status(),
    }
    sections: Dict[str, Dict[str, object]] = {
        "account": account,
        "positions": positions,
        "live_positions": live_positions,
        "performance": performance,
        "decision_history": decision_history,
        "watchdog": watchdog,
        "runtime": runtime,
    }
    available = 0
    for section in sections.values():
        if section.get("status") == "OK":
            available += 1
    return {
        "status": "OK" if available == len(sections) else "PARTIAL",
        "validation": "SERVER_PAPER_UI_SNAPSHOT_V1",
        "snapshot_at": snapshot_at.isoformat(),
        "sections": sections,
        "paper_only": True,
        "execution": False,
    }


async def init_candle_schema() -> None:
    """Create persistence tables if absent (idempotent). Raises on real DDL failure
    so it is NEVER swallowed into a silent false success."""
    async with engine.begin() as conn:
        await conn.run_sync(metadata.create_all)
        now = utcnow()
        stmt = pg_insert(paper_account_table).values(
            account_id=PAPER_ACCOUNT_ID,
            currency=PAPER_ACCOUNT_CURRENCY,
            initial_capital=PAPER_INITIAL_CAPITAL,
            created_at=now,
            updated_at=now,
        )
        stmt = stmt.on_conflict_do_nothing(index_elements=["account_id"])
        await conn.execute(stmt)


@dataclass(frozen=True)
class CandleRow:
    source: str
    product_id: str
    granularity: str
    bucket_start: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
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
        open=Decimal(str(o["open"])), high=Decimal(str(o["high"])),
        low=Decimal(str(o["low"])), close=Decimal(str(o["close"])),
        volume=Decimal(str(o["volume"])),
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
                open=Decimal(str(o)), high=Decimal(str(h)), low=Decimal(str(low)),
                close=Decimal(str(c)), volume=Decimal(str(v)),
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


class USEquityRTHCalendar(MarketCalendar):
    """Baseline U.S. cash-index regular-hours calendar.

    Massive documents most U.S. indices as updating Monday-Friday 09:30-16:00
    America/New_York. DST is handled by IANA ZoneInfo. This class intentionally
    does NOT fabricate holiday/early-close knowledge: it provides the documented
    regular-hours baseline only, and gap analysis remains UNKNOWN because Massive
    explicitly emits no aggregate when an index has no update.
    """

    policy = MarketCalendarPolicy.US_EQUITY_RTH
    timezone_name = "America/New_York"

    def is_market_expected_open(self, ts_unix: int) -> OpenState:
        ny = _zone(self.timezone_name)
        if ny is None:
            return OpenState.UNKNOWN
        try:
            now = datetime.fromtimestamp(int(ts_unix), tz=timezone.utc).astimezone(ny)
        except (OverflowError, OSError, ValueError):
            return OpenState.UNKNOWN
        if now.weekday() >= 5:
            return OpenState.CLOSED
        minutes = now.hour * 60 + now.minute
        return OpenState.OPEN if 570 <= minutes < 960 else OpenState.CLOSED

    def expected_bucket_starts(self, granularity: str, start: int, end: int) -> Optional[List[int]]:
        return None  # holidays/early closes/index-specific update cadence not fabricated

    def analyze_gaps(self, starts: List[int], bucket: int) -> GapReport:
        return GapReport("UNKNOWN", [])  # no index update != missing market data


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
_US_EQUITY_RTH = USEquityRTHCalendar()
_COINBASE_CALENDAR = _ALWAYS_24_7


def calendar_for(policy: MarketCalendarPolicy) -> MarketCalendar:
    """Resolve implemented calendar policies; unknown configuration stays UNKNOWN."""
    if policy == MarketCalendarPolicy.ALWAYS_OPEN_24_7:
        return _ALWAYS_24_7
    if policy == MarketCalendarPolicy.FOREX_WEEK:
        return _FOREX_WEEK
    if policy == MarketCalendarPolicy.US_EQUITY_RTH:
        return _US_EQUITY_RTH
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
        ("SOL-USD", "SOL", "USD", "Solana / US Dollar"),
        ("XRP-USD", "XRP", "USD", "XRP / US Dollar"),
        ("LTC-USD", "LTC", "USD", "Litecoin / US Dollar"),
        ("ADA-USD", "ADA", "USD", "Cardano / US Dollar"),
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


# V16-M5B28A-TOP100 — dynamic, live-verified Coinbase crypto universe.
# Coinbase public products are requested in explicit 24h quote-volume order.
# Only real SPOT *-USD products with valid sizing metadata are eligible.
# This is Coinbase's top USD spot universe by 24h quote volume, not a fabricated
# global market-cap ranking. The original six remain registered as the proven base.
CRYPTO_UNIVERSE_VERSION = "SERVER_CRYPTO_UNIVERSE_TOP100_V2"
CRYPTO_UNIVERSE_TARGET_SIZE = 100
CRYPTO_UNIVERSE_DISCOVERY_LIMIT = 250
crypto_universe_activation: Dict[str, Dict[str, object]] = {}


def coinbase_product_is_eligible(symbol: str, payload: object) -> Tuple[bool, str]:
    """Fail closed unless Coinbase proves a usable public USD SPOT product."""
    if not isinstance(payload, dict):
        return False, "PRODUCT_PAYLOAD_INVALID"
    canonical = symbol.upper().replace("/", "-")
    product_id = str(payload.get("product_id", "")).upper()
    if product_id != canonical:
        return False, "PRODUCT_ID_MISMATCH"
    if not canonical.endswith("-USD"):
        return False, "QUOTE_NOT_USD"
    product_type = str(payload.get("product_type", "SPOT")).upper()
    if product_type != "SPOT":
        return False, "PRODUCT_NOT_SPOT"
    if payload.get("trading_disabled") is True or payload.get("is_disabled") is True:
        return False, "TRADING_DISABLED"
    if payload.get("view_only") is True:
        return False, "VIEW_ONLY"
    specs = parse_coinbase_spot_specs(canonical, payload)
    if specs.get("status") != "VALID":
        return False, "PRODUCT_SPECS_INVALID"
    return True, "COINBASE_PRODUCT_VERIFIED"


def _positive_product_decimal(value: object) -> Optional[Decimal]:
    try:
        parsed = Decimal(str(value))
    except (ArithmeticError, ValueError):
        return None
    return parsed if parsed.is_finite() and parsed > 0 else None


def select_coinbase_top_usd_spot_products(
    payload: object, target: int = CRYPTO_UNIVERSE_TARGET_SIZE
) -> List[Dict[str, object]]:
    """Select up to target eligible USD SPOT products by real 24h quote volume."""
    if not isinstance(payload, dict) or not isinstance(payload.get("products"), list):
        return []
    ranked: List[Tuple[Decimal, Dict[str, object]]] = []
    seen: Set[str] = set()
    for raw in payload["products"]:
        if not isinstance(raw, dict):
            continue
        symbol = str(raw.get("product_id", "")).upper().replace("/", "-")
        if symbol in seen:
            continue
        eligible, _reason = coinbase_product_is_eligible(symbol, raw)
        if not eligible:
            continue
        volume = _positive_product_decimal(
            raw.get("approximate_quote_24h_volume", raw.get("volume_24h"))
        )
        if volume is None:
            continue
        seen.add(symbol)
        ranked.append((volume, raw))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [product for _volume, product in ranked[: max(1, int(target))]]


def register_verified_coinbase_crypto(symbol: str) -> bool:
    """Register only a symbol already verified against Coinbase public metadata."""
    canonical = symbol.upper().replace("/", "-")
    if not canonical.endswith("-USD"):
        return False
    base = canonical.removesuffix("-USD")
    instrument_registry.register(
        Instrument(
            canonical_symbol=canonical,
            asset_class=AssetClass.CRYPTO,
            base_asset=base,
            quote_asset="USD",
            display_name=f"{base} / US Dollar",
            timezone="UTC",
            market_calendar=MarketCalendarPolicy.ALWAYS_OPEN_24_7,
            volume_semantics=VolumeSemantics.BASE_ASSET_VOLUME,
            price_precision=None,
            tick_size=None,
        )
    )
    provider_symbol_map.add("coinbase", canonical, canonical)
    return True


async def activate_verified_crypto_universe() -> Dict[str, object]:
    """Discover Coinbase live and activate at most 100 verified USD spot products."""
    activated: List[str] = []
    rejected: Dict[str, str] = {}
    crypto_universe_activation.clear()
    try:
        payload = await market_provider.list_public_spot_products(
            CRYPTO_UNIVERSE_DISCOVERY_LIMIT
        )
    except asyncio.CancelledError:
        raise
    except (httpx.HTTPError, ValueError, KeyError):
        return {
            "status": "DEGRADED",
            "reason": "COINBASE_PRODUCT_DISCOVERY_UNAVAILABLE",
            "activated": [],
            "rejected": {},
            "target_count": CRYPTO_UNIVERSE_TARGET_SIZE,
            "marker": CRYPTO_UNIVERSE_VERSION,
            "paper_only": True,
            "execution": False,
        }
    products = select_coinbase_top_usd_spot_products(payload)
    for product in products:
        symbol = str(product.get("product_id", "")).upper()
        eligible, reason = coinbase_product_is_eligible(symbol, product)
        if eligible and register_verified_coinbase_crypto(symbol):
            activated.append(symbol)
        else:
            rejected[symbol] = reason
        crypto_universe_activation[symbol] = {
            "active": bool(eligible),
            "reason": reason,
            "volume_24h_quote": str(
                product.get("approximate_quote_24h_volume", product.get("volume_24h", ""))
            ),
        }
    return {
        "status": "READY",
        "activated": activated,
        "rejected": rejected,
        "target_count": CRYPTO_UNIVERSE_TARGET_SIZE,
        "discovered_eligible_count": len(products),
        "ranking": "COINBASE_24H_QUOTE_VOLUME_DESC",
        "marker": CRYPTO_UNIVERSE_VERSION,
        "paper_only": True,
        "execution": False,
    }


@api_router.get("/market/crypto-universe")
async def get_crypto_universe() -> Dict[str, object]:
    active = sorted(
        instrument.canonical_symbol
        for instrument in instrument_registry.all()
        if instrument.asset_class == AssetClass.CRYPTO
        if provider_symbol_map.to_provider(
            "coinbase", instrument.canonical_symbol
        ) is not None
    )
    return {
        "status": "READY",
        "active_symbols": active,
        "active_count": len(active),
        "target_count": CRYPTO_UNIVERSE_TARGET_SIZE,
        "candidate_activation": dict(crypto_universe_activation),
        "source": "coinbase_public_products",
        "ranking": "COINBASE_24H_QUOTE_VOLUME_DESC",
        "marker": CRYPTO_UNIVERSE_VERSION,
        "paper_only": True,
        "execution": False,
    }


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




# ============================ Massive Forex realtime WebSocket (V2-A) ===========
# Official Massive protocol: wss://socket.massive.com/forex, auth action,
# C.<PAIR> BBO quotes and CA.<PAIR> per-minute quote-derived OHLC.
# No synthetic midpoint and no synthetic missing bars.
@dataclass(frozen=True)
class ForexRealtimeQuote:
    canonical_symbol: str
    bid: Decimal
    ask: Decimal
    source_timestamp: datetime
    received_at: datetime
    quality: DataQualityStatus

    def to_dict(self) -> Dict[str, object]:
        return {"source": "massive", "canonical_symbol": self.canonical_symbol,
                "bid": str(self.bid), "ask": str(self.ask),
                "source_timestamp": self.source_timestamp.isoformat(),
                "received_at": self.received_at.isoformat(), "quality": self.quality.value}


@dataclass(frozen=True)
class ForexRealtimeCandle:
    canonical_symbol: str
    start: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    received_at: datetime
    quality: DataQualityStatus

    def to_dict(self) -> Dict[str, object]:
        return {"source": "massive", "canonical_symbol": self.canonical_symbol,
                "granularity": "1m", "start": self.start.isoformat(),
                "open": str(self.open), "high": str(self.high), "low": str(self.low),
                "close": str(self.close), "volume": str(self.volume),
                "received_at": self.received_at.isoformat(), "quality": self.quality.value}


def _positive_decimal(value: Any) -> Optional[Decimal]:
    try:
        d = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    return d if d > 0 and d.is_finite() else None


def _massive_pair_to_canonical(pair: Any) -> Optional[str]:
    if not isinstance(pair, str) or "/" not in pair:
        return None
    canonical = pair.replace("/", "-").upper()
    mapped = provider_symbol_map.to_provider("massive", canonical)
    return canonical if mapped is not None else None


def parse_massive_forex_quote(
    item: Any, received_at: Optional[datetime] = None
) -> Optional[ForexRealtimeQuote]:
    if not isinstance(item, dict) or item.get("ev") != "C":
        return None
    canonical = _massive_pair_to_canonical(item.get("p"))
    bid, ask = _positive_decimal(item.get("b")), _positive_decimal(item.get("a"))
    ts = _unix_ms_to_dt(item.get("t"))
    if canonical is None or bid is None or ask is None or ts is None or ask < bid:
        return None
    recv = received_at or utcnow()
    return ForexRealtimeQuote(canonical, bid, ask, ts, recv,
                              classify_freshness(ts, settings.ticker_max_age_seconds, now=recv))


def parse_massive_forex_minute(
    item: Any, received_at: Optional[datetime] = None
) -> Optional[ForexRealtimeCandle]:
    if not isinstance(item, dict) or item.get("ev") != "CA":
        return None
    canonical = _massive_pair_to_canonical(item.get("pair"))
    start = _unix_ms_to_dt(item.get("s"))
    vals = [_positive_decimal(item.get(k)) for k in ("o", "h", "l", "c", "v")]
    if canonical is None or start is None or any(v is None for v in vals):
        return None
    open_, high, low, close, volume = vals
    assert (
        open_ is not None
        and high is not None
        and low is not None
        and close is not None
        and volume is not None
    )
    if high < low or not (low <= open_ <= high) or not (low <= close <= high):
        return None
    recv = received_at or utcnow()
    return ForexRealtimeCandle(canonical, start, open_, high, low, close, volume, recv,
                               classify_freshness(start, 120.0, now=recv))


class MassiveForexWsManager:
    SOURCE = "massive"

    def __init__(self, url: Optional[str] = None, api_key: Optional[str] = None) -> None:
        self.url = url or settings.massive_forex_ws_url
        self.api_key = settings.massive_api_key if api_key is None else api_key
        self.running = False
        self.websocket: Any = None
        self.authenticated = False
        self.last_message_at: Optional[datetime] = None
        self.last_error: Optional[str] = None
        self._task: Optional[asyncio.Task] = None
        self.quotes: Dict[str, ForexRealtimeQuote] = {}
        self.candles: Dict[str, ForexRealtimeCandle] = {}

    def _topics(self) -> List[str]:
        topics: List[str] = []
        for canonical in EXPECTED_MASSIVE_FOREX:
            if provider_symbol_map.to_provider("massive", canonical) is not None:
                pair = canonical.replace("-", "/")
                topics.extend((f"C.{pair}", f"CA.{pair}"))
        return topics

    async def start(self) -> None:
        if not self.api_key:
            raise RuntimeError("MASSIVE_API_KEY not set")
        if websockets is None:
            raise RuntimeError("websockets dependency is not installed")
        if self.running:
            return
        if not self._topics():
            raise RuntimeError("no verified Massive forex mappings available")
        self.running = True
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        self.running = False
        self.authenticated = False
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

    async def _run_loop(self) -> None:  # pragma: no cover - live provider socket
        attempt = 0
        while self.running:
            try:
                async with websockets.connect(
                    self.url,
                    ping_interval=20,
                    ping_timeout=20,
                    close_timeout=5,
                ) as ws:
                    self.websocket = ws
                    self.authenticated = False
                    attempt = 0
                    await ws.send(json.dumps({"action": "auth", "params": self.api_key}))
                    async for raw in ws:
                        self.last_message_at = utcnow()
                        await self._handle(raw)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                self.websocket = None
                self.authenticated = False
                self.last_error = _redact_secret(str(exc))[:200]
                if not self.running:
                    break
                attempt += 1
                await asyncio.sleep(ws_backoff(attempt))

    async def _handle(self, raw: str | bytes) -> None:
        try:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            payload = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError):
            return
        items = payload if isinstance(payload, list) else [payload]
        for item in items:
            if not isinstance(item, dict):
                continue
            if item.get("ev") == "status":
                status_value = str(item.get("status", ""))
                if status_value == "auth_success":
                    self.authenticated = True
                    topics = self._topics()
                    if self.websocket is not None and topics:
                        await self.websocket.send(
                            json.dumps(
                                {"action": "subscribe", "params": ",".join(topics)}
                            )
                        )
                elif status_value in {"auth_failed", "error"}:
                    self.last_error = str(item.get("message") or status_value)[:200]
                continue
            q = parse_massive_forex_quote(item, self.last_message_at)
            if q is not None:
                old = self.quotes.get(q.canonical_symbol)
                if old is None or q.source_timestamp >= old.source_timestamp:
                    self.quotes[q.canonical_symbol] = q
                continue
            c = parse_massive_forex_minute(item, self.last_message_at)
            if c is not None:
                oldc = self.candles.get(c.canonical_symbol)
                if oldc is None or c.start >= oldc.start:
                    self.candles[c.canonical_symbol] = c

    def realtime(self, canonical_symbol: str) -> Dict[str, object]:
        canonical = canonical_symbol.upper()
        q, c = self.quotes.get(canonical), self.candles.get(canonical)
        transport = (
            "WEBSOCKET"
            if self.authenticated
            else ("CONNECTING" if self.running else "STOPPED")
        )
        return {
            "source": "massive",
            "canonical_symbol": canonical,
            "status": "OK" if (q or c) else "MISSING",
            "transport": transport,
            "quote": q.to_dict() if q else None,
            "candle": c.to_dict() if c else None,
        }

    def health(self) -> Dict[str, object]:
        return {
            "source": "massive",
            "running": self.running,
            "connected": self.websocket is not None,
            "authenticated": self.authenticated,
            "last_message_at": (
                self.last_message_at.isoformat() if self.last_message_at else None
            ),
            "last_error": self.last_error,
        }


massive_forex_ws = MassiveForexWsManager()


@api_router.post("/market/forex/websocket/start")
async def market_forex_ws_start() -> dict:
    try:
        await massive_forex_ws.start()
    except RuntimeError as exc:
        reason = str(exc)
        code = 503 if "API_KEY" in reason else 409
        raise HTTPException(
            status_code=code,
            detail={"status": "UNAVAILABLE", "reason": reason},
        ) from exc
    return {"status": "started", "source": "massive", "transport": "WEBSOCKET"}


@api_router.post("/market/forex/websocket/stop")
async def market_forex_ws_stop() -> dict:
    await massive_forex_ws.stop()
    return {"status": "stopped", "source": "massive"}


@api_router.get("/market/forex/websocket/health")
async def market_forex_ws_health() -> dict:
    return massive_forex_ws.health()


@api_router.get("/market/forex/{symbol}/realtime")
async def market_forex_realtime(symbol: str) -> dict:
    canonical = symbol.upper()
    inst = instrument_registry.get(canonical)
    if inst is None or inst.asset_class != AssetClass.FOREX:
        raise HTTPException(
            status_code=404,
            detail={"status": "NOT_SUPPORTED", "reason": "unknown forex instrument"},
        )
    if provider_symbol_map.to_provider("massive", canonical) is None:
        raise HTTPException(
            status_code=409,
            detail={"status": "NOT_MAPPED", "reason": "no verified Massive mapping"},
        )
    return massive_forex_ws.realtime(canonical)


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


@dataclass(frozen=True)
class TwelveDataQuoteResult:
    """Latest /quote outcome. price is an exact Decimal (never float); is_market_open
    is a provider flag (NOT a Gold calendar, NOT a data-quality decision by itself)."""
    status: str  # OK | NOT_MAPPED | NO_KEY | ACCESS_DENIED | RATE_LIMITED | UNAVAILABLE
    price: Optional[Decimal]
    is_market_open: Optional[bool]
    timestamp_utc: Optional[datetime]
    reason: Optional[str] = None


def parse_twelvedata_quote(payload: Any) -> TwelveDataQuoteResult:
    """Parse a /quote body. HTTP 200 + {status:error} is a provider error. Price from
    `close` parsed to Decimal directly. is_market_open kept as a provider boolean."""
    if not isinstance(payload, dict):
        return TwelveDataQuoteResult("UNAVAILABLE", None, None, None, "malformed provider response")
    if payload.get("status") == "error":
        code = payload.get("code")
        if code == 429:
            return TwelveDataQuoteResult("RATE_LIMITED", None, None, None, "rate limited (429)")
        if code in (401, 403):
            return TwelveDataQuoteResult(
                "ACCESS_DENIED", None, None, None, f"access denied by provider ({code})")
        return TwelveDataQuoteResult("UNAVAILABLE", None, None, None, "provider error status")
    price = _to_decimal(payload.get("close"))
    if price is None or price < 0:
        return TwelveDataQuoteResult("UNAVAILABLE", None, None, None, "no usable price in quote")
    raw_open = payload.get("is_market_open")
    is_open = raw_open if isinstance(raw_open, bool) else None
    ts = payload.get("timestamp")
    ts_utc: Optional[datetime] = None
    if isinstance(ts, int) and ts > 0:
        try:
            ts_utc = datetime.fromtimestamp(ts, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            ts_utc = None
    return TwelveDataQuoteResult("OK", price, is_open, ts_utc, None)


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

    async def get_quote(self, canonical_symbol: str) -> "TwelveDataQuoteResult":
        """Fetch the latest /quote (price + is_market_open). Price parsed to Decimal
        (never float). Same explicit statuses as get_time_series."""
        symbol = provider_symbol_map.to_provider(self.SOURCE, canonical_symbol)
        if symbol is None:
            return TwelveDataQuoteResult("NOT_MAPPED", None, None, None,
                                         f"no verified twelvedata symbol for {canonical_symbol}")
        if not self.api_key:
            return TwelveDataQuoteResult("NO_KEY", None, None, None, "TWELVEDATA_API_KEY not set")
        if self.client is None:
            await self.connect()
        assert self.client is not None
        try:
            resp = await self.client.get("/quote", params={"symbol": symbol, "timezone": "UTC"})
        except httpx.TimeoutException:
            return TwelveDataQuoteResult("UNAVAILABLE", None, None, None, "provider timeout")
        except httpx.HTTPError:
            return TwelveDataQuoteResult("UNAVAILABLE", None, None, None, "provider unreachable")
        code = resp.status_code
        if code in (401, 403):
            return TwelveDataQuoteResult(
                "ACCESS_DENIED", None, None, None, f"access denied by provider ({code})")
        if code == 429:
            return TwelveDataQuoteResult(
                "RATE_LIMITED", None, None, None, "rate limited by provider (429)")
        if code >= 500:
            return TwelveDataQuoteResult(
                "UNAVAILABLE", None, None, None, f"provider server error ({code})")
        try:
            payload = resp.json()
        except (ValueError, json.JSONDecodeError):
            return TwelveDataQuoteResult(
                "UNAVAILABLE", None, None, None, "malformed provider response")
        return parse_twelvedata_quote(payload)


# Officially-catalogued Twelve Data symbol for gold spot -> verified provider
# mapping (a mapping is not an entitlement: MAPPED can coexist with NOT_ENTITLED).
provider_symbol_map.add("twelvedata", "XAU-USD", "XAU/USD")

twelvedata_provider = TwelveDataProvider()

# Twelve Data WebSocket price stream for Gold Spot. Officially, /v1/quotes/price
# emits price ticks only: it does NOT provide OHLC or bid/ask. Historical/chart
# OHLC therefore remains sourced from the verified REST /time_series endpoint.
TWELVEDATA_WS_PRICE_MAX_AGE_SECONDS = 30.0


@dataclass(frozen=True)
class TwelveDataRealtimePrice:
    canonical_symbol: str
    provider_symbol: str
    price: Decimal
    source_timestamp: datetime
    received_at: datetime
    quality: DataQualityStatus

    def to_dict(self) -> Dict[str, object]:
        return {
            "source": "twelvedata",
            "canonical_symbol": self.canonical_symbol,
            "provider_symbol": self.provider_symbol,
            "price": str(self.price),
            "source_timestamp": self.source_timestamp.isoformat(),
            "received_at": self.received_at.isoformat(),
            "quality": self.quality.value,
        }


def parse_twelvedata_ws_price(
    item: Any,
    canonical_symbol: str,
    received_at: Optional[datetime] = None,
) -> Optional[TwelveDataRealtimePrice]:
    """Parse one official Twelve Data `price` WebSocket event without synthesis."""
    if not isinstance(item, dict) or item.get("event") != "price":
        return None
    provider_symbol = item.get("symbol")
    expected = provider_symbol_map.to_provider("twelvedata", canonical_symbol)
    if not isinstance(provider_symbol, str) or provider_symbol != expected:
        return None
    price = _to_decimal(item.get("price"))
    if price is None or price <= 0:
        return None
    raw_ts = item.get("timestamp")
    if not isinstance(raw_ts, (int, float)) or isinstance(raw_ts, bool) or raw_ts <= 0:
        return None
    try:
        source_timestamp = datetime.fromtimestamp(float(raw_ts), tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None
    observed = received_at or datetime.now(timezone.utc)
    quality = classify_freshness(
        source_timestamp,
        TWELVEDATA_WS_PRICE_MAX_AGE_SECONDS,
        now=observed,
    )
    return TwelveDataRealtimePrice(
        canonical_symbol=canonical_symbol,
        provider_symbol=provider_symbol,
        price=price,
        source_timestamp=source_timestamp,
        received_at=observed,
        quality=quality,
    )


class TwelveDataGoldWsManager:
    """Server-side XAU/USD price stream. API key never reaches the frontend."""

    def __init__(self) -> None:
        self.running = False
        self.connected = False
        self.subscribed = False
        self.task: Optional[asyncio.Task[None]] = None
        self.websocket: Any = None
        self.last_price: Optional[TwelveDataRealtimePrice] = None
        self.last_message_at: Optional[datetime] = None
        self.last_error: Optional[str] = None

    async def start(self) -> None:
        if self.running:
            return
        if not settings.twelvedata_api_key:
            raise RuntimeError("TWELVEDATA_API_KEY not set")
        if provider_symbol_map.to_provider("twelvedata", "XAU-USD") != "XAU/USD":
            raise RuntimeError("XAU-USD has no verified Twelve Data mapping")
        self.running = True
        self.last_error = None
        self.task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self.running = False
        if self.websocket is not None:
            await self.websocket.close()
        if self.task is not None:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        self.task = None
        self.websocket = None
        self.connected = False
        self.subscribed = False

    async def _run(self) -> None:
        delay = 1.0
        while self.running:
            try:
                key = url_quote(settings.twelvedata_api_key, safe="")
                connect_url = f"{settings.twelvedata_ws_url}?apikey={key}"
                async with websockets.connect(
                    connect_url,
                    ping_interval=20,
                    ping_timeout=20,
                    close_timeout=5,
                ) as ws:
                    self.websocket = ws
                    self.connected = True
                    self.subscribed = False
                    self.last_error = None
                    await ws.send(json.dumps({
                        "action": "subscribe",
                        "params": {"symbols": "XAU/USD"},
                    }))
                    delay = 1.0
                    while self.running:
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=10.0)
                        except asyncio.TimeoutError:
                            await ws.send(json.dumps({"action": "heartbeat"}))
                            continue
                        self.last_message_at = datetime.now(timezone.utc)
                        await self._handle(raw)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - provider transport boundary
                # Do not retain str(exc): WebSocket connection exceptions can include
                # the URI, and the Twelve Data URI contains the server-side API key.
                self.last_error = f"websocket {type(exc).__name__}"
            finally:
                self.websocket = None
                self.connected = False
                self.subscribed = False
            if self.running:
                await asyncio.sleep(delay)
                delay = min(delay * 2.0, 30.0)

    async def _handle(self, raw: Any) -> None:
        try:
            item = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
        except (json.JSONDecodeError, UnicodeDecodeError):
            return
        if not isinstance(item, dict):
            return
        if item.get("event") == "subscribe-status":
            status_value = str(item.get("status") or "").lower()
            if status_value in {"ok", "success"}:
                self.subscribed = True
            elif status_value in {"error", "failed"}:
                self.last_error = "subscription rejected by provider"
            return
        parsed = parse_twelvedata_ws_price(item, "XAU-USD")
        if parsed is not None:
            self.last_price = parsed

    def realtime(self) -> Dict[str, object]:
        transport = "STOPPED"
        if self.running:
            transport = "WEBSOCKET" if self.connected else "CONNECTING"
        return {
            "source": "twelvedata",
            "canonical_symbol": "XAU-USD",
            "status": "OK" if self.last_price is not None else "MISSING",
            "transport": transport,
            "subscribed": self.subscribed,
            "price": self.last_price.to_dict() if self.last_price else None,
            "ohlc_transport": "REST",
            "last_error": self.last_error,
        }

    def health(self) -> Dict[str, object]:
        return {
            "source": "twelvedata",
            "running": self.running,
            "connected": self.connected,
            "subscribed": self.subscribed,
            "last_message_at": (
                self.last_message_at.isoformat() if self.last_message_at else None
            ),
            "last_error": self.last_error,
        }


twelvedata_gold_ws = TwelveDataGoldWsManager()


@api_router.post("/market/metal/websocket/start")
async def market_metal_ws_start() -> dict:
    try:
        await twelvedata_gold_ws.start()
    except RuntimeError as exc:
        reason = str(exc)
        code = 503 if "API_KEY" in reason else 409
        raise HTTPException(
            status_code=code,
            detail={"status": "UNAVAILABLE", "reason": reason},
        ) from exc
    return {
        "status": "started",
        "source": "twelvedata",
        "transport": "WEBSOCKET_PRICE_ONLY",
        "ohlc_transport": "REST",
    }


@api_router.post("/market/metal/websocket/stop")
async def market_metal_ws_stop() -> dict:
    await twelvedata_gold_ws.stop()
    return {"status": "stopped", "source": "twelvedata"}


@api_router.get("/market/metal/websocket/health")
async def market_metal_ws_health() -> dict:
    return twelvedata_gold_ws.health()


@api_router.get("/market/metal/{symbol}/realtime")
async def market_metal_realtime(symbol: str) -> dict:
    canonical = symbol.upper().replace("/", "-")
    if canonical != "XAU-USD":
        raise HTTPException(
            status_code=404,
            detail={"status": "NOT_SUPPORTED", "reason": "unknown metal instrument"},
        )
    return twelvedata_gold_ws.realtime()


def _register_metal_instruments() -> None:
    """Register the canonical Gold Spot instrument. Calendar NOT_CONFIGURED (no Gold
    calendar invented in this increment); volume UNKNOWN; precision/tick None."""
    instrument_registry.register(
        Instrument(
            canonical_symbol="XAU-USD",
            asset_class=AssetClass.METAL,
            base_asset="XAU",
            quote_asset="USD",
            display_name="Gold Spot",
            timezone="UTC",
            market_calendar=MarketCalendarPolicy.NOT_CONFIGURED,
            volume_semantics=VolumeSemantics.UNKNOWN,
            price_precision=None,
            tick_size=None,
        )
    )


_register_metal_instruments()


@dataclass(frozen=True)
class MetalHistory:
    """Assembled metal history: the JSON-serialisable `result` for the API, and the
    Decimal-exact `rows` ready for persistence (never routed through float)."""
    result: Dict[str, object]
    rows: List[CandleRow]


def _td_bar_dict(bar: TwelveDataBar) -> Dict[str, object]:
    """JSON-safe view of a bar: Decimals as strings (exact), datetime as ISO UTC."""
    def s(v: Optional[Decimal]) -> Optional[str]:
        return str(v) if v is not None else None
    return {
        "start": bar.datetime_utc.isoformat() if bar.datetime_utc else None,
        "open": s(bar.open), "high": s(bar.high), "low": s(bar.low),
        "close": s(bar.close), "volume": s(bar.volume), "quality": bar.status.value,
    }


def _metal_latest_quality(
    bars: List[TwelveDataBar], granularity: str, now: Optional[datetime] = None
) -> str:
    """Freshness of the MOST RECENT bar (market state is separate). MISSING if none."""
    dts = [b.datetime_utc for b in bars if b.datetime_utc is not None]
    if not dts:
        return DataQualityStatus.MISSING.value
    bucket = GRANULARITIES[granularity][1] if granularity in GRANULARITIES else 3600
    return classify_freshness(max(dts), bucket * 2, now=now).value


def _metal_bars_to_rows(
    source: str, product_id: Optional[str], granularity: str, bars: List[TwelveDataBar],
    observed_at: datetime,
) -> List[CandleRow]:
    """Build Decimal-exact CandleRows from parsed bars. A bar without a provider
    volume is NOT persisted (the column is NOT NULL and we never fabricate a 0)."""
    rows: List[CandleRow] = []
    if product_id is None:
        return rows
    for b in bars:
        if b.status == DataQualityStatus.INVALID or b.datetime_utc is None:
            continue
        if b.open is None or b.high is None or b.low is None or b.close is None:
            continue
        if b.volume is None:  # cannot persist without a volume; never fabricate one
            continue
        rows.append(
            CandleRow(
                source=source, product_id=product_id, granularity=granularity,
                bucket_start=b.datetime_utc,
                open=b.open, high=b.high, low=b.low, close=b.close, volume=b.volume,
                quality=b.status, origin="rest", source_timestamp=None,
                observed_at=observed_at,
            )
        )
    return rows


async def fetch_metal_history(
    canonical_symbol: str, granularity: str, start: int, end: int
) -> MetalHistory:
    """Assemble Gold XAU/USD history from Twelve Data. [start, end) half-open, dedup
    by timestamp, ascending, INVALID excluded. Gaps stay UNKNOWN (no Gold calendar):
    a missing bar is never a gap. Provider/business status is surfaced explicitly."""
    inst = instrument_registry.get(canonical_symbol)
    if inst is None or inst.asset_class != AssetClass.METAL:
        raise ValueError(f"unknown metal instrument: {canonical_symbol}")
    if start >= end:
        raise ValueError("start must be strictly before end")
    provider_symbol = provider_symbol_map.to_provider("twelvedata", canonical_symbol)
    start_s = datetime.fromtimestamp(int(start), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    end_s = datetime.fromtimestamp(int(end), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    tdr = await twelvedata_provider.get_time_series(
        canonical_symbol, granularity, outputsize=5000, start=start_s, end=end_s
    )
    base: Dict[str, object] = {
        "source": "twelvedata",
        "canonical_symbol": canonical_symbol,
        "provider_symbol": provider_symbol,
        "granularity": granularity,
        "requested_range": {"start": start, "end": end},
        "timezone_internal": "UTC",
        "display_timezone": "America/Toronto",
        "market_timezone": inst.timezone,
        "market_calendar": inst.market_calendar.value,
        "volume_semantics": inst.volume_semantics.value,  # UNKNOWN (never invented)
    }
    if tdr.status != "OK":
        base.update({"status": tdr.status, "reason": tdr.reason, "count": 0, "candles": []})
        return MetalHistory(base, [])
    collected: Dict[int, TwelveDataBar] = {}
    invalid = 0
    for bar in tdr.bars:
        if bar.datetime_utc is None or bar.status == DataQualityStatus.INVALID:
            invalid += 1
            continue
        key = int(bar.datetime_utc.timestamp())
        if key < start or key >= end:
            continue
        collected[key] = bar
    kept = [collected[k] for k in sorted(collected)]
    rows = _metal_bars_to_rows("twelvedata", provider_symbol, granularity, kept, utcnow())
    base.update({
        "status": "EMPTY" if not kept else "OK",
        "count": len(kept),
        "invalid_candles_count": invalid,
        "latest_quality": _metal_latest_quality(kept, granularity),
        "gaps_status": "UNKNOWN",   # no Gold calendar -> absence is not a gap
        "candles": [_td_bar_dict(b) for b in kept],
    })
    return MetalHistory(base, rows)


_METAL_HTTP_STATUS = {
    "NOT_MAPPED": 409, "NOT_SUPPORTED": 409, "NO_KEY": 503, "ACCESS_DENIED": 403,
    "RATE_LIMITED": 429, "UNAVAILABLE": 503,
}


@api_router.get("/market/metal/{symbol}/history")
async def market_metal_history(
    symbol: str, start: int, end: int, granularity: str = "1h"
) -> dict:
    try:
        hist = await fetch_metal_history(symbol.upper(), granularity, start, end)
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail={"status": "INVALID", "reason": str(exc)}
        ) from exc
    status = str(hist.result.get("status"))
    if status == "OK" and persistence_state.ready and hist.rows:
        try:
            await persist_candles(hist.rows)  # source="twelvedata"; INVALID never sent
        except Exception as exc:  # noqa: BLE001 - a read must never fail on a write error
            log.warning("Metal persistence error: %s", exc)
    http = _METAL_HTTP_STATUS.get(status)
    if http is not None:
        raise HTTPException(
            status_code=http, detail={"status": status, "reason": hist.result.get("reason")}
        )
    return hist.result


async def fetch_metal_quote(canonical_symbol: str) -> Dict[str, object]:
    """Latest Gold quote from Twelve Data. price is Decimal (serialised as string);
    is_market_open is a provider flag, kept SEPARATE from data quality and from any
    (future) Gold calendar. Quality here is the freshness of the quote timestamp."""
    inst = instrument_registry.get(canonical_symbol)
    if inst is None or inst.asset_class != AssetClass.METAL:
        raise ValueError(f"unknown metal instrument: {canonical_symbol}")
    q = await twelvedata_provider.get_quote(canonical_symbol)
    provider_symbol = provider_symbol_map.to_provider("twelvedata", canonical_symbol)
    base: Dict[str, object] = {
        "source": "twelvedata",
        "canonical_symbol": canonical_symbol,
        "provider_symbol": provider_symbol,
        "timezone_internal": "UTC",
        "display_timezone": "America/Toronto",
    }
    if q.status != "OK":
        base.update({"status": q.status, "reason": q.reason, "price": None,
                     "is_market_open": None, "quality": DataQualityStatus.MISSING.value})
        return base
    # Quality = freshness of the quote's OWN provider timestamp, judged against the
    # ticker freshness budget (ticker_max_age_seconds). It is deliberately NOT driven
    # by is_market_open: a spot-metal /quote can be legitimately STALE while the market
    # is open (sparse ticks and/or a delayed data plan return an old timestamp). We do
    # NOT relax the threshold to force LIVE; STALE truthfully reflects an old timestamp.
    # quote_age_seconds is exposed so the reason (e.g. "il y a 144 min") is transparent.
    age = compute_age_seconds(q.timestamp_utc) if q.timestamp_utc is not None else None
    quality = (
        classify_freshness(q.timestamp_utc, settings.ticker_max_age_seconds).value
        if q.timestamp_utc is not None else DataQualityStatus.UNKNOWN.value
    )
    base.update({
        "status": "OK",
        "price": str(q.price) if q.price is not None else None,  # Decimal -> string
        "is_market_open": q.is_market_open,        # provider flag, informational only
        "quote_time": q.timestamp_utc.isoformat() if q.timestamp_utc else None,
        "quote_age_seconds": age,                  # transparency for the STALE reason
        "quality": quality,
        "volume_semantics": inst.volume_semantics.value,
    })
    return base


@api_router.get("/market/metal/{symbol}/quote")
async def market_metal_quote(symbol: str) -> dict:
    try:
        result = await fetch_metal_quote(symbol.upper())
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail={"status": "INVALID", "reason": str(exc)}
        ) from exc
    status = str(result.get("status"))
    http = _METAL_HTTP_STATUS.get(status)
    if http is not None:
        raise HTTPException(
            status_code=http, detail={"status": status, "reason": result.get("reason")}
        )
    return result


# ==================== Massive US Cash Indices REST ============================
# Official cash indices (I: prefix), verified (massive.com/docs/rest/indices):
# I:SPX (S&P 500), I:NDX (Nasdaq-100), I:DJI (Dow). NOT ETFs (SPY/QQQ/DIA), NOT
# futures (ES/NQ/YM), NOT CFDs. GET /v2/aggs/ticker/{I:XXX}/range/{mult}/{timespan}
# /{from}/{to} -> results[{o,h,l,c,t}] with NO volume (index aggregates are derived
# from index VALUES, not trades); t = Unix ms, bars aligned in Eastern Time; no bar
# when no index update (absence != gap). Same account/key as Massive Forex (Bearer),
# but Indices is a SEPARATE entitlement: MAPPED != entitlement (403 -> NOT_ENTITLED).
# D2: indices are served/qualified live only; NOT persisted (candles.volume is NOT
# NULL and indices have no volume; no volume=0 sentinel is ever written).

MASSIVE_INDEX_GRANULARITIES: Dict[str, tuple] = {
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


@dataclass(frozen=True)
class IndexBar:
    """A parsed index aggregate. OHLC are exact Decimals (never float). Indices have
    NO volume (aggregates are derived from index values, not trades)."""
    datetime_utc: Optional[datetime]
    open: Optional[Decimal]
    high: Optional[Decimal]
    low: Optional[Decimal]
    close: Optional[Decimal]
    status: DataQualityStatus


@dataclass(frozen=True)
class MassiveIndexResult:
    status: str  # OK | EMPTY | NOT_MAPPED | NOT_SUPPORTED | NO_KEY |
    #              ACCESS_DENIED | RATE_LIMITED | UNAVAILABLE
    bars: List[IndexBar]
    reason: Optional[str] = None


def index_bar_from_agg(
    item: Any, max_age_seconds: float, now: Optional[datetime] = None
) -> IndexBar:
    """Parse one index aggregate (t ms + o/h/l/c). OHLC via Decimal only; no volume.
    Missing/non-numeric/negative OHLC or timestamp -> INVALID (never fabricated)."""
    if not isinstance(item, dict):
        return IndexBar(None, None, None, None, None, DataQualityStatus.INVALID)
    dt = _unix_ms_to_dt(item.get("t"))
    open_ = _to_decimal(item.get("o"))
    high = _to_decimal(item.get("h"))
    low = _to_decimal(item.get("l"))
    close = _to_decimal(item.get("c"))
    if dt is None or open_ is None or high is None or low is None or close is None:
        return IndexBar(dt, open_, high, low, close, DataQualityStatus.INVALID)
    if open_ < 0 or high < 0 or low < 0 or close < 0:
        return IndexBar(dt, open_, high, low, close, DataQualityStatus.INVALID)
    return IndexBar(dt, open_, high, low, close, classify_freshness(dt, max_age_seconds, now=now))


def parse_massive_index_aggs(
    payload: Any, max_age_seconds: float, now: Optional[datetime] = None
) -> MassiveIndexResult:
    """Parse an index aggregates body. Empty results -> EMPTY (a legitimate no-update
    period, never a gap). Malformed -> UNAVAILABLE."""
    if not isinstance(payload, dict):
        return MassiveIndexResult("UNAVAILABLE", [], "malformed provider response")
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        return MassiveIndexResult("EMPTY", [], None)
    bars = [index_bar_from_agg(x, max_age_seconds, now=now) for x in results]
    return MassiveIndexResult("OK", bars, None)


class MassiveIndicesProvider:
    """Massive Indices REST adapter. Same account/key as Forex (auth header, never
    in URL/logs/response); no network without a key. OHLC decoded with
    parse_float=Decimal so index values are exact (never float)."""

    SOURCE = "massive"

    def __init__(self, rest_url: Optional[str] = None, api_key: Optional[str] = None) -> None:
        self.rest_url = (rest_url or settings.massive_rest_url).rstrip("/")
        self.api_key = api_key if api_key is not None else settings.massive_api_key
        self.client: Optional[httpx.AsyncClient] = None

    async def connect(self) -> None:
        if self.client is None:
            headers = {"Accept": "application/json"}
            if self.api_key:
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

    async def get_index_aggregates(
        self, canonical_symbol: str, granularity: str, start: int, end: int
    ) -> MassiveIndexResult:
        """Fetch index aggregates over [start, end] UNIX seconds. Returns an explicit
        MassiveIndexResult; never raises for provider/HTTP errors."""
        ticker = provider_symbol_map.to_provider(self.SOURCE, canonical_symbol)
        if ticker is None:
            return MassiveIndexResult(
                "NOT_MAPPED", [], f"no verified Massive index symbol for {canonical_symbol}")
        if granularity not in MASSIVE_INDEX_GRANULARITIES:
            return MassiveIndexResult(
                "NOT_SUPPORTED", [], f"granularity {granularity} not supported (massive indices)")
        if not self.api_key:
            return MassiveIndexResult("NO_KEY", [], "MASSIVE_API_KEY not set")
        if self.client is None:
            await self.connect()
        assert self.client is not None
        multiplier, timespan = MASSIVE_INDEX_GRANULARITIES[granularity]
        start_ms = int(start) * 1000
        end_ms = int(end) * 1000
        try:
            resp = await self.client.get(
                f"/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{start_ms}/{end_ms}",
                params={"adjusted": "true", "sort": "asc", "limit": 50000},
            )
        except httpx.TimeoutException:
            return MassiveIndexResult("UNAVAILABLE", [], "provider timeout")
        except httpx.HTTPError:
            return MassiveIndexResult("UNAVAILABLE", [], "provider unreachable")
        code = resp.status_code
        if code in (401, 403):
            return MassiveIndexResult(
                "ACCESS_DENIED", [], f"access denied by provider ({code})")
        if code == 429:
            return MassiveIndexResult("RATE_LIMITED", [], "rate limited by provider (429)")
        if code >= 500:
            return MassiveIndexResult("UNAVAILABLE", [], f"provider server error ({code})")
        try:
            # parse_float=Decimal -> exact index OHLC, no intermediate float
            payload = json.loads(resp.text, parse_float=Decimal)
        except (ValueError, json.JSONDecodeError):
            return MassiveIndexResult("UNAVAILABLE", [], "malformed provider response")
        return parse_massive_index_aggs(payload, max_age_seconds=float("inf"))


def _register_index_instruments() -> None:
    """Register the 3 canonical US cash indices. INDEX class, quote in USD points,
    volume NOT_AVAILABLE (indices have no volume), documented U.S. equity RTH baseline
    calendar (holidays/early closes intentionally not inferred), precision/tick None.
    Mappings are documentation-verified -> MAPPED (independent of entitlement)."""
    indices = (
        ("SPX", "I:SPX", "S&P 500"),
        ("NDX", "I:NDX", "Nasdaq-100"),
        ("US30", "I:DJI", "Dow Jones Industrial Average"),
    )
    for canonical, provider_ticker, name in indices:
        instrument_registry.register(
            Instrument(
                canonical_symbol=canonical,
                asset_class=AssetClass.INDEX,
                base_asset=None,
                quote_asset="USD",
                display_name=name,
                timezone="America/New_York",  # US cash index (points); internal stays UTC
                market_calendar=MarketCalendarPolicy.US_EQUITY_RTH,
                volume_semantics=VolumeSemantics.NOT_AVAILABLE,
                price_precision=None,
                tick_size=None,
            )
        )
        provider_symbol_map.add("massive", canonical, provider_ticker)


massive_indices_provider = MassiveIndicesProvider()
_register_index_instruments()


def _index_bar_dict(bar: IndexBar) -> Dict[str, object]:
    """JSON-safe index bar: Decimals as strings, datetime ISO UTC. No volume key."""
    def s(v: Optional[Decimal]) -> Optional[str]:
        return str(v) if v is not None else None
    return {
        "start": bar.datetime_utc.isoformat() if bar.datetime_utc else None,
        "open": s(bar.open), "high": s(bar.high), "low": s(bar.low),
        "close": s(bar.close), "quality": bar.status.value,
    }


async def fetch_index_history(
    canonical_symbol: str, granularity: str, start: int, end: int
) -> Dict[str, object]:
    """Assemble US cash index history from Massive (live only, NOT persisted). Half-open
    [start, end), dedup by timestamp, ascending, INVALID excluded. Gaps UNKNOWN (no
    RTH calendar): a missing bar is never a gap. No volume (NOT_AVAILABLE)."""
    inst = instrument_registry.get(canonical_symbol)
    if inst is None or inst.asset_class != AssetClass.INDEX:
        raise ValueError(f"unknown index instrument: {canonical_symbol}")
    if start >= end:
        raise ValueError("start must be strictly before end")
    provider_symbol = provider_symbol_map.to_provider("massive", canonical_symbol)
    res = await massive_indices_provider.get_index_aggregates(
        canonical_symbol, granularity, start, end)
    base: Dict[str, object] = {
        "source": "massive",
        "canonical_symbol": canonical_symbol,
        "provider_symbol": provider_symbol,
        "granularity": granularity,
        "requested_range": {"start": start, "end": end},
        "timezone_internal": "UTC",
        "display_timezone": "America/Toronto",
        "market_timezone": inst.timezone,
        "market_calendar": inst.market_calendar.value,
        "volume_semantics": inst.volume_semantics.value,  # NOT_AVAILABLE
        "persisted": False,  # D2: indices are never written to candles this increment
    }
    if res.status != "OK":
        base.update({"status": res.status, "reason": res.reason, "count": 0, "candles": []})
        return base
    collected: Dict[int, IndexBar] = {}
    invalid = 0
    for bar in res.bars:
        if bar.datetime_utc is None or bar.status == DataQualityStatus.INVALID:
            invalid += 1
            continue
        key = int(bar.datetime_utc.timestamp())
        if key < start or key >= end:
            continue
        collected[key] = bar
    kept = [collected[k] for k in sorted(collected)]
    dts = [b.datetime_utc for b in kept if b.datetime_utc is not None]
    bucket = GRANULARITIES[granularity][1] if granularity in GRANULARITIES else 86400
    latest_quality = (
        classify_freshness(max(dts), bucket * 2).value if dts
        else DataQualityStatus.MISSING.value
    )
    base.update({
        "status": "EMPTY" if not kept else "OK",
        "count": len(kept),
        "invalid_candles_count": invalid,
        "latest_quality": latest_quality,   # never forced LIVE; EOD data is often STALE
        "gaps_status": calendar_for(inst.market_calendar).analyze_gaps([], bucket).status,
        "candles": [_index_bar_dict(b) for b in kept],
    })
    return base




@dataclass(frozen=True)
class IndexRealtimeValue:
    canonical_symbol: str
    value: Decimal
    source_timestamp: datetime
    received_at: datetime
    quality: DataQualityStatus

    def to_dict(self) -> Dict[str, object]:
        return {
            "canonical_symbol": self.canonical_symbol,
            "value": str(self.value),
            "source_timestamp": self.source_timestamp.isoformat(),
            "received_at": self.received_at.isoformat(),
            "quality": self.quality.value,
        }


@dataclass(frozen=True)
class IndexRealtimeCandle:
    canonical_symbol: str
    start: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    received_at: datetime
    quality: DataQualityStatus

    def to_dict(self) -> Dict[str, object]:
        return {
            "canonical_symbol": self.canonical_symbol,
            "granularity": "1m",
            "start": self.start.isoformat(),
            "open": str(self.open),
            "high": str(self.high),
            "low": str(self.low),
            "close": str(self.close),
            "received_at": self.received_at.isoformat(),
            "quality": self.quality.value,
        }


def _massive_index_to_canonical(provider_symbol: Any) -> Optional[str]:
    if not isinstance(provider_symbol, str):
        return None
    for canonical in ("SPX", "NDX", "US30"):
        if provider_symbol_map.to_provider("massive", canonical) == provider_symbol:
            return canonical
    return None


def parse_massive_index_value(
    item: Any, received_at: Optional[datetime] = None
) -> Optional[IndexRealtimeValue]:
    if not isinstance(item, dict) or item.get("ev") != "V":
        return None
    canonical = _massive_index_to_canonical(item.get("T"))
    value = _positive_decimal(item.get("val"))
    ts = _unix_ms_to_dt(item.get("t"))
    if canonical is None or value is None or ts is None:
        return None
    recv = received_at or utcnow()
    quality = classify_freshness(ts, 16 * 60.0, now=recv)
    return IndexRealtimeValue(canonical, value, ts, recv, quality)


def parse_massive_index_minute(
    item: Any, received_at: Optional[datetime] = None
) -> Optional[IndexRealtimeCandle]:
    if not isinstance(item, dict) or item.get("ev") != "AM":
        return None
    canonical = _massive_index_to_canonical(item.get("sym"))
    start = _unix_ms_to_dt(item.get("s"))
    vals = [_positive_decimal(item.get(k)) for k in ("o", "h", "l", "c")]
    if canonical is None or start is None or any(v is None for v in vals):
        return None
    open_, high, low, close = vals
    assert open_ is not None and high is not None and low is not None and close is not None
    if high < low or not (low <= open_ <= high) or not (low <= close <= high):
        return None
    recv = received_at or utcnow()
    quality = classify_freshness(start, 17 * 60.0, now=recv)
    return IndexRealtimeCandle(canonical, start, open_, high, low, close, recv, quality)


class MassiveIndicesWsManager:
    SOURCE = "massive"
    FEED_RECENCY = "15_MIN_DELAYED"

    def __init__(self, url: Optional[str] = None, api_key: Optional[str] = None) -> None:
        self.url = url or settings.massive_indices_ws_url
        self.api_key = settings.massive_api_key if api_key is None else api_key
        self.running = False
        self.websocket: Any = None
        self.authenticated = False
        self.last_message_at: Optional[datetime] = None
        self.last_error: Optional[str] = None
        self._task: Optional[asyncio.Task] = None
        self.values: Dict[str, IndexRealtimeValue] = {}
        self.candles: Dict[str, IndexRealtimeCandle] = {}

    def _topics(self) -> List[str]:
        tickers = [
            provider_symbol_map.to_provider("massive", canonical)
            for canonical in ("SPX", "NDX", "US30")
        ]
        verified = [ticker for ticker in tickers if ticker is not None]
        return [f"{channel}.{ticker}" for ticker in verified for channel in ("V", "AM")]

    async def start(self) -> None:
        if not self.api_key:
            raise RuntimeError("MASSIVE_API_KEY not set")
        if websockets is None:
            raise RuntimeError("websockets dependency is not installed")
        if self.running:
            return
        if not self._topics():
            raise RuntimeError("no verified Massive index mappings available")
        self.running = True
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        self.running = False
        self.authenticated = False
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

    async def _run_loop(self) -> None:  # pragma: no cover - live provider socket
        attempt = 0
        while self.running:
            try:
                async with websockets.connect(
                    self.url, ping_interval=20, ping_timeout=20, close_timeout=5
                ) as ws:
                    self.websocket = ws
                    self.authenticated = False
                    attempt = 0
                    await ws.send(json.dumps({"action": "auth", "params": self.api_key}))
                    async for raw in ws:
                        self.last_message_at = utcnow()
                        await self._handle(raw)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                self.websocket = None
                self.authenticated = False
                self.last_error = _redact_secret(str(exc))[:200]
                if not self.running:
                    break
                attempt += 1
                await asyncio.sleep(ws_backoff(attempt))

    async def _handle(self, raw: str | bytes) -> None:
        try:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            payload = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError):
            return
        items = payload if isinstance(payload, list) else [payload]
        for item in items:
            if not isinstance(item, dict):
                continue
            if item.get("ev") == "status":
                status_value = str(item.get("status", ""))
                if status_value == "auth_success":
                    self.authenticated = True
                    if self.websocket is not None:
                        await self.websocket.send(
                            json.dumps({"action": "subscribe", "params": ",".join(self._topics())})
                        )
                elif status_value in {"auth_failed", "error"}:
                    self.last_error = str(item.get("message") or status_value)[:200]
                continue
            value = parse_massive_index_value(item, self.last_message_at)
            if value is not None:
                old = self.values.get(value.canonical_symbol)
                if old is None or value.source_timestamp >= old.source_timestamp:
                    self.values[value.canonical_symbol] = value
                continue
            candle = parse_massive_index_minute(item, self.last_message_at)
            if candle is not None:
                oldc = self.candles.get(candle.canonical_symbol)
                if oldc is None or candle.start >= oldc.start:
                    self.candles[candle.canonical_symbol] = candle

    def realtime(self, canonical_symbol: str) -> Dict[str, object]:
        canonical = canonical_symbol.upper()
        value = self.values.get(canonical)
        candle = self.candles.get(canonical)
        transport = (
            "WEBSOCKET" if self.authenticated else ("CONNECTING" if self.running else "STOPPED")
        )
        return {
            "source": "massive",
            "canonical_symbol": canonical,
            "status": "OK" if (value or candle) else "MISSING",
            "transport": transport,
            "feed_recency": self.FEED_RECENCY,
            "value": value.to_dict() if value else None,
            "candle": candle.to_dict() if candle else None,
        }

    def health(self) -> Dict[str, object]:
        return {
            "source": "massive",
            "running": self.running,
            "connected": self.websocket is not None,
            "authenticated": self.authenticated,
            "feed_recency": self.FEED_RECENCY,
            "last_message_at": self.last_message_at.isoformat() if self.last_message_at else None,
            "last_error": self.last_error,
        }


massive_indices_ws = MassiveIndicesWsManager()


@api_router.post("/market/index/websocket/start")
async def market_index_ws_start() -> dict:
    try:
        await massive_indices_ws.start()
    except RuntimeError as exc:
        reason = str(exc)
        code = 503 if "API_KEY" in reason else 409
        raise HTTPException(
            status_code=code, detail={"status": "UNAVAILABLE", "reason": reason}
        ) from exc
    return {
        "status": "started",
        "source": "massive",
        "transport": "WEBSOCKET",
        "feed_recency": massive_indices_ws.FEED_RECENCY,
    }


@api_router.post("/market/index/websocket/stop")
async def market_index_ws_stop() -> dict:
    await massive_indices_ws.stop()
    return {"status": "stopped", "source": "massive"}


@api_router.get("/market/index/websocket/health")
async def market_index_ws_health() -> dict:
    return massive_indices_ws.health()


@api_router.get("/market/index/{symbol}/realtime")
async def market_index_realtime(symbol: str) -> dict:
    canonical = symbol.upper()
    inst = instrument_registry.get(canonical)
    if inst is None or inst.asset_class != AssetClass.INDEX:
        raise HTTPException(
            status_code=404,
            detail={"status": "NOT_MAPPED", "reason": "unknown index instrument"},
        )
    return massive_indices_ws.realtime(canonical)


@api_router.get("/market/index/{symbol}/history")
async def market_index_history(
    symbol: str, start: int, end: int, granularity: str = "1d"
) -> dict:
    try:
        result = await fetch_index_history(symbol.upper(), granularity, start, end)
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail={"status": "INVALID", "reason": str(exc)}
        ) from exc
    status = str(result.get("status"))
    http = _METAL_HTTP_STATUS.get(status)  # same status->HTTP mapping
    if http is not None:
        raise HTTPException(
            status_code=http, detail={"status": status, "reason": result.get("reason")}
        )
    return result


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
async def start_server_crypto_market_stream() -> bool:
    """Start the Coinbase ticker stream server-side for autonomous paper monitoring."""
    products = sorted(
        provider_symbol
        for instrument in instrument_registry.all()
        if instrument.asset_class == AssetClass.CRYPTO
        if (
            provider_symbol := provider_symbol_map.to_provider(
                "coinbase", instrument.canonical_symbol
            )
        ) is not None
    )
    if not products:
        log.warning("No Coinbase products registered for server paper monitoring")
        return False
    try:
        await market_ws.subscribe("ticker", products)
        await market_ws.start()
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - REST fallback keeps paper monitoring safe
        log.warning("Server Coinbase WS auto-start failed; REST fallback remains active: %s", exc)
        return False
    log.info("Server Coinbase WS auto-started for paper monitoring: %s", products)
    return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting %s (env=%s)", settings.app_name, settings.environment)
    if settings.is_production and settings.live_trading_enabled:
        raise RuntimeError(
            "live_trading_enabled=True but this build supports paper trading only."
        )
    await market_provider.connect()
    await massive_forex_provider.connect()
    await twelvedata_provider.connect()
    await massive_indices_provider.connect()
    activation = await activate_massive_forex_mappings()
    log.info("Massive forex mapping activation: %s", activation)
    crypto_activation = await activate_verified_crypto_universe()
    log.info("Coinbase crypto universe activation: %s", crypto_activation)
    await start_server_crypto_market_stream()
    try:
        await init_candle_schema()
        persistence_state.mark_ready()
        market_bus.subscribe(persistence_consumer)
    except Exception as exc:  # noqa: BLE001 - explicit, never a silent false success
        persistence_state.mark_init_failed(str(exc))
        log.error("Candle schema init failed; persistence UNAVAILABLE: %s", exc)
    paper_monitor_stop = asyncio.Event()
    paper_monitor_task = asyncio.create_task(
        paper_monitor_loop(paper_monitor_stop), name="paper-monitor"
    )
    global auto_entry_orchestrator_task
    auto_entry_orchestrator_task = asyncio.create_task(
        auto_entry_orchestrator_loop(), name="auto-entry-orchestrator"
    )
    try:
        yield
    finally:
        paper_monitor_stop.set()
        try:
            await paper_monitor_task
        except asyncio.CancelledError:
            pass
        if auto_entry_orchestrator_task is not None:
            auto_entry_orchestrator_task.cancel()
            try:
                await auto_entry_orchestrator_task
            except asyncio.CancelledError:
                pass
            auto_entry_orchestrator_task = None
        await massive_indices_ws.stop()
        await twelvedata_gold_ws.stop()
        await massive_forex_ws.stop()
        await market_ws.stop()
        await market_provider.disconnect()
        await massive_forex_provider.disconnect()
        await twelvedata_provider.disconnect()
        await massive_indices_provider.disconnect()
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


# V16-M5B4-FIX2 — fresh synchronized copy

# V16-M5B17: paper performance analytics from persisted CLOSED trades

# V16-M5B20: UTC DAY/WEEK/MONTH/YEAR paper performance windows


# ============================ V16-M5B23 Market Sessions & Calendar ============
def market_session_context(
    canonical_symbol: str, now_utc: Optional[datetime] = None
) -> Dict[str, object]:
    """Unified, fail-safe session/calendar context for a registered instrument.

    Calendar state never implies data quality or trade permission. Holiday knowledge
    is not fabricated: policies without a verified holiday calendar expose UNKNOWN.
    """
    symbol = canonical_symbol.upper().replace("/", "-")
    inst = instrument_registry.get(symbol)
    if inst is None:
        return {
            "status": "NOT_SUPPORTED",
            "symbol": symbol,
            "market_state": "UNKNOWN",
            "reason": "unknown instrument",
            "paper_only": True,
            "execution": False,
        }
    now = now_utc or utcnow()
    if now.tzinfo is None:
        return {
            "status": "INVALID_TIME",
            "symbol": symbol,
            "market_state": "UNKNOWN",
            "reason": "timezone-aware UTC timestamp required",
            "paper_only": True,
            "execution": False,
        }
    now = now.astimezone(timezone.utc)
    calendar = calendar_for(inst.market_calendar)
    open_state = calendar.is_market_expected_open(int(now.timestamp())).value
    sessions: List[Dict[str, object]] = []
    current_session: Optional[str] = None
    next_open: Optional[str] = None
    next_close: Optional[str] = None
    holidays = "NOT_IMPLEMENTED"

    if inst.market_calendar == MarketCalendarPolicy.ALWAYS_OPEN_24_7:
        market_state = "OPEN"
        current_session = "24_7"
        reason = "registered 24/7 market calendar"
        holidays = "NOT_APPLICABLE"
    elif inst.market_calendar == MarketCalendarPolicy.FOREX_WEEK:
        state = forex_market_state(now)
        market_state = str(state["market_state"])
        reason = str(state["reason"])
        state_sessions = state.get("sessions")
        if isinstance(state_sessions, list):
            sessions = [item for item in state_sessions if isinstance(item, dict)]
        value = state.get("current_session")
        current_session = str(value) if value is not None else None
        open_value = state.get("next_open")
        close_value = state.get("next_close")
        next_open = open_value if isinstance(open_value, str) else None
        next_close = close_value if isinstance(close_value, str) else None
    elif inst.market_calendar == MarketCalendarPolicy.US_EQUITY_RTH:
        market_state = open_state
        reason = "U.S. regular-hours baseline; holidays and early closes unknown"
        if market_state == "OPEN":
            current_session = "US_RTH"
    else:
        market_state = "UNKNOWN"
        reason = "market calendar not configured"

    return {
        "status": "READY",
        "symbol": symbol,
        "asset_class": inst.asset_class.value,
        "calendar_policy": inst.market_calendar.value,
        "market_state": market_state,
        "calendar_open_state": open_state,
        "current_session": current_session,
        "sessions": sessions,
        "next_open": next_open,
        "next_close": next_close,
        "holidays": holidays,
        "observed_at": now.isoformat(),
        "timezone_internal": "UTC",
        "reason": reason,
        "data_quality_independent": True,
        "trade_authorization": False,
        "paper_only": True,
        "execution": False,
        "marker": "SERVER_MARKET_SESSIONS_CALENDAR_V1",
    }


@api_router.get("/market/session-context/{symbol}")
async def market_session_context_endpoint(symbol: str) -> dict:
    result = market_session_context(symbol)
    if result["status"] == "NOT_SUPPORTED":
        raise HTTPException(status_code=404, detail=result)
    return result


# V16-M5B26 — Multi-strategy candidate foundation (paper-only, non-executing)
# New strategy families are observational candidates until independently validated.
MULTI_STRATEGY_ENGINE_VERSION = "SERVER_MULTI_STRATEGY_CANDIDATES_V1"
STRATEGY_MIN_VALIDATION_TRADES = 30


def server_strategy_registry() -> List[Dict[str, object]]:
    """Return the objective server-side strategy catalogue.

    SMC remains the only execution-capable family inherited from the validated
    pipeline. New families are CANDIDATE and cannot auto-queue paper entries.
    """
    return [
        {
            "strategy_id": "SMC_LIQUIDITY_REVERSAL",
            "version": "1.0",
            "status": "ACTIVE_VALIDATED_PIPELINE",
            "family": "REVERSAL",
            "preferred_regimes": ["TREND", "RANGE", "TRANSITION"],
            "execution_eligible": True,
            "paper_only": True,
        },
        {
            "strategy_id": "TREND_PULLBACK",
            "version": TREND_PULLBACK_PAPER_VERSION,
            "status": "ACTIVE_PAPER_UNVALIDATED",
            "family": "CONTINUATION",
            "preferred_regimes": ["TREND"],
            "execution_eligible": True,
            "paper_only": True,
        },
        {
            "strategy_id": "BREAKOUT_EXPANSION",
            "version": "0.1-candidate",
            "status": "CANDIDATE",
            "family": "BREAKOUT",
            "preferred_regimes": ["TREND", "TRANSITION"],
            "preferred_volatility": ["EXPANSION"],
            "execution_eligible": False,
            "paper_only": True,
        },
    ]


def evaluate_candidate_strategy_context(
    strategy_id: str, regime: Dict[str, object]
) -> Dict[str, object]:
    """Regime gate only; this function never creates a trade or signal."""
    catalogue = {str(item["strategy_id"]): item for item in server_strategy_registry()}
    strategy = catalogue.get(strategy_id.upper())
    if strategy is None:
        return {
            "status": "NOT_SUPPORTED",
            "reason": "STRATEGY_NOT_REGISTERED",
            "auto_queue": False,
            "execution": False,
        }
    if regime.get("status") != "READY":
        return {
            "status": "WAIT",
            "strategy_id": strategy["strategy_id"],
            "reason": "REGIME_NOT_READY",
            "auto_queue": False,
            "execution": False,
        }
    allowed_regimes = strategy.get("preferred_regimes", [])
    regime_name = regime.get("regime")
    if isinstance(allowed_regimes, list) and regime_name not in allowed_regimes:
        return {
            "status": "WAIT",
            "strategy_id": strategy["strategy_id"],
            "reason": "REGIME_NOT_ELIGIBLE",
            "regime": regime_name,
            "auto_queue": False,
            "execution": False,
        }
    allowed_volatility = strategy.get("preferred_volatility")
    volatility = regime.get("volatility")
    if isinstance(allowed_volatility, list) and volatility not in allowed_volatility:
        return {
            "status": "WAIT",
            "strategy_id": strategy["strategy_id"],
            "reason": "VOLATILITY_NOT_ELIGIBLE",
            "regime": regime_name,
            "volatility": volatility,
            "auto_queue": False,
            "execution": False,
        }
    return {
        "status": "CONTEXT_ELIGIBLE",
        "strategy_id": strategy["strategy_id"],
        "reason": "REGIME_CONTEXT_MATCH",
        "regime": regime_name,
        "volatility": volatility,
        "candidate_only": not bool(strategy["execution_eligible"]),
        "auto_queue": False,
        "execution": False,
    }


def strategy_validation_gate(closed_trades: int) -> Dict[str, object]:
    """Anti-small-sample gate for future promotion; no performance is invented."""
    count = max(int(closed_trades), 0)
    ready = count >= STRATEGY_MIN_VALIDATION_TRADES
    return {
        "status": "SAMPLE_READY" if ready else "INSUFFICIENT_SAMPLE",
        "closed_trades": count,
        "minimum_closed_trades": STRATEGY_MIN_VALIDATION_TRADES,
        "promotion_authorized": False,
        "execution": False,
        "marker": MULTI_STRATEGY_ENGINE_VERSION,
    }


@api_router.get("/strategies")
async def get_server_strategies() -> Dict[str, object]:
    return {
        "status": "READY",
        "strategies": server_strategy_registry(),
        "marker": MULTI_STRATEGY_ENGINE_VERSION,
        "paper_only": True,
        "live_trading": False,
    }


@api_router.get("/strategies/context/{symbol}")
async def get_strategy_context(symbol: str) -> Dict[str, object]:
    canonical = symbol.upper().replace("/", "-")
    regime = await get_server_market_regime(canonical)
    evaluations = [
        evaluate_candidate_strategy_context(str(item["strategy_id"]), regime)
        for item in server_strategy_registry()
    ]
    return {
        "status": "READY" if regime.get("status") == "READY" else "WAIT",
        "symbol": canonical,
        "regime": regime,
        "strategies": evaluations,
        "auto_queue": False,
        "execution": False,
        "marker": MULTI_STRATEGY_ENGINE_VERSION,
    }


# V16-M5B27 — objective candidate detectors (observational, non-executing)
CANDIDATE_DETECTOR_VERSION = "SERVER_CANDIDATE_DETECTORS_V1"
TREND_PULLBACK_EMA_PERIOD = 20
BREAKOUT_LOOKBACK = 20
BREAKOUT_BODY_MULTIPLIER = 1.5
BREAKOUT_MIN_BODY_RANGE_RATIO = 0.70


def _ema(values: List[float], period: int) -> List[float]:
    """Deterministic EMA series using only values available at each index."""
    if not values or period <= 0:
        return []
    alpha = 2.0 / (period + 1.0)
    result = [values[0]]
    for value in values[1:]:
        result.append(alpha * value + (1.0 - alpha) * result[-1])
    return result


def detect_trend_pullback_candidate(
    candles: List[Candle], now: datetime, regime: Dict[str, object]
) -> Dict[str, object]:
    """Detect a closed-candle trend pullback confirmation without execution."""
    base = {
        "strategy_id": "TREND_PULLBACK",
        "marker": CANDIDATE_DETECTOR_VERSION,
        "candidate_only": True,
        "auto_queue": False,
        "execution": False,
        "no_lookahead": True,
    }
    context = evaluate_candidate_strategy_context("TREND_PULLBACK", regime)
    if context.get("status") != "CONTEXT_ELIGIBLE":
        reason = str(context.get("reason", "CONTEXT_NOT_ELIGIBLE"))
        return {**base, "status": "WAIT", "reason": reason}
    closed = closed_valid_candles(candles, now)
    if len(closed) < TREND_PULLBACK_EMA_PERIOD + 2:
        return {**base, "status": "WAIT", "reason": "INSUFFICIENT_CLOSED_CANDLES"}
    sample = closed[-(TREND_PULLBACK_EMA_PERIOD + 2):]
    closes = [float(item.close) for item in sample if item.close is not None]
    if len(closes) != len(sample) or any(value <= 0 for value in closes):
        return {**base, "status": "WAIT", "reason": "INVALID_CLOSE_SERIES"}
    ema = _ema(closes, TREND_PULLBACK_EMA_PERIOD)
    previous, latest = sample[-2], sample[-1]
    previous_ema, latest_ema = ema[-2], ema[-1]
    if previous.low is None or previous.high is None or latest.close is None:
        return {**base, "status": "WAIT", "reason": "INVALID_PULLBACK_CANDLE"}
    direction = str(regime.get("direction") or "")
    if direction == "BULLISH":
        touched = previous.low <= previous_ema
        confirmed = latest.close > latest_ema and latest.close > previous.high
    elif direction == "BEARISH":
        touched = previous.high >= previous_ema
        confirmed = latest.close < latest_ema and latest.close < previous.low
    else:
        return {**base, "status": "WAIT", "reason": "TREND_DIRECTION_NOT_READY"}
    if not touched:
        return {**base, "status": "WAIT", "reason": "PULLBACK_NOT_TOUCHED"}
    if not confirmed:
        return {**base, "status": "WAIT", "reason": "PULLBACK_NOT_CONFIRMED"}
    return {
        **base,
        "status": "SETUP",
        "reason": "TREND_PULLBACK_CONFIRMED",
        "direction": direction,
        "ema_period": TREND_PULLBACK_EMA_PERIOD,
        "ema": round(latest_ema, 8),
        "latest_closed_timestamp": latest.start.isoformat() if latest.start else None,
    }


def detect_breakout_expansion_candidate(
    candles: List[Candle], now: datetime, regime: Dict[str, object]
) -> Dict[str, object]:
    """Detect a closed-candle range breakout with objective displacement."""
    base = {
        "strategy_id": "BREAKOUT_EXPANSION",
        "marker": CANDIDATE_DETECTOR_VERSION,
        "candidate_only": True,
        "auto_queue": False,
        "execution": False,
        "no_lookahead": True,
    }
    context = evaluate_candidate_strategy_context("BREAKOUT_EXPANSION", regime)
    if context.get("status") != "CONTEXT_ELIGIBLE":
        reason = str(context.get("reason", "CONTEXT_NOT_ELIGIBLE"))
        return {**base, "status": "WAIT", "reason": reason}
    closed = closed_valid_candles(candles, now)
    if len(closed) < BREAKOUT_LOOKBACK + 1:
        return {**base, "status": "WAIT", "reason": "INSUFFICIENT_CLOSED_CANDLES"}
    prior = closed[-(BREAKOUT_LOOKBACK + 1):-1]
    latest = closed[-1]
    if any(
        item.high is None
        or item.low is None
        or item.open is None
        or item.close is None
        for item in prior
    ):
        return {**base, "status": "WAIT", "reason": "INVALID_BREAKOUT_HISTORY"}
    if latest.high is None or latest.low is None or latest.open is None or latest.close is None:
        return {**base, "status": "WAIT", "reason": "INVALID_BREAKOUT_CANDLE"}
    range_high = max(float(item.high) for item in prior if item.high is not None)
    range_low = min(float(item.low) for item in prior if item.low is not None)
    bodies = [
        abs(float(item.close) - float(item.open))
        for item in prior
        if item.close is not None and item.open is not None
    ]
    mean_body = sum(bodies) / len(bodies) if bodies else 0.0
    body = abs(float(latest.close) - float(latest.open))
    candle_range = float(latest.high) - float(latest.low)
    body_ratio = body / candle_range if candle_range > 0 else 0.0
    displaced = (
        mean_body > 0
        and body >= BREAKOUT_BODY_MULTIPLIER * mean_body
        and body_ratio >= BREAKOUT_MIN_BODY_RANGE_RATIO
    )
    bullish = float(latest.close) > range_high and float(latest.close) > float(latest.open)
    bearish = float(latest.close) < range_low and float(latest.close) < float(latest.open)
    if not bullish and not bearish:
        return {**base, "status": "WAIT", "reason": "BREAKOUT_NOT_CONFIRMED"}
    if not displaced:
        return {**base, "status": "WAIT", "reason": "BREAKOUT_NO_DISPLACEMENT"}
    direction = "BULLISH" if bullish else "BEARISH"
    return {
        **base,
        "status": "SETUP",
        "reason": "BREAKOUT_EXPANSION_CONFIRMED",
        "direction": direction,
        "range_high": range_high,
        "range_low": range_low,
        "body_ratio": round(body_ratio, 6),
        "body_multiple": round(body / mean_body, 6),
        "latest_closed_timestamp": latest.start.isoformat() if latest.start else None,
    }


@api_router.get("/strategies/detect/{symbol}")
async def get_candidate_strategy_detections(symbol: str) -> Dict[str, object]:
    """Run candidate detectors on one real Coinbase closed-candle snapshot."""
    canonical = symbol.upper().replace("/", "-")
    instrument = instrument_registry.get(canonical)
    if instrument is None:
        return {"status": "UNAVAILABLE", "symbol": canonical, "reason": "INSTRUMENT_NOT_REGISTERED"}
    if instrument.asset_class != AssetClass.CRYPTO:
        return {
            "status": "NOT_SUPPORTED",
            "symbol": canonical,
            "reason": "CANDIDATE_DETECTORS_CRYPTO_ONLY_V1",
        }
    provider_symbol = provider_symbol_map.to_provider("coinbase", canonical)
    if provider_symbol is None:
        return {
            "status": "UNAVAILABLE",
            "symbol": canonical,
            "reason": "PROVIDER_SYMBOL_NOT_MAPPED",
        }
    try:
        candles, quality = await market_provider.get_candles(
            provider_symbol, SERVER_SETUP_GRANULARITY, SERVER_SETUP_CANDLE_LIMIT
        )
    except (httpx.HTTPError, ValueError):
        return {"status": "UNAVAILABLE", "symbol": canonical, "reason": "CANDLES_UNAVAILABLE"}
    if quality != DataQualityStatus.VALID:
        return {
            "status": "WAIT",
            "symbol": canonical,
            "reason": "CANDLES_NOT_VALID",
            "quality": quality.value,
        }
    latest_quality = _latest_quality(candles)
    if latest_quality != DataQualityStatus.VALID.value:
        return {
            "status": "WAIT",
            "symbol": canonical,
            "reason": "LATEST_CANDLE_NOT_FRESH",
            "quality": latest_quality,
        }
    now = utcnow()
    regime = classify_server_market_regime(candles, now)
    detections = [
        detect_trend_pullback_candidate(candles, now, regime),
        detect_breakout_expansion_candidate(candles, now, regime),
    ]
    return {
        "status": "READY" if regime.get("status") == "READY" else "WAIT",
        "symbol": canonical,
        "source": "coinbase",
        "granularity": SERVER_SETUP_GRANULARITY,
        "quality": quality.value,
        "regime": regime,
        "detections": detections,
        "candidate_only": True,
        "auto_queue": False,
        "execution": False,
        "paper_only": True,
        "marker": CANDIDATE_DETECTOR_VERSION,
    }



# V16-M5B29B — Multi-Asset Strategy Analysis (observational, non-executing)
# ---------------------------------------------------------------------------
# Purpose: run the existing objective SMC / Trend Pullback / Breakout Expansion
# detectors against REAL provider candles for FOREX, METAL and INDEX instruments.
# This layer deliberately does NOT create, queue or size a multi-asset paper trade.
# M5B29A fail-closed sizing rules remain authoritative until a later execution
# milestone independently validates conversions, contract values and SL/TP P&L.
MULTI_ASSET_ANALYSIS_VERSION = "SERVER_MULTI_ASSET_STRATEGY_ANALYSIS_V1_FIX1"
MULTI_ASSET_ANALYSIS_INTERVAL_SECONDS = 300.0
_multi_asset_analysis_last_run_monotonic = 0.0
_multi_asset_decision_fingerprints: Dict[str, str] = {}
multi_asset_analysis_runtime: Dict[str, object] = {
    "runs": 0,
    "last_started_at": None,
    "last_completed_at": None,
    "last_error": None,
    "last_summary": None,
}


def _analysis_candle_from_dict(item: Any) -> Optional[Candle]:
    """Convert a provider-neutral history row to the established Candle model.

    No OHLC value, timestamp or quality is invented. Volume is optional because
    Gold/index feeds legitimately do not always expose it and the three detectors
    used here do not depend on volume.
    """
    if not isinstance(item, dict):
        return None
    start = parse_iso8601(item.get("start"))
    open_ = _to_float(item.get("open"))
    high = _to_float(item.get("high"))
    low = _to_float(item.get("low"))
    close = _to_float(item.get("close"))
    volume = _to_float(item.get("volume")) if item.get("volume") is not None else None
    if start is None or open_ is None or high is None or low is None or close is None:
        return None
    if min(open_, high, low, close) < 0 or high < low:
        return None
    if not (low <= open_ <= high and low <= close <= high):
        return None
    raw_quality = str(item.get("quality") or DataQualityStatus.VALID.value)
    try:
        status_value = DataQualityStatus(raw_quality)
    except ValueError:
        status_value = DataQualityStatus.UNKNOWN
    if status_value in {
        DataQualityStatus.INVALID,
        DataQualityStatus.MISSING,
        DataQualityStatus.CONFLICTED,
        DataQualityStatus.UNKNOWN,
    }:
        return None
    return Candle(start, low, high, open_, close, volume, status_value)


def _analysis_closed_candles(
    candles: List[Candle], granularity: str, now: datetime, limit: int
) -> List[Candle]:
    """Keep only genuinely closed, structurally valid bars for one timeframe."""
    bucket = GRANULARITIES[granularity][1]
    usable = {DataQualityStatus.VALID, DataQualityStatus.STALE}
    kept = [
        candle
        for candle in candles
        if candle.start is not None
        and candle.status in usable
        and candle.open is not None
        and candle.high is not None
        and candle.low is not None
        and candle.close is not None
        and candle.start + timedelta(seconds=bucket) <= now
    ]
    kept.sort(key=lambda item: item.start or datetime.min.replace(tzinfo=timezone.utc))
    return kept[-limit:]


def _analysis_history_span_seconds(
    asset_class: AssetClass, granularity: str, limit: int
) -> int:
    bucket = GRANULARITIES[granularity][1]
    bare = bucket * max(limit + 10, 1)
    # Session-aware markets need a wider wall-clock range than their bar count.
    # These are retrieval windows only, never fabricated candles.
    if asset_class == AssetClass.INDEX:
        return max(bare * 6, 21 * 86400)
    if asset_class in {AssetClass.FOREX, AssetClass.METAL}:
        return max(bare * 3, 10 * 86400)
    return bare * 2


def _analysis_latest_quality(
    candles: List[Candle], granularity: str, asset_class: AssetClass, now: datetime
) -> str:
    if not candles or candles[-1].start is None:
        return DataQualityStatus.MISSING.value
    bucket = GRANULARITIES[granularity][1]
    # Massive index values may be 15-minute delayed. This budget only controls
    # observational analysis and never authorizes an entry.
    budget = bucket * 4
    if asset_class == AssetClass.INDEX:
        budget = max(budget, 25 * 60)
    closed_at = candles[-1].start + timedelta(seconds=bucket)
    age = max(0.0, (now - closed_at).total_seconds())
    return DataQualityStatus.VALID.value if age <= budget else DataQualityStatus.STALE.value


async def _fetch_multi_asset_analysis_candles(
    canonical: str, granularity: str, limit: int, now: datetime
) -> Dict[str, object]:
    """Fetch REAL candles from the provider mapped to the instrument class."""
    instrument = instrument_registry.get(canonical)
    if instrument is None:
        return {"status": "UNAVAILABLE", "reason": "INSTRUMENT_NOT_REGISTERED"}
    if granularity not in GRANULARITIES:
        return {"status": "UNAVAILABLE", "reason": "GRANULARITY_NOT_SUPPORTED"}

    source: Optional[str] = None
    raw_rows: List[Dict[str, object]] = []
    provider_status = "OK"
    provider_reason: Optional[str] = None

    try:
        if instrument.asset_class == AssetClass.CRYPTO:
            source = "coinbase"
            provider_symbol = provider_symbol_map.to_provider(source, canonical)
            if provider_symbol is None:
                return {"status": "UNAVAILABLE", "reason": "PROVIDER_SYMBOL_NOT_MAPPED"}
            candles, quality = await market_provider.get_candles(
                provider_symbol, granularity, limit
            )
            closed = _analysis_closed_candles(candles, granularity, now, limit)
            return {
                "status": "OK" if closed else "EMPTY",
                "source": source,
                "quality": quality.value,
                "latest_quality": _analysis_latest_quality(
                    closed, granularity, instrument.asset_class, now
                ),
                "candles": closed,
            }

        span = _analysis_history_span_seconds(instrument.asset_class, granularity, limit)
        end = int(now.timestamp())
        start = end - span

        if instrument.asset_class == AssetClass.FOREX:
            source = "massive"
            forex_history = await fetch_forex_history(canonical, granularity, start, end)
            provider_status = str(forex_history.get("status") or "UNAVAILABLE")
            provider_reason = (
                str(forex_history.get("reason"))
                if forex_history.get("reason") is not None
                else None
            )
            rows = forex_history.get("candles")
            raw_rows = rows if isinstance(rows, list) else []
        elif instrument.asset_class == AssetClass.METAL:
            source = "twelvedata"
            metal_history = await fetch_metal_history(canonical, granularity, start, end)
            provider_status = str(metal_history.result.get("status") or "UNAVAILABLE")
            provider_reason = (
                str(metal_history.result.get("reason"))
                if metal_history.result.get("reason") is not None
                else None
            )
            rows = metal_history.result.get("candles")
            raw_rows = rows if isinstance(rows, list) else []
        elif instrument.asset_class == AssetClass.INDEX:
            source = "massive"
            index_history = await fetch_index_history(canonical, granularity, start, end)
            provider_status = str(index_history.get("status") or "UNAVAILABLE")
            provider_reason = (
                str(index_history.get("reason"))
                if index_history.get("reason") is not None
                else None
            )
            rows = index_history.get("candles")
            raw_rows = rows if isinstance(rows, list) else []
        else:
            return {"status": "UNAVAILABLE", "reason": "ASSET_CLASS_NOT_SUPPORTED"}
    except (httpx.HTTPError, ValueError) as exc:
        log.warning(
            "Multi-asset analysis candle fetch failed for %s/%s: %s",
            canonical,
            granularity,
            type(exc).__name__,
        )
        return {
            "status": "UNAVAILABLE",
            "source": source,
            "reason": "PROVIDER_CANDLES_UNAVAILABLE",
        }

    if provider_status != "OK":
        return {
            "status": "UNAVAILABLE" if provider_status not in {"EMPTY"} else "EMPTY",
            "source": source,
            "provider_status": provider_status,
            "reason": provider_reason or f"PROVIDER_{provider_status}",
            "candles": [],
        }

    parsed = [c for item in raw_rows if (c := _analysis_candle_from_dict(item)) is not None]
    closed = _analysis_closed_candles(parsed, granularity, now, limit)
    return {
        "status": "OK" if closed else "EMPTY",
        "source": source,
        "provider_status": provider_status,
        "latest_quality": _analysis_latest_quality(
            closed, granularity, instrument.asset_class, now
        ),
        "candles": closed,
    }


def _multi_asset_detection_state(detector: Dict[str, object]) -> Tuple[str, str]:
    strategy_id = str(detector.get("strategy_id") or "")
    if strategy_id == "SMC_LIQUIDITY_REVERSAL":
        state = str(detector.get("setup_state") or detector.get("status") or "WAIT")
        return state, _decision_reason_from_detector(detector)
    status_value = str(detector.get("status") or "WAIT")
    state = "SETUP" if status_value == "SETUP" else "WAIT"
    return state, str(detector.get("reason") or "WAITING_FOR_VALID_SETUP")


async def analyze_multi_asset_strategy_symbol(symbol: str) -> Dict[str, object]:
    """Analyze one registered instrument with all three strategy families.

    The result is observational. Even an SMC ENTRY_NOW or candidate SETUP has
    auto_queue=False and execution=False in this milestone.
    """
    canonical = symbol.upper().replace("/", "-")
    instrument = instrument_registry.get(canonical)
    now = utcnow()
    if instrument is None:
        return {
            "status": "UNAVAILABLE",
            "symbol": canonical,
            "reason": "INSTRUMENT_NOT_REGISTERED",
            "execution": False,
        }

    session = market_session_context(canonical, now)
    market_state = str(session.get("market_state") or "UNKNOWN")
    if market_state == "CLOSED":
        return {
            "status": "WAIT",
            "symbol": canonical,
            "asset_class": instrument.asset_class.value,
            "reason": "MARKET_CLOSED",
            "session": session,
            "detections": [],
            "analysis_only": True,
            "auto_queue": False,
            "execution": False,
            "marker": MULTI_ASSET_ANALYSIS_VERSION,
        }

    ltf_fetch = await _fetch_multi_asset_analysis_candles(
        canonical, SERVER_SETUP_GRANULARITY, SERVER_SETUP_CANDLE_LIMIT, now
    )
    if ltf_fetch.get("status") != "OK":
        return {
            "status": "WAIT" if ltf_fetch.get("status") == "EMPTY" else "UNAVAILABLE",
            "symbol": canonical,
            "asset_class": instrument.asset_class.value,
            "source": ltf_fetch.get("source"),
            "reason": str(ltf_fetch.get("reason") or "CANDLES_UNAVAILABLE"),
            "provider_status": ltf_fetch.get("provider_status"),
            "session": session,
            "detections": [],
            "analysis_only": True,
            "auto_queue": False,
            "execution": False,
            "marker": MULTI_ASSET_ANALYSIS_VERSION,
        }

    ltf_candles = ltf_fetch.get("candles")
    if not isinstance(ltf_candles, list) or not ltf_candles:
        return {
            "status": "WAIT",
            "symbol": canonical,
            "asset_class": instrument.asset_class.value,
            "source": ltf_fetch.get("source"),
            "reason": "NO_CLOSED_CANDLES",
            "session": session,
            "detections": [],
            "analysis_only": True,
            "auto_queue": False,
            "execution": False,
            "marker": MULTI_ASSET_ANALYSIS_VERSION,
        }

    latest_quality = str(ltf_fetch.get("latest_quality") or DataQualityStatus.UNKNOWN.value)
    if latest_quality != DataQualityStatus.VALID.value:
        return {
            "status": "WAIT",
            "symbol": canonical,
            "asset_class": instrument.asset_class.value,
            "source": ltf_fetch.get("source"),
            "reason": "LATEST_CANDLE_NOT_FRESH",
            "quality": latest_quality,
            "session": session,
            "detections": [],
            "analysis_only": True,
            "auto_queue": False,
            "execution": False,
            "marker": MULTI_ASSET_ANALYSIS_VERSION,
        }

    regime = classify_server_market_regime(ltf_candles, now)
    smc_raw = detect_server_market_structure(ltf_candles, now)
    smc: Dict[str, object] = {
        **smc_raw,
        "strategy_id": "SMC_LIQUIDITY_REVERSAL",
        "strategy_version": "1.0",
        "analysis_only": True,
        "auto_queue": False,
        "execution": False,
    }

    # Preserve the existing HTF gate for an SMC ENTRY_NOW. We fetch 1h only when
    # the LTF chain has already reached ENTRY_NOW, reducing provider load.
    if smc.get("setup_state") == "ENTRY_NOW":
        htf_fetch = await _fetch_multi_asset_analysis_candles(
            canonical, SERVER_HTF_GRANULARITY, SERVER_HTF_CANDLE_LIMIT, now
        )
        if htf_fetch.get("status") != "OK":
            smc = apply_htf_context_to_ltf_setup(
                smc,
                {
                    "status": "WAIT",
                    "reason": str(htf_fetch.get("reason") or "HTF_CANDLES_UNAVAILABLE"),
                },
            )
        else:
            htf_candles = htf_fetch.get("candles")
            if isinstance(htf_candles, list) and htf_candles:
                htf_context = classify_server_htf_context(htf_candles, now)
                smc = apply_htf_context_to_ltf_setup(smc, htf_context)
            else:
                smc = apply_htf_context_to_ltf_setup(
                    smc, {"status": "WAIT", "reason": "HTF_CANDLES_UNAVAILABLE"}
                )
        smc.update(
            {
                "strategy_id": "SMC_LIQUIDITY_REVERSAL",
                "strategy_version": "1.0",
                "analysis_only": True,
                "auto_queue": False,
                "execution": False,
            }
        )

    trend = detect_trend_pullback_candidate(ltf_candles, now, regime)
    trend.update({"analysis_only": True, "auto_queue": False, "execution": False})
    breakout = detect_breakout_expansion_candidate(ltf_candles, now, regime)
    breakout.update({"analysis_only": True, "auto_queue": False, "execution": False})
    detections = [smc, trend, breakout]

    return {
        "status": "READY" if regime.get("status") == "READY" else "WAIT",
        "symbol": canonical,
        "asset_class": instrument.asset_class.value,
        "source": ltf_fetch.get("source"),
        "granularity": SERVER_SETUP_GRANULARITY,
        "quality": latest_quality,
        "session": session,
        "regime": regime,
        "detections": detections,
        "analysis_only": True,
        "paper_only": True,
        "auto_queue": False,
        "execution": False,
        "marker": MULTI_ASSET_ANALYSIS_VERSION,
    }


async def _persist_multi_asset_analysis_decisions(result: Dict[str, object]) -> int:
    """Persist only a strategy state change/new candle, avoiding 5-minute spam."""
    symbol = str(result.get("symbol") or "")
    detections = result.get("detections")
    if not symbol or not isinstance(detections, list):
        return 0
    written = 0
    for detector in detections:
        if not isinstance(detector, dict):
            continue
        strategy_id = str(detector.get("strategy_id") or "UNKNOWN_STRATEGY")
        state, reason = _multi_asset_detection_state(detector)
        latest = detector.get("latest_closed_timestamp")
        fingerprint_raw = f"{symbol}|{strategy_id}|{state}|{reason}|{latest}"
        fingerprint = hashlib.sha256(fingerprint_raw.encode("utf-8")).hexdigest()
        key = f"{symbol}|{strategy_id}"
        if _multi_asset_decision_fingerprints.get(key) == fingerprint:
            continue
        _multi_asset_decision_fingerprints[key] = fingerprint
        context = dict(detector)
        context.update(
            {
                "setup_state": state,
                "strategy_id": strategy_id,
                "asset_class": result.get("asset_class"),
                "source": result.get("source"),
                "granularity": result.get("granularity"),
                "regime": result.get("regime"),
                "session": result.get("session"),
                "analysis_only": True,
                "execution": False,
                "marker": MULTI_ASSET_ANALYSIS_VERSION,
            }
        )
        await record_and_persist_auto_decision_trace(symbol, state, reason, context)
        written += 1
    return written


async def run_multi_asset_strategy_analysis_once(force: bool = False) -> Dict[str, object]:
    """Scan registered non-crypto instruments at a safe 5-minute cadence."""
    global _multi_asset_analysis_last_run_monotonic
    now_mono = time.monotonic()
    elapsed = now_mono - _multi_asset_analysis_last_run_monotonic
    if (
        not force
        and _multi_asset_analysis_last_run_monotonic
        and elapsed < MULTI_ASSET_ANALYSIS_INTERVAL_SECONDS
    ):
        return {
            "status": "SKIPPED",
            "reason": "ANALYSIS_INTERVAL_NOT_ELAPSED",
            "next_in_seconds": round(MULTI_ASSET_ANALYSIS_INTERVAL_SECONDS - elapsed, 3),
            "execution": False,
        }

    _multi_asset_analysis_last_run_monotonic = now_mono
    multi_asset_analysis_runtime["last_started_at"] = utcnow().isoformat()
    multi_asset_analysis_runtime["last_error"] = None
    checked = ready = waiting = unavailable = persisted = 0
    by_class: Dict[str, int] = {}
    results: List[Dict[str, object]] = []
    try:
        instruments = sorted(
            (
                instrument
                for instrument in instrument_registry.all()
                if instrument.asset_class in {AssetClass.FOREX, AssetClass.METAL, AssetClass.INDEX}
            ),
            key=lambda item: item.canonical_symbol,
        )
        for instrument in instruments:
            checked += 1
            asset_class_value = instrument.asset_class.value
            by_class[asset_class_value] = by_class.get(asset_class_value, 0) + 1
            result = await analyze_multi_asset_strategy_symbol(instrument.canonical_symbol)
            results.append(result)
            status_value = str(result.get("status") or "UNAVAILABLE")
            if status_value == "READY":
                ready += 1
                persisted += await _persist_multi_asset_analysis_decisions(result)
            elif status_value == "WAIT":
                waiting += 1
            else:
                unavailable += 1
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - fail-safe observational scanner boundary
        multi_asset_analysis_runtime["last_error"] = type(exc).__name__
        log.error("Multi-asset strategy analysis iteration failed: %s", exc)

    summary: Dict[str, object] = {
        "status": "READY" if ready else ("WAIT" if waiting else "UNAVAILABLE"),
        "checked": checked,
        "ready": ready,
        "waiting": waiting,
        "unavailable": unavailable,
        "decisions_persisted": persisted,
        "by_asset_class": by_class,
        "results": results,
        "analysis_only": True,
        "paper_only": True,
        "auto_queue": False,
        "execution": False,
        "marker": MULTI_ASSET_ANALYSIS_VERSION,
    }
    runs = multi_asset_analysis_runtime.get("runs", 0)
    multi_asset_analysis_runtime["runs"] = int(runs) + 1 if isinstance(runs, int) else 1
    multi_asset_analysis_runtime["last_summary"] = summary
    multi_asset_analysis_runtime["last_completed_at"] = utcnow().isoformat()
    return summary


@api_router.get("/strategies/multi-asset/detect/{symbol}")
async def get_multi_asset_strategy_analysis(symbol: str) -> Dict[str, object]:
    return await analyze_multi_asset_strategy_symbol(symbol)


@api_router.get("/strategies/multi-asset/runtime")
async def get_multi_asset_strategy_analysis_runtime() -> Dict[str, object]:
    return {
        "status": "READY",
        "interval_seconds": MULTI_ASSET_ANALYSIS_INTERVAL_SECONDS,
        "runtime": dict(multi_asset_analysis_runtime),
        "analysis_only": True,
        "auto_queue": False,
        "execution": False,
        "marker": MULTI_ASSET_ANALYSIS_VERSION,
    }


@api_router.get("/strategies/multi-asset/universe")
async def get_multi_asset_strategy_analysis_universe() -> Dict[str, object]:
    instruments = [
        {
            "symbol": instrument.canonical_symbol,
            "asset_class": instrument.asset_class.value,
            "display_name": instrument.display_name,
            "market_calendar": instrument.market_calendar.value,
        }
        for instrument in sorted(
            instrument_registry.all(), key=lambda item: item.canonical_symbol
        )
        if instrument.asset_class in {AssetClass.FOREX, AssetClass.METAL, AssetClass.INDEX}
    ]
    return {
        "status": "READY",
        "instruments": instruments,
        "count": len(instruments),
        "strategies": ["SMC_LIQUIDITY_REVERSAL", "TREND_PULLBACK", "BREAKOUT_EXPANSION"],
        "analysis_only": True,
        "auto_queue": False,
        "execution": False,
        "marker": MULTI_ASSET_ANALYSIS_VERSION,
    }


# V16-M5B29D — FX conversion + Gold + cash-index paper sizing foundation
# ---------------------------------------------------------------------------
# Paper-only models below never claim broker lots/contracts. Gold is sized in
# native XAU units. Cash indices use an explicit INTERNAL PAPER POINT UNIT where
# one unit earns/loses USD 1 per index point; this is not a broker contract spec.
# Non-USD quoted FX is authorized only when a fresh quote-to-USD conversion can be
# snapshotted atomically with the paper position; close snapshots preserve USD P&L.
MULTI_ASSET_EXTENDED_SIZING_VERSION = "SERVER_MULTI_ASSET_EXTENDED_SIZING_V1"


def multi_asset_extended_sizing_readiness(symbol: str) -> Dict[str, object]:
    canonical = symbol.upper().replace("/", "-")
    instrument = instrument_registry.get(canonical)
    base: Dict[str, object] = {
        "validation": MULTI_ASSET_EXTENDED_SIZING_VERSION,
        "symbol": canonical,
        "paper_only": True,
        "live_trading": False,
        "broker_contract_size_applied": False,
    }
    if instrument is None:
        return {**base, "status": "BLOCKED", "reason": "INSTRUMENT_NOT_REGISTERED"}
    base.update({
        "asset_class": instrument.asset_class.value,
        "base_asset": instrument.base_asset,
        "quote_asset": instrument.quote_asset,
    })
    if instrument.asset_class == AssetClass.FOREX:
        if instrument.quote_asset == "USD":
            return {
                **base, "status": "PREVIEW_READY",
                "reason": "USD_QUOTED_SPOT_UNIT_PNL",
                "sizing_mode": "BASE_UNITS", "pnl_currency": "USD",
            }
        return {
            **base, "status": "CONVERSION_REQUIRED",
            "reason": "REALTIME_QUOTE_TO_USD_CONVERSION_REQUIRED",
            "sizing_mode": "BASE_UNITS",
            "quote_currency": instrument.quote_asset,
            "pnl_currency": "USD",
        }
    if instrument.asset_class == AssetClass.METAL and canonical == "XAU-USD":
        return {
            **base, "status": "PREVIEW_READY",
            "reason": "XAU_USD_NATIVE_UNIT_PNL",
            "sizing_mode": "XAU_UNITS", "pnl_currency": "USD",
            "session_model": "WEEKDAY_SPOT_BASELINE_PLUS_FRESH_PROVIDER_MARK",
        }
    if instrument.asset_class == AssetClass.INDEX and instrument.quote_asset == "USD":
        return {
            **base, "status": "PREVIEW_READY",
            "reason": "INTERNAL_PAPER_INDEX_POINT_MODEL",
            "sizing_mode": "PAPER_POINT_UNITS", "pnl_currency": "USD",
            "paper_point_value_usd": "1",
            "broker_equivalent": False,
        }
    return {**base, "status": "BLOCKED", "reason": "SIZING_MODEL_NOT_SUPPORTED"}


def _fx_quote_to_usd_conversion_symbol(symbol: str) -> Optional[Tuple[str, bool]]:
    instrument = instrument_registry.get(symbol.upper().replace("/", "-"))
    if instrument is None or instrument.asset_class != AssetClass.FOREX:
        return None
    quote = instrument.quote_asset
    if quote == "USD":
        return ("USD", False)
    direct = f"{quote}-USD"
    if instrument_registry.get(direct) is not None:
        return (direct, False)
    inverse = f"USD-{quote}"
    if instrument_registry.get(inverse) is not None:
        return (inverse, True)
    return None


async def realtime_fx_quote_to_usd(symbol: str) -> Dict[str, object]:
    canonical = symbol.upper().replace("/", "-")
    instrument = instrument_registry.get(canonical)
    base: Dict[str, object] = {
        "symbol": canonical, "account_currency": "USD",
        "paper_only": True, "execution": False,
        "validation": MULTI_ASSET_EXTENDED_SIZING_VERSION,
    }
    if instrument is None or instrument.asset_class != AssetClass.FOREX:
        return {**base, "status": "BLOCKED", "reason": "NOT_FOREX"}
    if instrument.quote_asset == "USD":
        return {
            **base, "status": "VALID", "quote_currency": "USD",
            "quote_to_usd": Decimal("1"), "conversion_symbol": "USD",
        }
    route = _fx_quote_to_usd_conversion_symbol(canonical)
    if route is None:
        return {**base, "status": "BLOCKED", "reason": "CONVERSION_ROUTE_NOT_REGISTERED"}
    conversion_symbol, inverse = route
    mark = await paper_mark_from_realtime(conversion_symbol)
    if mark is None or mark.current_price <= 0:
        return {**base, "status": "WAIT", "reason": "CONVERSION_MARK_UNAVAILABLE"}
    if classify_freshness(
        mark.source_timestamp, settings.ticker_max_age_seconds, now=utcnow()
    ) != DataQualityStatus.VALID:
        return {**base, "status": "WAIT", "reason": "CONVERSION_MARK_NOT_FRESH"}
    rate = Decimal("1") / mark.current_price if inverse else mark.current_price
    return {
        **base, "status": "VALID", "quote_currency": instrument.quote_asset,
        "quote_to_usd": rate, "conversion_symbol": conversion_symbol,
        "conversion_price": mark.current_price, "inverse": inverse,
        "source": mark.source, "source_timestamp": mark.source_timestamp,
    }


def calculate_extended_paper_size(
    capital: Decimal, risk_percent: Decimal, entry: Decimal, stop_loss: Decimal,
    readiness: Dict[str, object], quote_to_usd: Decimal = Decimal("1"),
) -> Dict[str, object]:
    if readiness.get("status") != "PREVIEW_READY":
        return {"status": "BLOCKED", "reason": str(readiness.get("reason") or "SIZING_NOT_READY")}
    distance = abs(entry - stop_loss)
    if capital <= 0 or risk_percent <= 0 or risk_percent > Decimal("100"):
        return {"status": "BLOCKED", "reason": "RISK_INVALID"}
    if distance <= 0 or quote_to_usd <= 0:
        return {"status": "BLOCKED", "reason": "STOP_OR_CONVERSION_INVALID"}
    risk_money = capital * risk_percent / Decimal("100")
    units = risk_money / (distance * quote_to_usd)
    if units <= 0:
        return {"status": "BLOCKED", "reason": "SIZE_INVALID"}
    return {
        "status": "VALID", "validation": MULTI_ASSET_EXTENDED_SIZING_VERSION,
        "size": units, "size_unit": readiness.get("sizing_mode"),
        "risk_money": risk_money, "risk_percent": risk_percent,
        "stop_distance": distance, "quote_to_usd": quote_to_usd,
        "pnl_currency": "USD", "paper_only": True, "live_trading": False,
    }


@api_router.get("/paper/multi-asset-sizing/extended-readiness/{symbol}")
async def get_multi_asset_extended_sizing_readiness(symbol: str) -> Dict[str, object]:
    return multi_asset_extended_sizing_readiness(symbol)


@api_router.get("/paper/fx-conversion/{symbol}")
async def get_realtime_fx_quote_to_usd(symbol: str) -> Dict[str, object]:
    return await realtime_fx_quote_to_usd(symbol)


@api_router.get("/paper/fx-conversion-snapshots/{position_id}")
async def get_paper_fx_conversion_snapshots(position_id: str) -> Dict[str, object]:
    if not persistence_state.ready:
        raise HTTPException(
            status_code=503,
            detail={"status": "UNAVAILABLE", "reason": "persistence not ready"},
        )
    async with engine.connect() as conn:
        result = await conn.execute(
            text(
                "SELECT phase, quote_currency, quote_to_usd, conversion_symbol, "
                "conversion_price, inverse, source, source_timestamp, created_at "
                "FROM paper_fx_conversion_snapshots WHERE position_id=:position_id "
                "ORDER BY phase"
            ),
            {"position_id": position_id},
        )
        rows = []
        for row in result.fetchall():
            item = dict(row._mapping)
            for key in ("quote_to_usd", "conversion_price"):
                if item.get(key) is not None:
                    item[key] = str(item[key])
            for key in ("source_timestamp", "created_at"):
                if item.get(key) is not None:
                    item[key] = item[key].isoformat()
            rows.append(item)
    return {
        "status": "OK",
        "position_id": position_id,
        "snapshots": rows,
        "paper_only": True,
        "execution": False,
    }


# V16-M5B29C — Multi-Asset Paper Execution Foundation
# First executable multi-asset slice: USD-quoted spot Forex in native base units.
# This is PAPER ONLY. Gold, cash indices, and non-USD-quoted FX remain fail-closed
# until their missing calendar / contract-value / account-currency rules are verified.
MULTI_ASSET_PAPER_EXECUTION_VERSION = "SERVER_MULTI_ASSET_PAPER_EXECUTION_V3_PERSISTED_FX"
MULTI_ASSET_PAPER_RISK_PERCENT = Decimal("1")


def multi_asset_paper_execution_readiness(symbol: str) -> Dict[str, object]:
    """Authorize only paper models whose USD P&L is deterministic in this build."""
    canonical = symbol.upper().replace("/", "-")
    instrument = instrument_registry.get(canonical)
    observed_at = utcnow()
    base: Dict[str, object] = {
        "validation": MULTI_ASSET_PAPER_EXECUTION_VERSION, "symbol": canonical,
        "observed_at": observed_at.isoformat(), "paper_only": True,
        "live_trading": False, "execution": False, "auto_entry_authorized": False,
    }
    if instrument is None:
        return {**base, "status": "BLOCKED", "reason": "INSTRUMENT_NOT_REGISTERED"}
    base.update({
        "asset_class": instrument.asset_class.value, "base_asset": instrument.base_asset,
        "quote_asset": instrument.quote_asset, "market_calendar": instrument.market_calendar.value,
    })
    if instrument.asset_class == AssetClass.CRYPTO:
        return {**base, "status": "DELEGATED", "reason": "USE_EXISTING_CRYPTO_PAPER_PIPELINE"}
    if instrument.asset_class == AssetClass.FOREX:
        if instrument.market_calendar != MarketCalendarPolicy.FOREX_WEEK:
            return {**base, "status": "BLOCKED", "reason": "MARKET_CALENDAR_NOT_VERIFIED"}
        if provider_symbol_map.to_provider("massive", canonical) is None:
            return {**base, "status": "BLOCKED", "reason": "PROVIDER_SYMBOL_NOT_MAPPED"}
        session = market_session_context(canonical, observed_at)
        if str(session.get("market_state") or "UNKNOWN") != "OPEN":
            return {**base, "status": "BLOCKED", "reason": "MARKET_CLOSED", "session": session}
        if instrument.quote_asset != "USD":
            route = _fx_quote_to_usd_conversion_symbol(canonical)
            if route is None:
                return {
                    **base,
                    "status": "BLOCKED",
                    "reason": "CONVERSION_ROUTE_NOT_REGISTERED",
                    "session": session,
                }
            return {
                **base,
                "status": "READY",
                "reason": "FX_CONVERSION_SNAPSHOT_REQUIRED_AT_ENTRY",
                "sizing_mode": "BASE_UNITS",
                "pnl_currency": "USD",
                "quote_currency": instrument.quote_asset,
                "conversion_symbol": route[0],
                "conversion_inverse": route[1],
                "risk_percent": str(MULTI_ASSET_PAPER_RISK_PERCENT),
                "session": session,
                "auto_entry_authorized": True,
                "requires_conversion_snapshot": True,
            }
        sizing = multi_asset_extended_sizing_readiness(canonical)
        return {
            **base, "status": "READY", "reason": "USD_QUOTED_SPOT_FOREX_PAPER_READY",
            "sizing_mode": sizing.get("sizing_mode"), "pnl_currency": "USD",
            "risk_percent": str(MULTI_ASSET_PAPER_RISK_PERCENT), "session": session,
            "auto_entry_authorized": True,
        }
    if instrument.asset_class == AssetClass.METAL:
        if (
            canonical != "XAU-USD"
            or provider_symbol_map.to_provider("twelvedata", canonical) is None
        ):
            return {**base, "status": "BLOCKED", "reason": "METAL_PROVIDER_NOT_MAPPED"}
        spot_session = forex_market_state(observed_at)
        if str(spot_session.get("market_state") or "UNKNOWN") != "OPEN":
            return {**base, "status": "BLOCKED", "reason": "MARKET_CLOSED", "session": spot_session}
        sizing = multi_asset_extended_sizing_readiness(canonical)
        return {
            **base, "status": "READY", "reason": "XAU_USD_PAPER_UNIT_MODEL_READY",
            "sizing_mode": sizing.get("sizing_mode"), "pnl_currency": "USD",
            "session": spot_session,
            "session_model": "WEEKDAY_SPOT_BASELINE_PLUS_FRESH_PROVIDER_MARK",
            "risk_percent": str(MULTI_ASSET_PAPER_RISK_PERCENT),
            "auto_entry_authorized": True,
        }
    if instrument.asset_class == AssetClass.INDEX:
        if provider_symbol_map.to_provider("massive", canonical) is None:
            return {**base, "status": "BLOCKED", "reason": "INDEX_PROVIDER_NOT_MAPPED"}
        session = market_session_context(canonical, observed_at)
        if str(session.get("market_state") or "UNKNOWN") != "OPEN":
            return {**base, "status": "BLOCKED", "reason": "MARKET_CLOSED", "session": session}
        sizing = multi_asset_extended_sizing_readiness(canonical)
        return {
            **base, "status": "READY", "reason": "INTERNAL_PAPER_INDEX_POINT_MODEL_READY",
            "sizing_mode": sizing.get("sizing_mode"), "pnl_currency": "USD",
            "paper_point_value_usd": "1", "broker_equivalent": False,
            "session": session, "risk_percent": str(MULTI_ASSET_PAPER_RISK_PERCENT),
            "auto_entry_authorized": True,
        }
    return {**base, "status": "BLOCKED", "reason": "ASSET_CLASS_NOT_SUPPORTED"}


@api_router.get("/paper/multi-asset-execution/readiness/{symbol}")
async def get_multi_asset_paper_execution_readiness(symbol: str) -> Dict[str, object]:
    return multi_asset_paper_execution_readiness(symbol)


def _paper_mark_to_market_datum(symbol: str, mark: PaperPositionMark) -> MarketDatum:
    return MarketDatum(
        source=mark.source,
        symbol=symbol,
        value=float(mark.current_price),
        timestamp=mark.source_timestamp,
        status=DataQualityStatus.VALID,
    )


def build_smc_multi_asset_paper_plan(
    detector: Dict[str, object], mark: PaperPositionMark
) -> Dict[str, object]:
    base: Dict[str, object] = {
        "strategy_id": "SMC_LIQUIDITY_REVERSAL",
        "strategy_version": "1.0-paper",
        "paper_only": True,
        "execution": False,
    }
    if detector.get("setup_state") != "ENTRY_NOW":
        return {**base, "status": "WAIT", "reason": "SMC_SETUP_NOT_ENTRY_NOW"}
    trade_plan = detector.get("trade_plan")
    if not isinstance(trade_plan, dict):
        return {**base, "status": "WAIT", "reason": "SMC_TRADE_PLAN_MISSING"}
    try:
        stop = Decimal(str(trade_plan["stop_loss"]))
        target = Decimal(str(trade_plan["take_profit"]))
        zone_low = Decimal(str(trade_plan["entry_zone_low"]))
        zone_high = Decimal(str(trade_plan["entry_zone_high"]))
    except (KeyError, ValueError, InvalidOperation):
        return {**base, "status": "WAIT", "reason": "SMC_TRADE_PLAN_INVALID"}
    entry = mark.current_price
    if not zone_low <= entry <= zone_high:
        return {**base, "status": "WAIT", "reason": "REALTIME_PRICE_OUTSIDE_ENTRY_ZONE"}
    direction = str(detector.get("direction") or "")
    if direction == "BULLISH":
        if not stop < entry < target:
            return {**base, "status": "WAIT", "reason": "SMC_LONG_LEVELS_INVALID"}
        side = "LONG"
    elif direction == "BEARISH":
        if not target < entry < stop:
            return {**base, "status": "WAIT", "reason": "SMC_SHORT_LEVELS_INVALID"}
        side = "SHORT"
    else:
        return {**base, "status": "WAIT", "reason": "SMC_DIRECTION_INVALID"}
    return {
        **base,
        "status": "ENTRY_NOW",
        "side": side,
        "direction": direction,
        "entry": entry,
        "stop_loss": stop,
        "take_profit": target,
        "source_timestamp": mark.source_timestamp,
        "setup_timestamp": detector.get("latest_closed_timestamp"),
        "plan_source": f"SMC_CLOSED_STRUCTURE+REAL_{mark.source.upper()}_MARK",
    }


async def execute_multi_asset_paper_plan(
    symbol: str, plan: Dict[str, object]
) -> Dict[str, object]:
    """Persist one authorized non-crypto PAPER position, fail-closed by class."""
    if not persistence_state.ready:
        return {"status": "BLOCKED", "reason": "PERSISTENCE_NOT_READY"}
    if plan.get("status") != "ENTRY_NOW":
        return {"status": "BLOCKED", "reason": "PLAN_NOT_ENTRY_NOW"}
    canonical = symbol.upper().replace("/", "-")
    readiness = multi_asset_paper_execution_readiness(canonical)
    if readiness.get("status") != "READY" or not readiness.get("auto_entry_authorized"):
        return {
            "status": "BLOCKED",
            "reason": str(readiness.get("reason") or "MULTI_ASSET_EXECUTION_NOT_READY"),
            "readiness": readiness,
        }
    try:
        entry = Decimal(str(plan["entry"]))
        stop = Decimal(str(plan["stop_loss"]))
        target = Decimal(str(plan["take_profit"]))
        source_timestamp = plan["source_timestamp"]
    except (KeyError, ValueError, InvalidOperation):
        return {"status": "BLOCKED", "reason": "PLAN_INVALID"}
    if not isinstance(source_timestamp, datetime) or source_timestamp.tzinfo is None:
        return {"status": "BLOCKED", "reason": "SOURCE_TIMESTAMP_INVALID"}
    side = str(plan.get("side") or "")
    if side == "LONG" and not stop < entry < target:
        return {"status": "BLOCKED", "reason": "LONG_LEVELS_INVALID"}
    if side == "SHORT" and not target < entry < stop:
        return {"status": "BLOCKED", "reason": "SHORT_LEVELS_INVALID"}
    if side not in {"LONG", "SHORT"}:
        return {"status": "BLOCKED", "reason": "SIDE_INVALID"}
    edge_gate = await evaluate_adaptive_edge_active_gate(
        canonical, str(plan.get("strategy_id") or "MULTI_ASSET"), plan.get("performance_timeframe"),
        plan.get("performance_session"), plan.get("performance_market_regime"),
    )
    if edge_gate.get("action") == "BLOCKED":
        return {
            "status": "BLOCKED", "reason": "ADAPTIVE_EDGE_WEAK",
            "adaptive_edge_gate": edge_gate,
        }

    if classify_freshness(
        source_timestamp, settings.ticker_max_age_seconds, now=utcnow()
    ) != DataQualityStatus.VALID:
        return {"status": "BLOCKED", "reason": "REALTIME_MARK_NOT_FRESH"}

    account = await get_paper_account()
    capital = Decimal(str(account["current_capital"]))
    sizing_readiness = multi_asset_extended_sizing_readiness(canonical)
    quote_to_usd = Decimal("1")
    fx_conversion: Optional[Dict[str, object]] = None
    fx_conversion_price: Optional[Decimal] = None
    fx_conversion_source_timestamp: Optional[datetime] = None
    if paper_position_quote_to_usd_required(canonical):
        fx_conversion = await realtime_fx_quote_to_usd(canonical)
        if fx_conversion.get("status") != "VALID":
            return {
                "status": "BLOCKED",
                "reason": str(
                    fx_conversion.get("reason") or "FX_CONVERSION_UNAVAILABLE"
                ),
                "conversion": fx_conversion,
            }
        conversion_rate = fx_conversion.get("quote_to_usd")
        if not isinstance(conversion_rate, Decimal) or conversion_rate <= 0:
            return {"status": "BLOCKED", "reason": "FX_CONVERSION_INVALID"}
        quote_to_usd = conversion_rate
        raw_conversion_price = fx_conversion.get("conversion_price")
        if isinstance(raw_conversion_price, Decimal):
            fx_conversion_price = raw_conversion_price
        raw_conversion_timestamp = fx_conversion.get("source_timestamp")
        if isinstance(raw_conversion_timestamp, datetime):
            fx_conversion_source_timestamp = raw_conversion_timestamp
        sizing_readiness = {**sizing_readiness, "status": "PREVIEW_READY"}
    sizing = calculate_extended_paper_size(
        capital,
        MULTI_ASSET_PAPER_RISK_PERCENT,
        entry,
        stop,
        sizing_readiness,
        quote_to_usd=quote_to_usd,
    )
    if sizing.get("status") != "VALID":
        return {"status": "BLOCKED", "reason": str(sizing.get("reason") or "SIZING_INVALID")}

    strategy_id = str(plan.get("strategy_id") or "MULTI_ASSET")
    strategy_version = str(plan.get("strategy_version") or "unknown")
    async with auto_paper_portfolio_lock:
        open_positions = await get_open_paper_risk_snapshot()
        guard = evaluate_paper_portfolio_risk_guard(
            open_positions,
            canonical,
            Decimal(str(sizing["risk_money"])),
            capital,
        )
        if guard.get("status") != "VALID":
            return {"status": "BLOCKED", "reason": str(guard.get("reason"))}
        position = PaperPositionCreate(
            position_id=build_strategy_paper_position_id(
                strategy_id,
                canonical,
                side,
                plan.get("setup_timestamp"),
            ),
            symbol=canonical,
            side=side,
            entry=entry,
            stop_loss=stop,
            take_profit=target,
            size=Decimal(str(sizing["size"])),
            size_unit=str(sizing.get("size_unit") or "BASE_UNITS"),
            risk_money=Decimal(str(sizing["risk_money"])),
            risk_percent=Decimal(str(sizing["risk_percent"])),
            capital_before=capital,
            source=f"strategy:{strategy_id}@{strategy_version}",
            source_timestamp=source_timestamp,
            opened_at=utcnow(),
            performance_strategy_id=strategy_id,
            performance_strategy_version=strategy_version,
            performance_timeframe=(
                str(plan.get("performance_timeframe"))
                if plan.get("performance_timeframe") is not None
                else None
            ),
            performance_session=(
                str(plan.get("performance_session"))
                if plan.get("performance_session") is not None
                else None
            ),
            performance_market_regime=(
                str(plan.get("performance_market_regime"))
                if plan.get("performance_market_regime") is not None
                else None
            ),
            performance_setup_context=(
                str(plan.get("performance_setup_context"))
                if plan.get("performance_setup_context") is not None
                else None
            ),
            fx_quote_currency=(
                str(fx_conversion.get("quote_currency")) if fx_conversion else None
            ),
            fx_quote_to_usd=quote_to_usd if fx_conversion else None,
            fx_conversion_symbol=(
                str(fx_conversion.get("conversion_symbol")) if fx_conversion else None
            ),
            fx_conversion_price=fx_conversion_price,
            fx_conversion_inverse=(
                bool(fx_conversion.get("inverse", False)) if fx_conversion else None
            ),
            fx_conversion_source=(
                str(fx_conversion.get("source") or "identity")
                if fx_conversion
                else None
            ),
            fx_conversion_source_timestamp=fx_conversion_source_timestamp,
        )
        created = await create_paper_position(position)
        await persist_adaptive_edge_trade_link(position.position_id, edge_gate)
    return {
        "status": "OPENED",
        "strategy_id": strategy_id,
        "strategy_version": strategy_version,
        "position": created,
        "sizing_model": sizing_readiness,
        "paper_only": True,
        "live_trading": False,
        "execution": False,
        "marker": MULTI_ASSET_PAPER_EXECUTION_VERSION,
    }


async def run_multi_asset_paper_generation_once() -> Dict[str, int]:
    """Auto-open verified paper setups from real Forex, Gold and index data."""
    stats = {
        "checked": 0,
        "ready": 0,
        "setups": 0,
        "opened": 0,
        "blocked": 0,
        "conflicts": 0,
    }
    if not persistence_state.ready:
        return stats
    for instrument in instrument_registry.all():
        if instrument.asset_class == AssetClass.CRYPTO:
            continue
        canonical = instrument.canonical_symbol
        stats["checked"] += 1
        readiness = multi_asset_paper_execution_readiness(canonical)
        if readiness.get("status") != "READY":
            continue
        stats["ready"] += 1
        try:
            analysis = await analyze_multi_asset_strategy_symbol(canonical)
            detections = analysis.get("detections")
            if not isinstance(detections, list):
                stats["blocked"] += 1
                continue
            candles_fetch = await _fetch_multi_asset_analysis_candles(
                canonical,
                SERVER_SETUP_GRANULARITY,
                SERVER_SETUP_CANDLE_LIMIT,
                utcnow(),
            )
            candles_obj = candles_fetch.get("candles")
            if not isinstance(candles_obj, list) or not candles_obj:
                stats["blocked"] += 1
                continue
            mark = await paper_mark_from_realtime(canonical)
            if mark is None:
                stats["blocked"] += 1
                continue
            market_datum = _paper_mark_to_market_datum(canonical, mark)
            for detector_obj in detections:
                if not isinstance(detector_obj, dict):
                    continue
                strategy_id = str(detector_obj.get("strategy_id") or "")
                plan: Optional[Dict[str, object]] = None
                if strategy_id == "SMC_LIQUIDITY_REVERSAL":
                    if detector_obj.get("setup_state") != "ENTRY_NOW":
                        continue
                    plan = build_smc_multi_asset_paper_plan(detector_obj, mark)
                elif strategy_id == "TREND_PULLBACK":
                    if detector_obj.get("status") != "SETUP":
                        continue
                    plan = build_trend_pullback_paper_plan(
                        candles_obj,
                        utcnow(),
                        detector_obj,
                        market_datum,
                    )
                    plan["plan_source"] = (
                        f"CLOSED_PULLBACK_STRUCTURE+REAL_{mark.source.upper()}_MARK"
                    )
                elif strategy_id == "BREAKOUT_EXPANSION":
                    if detector_obj.get("status") != "SETUP":
                        continue
                    plan = build_breakout_expansion_paper_plan(
                        detector_obj,
                        market_datum,
                        utcnow(),
                    )
                    plan["plan_source"] = f"CLOSED_BREAKOUT_RANGE+REAL_{mark.source.upper()}_MARK"
                if plan is None or plan.get("status") != "ENTRY_NOW":
                    continue
                session_obj = analysis.get("session")
                session_value = (
                    session_obj.get("current_session")
                    if isinstance(session_obj, dict)
                    else None
                )
                regime_obj = analysis.get("regime")
                regime_value = (
                    regime_obj.get("regime") if isinstance(regime_obj, dict) else None
                )
                plan["performance_timeframe"] = str(
                    analysis.get("granularity") or SERVER_SETUP_GRANULARITY
                )
                plan["performance_session"] = (
                    str(session_value) if session_value is not None else None
                )
                plan["performance_market_regime"] = (
                    str(regime_value) if regime_value is not None else None
                )
                plan["performance_setup_context"] = json.dumps(
                    {
                        "asset_class": analysis.get("asset_class"),
                        "strategy_id": strategy_id,
                        "detector": detector_obj,
                    },
                    default=str,
                    sort_keys=True,
                )
                stats["setups"] += 1
                try:
                    result = await execute_multi_asset_paper_plan(canonical, plan)
                except HTTPException as exc:
                    if exc.status_code == 409:
                        stats["conflicts"] += 1
                    else:
                        stats["blocked"] += 1
                    continue
                if result.get("status") == "OPENED":
                    stats["opened"] += 1
                else:
                    stats["blocked"] += 1
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            log.error("Multi-asset paper generation failed for %s: %s", canonical, exc)
            stats["blocked"] += 1
    return stats


@api_router.get("/paper/multi-asset-execution/status")
async def get_multi_asset_paper_execution_status() -> Dict[str, object]:
    instruments: List[Dict[str, object]] = []
    for instrument in instrument_registry.all():
        if instrument.asset_class == AssetClass.CRYPTO:
            continue
        instruments.append(multi_asset_paper_execution_readiness(instrument.canonical_symbol))
    return {
        "status": "ACTIVE_PAPER_FOUNDATION",
        "marker": MULTI_ASSET_PAPER_EXECUTION_VERSION,
        "risk_percent": str(MULTI_ASSET_PAPER_RISK_PERCENT),
        "authorized_scope": "USD_FX+XAU_UNITS+INTERNAL_INDEX_POINT_UNITS",
        "non_usd_fx": "CONVERSION_PREVIEW_ONLY_UNTIL_SNAPSHOT_PERSISTENCE",
        "strategies": ["SMC_LIQUIDITY_REVERSAL", "TREND_PULLBACK", "BREAKOUT_EXPANSION"],
        "instruments": instruments,
        "paper_only": True,
        "live_trading": False,
        "execution": False,
    }


# V16-M5B29F — Runtime validation + multi-asset monitoring
MULTI_ASSET_MONITORING_VERSION = "SERVER_MULTI_ASSET_RUNTIME_MONITORING_V1"


def _monitoring_detection_payload(detector: Dict[str, object]) -> Dict[str, object]:
    state, reason = _multi_asset_detection_state(detector)
    return {
        "strategy_id": str(detector.get("strategy_id") or "UNKNOWN_STRATEGY"),
        "strategy_version": str(detector.get("strategy_version") or "unknown"),
        "state": state,
        "reason": reason,
        "direction": detector.get("direction"),
        "latest_closed_timestamp": detector.get("latest_closed_timestamp"),
        "setup_timestamp": detector.get("setup_timestamp"),
    }


@api_router.get("/paper/multi-asset-monitoring")
async def get_multi_asset_runtime_monitoring() -> Dict[str, object]:
    """One fail-safe UI payload for observed multi-asset analysis/execution state."""
    observed_at = utcnow()
    runtime_summary = multi_asset_analysis_runtime.get("last_summary")
    analysis_by_symbol: Dict[str, Dict[str, object]] = {}
    if isinstance(runtime_summary, dict):
        raw_results = runtime_summary.get("results")
        if isinstance(raw_results, list):
            for item in raw_results:
                if isinstance(item, dict) and item.get("symbol"):
                    analysis_by_symbol[str(item["symbol"])] = item

    open_by_symbol: Dict[str, List[Dict[str, object]]] = {}
    persistence_reason: Optional[str] = None
    if persistence_state.ready:
        try:
            open_payload = await list_paper_positions(status_filter="OPEN")
            raw_positions = open_payload.get("positions")
            if isinstance(raw_positions, list):
                for row in raw_positions:
                    if not isinstance(row, dict):
                        continue
                    symbol = str(row.get("symbol") or "")
                    if symbol:
                        open_by_symbol.setdefault(symbol, []).append(row)
        except Exception as exc:  # noqa: BLE001 - monitoring must stay fail-safe
            persistence_reason = type(exc).__name__
    else:
        persistence_reason = "PERSISTENCE_NOT_READY"

    instruments: List[Dict[str, object]] = []
    for instrument in sorted(
        instrument_registry.all(), key=lambda item: item.canonical_symbol
    ):
        if instrument.asset_class == AssetClass.CRYPTO:
            continue
        canonical = instrument.canonical_symbol
        analysis = analysis_by_symbol.get(canonical)
        readiness = multi_asset_paper_execution_readiness(canonical)
        detections: List[Dict[str, object]] = []
        if isinstance(analysis, dict):
            raw_detections = analysis.get("detections")
            if isinstance(raw_detections, list):
                detections = [
                    _monitoring_detection_payload(detector)
                    for detector in raw_detections
                    if isinstance(detector, dict)
                ]
        quality = analysis.get("quality") if isinstance(analysis, dict) else None
        session = analysis.get("session") if isinstance(analysis, dict) else None
        instruments.append(
            {
                "symbol": canonical,
                "display_name": instrument.display_name,
                "asset_class": instrument.asset_class.value,
                "provider": analysis.get("source") if isinstance(analysis, dict) else None,
                "granularity": (
                    analysis.get("granularity") if isinstance(analysis, dict) else None
                ),
                "analysis_status": (
                    str(analysis.get("status") or "UNKNOWN")
                    if isinstance(analysis, dict)
                    else "NOT_SCANNED"
                ),
                "quality": quality,
                "session": session,
                "regime": analysis.get("regime") if isinstance(analysis, dict) else None,
                "detections": detections,
                "execution_readiness": readiness,
                "open_positions": open_by_symbol.get(canonical, []),
                "open_position_count": len(open_by_symbol.get(canonical, [])),
            }
        )

    return {
        "status": "READY" if instruments else "UNAVAILABLE",
        "marker": MULTI_ASSET_MONITORING_VERSION,
        "observed_at": observed_at.isoformat(),
        "analysis_last_completed_at": multi_asset_analysis_runtime.get(
            "last_completed_at"
        ),
        "analysis_runs": multi_asset_analysis_runtime.get("runs", 0),
        "persistence_status": "READY" if persistence_reason is None else "DEGRADED",
        "persistence_reason": persistence_reason,
        "instruments": instruments,
        "count": len(instruments),
        "paper_only": True,
        "live_trading": False,
        "execution": False,
    }


# V16-M5B28B1 — Trend Pullback paper execution (paper-only)
TREND_PULLBACK_PAPER_VERSION = "0.2-paper"
TREND_PULLBACK_RISK_REWARD = Decimal("2")


def build_trend_pullback_paper_plan(
    candles: List[Candle],
    now: datetime,
    detection: Dict[str, object],
    ticker: MarketDatum,
) -> Dict[str, object]:
    """Build an objective Trend Pullback plan from closed candles + real ticker."""
    base = {
        "strategy_id": "TREND_PULLBACK",
        "strategy_version": TREND_PULLBACK_PAPER_VERSION,
        "paper_only": True,
        "execution": False,
    }
    if detection.get("status") != "SETUP":
        return {**base, "status": "WAIT", "reason": "TREND_SETUP_NOT_READY"}
    if ticker.status != DataQualityStatus.VALID or ticker.value is None:
        return {**base, "status": "WAIT", "reason": "REALTIME_TICKER_NOT_VALID"}
    if ticker.timestamp is None or ticker.timestamp.tzinfo is None:
        return {**base, "status": "WAIT", "reason": "REALTIME_TICKER_TIMESTAMP_INVALID"}
    if classify_freshness(
        ticker.timestamp, settings.ticker_max_age_seconds, now=now
    ) != DataQualityStatus.VALID:
        return {**base, "status": "WAIT", "reason": "REALTIME_TICKER_NOT_FRESH"}
    closed = closed_valid_candles(candles, now)
    if len(closed) < 2:
        return {**base, "status": "WAIT", "reason": "PULLBACK_STRUCTURE_MISSING"}
    pullback = closed[-2]
    if pullback.low is None or pullback.high is None:
        return {**base, "status": "WAIT", "reason": "PULLBACK_STRUCTURE_INVALID"}
    entry = Decimal(str(ticker.value))
    direction = str(detection.get("direction") or "")
    if direction == "BULLISH":
        stop = Decimal(str(pullback.low))
        if stop >= entry:
            return {**base, "status": "WAIT", "reason": "BULLISH_STOP_INVALID"}
        risk = entry - stop
        target = entry + TREND_PULLBACK_RISK_REWARD * risk
        side = "LONG"
    elif direction == "BEARISH":
        stop = Decimal(str(pullback.high))
        if stop <= entry:
            return {**base, "status": "WAIT", "reason": "BEARISH_STOP_INVALID"}
        risk = stop - entry
        target = entry - TREND_PULLBACK_RISK_REWARD * risk
        if target <= 0:
            return {**base, "status": "WAIT", "reason": "BEARISH_TARGET_INVALID"}
        side = "SHORT"
    else:
        return {**base, "status": "WAIT", "reason": "TREND_DIRECTION_INVALID"}
    return {
        **base,
        "status": "ENTRY_NOW",
        "side": side,
        "direction": direction,
        "entry": entry,
        "stop_loss": stop,
        "take_profit": target,
        "risk_reward": TREND_PULLBACK_RISK_REWARD,
        "source_timestamp": ticker.timestamp,
        "setup_timestamp": detection.get("latest_closed_timestamp"),
        "plan_source": "CLOSED_PULLBACK_STRUCTURE+REAL_COINBASE_TICKER",
    }


def build_strategy_paper_position_id(
    strategy_id: str, symbol: str, side: str, setup_timestamp: object
) -> str:
    raw = f"{strategy_id}|{symbol}|{side}|{setup_timestamp}".encode()
    digest = hashlib.sha256(raw).hexdigest()[:24]
    return f"strategy-{strategy_id.lower()}-{digest}"


async def execute_trend_pullback_paper_plan(
    symbol: str, plan: Dict[str, object]
) -> Dict[str, object]:
    """Persist one prevalidated Trend Pullback position using existing risk guards."""
    if not persistence_state.ready:
        return {"status": "BLOCKED", "reason": "PERSISTENCE_NOT_READY"}
    if plan.get("status") != "ENTRY_NOW":
        return {"status": "BLOCKED", "reason": "TREND_PLAN_NOT_ENTRY_NOW"}
    canonical = symbol.upper().replace("/", "-")
    try:
        entry = Decimal(str(plan["entry"]))
        stop = Decimal(str(plan["stop_loss"]))
        target = Decimal(str(plan["take_profit"]))
        source_timestamp = plan["source_timestamp"]
    except (KeyError, ValueError, InvalidOperation):
        return {"status": "BLOCKED", "reason": "TREND_PLAN_INVALID"}
    if not isinstance(source_timestamp, datetime) or source_timestamp.tzinfo is None:
        return {"status": "BLOCKED", "reason": "TREND_SOURCE_TIMESTAMP_INVALID"}
    side = str(plan.get("side"))
    if side == "LONG" and not stop < entry < target:
        return {"status": "BLOCKED", "reason": "TREND_LONG_LEVELS_INVALID"}
    if side == "SHORT" and not target < entry < stop:
        return {"status": "BLOCKED", "reason": "TREND_SHORT_LEVELS_INVALID"}
    if side not in {"LONG", "SHORT"}:
        return {"status": "BLOCKED", "reason": "TREND_SIDE_INVALID"}
    edge_gate = await evaluate_adaptive_edge_active_gate(
        canonical, "TREND_PULLBACK", plan.get("performance_timeframe"),
        plan.get("performance_session"), plan.get("performance_market_regime"),
    )
    if edge_gate.get("action") == "BLOCKED":
        return {
            "status": "BLOCKED", "reason": "ADAPTIVE_EDGE_WEAK",
            "adaptive_edge_gate": edge_gate,
        }
    specs = await get_paper_instrument_specs(canonical)
    if specs.get("status") != "VALID":
        return {"status": "BLOCKED", "reason": "SERVER_INSTRUMENT_SPECS_NOT_VALID"}
    account = await get_paper_account()
    capital = Decimal(str(account["current_capital"]))
    sizing = calculate_verified_crypto_size(capital, Decimal("1"), entry, stop, specs)
    if sizing.get("status") != "VALID":
        return {"status": "BLOCKED", "reason": str(sizing.get("reason"))}
    async with auto_paper_portfolio_lock:
        open_positions = await get_open_paper_risk_snapshot()
        guard = evaluate_paper_portfolio_risk_guard(
            open_positions, canonical, Decimal(str(sizing["risk_money"])), capital
        )
        if guard.get("status") != "VALID":
            return {"status": "BLOCKED", "reason": str(guard.get("reason"))}
        position = PaperPositionCreate(
            position_id=build_strategy_paper_position_id(
                "TREND_PULLBACK", canonical, side, plan.get("setup_timestamp")
            ),
            symbol=canonical,
            side=side,
            entry=entry,
            stop_loss=stop,
            take_profit=target,
            size=Decimal(str(sizing["size"])),
            size_unit="BASE_UNITS",
            risk_money=Decimal(str(sizing["risk_money"])),
            risk_percent=Decimal(str(sizing["risk_percent"])),
            capital_before=capital,
            source=f"strategy:TREND_PULLBACK@{TREND_PULLBACK_PAPER_VERSION}",
            source_timestamp=source_timestamp,
            opened_at=utcnow(),
            performance_strategy_id="TREND_PULLBACK",
            performance_strategy_version=TREND_PULLBACK_PAPER_VERSION,
            performance_timeframe=(
                str(plan.get("performance_timeframe"))
                if plan.get("performance_timeframe") is not None
                else None
            ),
            performance_session=(
                str(plan.get("performance_session"))
                if plan.get("performance_session") is not None
                else None
            ),
            performance_market_regime=(
                str(plan.get("performance_market_regime"))
                if plan.get("performance_market_regime") is not None
                else None
            ),
            performance_setup_context=(
                str(plan.get("performance_setup_context"))
                if plan.get("performance_setup_context") is not None
                else None
            ),
        )
        created = await create_paper_position(position)
        await persist_adaptive_edge_trade_link(position.position_id, edge_gate)
    return {
        "status": "OPENED",
        "strategy_id": "TREND_PULLBACK",
        "strategy_version": TREND_PULLBACK_PAPER_VERSION,
        "position": created,
        "paper_only": True,
        "execution": False,
    }


async def run_trend_pullback_paper_generation_once() -> Dict[str, int]:
    """Scan the real crypto registry for Trend Pullback paper entries."""
    stats = {"checked": 0, "setups": 0, "opened": 0, "blocked": 0}
    if not persistence_state.ready:
        return stats
    for instrument in instrument_registry.all():
        if instrument.asset_class != AssetClass.CRYPTO:
            continue
        symbol = instrument.canonical_symbol
        stats["checked"] += 1
        provider_symbol = provider_symbol_map.to_provider("coinbase", symbol)
        if provider_symbol is None:
            stats["blocked"] += 1
            continue
        try:
            candles, quality = await market_provider.get_candles(
                provider_symbol, SERVER_SETUP_GRANULARITY, SERVER_SETUP_CANDLE_LIMIT
            )
            if quality != DataQualityStatus.VALID:
                continue
            now = utcnow()
            regime = classify_server_market_regime(candles, now)
            detection = detect_trend_pullback_candidate(candles, now, regime)
            if detection.get("status") != "SETUP":
                continue
            stats["setups"] += 1
            ticker = await market_provider.get_ticker(provider_symbol)
            plan = build_trend_pullback_paper_plan(candles, now, detection, ticker)
            session_snapshot = market_session_context(symbol, now)
            plan["performance_timeframe"] = SERVER_SETUP_GRANULARITY
            plan["performance_session"] = session_snapshot.get("current_session")
            plan["performance_market_regime"] = regime.get("regime")
            plan["performance_setup_context"] = json.dumps(
                {"detector": detection}, default=str, sort_keys=True
            )
            if plan.get("status") != "ENTRY_NOW":
                stats["blocked"] += 1
                continue
            result = await execute_trend_pullback_paper_plan(symbol, plan)
            if result.get("status") == "OPENED":
                stats["opened"] += 1
            else:
                stats["blocked"] += 1
        except HTTPException as exc:
            if exc.status_code != 409:
                stats["blocked"] += 1
        except Exception as exc:  # noqa: BLE001
            log.error("Trend Pullback paper generation failed for %s: %s", symbol, exc)
            stats["blocked"] += 1
    return stats


@api_router.get("/strategies/trend-pullback/paper-status")
async def trend_pullback_paper_status() -> Dict[str, object]:
    return {
        "status": "ACTIVE_PAPER",
        "strategy_id": "TREND_PULLBACK",
        "strategy_version": TREND_PULLBACK_PAPER_VERSION,
        "risk_percent": "1",
        "risk_reward_rule": str(TREND_PULLBACK_RISK_REWARD),
        "stop_rule": "CLOSED_PULLBACK_STRUCTURE",
        "entry_source": "REAL_COINBASE_TICKER",
        "paper_only": True,
        "live_trading": False,
        "execution": False,
    }


# V16-M5B28B2 — Breakout Expansion paper execution (paper-only)
BREAKOUT_EXPANSION_PAPER_VERSION = "0.2-paper"
BREAKOUT_EXPANSION_RISK_REWARD = Decimal("2")


def build_breakout_expansion_paper_plan(
    detection: Dict[str, object],
    ticker: MarketDatum,
    now: datetime,
) -> Dict[str, object]:
    """Build a breakout plan from confirmed closed-candle range + real ticker."""
    base = {
        "strategy_id": "BREAKOUT_EXPANSION",
        "strategy_version": BREAKOUT_EXPANSION_PAPER_VERSION,
        "paper_only": True,
        "execution": False,
    }
    if detection.get("status") != "SETUP":
        return {**base, "status": "WAIT", "reason": "BREAKOUT_SETUP_NOT_READY"}
    if ticker.status != DataQualityStatus.VALID or ticker.value is None:
        return {**base, "status": "WAIT", "reason": "REALTIME_TICKER_NOT_VALID"}
    if ticker.timestamp is None or ticker.timestamp.tzinfo is None:
        return {**base, "status": "WAIT", "reason": "REALTIME_TICKER_TIMESTAMP_INVALID"}
    if classify_freshness(
        ticker.timestamp, settings.ticker_max_age_seconds, now=now
    ) != DataQualityStatus.VALID:
        return {**base, "status": "WAIT", "reason": "REALTIME_TICKER_NOT_FRESH"}
    try:
        entry = Decimal(str(ticker.value))
        range_high = Decimal(str(detection["range_high"]))
        range_low = Decimal(str(detection["range_low"]))
    except (KeyError, ValueError, InvalidOperation):
        return {**base, "status": "WAIT", "reason": "BREAKOUT_STRUCTURE_INVALID"}
    direction = str(detection.get("direction") or "")
    if direction == "BULLISH":
        stop = range_high
        if stop <= 0 or entry <= stop:
            return {**base, "status": "WAIT", "reason": "BULLISH_BREAKOUT_NOT_HELD"}
        risk = entry - stop
        target = entry + BREAKOUT_EXPANSION_RISK_REWARD * risk
        side = "LONG"
    elif direction == "BEARISH":
        stop = range_low
        if entry <= 0 or stop <= entry:
            return {**base, "status": "WAIT", "reason": "BEARISH_BREAKOUT_NOT_HELD"}
        risk = stop - entry
        target = entry - BREAKOUT_EXPANSION_RISK_REWARD * risk
        if target <= 0:
            return {**base, "status": "WAIT", "reason": "BEARISH_TARGET_INVALID"}
        side = "SHORT"
    else:
        return {**base, "status": "WAIT", "reason": "BREAKOUT_DIRECTION_INVALID"}
    return {
        **base,
        "status": "ENTRY_NOW",
        "side": side,
        "direction": direction,
        "entry": entry,
        "stop_loss": stop,
        "take_profit": target,
        "risk_reward": BREAKOUT_EXPANSION_RISK_REWARD,
        "source_timestamp": ticker.timestamp,
        "setup_timestamp": detection.get("latest_closed_timestamp"),
        "plan_source": "CLOSED_BREAKOUT_RANGE+REAL_COINBASE_TICKER",
    }


async def execute_breakout_expansion_paper_plan(
    symbol: str, plan: Dict[str, object]
) -> Dict[str, object]:
    """Persist one prevalidated Breakout Expansion position with existing guards."""
    if not persistence_state.ready:
        return {"status": "BLOCKED", "reason": "PERSISTENCE_NOT_READY"}
    if plan.get("status") != "ENTRY_NOW":
        return {"status": "BLOCKED", "reason": "BREAKOUT_PLAN_NOT_ENTRY_NOW"}
    canonical = symbol.upper().replace("/", "-")
    try:
        entry = Decimal(str(plan["entry"]))
        stop = Decimal(str(plan["stop_loss"]))
        target = Decimal(str(plan["take_profit"]))
        source_timestamp = plan["source_timestamp"]
    except (KeyError, ValueError, InvalidOperation):
        return {"status": "BLOCKED", "reason": "BREAKOUT_PLAN_INVALID"}
    if not isinstance(source_timestamp, datetime) or source_timestamp.tzinfo is None:
        return {"status": "BLOCKED", "reason": "BREAKOUT_SOURCE_TIMESTAMP_INVALID"}
    side = str(plan.get("side"))
    if side == "LONG" and not stop < entry < target:
        return {"status": "BLOCKED", "reason": "BREAKOUT_LONG_LEVELS_INVALID"}
    if side == "SHORT" and not target < entry < stop:
        return {"status": "BLOCKED", "reason": "BREAKOUT_SHORT_LEVELS_INVALID"}
    if side not in {"LONG", "SHORT"}:
        return {"status": "BLOCKED", "reason": "BREAKOUT_SIDE_INVALID"}
    edge_gate = await evaluate_adaptive_edge_active_gate(
        canonical, "BREAKOUT_EXPANSION", plan.get("performance_timeframe"),
        plan.get("performance_session"), plan.get("performance_market_regime"),
    )
    if edge_gate.get("action") == "BLOCKED":
        return {
            "status": "BLOCKED", "reason": "ADAPTIVE_EDGE_WEAK",
            "adaptive_edge_gate": edge_gate,
        }
    specs = await get_paper_instrument_specs(canonical)
    if specs.get("status") != "VALID":
        return {"status": "BLOCKED", "reason": "SERVER_INSTRUMENT_SPECS_NOT_VALID"}
    account = await get_paper_account()
    capital = Decimal(str(account["current_capital"]))
    sizing = calculate_verified_crypto_size(capital, Decimal("1"), entry, stop, specs)
    if sizing.get("status") != "VALID":
        return {"status": "BLOCKED", "reason": str(sizing.get("reason"))}
    async with auto_paper_portfolio_lock:
        open_positions = await get_open_paper_risk_snapshot()
        guard = evaluate_paper_portfolio_risk_guard(
            open_positions, canonical, Decimal(str(sizing["risk_money"])), capital
        )
        if guard.get("status") != "VALID":
            return {"status": "BLOCKED", "reason": str(guard.get("reason"))}
        position = PaperPositionCreate(
            position_id=build_strategy_paper_position_id(
                "BREAKOUT_EXPANSION", canonical, side, plan.get("setup_timestamp")
            ),
            symbol=canonical,
            side=side,
            entry=entry,
            stop_loss=stop,
            take_profit=target,
            size=Decimal(str(sizing["size"])),
            size_unit="BASE_UNITS",
            risk_money=Decimal(str(sizing["risk_money"])),
            risk_percent=Decimal(str(sizing["risk_percent"])),
            capital_before=capital,
            source=f"strategy:BREAKOUT_EXPANSION@{BREAKOUT_EXPANSION_PAPER_VERSION}",
            source_timestamp=source_timestamp,
            opened_at=utcnow(),
            performance_strategy_id="BREAKOUT_EXPANSION",
            performance_strategy_version=BREAKOUT_EXPANSION_PAPER_VERSION,
            performance_timeframe=(
                str(plan.get("performance_timeframe"))
                if plan.get("performance_timeframe") is not None
                else None
            ),
            performance_session=(
                str(plan.get("performance_session"))
                if plan.get("performance_session") is not None
                else None
            ),
            performance_market_regime=(
                str(plan.get("performance_market_regime"))
                if plan.get("performance_market_regime") is not None
                else None
            ),
            performance_setup_context=(
                str(plan.get("performance_setup_context"))
                if plan.get("performance_setup_context") is not None
                else None
            ),
        )
        created = await create_paper_position(position)
        await persist_adaptive_edge_trade_link(position.position_id, edge_gate)
    return {
        "status": "OPENED",
        "strategy_id": "BREAKOUT_EXPANSION",
        "strategy_version": BREAKOUT_EXPANSION_PAPER_VERSION,
        "position": created,
        "paper_only": True,
        "execution": False,
    }


async def run_breakout_expansion_paper_generation_once() -> Dict[str, int]:
    """Scan the real crypto registry for Breakout Expansion paper entries."""
    stats = {"checked": 0, "setups": 0, "opened": 0, "blocked": 0}
    if not persistence_state.ready:
        return stats
    for instrument in instrument_registry.all():
        if instrument.asset_class != AssetClass.CRYPTO:
            continue
        symbol = instrument.canonical_symbol
        stats["checked"] += 1
        provider_symbol = provider_symbol_map.to_provider("coinbase", symbol)
        if provider_symbol is None:
            stats["blocked"] += 1
            continue
        try:
            candles, quality = await market_provider.get_candles(
                provider_symbol, SERVER_SETUP_GRANULARITY, SERVER_SETUP_CANDLE_LIMIT
            )
            if quality != DataQualityStatus.VALID:
                continue
            now = utcnow()
            regime = classify_server_market_regime(candles, now)
            detection = detect_breakout_expansion_candidate(candles, now, regime)
            if detection.get("status") != "SETUP":
                continue
            stats["setups"] += 1
            ticker = await market_provider.get_ticker(provider_symbol)
            plan = build_breakout_expansion_paper_plan(detection, ticker, now)
            session_snapshot = market_session_context(symbol, now)
            plan["performance_timeframe"] = SERVER_SETUP_GRANULARITY
            plan["performance_session"] = session_snapshot.get("current_session")
            plan["performance_market_regime"] = regime.get("regime")
            plan["performance_setup_context"] = json.dumps(
                {"detector": detection}, default=str, sort_keys=True
            )
            if plan.get("status") != "ENTRY_NOW":
                stats["blocked"] += 1
                continue
            result = await execute_breakout_expansion_paper_plan(symbol, plan)
            if result.get("status") == "OPENED":
                stats["opened"] += 1
            else:
                stats["blocked"] += 1
        except HTTPException as exc:
            if exc.status_code != 409:
                stats["blocked"] += 1
        except Exception as exc:  # noqa: BLE001
            log.error("Breakout Expansion paper generation failed for %s: %s", symbol, exc)
            stats["blocked"] += 1
    return stats


@api_router.get("/strategies/breakout-expansion/paper-status")
async def breakout_expansion_paper_status() -> Dict[str, object]:
    return {
        "status": "ACTIVE_PAPER",
        "strategy_id": "BREAKOUT_EXPANSION",
        "strategy_version": BREAKOUT_EXPANSION_PAPER_VERSION,
        "risk_percent": "1",
        "risk_reward_rule": str(BREAKOUT_EXPANSION_RISK_REWARD),
        "stop_rule": "PRIOR_CLOSED_RANGE_BOUNDARY",
        "entry_source": "REAL_COINBASE_TICKER",
        "paper_only": True,
        "live_trading": False,
        "execution": False,
    }



# V16-M5B28B7-FIX3 — same-origin dynamic crypto logo proxy.
# The frontend never calls third-party origins directly. Logos are resolved on the
# backend from a small deterministic provider chain, cached in memory, and
# returned as image bytes. Missing logos fail closed with 404 so the UI can keep
# its ticker-initial fallback.
CRYPTO_LOGO_CACHE_TTL_SECONDS = 24 * 60 * 60
CRYPTO_LOGO_NEGATIVE_TTL_SECONDS = 60 * 60
CRYPTO_LOGO_MAX_BYTES = 512 * 1024
CRYPTO_LOGO_SYMBOL_ALIASES = {
    'JUPITER': 'JUP',
}
_crypto_logo_cache: Dict[str, Tuple[float, bytes, str]] = {}
_crypto_logo_missing_until: Dict[str, float] = {}
_crypto_logo_locks: Dict[str, asyncio.Lock] = {}


def _normalize_crypto_logo_symbol(symbol: str) -> str:
    base = str(symbol or '').split('-')[0].strip().upper()
    if not re.fullmatch(r'[A-Z0-9]{1,15}', base):
        raise HTTPException(status_code=400, detail='INVALID_CRYPTO_SYMBOL')
    return base


def _crypto_logo_lock(base: str) -> asyncio.Lock:
    lock = _crypto_logo_locks.get(base)
    if lock is None:
        lock = asyncio.Lock()
        _crypto_logo_locks[base] = lock
    return lock


def _usable_logo_response(resp: httpx.Response) -> bool:
    if resp.status_code != 200:
        return False
    content_type = str(resp.headers.get('content-type') or '').split(';')[0].lower()
    if not content_type.startswith('image/'):
        return False
    length = len(resp.content)
    return 0 < length <= CRYPTO_LOGO_MAX_BYTES


async def _fetch_logo_bytes(url: str) -> Optional[Tuple[bytes, str]]:
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(5.0),
            follow_redirects=True,
            headers={'User-Agent': 'Crypto-Intelligence-Engine/1.0'},
        ) as client:
            resp = await client.get(url)
        if not _usable_logo_response(resp):
            return None
        media_type = str(resp.headers.get('content-type') or 'image/png').split(';')[0]
        return resp.content, media_type
    except httpx.HTTPError:
        return None


async def _resolve_crypto_logo(base: str) -> Optional[Tuple[bytes, str]]:
    lower = base.lower()
    static_candidates = (
        'https://cdn.jsdelivr.net/gh/spothq/cryptocurrency-icons@master/128/color/'
        f'{lower}.png',
        'https://cdn.jsdelivr.net/npm/cryptocurrency-icons@0.18.1/128/color/'
        f'{lower}.png',
    )
    for candidate in static_candidates:
        resolved = await _fetch_logo_bytes(candidate)
        if resolved is not None:
            return resolved

    # Dynamic fallback for newer assets absent from the static icon packages.
    # Search results are filtered to an exact ticker match before the returned
    # image URL is fetched, which avoids guessing an asset by name.
    search_symbols = [base]
    alias = CRYPTO_LOGO_SYMBOL_ALIASES.get(base)
    if alias and alias not in search_symbols:
        search_symbols.append(alias)
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(5.0),
            follow_redirects=True,
            headers={'User-Agent': 'Crypto-Intelligence-Engine/1.0'},
        ) as client:
            for search_symbol in search_symbols:
                search_url = (
                    'https://api.coingecko.com/api/v3/search?query='
                    + url_quote(search_symbol)
                )
                search_resp = await client.get(search_url)
                if search_resp.status_code != 200:
                    continue
                payload = search_resp.json()
                coins = payload.get('coins') if isinstance(payload, dict) else None
                if not isinstance(coins, list):
                    continue
                exact = [
                    coin
                    for coin in coins
                    if isinstance(coin, dict)
                    and str(coin.get('symbol') or '').upper() == search_symbol
                ]
                exact.sort(
                    key=lambda coin: (
                        coin.get('market_cap_rank') is None,
                        coin.get('market_cap_rank') or 10**9,
                    )
                )
                for coin in exact[:4]:
                    image_url = (
                        coin.get('large') or coin.get('small') or coin.get('thumb')
                    )
                    if not isinstance(image_url, str) or not image_url.startswith('https://'):
                        continue
                    image_resp = await client.get(image_url)
                    if _usable_logo_response(image_resp):
                        media_type = str(
                            image_resp.headers.get('content-type') or 'image/png'
                        ).split(';')[0]
                        return image_resp.content, media_type
    except (httpx.HTTPError, ValueError, TypeError):
        return None
    return None


@api_router.get('/market/crypto-logo/{symbol}')
async def crypto_logo(symbol: str) -> Response:
    base = _normalize_crypto_logo_symbol(symbol)
    now = time.monotonic()
    cached = _crypto_logo_cache.get(base)
    if cached is not None and now - cached[0] < CRYPTO_LOGO_CACHE_TTL_SECONDS:
        return Response(
            content=cached[1],
            media_type=cached[2],
            headers={'Cache-Control': 'public, max-age=86400, immutable'},
        )
    if _crypto_logo_missing_until.get(base, 0.0) > now:
        raise HTTPException(status_code=404, detail='CRYPTO_LOGO_NOT_FOUND')

    async with _crypto_logo_lock(base):
        now = time.monotonic()
        cached = _crypto_logo_cache.get(base)
        if cached is not None and now - cached[0] < CRYPTO_LOGO_CACHE_TTL_SECONDS:
            return Response(
                content=cached[1],
                media_type=cached[2],
                headers={'Cache-Control': 'public, max-age=86400, immutable'},
            )
        resolved = await _resolve_crypto_logo(base)
        if resolved is None:
            _crypto_logo_missing_until[base] = now + CRYPTO_LOGO_NEGATIVE_TTL_SECONDS
            raise HTTPException(status_code=404, detail='CRYPTO_LOGO_NOT_FOUND')
        content, media_type = resolved
        _crypto_logo_cache[base] = (now, content, media_type)
        _crypto_logo_missing_until.pop(base, None)
        return Response(
            content=content,
            media_type=media_type,
            headers={'Cache-Control': 'public, max-age=86400, immutable'},
        )


# App must be built only after every router decorator above has executed.
app = create_app()

# V16-M5B24A — autonomous crypto WS + fail-safe REST paper-mark fallback

# V16-M5B24B — crypto registry alignment: BTC/ETH/SOL/XRP/LTC/ADA
