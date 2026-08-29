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
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

import httpx
import redis.asyncio as aioredis
from fastapi import APIRouter, FastAPI, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import text
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
    return {
        "source": "coinbase",
        "symbol": symbol.upper(),
        "granularity": granularity,
        "count": len(candles),
        "quality": status.value,
        "candles": [c.to_dict() for c in candles],
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
    gaps = _missing_buckets_24_7(starts_sorted, bucket)

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
    try:
        yield
    finally:
        await market_ws.stop()
        await market_provider.disconnect()
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