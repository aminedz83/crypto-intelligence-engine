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
from typing import Any, AsyncIterator, Dict, List, Optional

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
                    for channel, prods in list(self.subscriptions.items()):
                        await ws.send(json.dumps(self.build_subscribe(channel, sorted(prods))))
                    # Heartbeats keep sparse subscriptions open (Coinbase docs).
                    await ws.send(json.dumps({"type": "subscribe", "channel": "heartbeats"}))
                    async for raw in ws:
                        self.last_message_at = utcnow()
                        self._handle(raw)
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

    def _handle(self, raw: str | bytes) -> None:  # pragma: no cover - needs a live socket
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            log.warning("Ignoring invalid JSON from Coinbase WS")
            return
        if isinstance(msg, dict) and msg.get("type") == "error":
            log.error("Coinbase WS error: %s", msg)
        # Distribution of real ticks to consumers is a later increment; nothing is
        # fabricated here.

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
    return await market_ws.health_check()


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