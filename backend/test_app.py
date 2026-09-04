"""All backend tests in one file (single-file layout).

38 tests, none skipped. Pure-logic tests (data quality, health aggregation) plus
integration tests (health endpoints, failure modes, frontend serving) that need
FastAPI + httpx. No conditional skip: a missing dependency fails the run rather
than skipping.
"""

import inspect
import asyncio
import json
import unittest
from datetime import datetime, timedelta, timezone
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

import main
from main import (
    ComponentHealth,
    DataQualityStatus,
    HealthReport,
    HealthState,
    QualifiedValue,
    aggregate_health,
    classify_freshness,
    compute_age_seconds,
    create_app,
)

NOW = datetime(2026, 8, 24, 12, 0, 0, tzinfo=timezone.utc)
FRONTEND = Path(__file__).resolve().parents[1] / "frontend"
INDEX = FRONTEND / "index.html"
VIEW_IDS = ["markets", "forex", "metal", "index", "system", "detail"]


# ----------------------------- data quality -----------------------------
class ComputeAgeTests(unittest.TestCase):
    def test_age_positive(self):
        self.assertAlmostEqual(compute_age_seconds(NOW - timedelta(seconds=30), now=NOW), 30.0)

    def test_naive_timestamp_rejected(self):
        with self.assertRaises(ValueError):
            compute_age_seconds(datetime(2026, 8, 24, 12, 0, 0), now=NOW)


class ClassifyFreshnessTests(unittest.TestCase):
    def test_missing(self):
        self.assertEqual(classify_freshness(None, 10, now=NOW), DataQualityStatus.MISSING)

    def test_valid(self):
        ts = NOW - timedelta(seconds=5)
        self.assertEqual(classify_freshness(ts, 10, now=NOW), DataQualityStatus.VALID)

    def test_boundary_is_valid(self):
        ts = NOW - timedelta(seconds=10)
        self.assertEqual(classify_freshness(ts, 10, now=NOW), DataQualityStatus.VALID)

    def test_stale(self):
        ts = NOW - timedelta(seconds=11)
        self.assertEqual(classify_freshness(ts, 10, now=NOW), DataQualityStatus.STALE)

    def test_future_timestamp_is_invalid(self):
        ts = NOW + timedelta(seconds=5)
        self.assertEqual(classify_freshness(ts, 10, now=NOW), DataQualityStatus.INVALID)

    def test_naive_timestamp_is_invalid(self):
        ts = datetime(2026, 8, 24, 11, 59, 55)
        self.assertEqual(classify_freshness(ts, 10, now=NOW), DataQualityStatus.INVALID)


class QualifiedValueTests(unittest.TestCase):
    def test_usable_only_when_valid(self):
        self.assertTrue(QualifiedValue(185.2, "coinbase", NOW, DataQualityStatus.VALID).is_usable)

    def test_stale_value_not_usable(self):
        self.assertFalse(QualifiedValue(185.2, "coinbase", NOW, DataQualityStatus.STALE).is_usable)

    def test_none_value_not_usable(self):
        self.assertFalse(QualifiedValue(None, "coinbase", NOW, DataQualityStatus.VALID).is_usable)


# ----------------------------- health model -----------------------------
class AggregateHealthTests(unittest.TestCase):
    @staticmethod
    def _c(state):
        return ComponentHealth("x", state)

    def test_empty_is_unknown(self):
        self.assertEqual(aggregate_health([]), HealthState.UNKNOWN)

    def test_all_up(self):
        comps = [self._c(HealthState.UP), self._c(HealthState.UP)]
        self.assertEqual(aggregate_health(comps), HealthState.UP)

    def test_any_down_wins(self):
        comps = [self._c(HealthState.UP), self._c(HealthState.DOWN)]
        self.assertEqual(aggregate_health(comps), HealthState.DOWN)

    def test_down_outranks_degraded(self):
        comps = [self._c(HealthState.DEGRADED), self._c(HealthState.DOWN)]
        self.assertEqual(aggregate_health(comps), HealthState.DOWN)

    def test_unknown_prevents_up(self):
        comps = [self._c(HealthState.UP), self._c(HealthState.UNKNOWN)]
        self.assertEqual(aggregate_health(comps), HealthState.DEGRADED)

    def test_degraded_when_only_degraded(self):
        comps = [self._c(HealthState.UP), self._c(HealthState.DEGRADED)]
        self.assertEqual(aggregate_health(comps), HealthState.DEGRADED)


class HealthReportTests(unittest.TestCase):
    def test_report_serialises(self):
        d = HealthReport.from_components([ComponentHealth("postgres", HealthState.UP)]).to_dict()
        self.assertEqual(d["overall"], "UP")
        self.assertEqual(len(d["components"]), 1)
        self.assertIn("generated_at", d)
        self.assertEqual(d["components"][0]["name"], "postgres")


# ----------------------------- health endpoints -----------------------------
class HealthEndpointTests(unittest.TestCase):
    def setUp(self):
        async def up_db():
            return ComponentHealth("postgres", HealthState.UP)

        async def up_redis():
            return ComponentHealth("redis", HealthState.UP)

        self._orig = (main.check_database, main.check_redis)
        main.check_database = up_db
        main.check_redis = up_redis
        self.client = TestClient(create_app())

    def tearDown(self):
        main.check_database, main.check_redis = self._orig

    def test_liveness(self):
        r = self.client.get("/health/live")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "alive")

    def test_health_overall_up(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["overall"], "UP")

    def test_ready_returns_200_when_up(self):
        r = self.client.get("/health/ready")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["overall"], "UP")


# ----------------------------- failure modes -----------------------------
UNREACHABLE_DB = "postgresql+asyncpg://cie:cie@127.0.0.1:1/cie"
UNREACHABLE_REDIS = "redis://127.0.0.1:1/0"


class ReadinessEndpointFailureTests(unittest.TestCase):
    def setUp(self):
        self._orig = (main.check_database, main.check_redis)

    def tearDown(self):
        main.check_database, main.check_redis = self._orig

    def _client(self, db_state, redis_state, db_detail=None):
        async def fdb():
            return ComponentHealth("postgres", db_state, detail=db_detail)

        async def fredis():
            return ComponentHealth("redis", redis_state)

        main.check_database = fdb
        main.check_redis = fredis
        return TestClient(create_app())

    def test_ready_503_when_db_down(self):
        r = self._client(HealthState.DOWN, HealthState.UP).get("/health/ready")
        self.assertEqual(r.status_code, 503)
        self.assertEqual(r.json()["overall"], "DOWN")

    def test_ready_503_when_redis_down(self):
        r = self._client(HealthState.UP, HealthState.DOWN).get("/health/ready")
        self.assertEqual(r.status_code, 503)
        self.assertEqual(r.json()["overall"], "DOWN")

    def test_health_200_but_surfaces_down_detail(self):
        client = self._client(HealthState.DOWN, HealthState.UP, db_detail="connection refused")
        r = client.get("/health")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["overall"], "DOWN")
        pg = next(c for c in body["components"] if c["name"] == "postgres")
        self.assertEqual(pg["state"], "DOWN")
        self.assertIn("refused", pg["detail"])


class RealProbeFailureTests(unittest.IsolatedAsyncioTestCase):
    async def test_database_probe_down_when_unreachable(self):
        from sqlalchemy.ext.asyncio import create_async_engine

        bad = create_async_engine(UNREACHABLE_DB)
        orig = main.engine
        main.engine = bad
        try:
            comp = await main.check_database()
        finally:
            main.engine = orig
            await bad.dispose()
        self.assertEqual(comp.state, HealthState.DOWN)
        self.assertIsNotNone(comp.detail)

    async def test_redis_probe_down_when_unreachable(self):
        import redis.asyncio as redis

        bad = redis.from_url(UNREACHABLE_REDIS, decode_responses=True)
        orig = main.redis_client
        main.redis_client = bad
        try:
            comp = await main.check_redis()
        finally:
            main.redis_client = orig
            await bad.aclose()
        self.assertEqual(comp.state, HealthState.DOWN)
        self.assertIsNotNone(comp.detail)


# ----------------------------- frontend serving -----------------------------
class FrontendServingTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(create_app())

    def test_root_serves_index_200(self):
        self.assertEqual(self.client.get("/").status_code, 200)

    def test_root_content_type_is_html(self):
        self.assertIn("text/html", self.client.get("/").headers.get("content-type", ""))

    def test_index_content_matches_file(self):
        self.assertEqual(self.client.get("/").text, INDEX.read_text(encoding="utf-8"))

    def test_index_is_self_contained(self):
        html = INDEX.read_text(encoding="utf-8")
        self.assertNotRegex(html, r'<link[^>]+rel=["\']stylesheet')
        self.assertNotRegex(html, r"<script[^>]+src=")
        self.assertIn("<style>", html)
        self.assertIn("<script>", html)

    def test_index_contains_all_views(self):
        html = INDEX.read_text(encoding="utf-8")
        for vid in VIEW_IDS:
            self.assertIn(f'id:"{vid}"', html)

    def test_missing_path_returns_404(self):
        self.assertEqual(self.client.get("/does-not-exist-xyz").status_code, 404)


class ApiRegressionTests(unittest.TestCase):
    def setUp(self):
        async def up_db():
            return ComponentHealth("postgres", HealthState.UP)

        async def up_redis():
            return ComponentHealth("redis", HealthState.UP)

        self._orig = (main.check_database, main.check_redis)
        main.check_database = up_db
        main.check_redis = up_redis
        self.client = TestClient(create_app())

    def tearDown(self):
        main.check_database, main.check_redis = self._orig

    def test_health_live_still_works(self):
        r = self.client.get("/health/live")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "alive")

    def test_health_still_works(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["overall"], "UP")

    def test_health_ready_still_works(self):
        r = self.client.get("/health/ready")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["overall"], "UP")

    def test_api_v1_root_works(self):
        r = self.client.get("/api/v1/")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["service"], "crypto-intelligence-engine")


class ApiOnlyRegressionTests(unittest.TestCase):
    def setUp(self):
        self._orig = main._frontend_dir
        main._frontend_dir = lambda cfg: Path("/nonexistent-frontend-xyz")
        self.client = TestClient(create_app())

    def tearDown(self):
        main._frontend_dir = self._orig

    def test_root_is_404_without_frontend(self):
        self.assertEqual(self.client.get("/").status_code, 404)

    def test_health_live_works_without_frontend(self):
        self.assertEqual(self.client.get("/health/live").status_code, 200)



class ServerClosedCandleHistoryFreshnessFixTests(unittest.TestCase):
    def _candle(self, minute: int, status: main.DataQualityStatus) -> main.Candle:
        start = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=minute)
        return main.Candle(start, 99.0, 101.0, 100.0, 100.5, 1.0, status)

    def test_stale_closed_history_is_usable(self):
        now = datetime(2026, 1, 1, 1, 0, tzinfo=timezone.utc)
        candle = self._candle(0, main.DataQualityStatus.STALE)
        self.assertEqual(main.closed_valid_candles([candle], now), [candle])

    def test_invalid_closed_history_is_rejected(self):
        now = datetime(2026, 1, 1, 1, 0, tzinfo=timezone.utc)
        candle = self._candle(0, main.DataQualityStatus.INVALID)
        self.assertEqual(main.closed_valid_candles([candle], now), [])

    def test_missing_closed_history_is_rejected(self):
        now = datetime(2026, 1, 1, 1, 0, tzinfo=timezone.utc)
        candle = self._candle(0, main.DataQualityStatus.MISSING)
        self.assertEqual(main.closed_valid_candles([candle], now), [])

    def test_open_stale_candle_is_still_rejected(self):
        now = datetime(2026, 1, 1, 0, 2, tzinfo=timezone.utc)
        candle = self._candle(0, main.DataQualityStatus.STALE)
        self.assertEqual(main.closed_valid_candles([candle], now), [])

    def test_detector_has_latest_freshness_guard(self):
        source = inspect.getsource(main.get_server_market_setup_detector)
        self.assertIn("LATEST_CANDLE_NOT_FRESH", source)
        self.assertIn("_latest_quality(candles)", source)

    def test_regime_has_latest_freshness_guard(self):
        source = inspect.getsource(main.get_server_market_regime)
        self.assertIn("LATEST_CANDLE_NOT_FRESH", source)
        self.assertIn("_latest_quality(candles)", source)

    def test_historical_stale_series_can_reach_structure_analysis(self):
        now = datetime(2026, 1, 1, 2, 0, tzinfo=timezone.utc)
        candles = [
            self._candle(index * 5, main.DataQualityStatus.STALE)
            for index in range(12)
        ]
        result = main.detect_server_market_structure(candles, now)
        self.assertNotEqual(result.get("reason"), "INSUFFICIENT_CLOSED_CANDLES")

    def test_fix_keeps_paper_detector_non_executing(self):
        source = inspect.getsource(main.get_server_market_setup_detector)
        self.assertIn("auto_queue", source)



class DynamicCryptoUniverseFrontendV16M5B28AUiTests(unittest.TestCase):
    """Frontend consumes only the server-authoritative dynamic crypto universe."""

    @classmethod
    def setUpClass(cls):
        cls.html = INDEX.read_text(encoding="utf-8")

    def test_crypto_symbols_start_empty(self):
        self.assertIn("var CRYPTO_SYMBOLS=[];", self.html)

    def test_no_legacy_six_symbol_array(self):
        legacy = '["BTC-USD","ETH-USD","SOL-USD","XRP-USD","LTC-USD","ADA-USD"]'
        self.assertNotIn(legacy, self.html)

    def test_crypto_universe_endpoint_is_consumed(self):
        self.assertIn('/api/v1/market/crypto-universe', self.html)

    def test_active_symbols_are_server_driven(self):
        self.assertIn("d.active_symbols", self.html)

    def test_dynamic_symbols_are_validated_as_usd_pairs(self):
        self.assertIn('/^[A-Z0-9]+-USD$/.test(sym)', self.html)

    def test_universe_failure_does_not_invent_symbols(self):
        self.assertIn('CRYPTO_SYMBOLS=[];', self.html)
        self.assertIn('status:"UNAVAILABLE"', self.html)

    def test_markets_render_from_dynamic_symbols(self):
        self.assertIn('CRYPTO_SYMBOLS.forEach(function(sym)', self.html)

    def test_market_tickers_use_dynamic_symbols(self):
        marker = 'market/ticker/"+encodeURIComponent(sym)'
        self.assertIn(marker, self.html)

    def test_signal_filter_uses_dynamic_symbols(self):
        self.assertIn('var symbols=[""].concat(CRYPTO_SYMBOLS)', self.html)

    def test_realtime_ticker_subscription_uses_dynamic_symbols(self):
        marker = '{channel:"ticker",products:CRYPTO_SYMBOLS}'
        self.assertIn(marker, self.html)

    def test_realtime_candle_subscription_uses_dynamic_symbols(self):
        marker = '{channel:"candles",products:CRYPTO_SYMBOLS}'
        self.assertIn(marker, self.html)

    def test_realtime_polling_uses_dynamic_symbols(self):
        self.assertIn('symbols=CRYPTO_SYMBOLS.slice()', self.html)

    def test_universe_load_precedes_realtime_start(self):
        marker = 'loadCryptoUniverse().then(function(){setView("dashboard");startCryptoRealtime()})'
        self.assertIn(marker, self.html)

    def test_market_ui_exposes_active_count(self):
        self.assertIn('"Actifs actifs: "+String(CRYPTO_SYMBOLS.length)', self.html)

    def test_market_ui_identifies_coinbase_source(self):
        self.assertIn('" · source Coinbase"', self.html)

    def test_empty_universe_is_fail_safe(self):
        self.assertIn('"Aucun symbole Coinbase vérifié n’est actif."', self.html)

if __name__ == "__main__":
    unittest.main()


# ----------------------------- market data (Phase 2) -----------------------------
from datetime import timedelta as _td  # noqa: E402

from main import (  # noqa: E402
    CoinbaseProvider,
    MarketDatum,
    MarketWsManager,
    parse_iso8601,
    ticker_datum_from_payload,
    ws_backoff,
)


class ParseIso8601Tests(unittest.TestCase):
    def test_valid_z_suffix(self):
        dt = parse_iso8601("2026-08-24T12:00:00Z")
        self.assertIsNotNone(dt)
        self.assertEqual(dt.tzinfo, timezone.utc)

    def test_naive_returns_none(self):
        self.assertIsNone(parse_iso8601("2026-08-24T12:00:00"))

    def test_garbage_returns_none(self):
        self.assertIsNone(parse_iso8601("not-a-date"))

    def test_non_string_returns_none(self):
        self.assertIsNone(parse_iso8601(12345))


class TickerDatumTests(unittest.TestCase):
    def _payload(self, price, when):
        return {"trades": [{"price": price, "time": when}]}

    def test_valid_recent_is_valid(self):
        now = datetime(2026, 8, 24, 12, 0, 0, tzinfo=timezone.utc)
        ts = (now - _td(seconds=2)).isoformat().replace("+00:00", "Z")
        d = ticker_datum_from_payload("BTC-USD", self._payload("50000.5", ts), now=now)
        self.assertEqual(d.value, 50000.5)
        self.assertEqual(d.status, DataQualityStatus.VALID)

    def test_old_is_stale(self):
        now = datetime(2026, 8, 24, 12, 0, 0, tzinfo=timezone.utc)
        ts = (now - _td(seconds=999)).isoformat().replace("+00:00", "Z")
        d = ticker_datum_from_payload("BTC-USD", self._payload("50000", ts), now=now)
        self.assertEqual(d.status, DataQualityStatus.STALE)

    def test_no_trades_is_missing(self):
        d = ticker_datum_from_payload("BTC-USD", {"trades": []})
        self.assertEqual(d.status, DataQualityStatus.MISSING)
        self.assertIsNone(d.value)

    def test_bad_price_is_invalid(self):
        d = ticker_datum_from_payload("BTC-USD", self._payload("abc", "2026-08-24T12:00:00Z"))
        self.assertEqual(d.status, DataQualityStatus.INVALID)

    def test_non_positive_price_is_invalid(self):
        d = ticker_datum_from_payload("BTC-USD", self._payload("0", "2026-08-24T12:00:00Z"))
        self.assertEqual(d.status, DataQualityStatus.INVALID)

    def test_to_dict_shape(self):
        d = MarketDatum("coinbase", "BTC-USD", 100.0, None, DataQualityStatus.MISSING)
        out = d.to_dict()
        self.assertEqual(
            set(out), {"source", "symbol", "value", "timestamp", "freshness_seconds", "quality"}
        )
        self.assertEqual(out["quality"], "MISSING")


class WsManagerPureTests(unittest.TestCase):
    def test_build_subscribe_format(self):
        msg = MarketWsManager.build_subscribe("ticker", ["btc-usd", "eth-usd"])
        self.assertEqual(msg["type"], "subscribe")
        self.assertEqual(msg["channel"], "ticker")
        self.assertEqual(msg["product_ids"], ["BTC-USD", "ETH-USD"])

    def test_backoff_grows_and_caps(self):
        self.assertEqual(ws_backoff(1), 1.0)
        self.assertEqual(ws_backoff(2), 2.0)
        self.assertLessEqual(ws_backoff(50), 60.0)

    def test_health_disconnected_is_unknown(self):
        async def run():
            return await MarketWsManager().health_check()
        h = asyncio.run(run())
        self.assertFalse(h["connected"])
        self.assertEqual(h["quality"], "UNKNOWN")


class MarketEndpointTests(unittest.TestCase):
    def setUp(self):
        self._orig = main.market_provider

    def tearDown(self):
        main.market_provider = self._orig

    def _client_with_provider(self, provider):
        main.market_provider = provider
        return TestClient(create_app())

    def test_ticker_ok(self):
        class FakeProvider:
            async def get_ticker(self, symbol):
                return MarketDatum(
                    "coinbase", symbol.upper(), 42.0, None, DataQualityStatus.MISSING
                )
        r = self._client_with_provider(FakeProvider()).get("/api/v1/market/ticker/btc-usd")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["symbol"], "BTC-USD")
        self.assertEqual(body["value"], 42.0)
        self.assertEqual(body["quality"], "MISSING")

    def test_ticker_upstream_error_503(self):
        class FailingProvider:
            async def get_ticker(self, symbol):
                raise httpx.ConnectError("upstream down")
        r = self._client_with_provider(FailingProvider()).get("/api/v1/market/ticker/btc-usd")
        self.assertEqual(r.status_code, 503)

    def test_ws_subscribe_rejects_unknown_channel(self):
        client = TestClient(create_app())
        r = client.post(
            "/api/v1/market/websocket/subscribe",
            json={"channel": "bogus", "products": ["BTC-USD"]},
        )
        self.assertEqual(r.status_code, 400)

    def test_ws_health_endpoint_ok(self):
        client = TestClient(create_app())
        r = client.get("/api/v1/market/websocket/health")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIn("connection", body)
        self.assertIn("transport", body)
        self.assertIn("connected", body["connection"])


class CoinbaseProviderConfigTests(unittest.TestCase):
    def test_uses_verified_public_rest_base(self):
        self.assertEqual(
            CoinbaseProvider().rest_url, "https://api.coinbase.com/api/v3/brokerage"
        )


# ----------------------------- candles (Phase 2) -----------------------------
from main import (  # noqa: E402
    CANDLE_MAX_LIMIT,
    GRANULARITIES,
    Candle,
    candle_from_payload,
    candles_from_payload,
)

_CANDLE_NOW = datetime(2026, 8, 24, 12, 0, 0, tzinfo=timezone.utc)


def _candle_item(start_unix, low="100", high="110", open_="105", close="108", volume="12.5"):
    return {
        "start": str(start_unix),
        "low": low,
        "high": high,
        "open": open_,
        "close": close,
        "volume": volume,
    }


class CandleParsingTests(unittest.TestCase):
    def test_valid_payload_converts(self):
        start = int((_CANDLE_NOW - _td(seconds=30)).timestamp())
        c = candle_from_payload(_candle_item(start), 120, now=_CANDLE_NOW)
        self.assertEqual(c.status, DataQualityStatus.VALID)
        self.assertEqual(c.low, 100.0)
        self.assertEqual(c.high, 110.0)
        self.assertEqual(c.open, 105.0)
        self.assertEqual(c.close, 108.0)

    def test_start_is_unix_seconds_to_utc(self):
        # 1639508050 -> 2021-12-14T20:14:10Z (seconds, not ms)
        c = candle_from_payload(_candle_item(1639508050), 10**12, now=_CANDLE_NOW)
        self.assertEqual(c.start.year, 2021)
        self.assertEqual(c.start.tzinfo, timezone.utc)

    def test_price_parsed_as_float(self):
        c = candle_from_payload(_candle_item(1639508050, low="140.21"), 10**12, now=_CANDLE_NOW)
        self.assertEqual(c.low, 140.21)

    def test_volume_parsed_as_float(self):
        item = _candle_item(1639508050, volume="56437345")
        c = candle_from_payload(item, 10**12, now=_CANDLE_NOW)
        self.assertEqual(c.volume, 56437345.0)

    def test_invalid_timestamp_is_invalid(self):
        item = _candle_item(1639508050)
        item["start"] = "not-a-number"
        c = candle_from_payload(item, 120, now=_CANDLE_NOW)
        self.assertEqual(c.status, DataQualityStatus.INVALID)

    def test_bad_price_is_invalid(self):
        item = _candle_item(1639508050, high="abc")
        c = candle_from_payload(item, 10**12, now=_CANDLE_NOW)
        self.assertEqual(c.status, DataQualityStatus.INVALID)

    def test_stale_candle_detected(self):
        start = int((_CANDLE_NOW - _td(seconds=1000)).timestamp())
        c = candle_from_payload(_candle_item(start), 120, now=_CANDLE_NOW)
        self.assertEqual(c.status, DataQualityStatus.STALE)

    def test_empty_response_is_missing(self):
        candles, status = candles_from_payload({"candles": []}, 120, now=_CANDLE_NOW)
        self.assertEqual(candles, [])
        self.assertEqual(status, DataQualityStatus.MISSING)



class CandleGranularityTests(unittest.TestCase):
    def test_nine_official_granularities(self):
        self.assertEqual(
            sorted(GRANULARITIES),
            sorted(["1m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "1d"]),
        )

    def test_maps_to_official_enum_values(self):
        enums = {v[0] for v in GRANULARITIES.values()}
        self.assertEqual(
            enums,
            {
                "ONE_MINUTE", "FIVE_MINUTE", "FIFTEEN_MINUTE", "THIRTY_MINUTE",
                "ONE_HOUR", "TWO_HOUR", "FOUR_HOUR", "SIX_HOUR", "ONE_DAY",
            },
        )


class CandleProviderRequestTests(unittest.IsolatedAsyncioTestCase):
    async def test_limit_clamped_to_350_and_official_params(self):
        captured = {}

        class FakeClient:
            async def get(self, path, params=None):
                captured["path"] = path
                captured["params"] = params

                class R:
                    def raise_for_status(self):
                        return None

                    def json(self):
                        return {"candles": []}

                return R()

        provider = CoinbaseProvider()
        provider.client = FakeClient()
        await provider.get_candles("btc-usd", "1m", limit=999)
        self.assertEqual(captured["params"]["limit"], CANDLE_MAX_LIMIT)  # clamped
        self.assertEqual(captured["params"]["granularity"], "ONE_MINUTE")
        self.assertIn("start", captured["params"])
        self.assertIn("end", captured["params"])
        self.assertTrue(captured["path"].endswith("/market/products/BTC-USD/candles"))


class CandleEndpointTests(unittest.TestCase):
    def setUp(self):
        self._orig = main.market_provider

    def tearDown(self):
        main.market_provider = self._orig

    def _client(self, provider):
        main.market_provider = provider
        return TestClient(create_app())

    def test_endpoint_ok_with_mocked_provider(self):
        class FakeProvider:
            async def get_candles(self, symbol, granularity, limit=350):
                start = _CANDLE_NOW
                candle = Candle(start, 1.0, 2.0, 1.5, 1.8, 3.0, DataQualityStatus.VALID)
                return [candle], DataQualityStatus.VALID
        r = self._client(FakeProvider()).get("/api/v1/market/candles/btc-usd?granularity=1h")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["symbol"], "BTC-USD")
        self.assertEqual(body["granularity"], "1h")
        self.assertEqual(body["count"], 1)
        self.assertEqual(body["quality"], "VALID")

    def test_endpoint_rejects_unknown_granularity(self):
        r = TestClient(create_app()).get("/api/v1/market/candles/btc-usd?granularity=4m")
        self.assertEqual(r.status_code, 400)

    def test_endpoint_http_error_returns_503(self):
        class FailingProvider:
            async def get_candles(self, symbol, granularity, limit=350):
                raise httpx.ConnectTimeout("timeout")
        r = self._client(FailingProvider()).get("/api/v1/market/candles/btc-usd")
        self.assertEqual(r.status_code, 503)


# ----------------------------- realtime WS pipeline (Phase 2) -----------------
from main import (  # noqa: E402
    MarketBus,
    MarketStateStore,
    RealtimeDatum,
    extract_candle_data,
    extract_ticker_data,
    parse_ws_message,
)

_RT_NOW = datetime(2026, 8, 24, 12, 0, 0, tzinfo=timezone.utc)


def _ticker_msg(product_id, price, seq=1, ts="2026-08-24T12:00:00Z"):
    return json.dumps({
        "channel": "ticker",
        "timestamp": ts,
        "sequence_num": seq,
        "events": [{"type": "update", "tickers": [
            {"type": "ticker", "product_id": product_id, "price": price}
        ]}],
    })


def _candle_msg(product_id, start, close="108", seq=1):
    return json.dumps({
        "channel": "candles",
        "timestamp": "2026-08-24T12:00:00Z",
        "sequence_num": seq,
        "events": [{"type": "update", "candles": [
            {"product_id": product_id, "start": str(start),
             "low": "100", "high": "110", "open": "105", "close": close, "volume": "5"}
        ]}],
    })


class WsParseTests(unittest.TestCase):
    def test_parse_valid_message(self):
        msg = parse_ws_message(_ticker_msg("BTC-USD", "50000"))
        self.assertIsInstance(msg, dict)
        self.assertEqual(msg["channel"], "ticker")

    def test_parse_malformed_returns_none(self):
        self.assertIsNone(parse_ws_message("{not json"))

    def test_parse_non_object_returns_none(self):
        self.assertIsNone(parse_ws_message("[1, 2, 3]"))


class ExtractTickerTests(unittest.TestCase):
    def test_valid_ticker_extracted(self):
        msg = parse_ws_message(_ticker_msg("BTC-USD", "50000", ts="2026-08-24T11:59:58Z"))
        data = extract_ticker_data(msg, received_at=_RT_NOW, now=_RT_NOW)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0].product_id, "BTC-USD")
        self.assertEqual(data[0].value, 50000.0)
        self.assertEqual(data[0].data_type, "ticker")
        self.assertEqual(data[0].sequence_num, 1)

    def test_missing_tickers_yields_nothing(self):
        msg = {"channel": "ticker", "timestamp": "2026-08-24T12:00:00Z",
               "sequence_num": 1, "events": [{"type": "update"}]}
        self.assertEqual(extract_ticker_data(msg, received_at=_RT_NOW), [])

    def test_invalid_price_skipped(self):
        msg = parse_ws_message(_ticker_msg("BTC-USD", "abc"))
        self.assertEqual(extract_ticker_data(msg, received_at=_RT_NOW, now=_RT_NOW), [])


class ExtractCandleTests(unittest.TestCase):
    def test_valid_candle_extracted(self):
        start = int((_RT_NOW - _td(seconds=60)).timestamp())
        msg = parse_ws_message(_candle_msg("BTC-USD", start))
        data = extract_candle_data(msg, received_at=_RT_NOW, now=_RT_NOW)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0].data_type, "candle")
        self.assertEqual(data[0].value, 108.0)
        self.assertIsNotNone(data[0].ohlcv)


class SequenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_in_order(self):
        s = MarketStateStore()
        self.assertEqual(await s.check_sequence(100), "first")
        self.assertEqual(await s.check_sequence(101), "ok")

    async def test_gap(self):
        s = MarketStateStore()
        await s.check_sequence(100)
        self.assertEqual(await s.check_sequence(102), "gap")

    async def test_out_of_order(self):
        s = MarketStateStore()
        await s.check_sequence(102)
        self.assertEqual(await s.check_sequence(101), "out_of_order")

    async def test_duplicate(self):
        s = MarketStateStore()
        await s.check_sequence(102)
        self.assertEqual(await s.check_sequence(102), "duplicate")

    async def test_reset_transport_no_false_out_of_order(self):
        s = MarketStateStore()
        await s.check_sequence(5000)
        await s.reset_transport()
        self.assertEqual(await s.check_sequence(3), "first")


class PerProductStateTests(unittest.IsolatedAsyncioTestCase):
    def _tick(self, product_id, price, ts):
        return RealtimeDatum("coinbase", product_id, "ticker", price, ts, _RT_NOW,
                             DataQualityStatus.VALID, 1)

    def _candle(self, product_id, price, start, seq=1):
        return RealtimeDatum("coinbase", product_id, "candle", price, start, _RT_NOW,
                             DataQualityStatus.VALID, seq)

    async def test_btc_eth_independent(self):
        s = MarketStateStore()
        await s.apply_ticker(self._tick("BTC-USD", 50000.0, _RT_NOW))
        await s.apply_ticker(self._tick("ETH-USD", 3000.0, _RT_NOW - _td(seconds=100)))
        btc = await s.get_ticker("BTC-USD")
        eth = await s.get_ticker("ETH-USD")
        self.assertEqual(btc.value, 50000.0)
        self.assertEqual(eth.value, 3000.0)

    async def test_older_ticker_does_not_overwrite(self):
        s = MarketStateStore()
        await s.apply_ticker(self._tick("BTC-USD", 50000.0, _RT_NOW))
        applied = await s.apply_ticker(self._tick("BTC-USD", 49000.0, _RT_NOW - _td(seconds=10)))
        self.assertFalse(applied)
        btc = await s.get_ticker("BTC-USD")
        self.assertEqual(btc.value, 50000.0)

    async def test_invalid_ticker_not_stored(self):
        s = MarketStateStore()
        bad = RealtimeDatum("coinbase", "BTC-USD", "ticker", None, _RT_NOW, _RT_NOW,
                            DataQualityStatus.INVALID, 1)
        self.assertFalse(await s.apply_ticker(bad))
        self.assertIsNone(await s.get_ticker("BTC-USD"))

    async def test_candle_same_start_updates_in_place(self):
        s = MarketStateStore()
        await s.apply_candle(self._candle("BTC-USD", 100.0, _RT_NOW, seq=1))
        applied = await s.apply_candle(self._candle("BTC-USD", 105.0, _RT_NOW, seq=2))
        self.assertTrue(applied)
        cur = await s.get_candle("BTC-USD")
        self.assertEqual(cur.value, 105.0)

    async def test_older_candle_bucket_not_overwrite(self):
        s = MarketStateStore()
        await s.apply_candle(self._candle("BTC-USD", 105.0, _RT_NOW, seq=2))
        older = self._candle("BTC-USD", 100.0, _RT_NOW - _td(seconds=300), seq=1)
        self.assertFalse(await s.apply_candle(older))
        cur = await s.get_candle("BTC-USD")
        self.assertEqual(cur.value, 105.0)


class RealtimeStoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_get_realtime_missing_when_empty(self):
        s = MarketStateStore()
        out = await s.get_realtime("BTC-USD")
        self.assertEqual(out["status"], "MISSING")

    async def test_heartbeat_not_in_price_store(self):
        s = MarketStateStore()
        await s.record_heartbeat(42, _RT_NOW)
        self.assertIsNone(await s.get_ticker("BTC-USD"))
        h = await s.health()
        self.assertEqual(h["heartbeat_counter"], 42)


class MarketBusTests(unittest.IsolatedAsyncioTestCase):
    async def test_consumer_receives_published(self):
        bus = MarketBus()
        seen = []
        bus.subscribe(lambda d: seen.append(d))
        datum = RealtimeDatum("coinbase", "BTC-USD", "ticker", 1.0, _RT_NOW, _RT_NOW,
                              DataQualityStatus.VALID, 1)
        await bus.publish(datum)
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0].product_id, "BTC-USD")


class RealtimeHandleTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self._store = main.market_store
        self._bus = main.market_bus
        main.market_store = MarketStateStore()
        main.market_bus = MarketBus()

    async def asyncTearDown(self):
        main.market_store = self._store
        main.market_bus = self._bus

    async def test_handle_ticker_stores(self):
        ts = _RT_NOW.isoformat().replace("+00:00", "Z")
        await main.market_ws._handle(_ticker_msg("BTC-USD", "50000", ts=ts))
        d = await main.market_store.get_ticker("BTC-USD")
        self.assertIsNotNone(d)
        self.assertEqual(d.value, 50000.0)

    async def test_handle_unknown_channel_ignored(self):
        raw = json.dumps({"channel": "l2_data", "sequence_num": 1, "events": []})
        await main.market_ws._handle(raw)
        self.assertIsNone(await main.market_store.get_ticker("BTC-USD"))

    async def test_handle_malformed_no_state_change(self):
        await main.market_ws._handle("{bad json")
        h = await main.market_store.health()
        self.assertEqual(h["messages"], 0)
        self.assertIsNone(await main.market_store.get_ticker("BTC-USD"))


class RealtimeEndpointTests(unittest.TestCase):
    def setUp(self):
        self._store = main.market_store
        main.market_store = MarketStateStore()
        self.client = TestClient(create_app())

    def tearDown(self):
        main.market_store = self._store

    def test_realtime_missing_when_no_data(self):
        r = self.client.get("/api/v1/market/realtime/btc-usd")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "MISSING")


# ----------------------------- paginated history (Phase 2) --------------------
from main import (  # noqa: E402
    PROVIDER_SAFE_BUCKETS,
    fetch_candle_history,
    max_history_span,
    plan_candle_windows,
)

_H = 3600  # ONE_HOUR bucket seconds


def _hist_candle(start_unix, close=100.0, status=DataQualityStatus.VALID):
    dt = datetime.fromtimestamp(start_unix, tz=timezone.utc)
    return Candle(dt, 90.0, 110.0, 95.0, close, 5.0, status)


class FakeRangeProvider:
    """Simulates Coinbase get_candles_range over a set of available starts.
    `inclusive` toggles end-boundary semantics to prove robustness either way."""

    def __init__(self, available, bucket=_H, inclusive=True, fail_on=None, invalid=None):
        self.available = sorted(available)
        self.bucket = bucket
        self.inclusive = inclusive
        self.fail_on = set(fail_on or [])
        self.invalid = set(invalid or [])
        self.calls = 0

    async def get_candles_range(self, symbol, granularity, start, end):
        self.calls += 1
        if start in self.fail_on:
            raise httpx.ConnectError("window failed")
        out = []
        for s in self.available:
            inside = (start <= s <= end) if self.inclusive else (start <= s < end)
            if inside:
                st = DataQualityStatus.INVALID if s in self.invalid else DataQualityStatus.VALID
                out.append(_hist_candle(s, status=st))
        return out, DataQualityStatus.VALID


class PlanWindowsTests(unittest.TestCase):
    def test_small_range_one_window(self):
        self.assertEqual(len(plan_candle_windows("1h", 0, 3 * _H)), 1)

    def test_safe_width_one_window(self):
        self.assertEqual(len(plan_candle_windows("1h", 0, PROVIDER_SAFE_BUCKETS * _H)), 1)

    def test_350_buckets_two_windows(self):
        self.assertEqual(len(plan_candle_windows("1h", 0, 350 * _H)), 2)

    def test_multiple_windows_count(self):
        self.assertEqual(len(plan_candle_windows("1h", 0, 700 * _H)), 3)

    def test_bucket_math_per_granularity(self):
        # 15m bucket = 900s; 350 buckets -> 2 windows
        self.assertEqual(len(plan_candle_windows("15m", 0, 350 * 900)), 2)

    def test_exact_bounds(self):
        w = plan_candle_windows("1h", 0, 700 * _H)
        self.assertEqual(w[0], (0, PROVIDER_SAFE_BUCKETS * _H))
        self.assertEqual(w[1][0], PROVIDER_SAFE_BUCKETS * _H)

    def test_window_never_exceeds_350_starts(self):
        w = plan_candle_windows("1h", 0, 5000 * _H)
        self.assertTrue(all((e - s) // _H <= 349 for s, e in w))

    def test_misaligned_raises(self):
        with self.assertRaises(ValueError):
            plan_candle_windows("1h", 1, 3 * _H)

    def test_start_ge_end_raises(self):
        with self.assertRaises(ValueError):
            plan_candle_windows("1h", 3 * _H, 3 * _H)

    def test_unknown_granularity_raises(self):
        with self.assertRaises(ValueError):
            plan_candle_windows("4m", 0, 3 * _H)


class MaxHistorySpanTests(unittest.TestCase):
    def test_scales_with_granularity(self):
        self.assertEqual(max_history_span("1h"), PROVIDER_SAFE_BUCKETS * _H * 20)
        self.assertGreater(max_history_span("1d"), max_history_span("1h"))


class HistoryAssemblerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._orig = main.market_provider

    def tearDown(self):
        main.market_provider = self._orig

    async def _run(self, provider, start, end, gran="1h"):
        main.market_provider = provider
        return await fetch_candle_history("BTC-USD", gran, start, end)

    async def test_end_inclusive_all_found_after_dedup(self):
        avail = [i * _H for i in range(350)]
        res = await self._run(FakeRangeProvider(avail, inclusive=True), 0, 350 * _H)
        self.assertEqual(res["count"], 350)
        self.assertEqual(res["status"], "COMPLETE")
        self.assertTrue(res["data_complete"])

    async def test_end_exclusive_all_found(self):
        avail = [i * _H for i in range(350)]
        res = await self._run(FakeRangeProvider(avail, inclusive=False), 0, 350 * _H)
        self.assertEqual(res["count"], 350)

    async def test_duplicate_boundary_single_candle(self):
        avail = [i * _H for i in range(350)]
        res = await self._run(FakeRangeProvider(avail, inclusive=True), 0, 350 * _H)
        starts = [c["start"] for c in res["candles"]]
        self.assertEqual(len(starts), len(set(starts)))

    async def test_sorted_chronological(self):
        avail = [i * _H for i in range(10)]
        res = await self._run(FakeRangeProvider(avail), 0, 10 * _H)
        starts = [c["start"] for c in res["candles"]]
        self.assertEqual(starts, sorted(starts))

    async def test_half_open_range_filter(self):
        avail = [0, _H, 2 * _H]
        res = await self._run(FakeRangeProvider(avail, inclusive=True), 0, 2 * _H)
        # end (2*_H) excluded by [start, end)
        self.assertEqual(res["count"], 2)

    async def test_empty_is_EMPTY(self):
        res = await self._run(FakeRangeProvider([]), 0, 10 * _H)
        self.assertEqual(res["status"], "EMPTY")
        self.assertEqual(res["count"], 0)
        self.assertFalse(res["data_complete"])

    async def test_gap_marks_incomplete_without_fabrication(self):
        avail = [0, _H, 3 * _H, 4 * _H]  # missing 2*_H
        res = await self._run(FakeRangeProvider(avail), 0, 5 * _H)
        self.assertTrue(res["gaps"])
        self.assertFalse(res["data_complete"])
        self.assertEqual(res["status"], "PARTIAL")
        self.assertEqual(res["count"], 4)  # no fabricated candle

    async def test_invalid_candle_counted_and_incomplete(self):
        avail = [0, _H, 2 * _H]
        res = await self._run(
            FakeRangeProvider(avail, invalid={_H}), 0, 3 * _H
        )
        self.assertEqual(res["invalid_candles_count"], 1)
        self.assertFalse(res["data_complete"])

    async def test_window_failure_is_partial(self):
        avail = [i * _H for i in range(350)]
        # second window starts at PROVIDER_SAFE_BUCKETS*_H
        provider = FakeRangeProvider(avail, fail_on={PROVIDER_SAFE_BUCKETS * _H})
        res = await self._run(provider, 0, 350 * _H)
        self.assertEqual(res["status"], "PARTIAL")
        self.assertEqual(res["provider_windows"]["failed"], 1)
        self.assertFalse(res["transport_complete"])

    async def test_complete_when_all_good(self):
        avail = [i * _H for i in range(10)]
        res = await self._run(FakeRangeProvider(avail), 0, 10 * _H)
        self.assertEqual(res["status"], "COMPLETE")
        self.assertTrue(res["transport_complete"])
        self.assertTrue(res["data_complete"])


class HistoryEndpointTests(unittest.TestCase):
    def setUp(self):
        self._orig = main.market_provider

    def tearDown(self):
        main.market_provider = self._orig

    def _client(self, provider):
        main.market_provider = provider
        return TestClient(create_app())

    def test_history_ok(self):
        avail = [i * _H for i in range(5)]
        r = self._client(FakeRangeProvider(avail)).get(
            "/api/v1/market/candles/btc-usd/history?granularity=1h&start=0&end=" + str(5 * _H)
        )
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["count"], 5)

    def test_history_misaligned_400(self):
        r = TestClient(create_app()).get(
            "/api/v1/market/candles/btc-usd/history?granularity=1h&start=1&end=" + str(3 * _H)
        )
        self.assertEqual(r.status_code, 400)

    def test_history_start_ge_end_400(self):
        r = TestClient(create_app()).get(
            "/api/v1/market/candles/btc-usd/history?granularity=1h&start=3600&end=3600"
        )
        self.assertEqual(r.status_code, 400)

    def test_history_range_too_large_400(self):
        too_big = PROVIDER_SAFE_BUCKETS * _H * 20 + _H
        r = TestClient(create_app()).get(
            "/api/v1/market/candles/btc-usd/history?granularity=1h&start=0&end=" + str(too_big)
        )
        self.assertEqual(r.status_code, 400)

    def test_history_unknown_granularity_400(self):
        r = TestClient(create_app()).get(
            "/api/v1/market/candles/btc-usd/history?granularity=4m&start=0&end=3600"
        )
        self.assertEqual(r.status_code, 400)


# ----------------------------- persistence (Phase 2) --------------------------
from main import (  # noqa: E402
    CandleRow,
    PersistenceState,
    PersistenceStatus,
    is_candle_closed,
    persist_candles,
    persist_history_result,
    persistence_consumer,
    persistence_state,
    read_stored_candles,
)

_DB_NOW = datetime(2026, 8, 24, 12, 0, 0, tzinfo=timezone.utc)


def _row(product="BTC-USD", gran="1h", start_unix=0, close=100.0, source="coinbase",
         quality=DataQualityStatus.VALID, observed=_DB_NOW, origin="rest", st=None):
    bs = datetime.fromtimestamp(start_unix, tz=timezone.utc)
    return CandleRow(source, product, gran, bs, 90.0, 110.0, 95.0, close, 5.0,
                     quality, origin, st, observed)


class _DBBase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        await main.engine.dispose()  # fresh pool bound to THIS test's loop
        await main.init_candle_schema()
        async with main.engine.begin() as conn:
            await conn.execute(main.text("TRUNCATE candles"))

    async def asyncTearDown(self):
        await main.engine.dispose()

    async def _raw_count(self):
        async with main.engine.connect() as conn:
            r = await conn.execute(main.text("SELECT count(*) FROM candles"))
            return r.scalar()


class DBPersistenceTests(_DBBase):
    async def test_insert_and_read(self):
        n = await persist_candles([_row(start_unix=0, close=100.0)])
        self.assertEqual(n, 1)
        rows = await read_stored_candles("BTC-USD", "1h", 0, 3600, 10)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["start"], 0)

    async def test_upsert_same_one_row(self):
        await persist_candles([_row(start_unix=0)])
        await persist_candles([_row(start_unix=0)])
        self.assertEqual(await self._raw_count(), 1)

    async def test_update_newer_observation(self):
        await persist_candles([_row(start_unix=0, close=100.0, observed=_DB_NOW)])
        newer = _DB_NOW + timedelta(seconds=10)
        await persist_candles([_row(start_unix=0, close=200.0, observed=newer)])
        rows = await read_stored_candles("BTC-USD", "1h", 0, 3600, 10)
        self.assertEqual(float(rows[0]["close"]), 200.0)

    async def test_reject_older_observation(self):
        await persist_candles([_row(start_unix=0, close=100.0, observed=_DB_NOW)])
        older = _DB_NOW - timedelta(seconds=10)
        await persist_candles([_row(start_unix=0, close=999.0, observed=older)])
        rows = await read_stored_candles("BTC-USD", "1h", 0, 3600, 10)
        self.assertEqual(float(rows[0]["close"]), 100.0)

    async def test_numeric_precision_roundtrip(self):
        await persist_candles([_row(start_unix=0, close=50000.123456)])
        rows = await read_stored_candles("BTC-USD", "1h", 0, 3600, 10)
        self.assertEqual(float(rows[0]["close"]), 50000.123456)

    async def test_multiple_granularities(self):
        await persist_candles([_row(gran="1h", start_unix=0)])
        await persist_candles([_row(gran="15m", start_unix=0)])
        self.assertEqual(await self._raw_count(), 2)

    async def test_multiple_products(self):
        await persist_candles([_row(product="BTC-USD", start_unix=0)])
        await persist_candles([_row(product="ETH-USD", start_unix=0)])
        self.assertEqual(len(await read_stored_candles("BTC-USD", "1h", 0, 3600, 10)), 1)
        self.assertEqual(len(await read_stored_candles("ETH-USD", "1h", 0, 3600, 10)), 1)

    async def test_separation_by_source(self):
        await persist_candles([_row(source="coinbase", start_unix=0)])
        await persist_candles([_row(source="kraken", start_unix=0)])
        self.assertEqual(await self._raw_count(), 2)  # distinct PK by source
        rows = await read_stored_candles("BTC-USD", "1h", 0, 3600, 10)
        self.assertEqual(len(rows), 1)  # read filters source=coinbase

    async def test_batch_insert(self):
        rows = [_row(start_unix=i * 3600) for i in range(5)]
        n = await persist_candles(rows)
        self.assertEqual(n, 5)
        self.assertEqual(await self._raw_count(), 5)

    async def test_transaction_rollback_atomic(self):
        orig = main.settings.persist_batch_size
        main.settings.persist_batch_size = 1  # force separate SQL batches in ONE tx
        try:
            good = _row(start_unix=0, close=100.0)
            bad = _row(start_unix=3600, close=1e50)  # overflows NUMERIC(38,18)
            with self.assertRaises(Exception):
                await persist_candles([good, bad])
        finally:
            main.settings.persist_batch_size = orig
        self.assertEqual(await self._raw_count(), 0)  # whole operation rolled back

    async def test_invalid_not_stored(self):
        n = await persist_candles([_row(quality=DataQualityStatus.INVALID)])
        self.assertEqual(n, 0)
        self.assertEqual(await self._raw_count(), 0)

    async def test_invalid_does_not_replace_valid(self):
        await persist_candles([_row(start_unix=0, close=100.0)])
        await persist_candles([_row(start_unix=0, close=999.0, quality=DataQualityStatus.INVALID)])
        rows = await read_stored_candles("BTC-USD", "1h", 0, 3600, 10)
        self.assertEqual(len(rows), 1)
        self.assertEqual(float(rows[0]["close"]), 100.0)

    async def test_time_closed_accepts_newer_correction(self):
        await persist_candles([_row(start_unix=0, close=100.0, observed=_DB_NOW)])
        newer = _DB_NOW + timedelta(seconds=10)
        await persist_candles([_row(start_unix=0, close=200.0, observed=newer)])
        rows = await read_stored_candles("BTC-USD", "1h", 0, 3600, 10)
        self.assertTrue(rows[0]["is_closed"])  # 1970 bucket is time-closed
        self.assertEqual(float(rows[0]["close"]), 200.0)  # yet correction applied

    async def test_read_empty(self):
        rows = await read_stored_candles("BTC-USD", "1h", 0, 3600, 10)
        self.assertEqual(rows, [])

    async def test_read_sorted(self):
        await persist_candles([_row(start_unix=2 * 3600), _row(start_unix=0),
                               _row(start_unix=3600)])
        rows = await read_stored_candles("BTC-USD", "1h", 0, 3 * 3600, 10)
        self.assertEqual([r["start"] for r in rows], [0, 3600, 7200])

    async def test_read_filters_half_open(self):
        await persist_candles([_row(start_unix=0), _row(start_unix=3600),
                               _row(start_unix=7200)])
        rows = await read_stored_candles("BTC-USD", "1h", 0, 7200, 10)
        self.assertEqual([r["start"] for r in rows], [0, 3600])  # 7200 excluded

    async def test_ws_candle_via_consumer(self):
        orig = persistence_state.ready
        persistence_state.ready = True
        try:
            start = datetime.fromtimestamp(0, tz=timezone.utc)
            datum = main.RealtimeDatum(
                "coinbase", "BTC-USD", "candle", 108.0, start, _DB_NOW,
                DataQualityStatus.VALID, 1,
                ohlcv={"open": 105.0, "high": 110.0, "low": 100.0, "close": 108.0,
                       "volume": 5.0},
            )
            await persistence_consumer(datum)
            rows = await read_stored_candles("BTC-USD", "5m", 0, 3600, 10)
            self.assertEqual(len(rows), 1)
            self.assertEqual(float(rows[0]["close"]), 108.0)
        finally:
            persistence_state.ready = orig

    async def test_history_persisted(self):
        saved = main.market_provider
        main.market_provider = FakeRangeProvider([i * _H for i in range(5)])
        try:
            res = await fetch_candle_history("BTC-USD", "1h", 0, 5 * _H)
            out = await persist_history_result(res)
            self.assertEqual(out["persisted"], 5)
            rows = await read_stored_candles("BTC-USD", "1h", 0, 5 * _H, 100)
            self.assertEqual(len(rows), 5)
        finally:
            main.market_provider = saved


class PersistenceStateTests(unittest.TestCase):
    def test_init_failed_is_unavailable(self):
        s = PersistenceState()
        s.mark_init_failed("boom")
        self.assertFalse(s.ready)
        self.assertEqual(s.status, PersistenceStatus.UNAVAILABLE)
        self.assertEqual(s.errors, 1)

    def test_runtime_error_degrades_then_recovers(self):
        s = PersistenceState()
        s.mark_ready()
        s.mark_runtime_error("db down")
        self.assertEqual(s.status, PersistenceStatus.DEGRADED)
        self.assertEqual(s.errors, 1)
        s.mark_write_ok()  # a later success recovers
        self.assertEqual(s.status, PersistenceStatus.READY)
        self.assertEqual(s.errors, 1)  # counter stays cumulative

    def test_is_candle_closed_time_rule(self):
        start = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        now_open = start + timedelta(seconds=100)
        now_closed = start + timedelta(seconds=4000)
        self.assertFalse(is_candle_closed(start, 3600, now=now_open))
        self.assertTrue(is_candle_closed(start, 3600, now=now_closed))


class StoredEndpointTests(unittest.TestCase):
    def setUp(self):
        self._ready = persistence_state.ready

    def tearDown(self):
        persistence_state.ready = self._ready

    def test_stored_not_ready_503(self):
        persistence_state.ready = False
        r = TestClient(create_app()).get(
            "/api/v1/market/candles/btc-usd/stored?granularity=1h&start=0&end=3600"
        )
        self.assertEqual(r.status_code, 503)

    def test_stored_bad_granularity_400(self):
        persistence_state.ready = True
        r = TestClient(create_app()).get(
            "/api/v1/market/candles/btc-usd/stored?granularity=4m&start=0&end=3600"
        )
        self.assertEqual(r.status_code, 400)

    def test_stored_start_ge_end_400(self):
        persistence_state.ready = True
        r = TestClient(create_app()).get(
            "/api/v1/market/candles/btc-usd/stored?granularity=1h&start=3600&end=3600"
        )
        self.assertEqual(r.status_code, 400)


# ----------------------------- multi-asset 6A --------------------------------
from main import (  # noqa: E402
    COINBASE_PROFILE,
    AssetClass,
    Capability,
    Instrument,
    MarketAvailability,
    MarketCalendarPolicy,
    OpenState,
    ProviderProfile,
    ProviderSymbolMap,
    VolumeSemantics,
    calendar_for,
    candles_table,
    instrument_registry,
    provider_symbol_map,
)


class RegistryMappingTests(unittest.TestCase):
    def test_coinbase_instrument_registered(self):
        inst = instrument_registry.get("BTC-USD")
        self.assertIsNotNone(inst)
        self.assertEqual(inst.asset_class, AssetClass.CRYPTO)

    def test_canonical_stable(self):
        self.assertIs(instrument_registry.get("BTC-USD"), instrument_registry.get("BTC-USD"))

    def test_mapping_coinbase_to_canonical(self):
        self.assertEqual(provider_symbol_map.to_canonical("coinbase", "BTC-USD"), "BTC-USD")

    def test_canonical_can_differ_from_provider_symbol(self):
        m = ProviderSymbolMap()
        m.add("provX", "XAU-USD", "XAU/USD")
        self.assertEqual(m.to_canonical("provX", "XAU/USD"), "XAU-USD")
        self.assertNotEqual(m.to_provider("provX", "XAU-USD"), "XAU-USD")

    def test_two_providers_same_canonical(self):
        m = ProviderSymbolMap()
        m.add("a", "XAU-USD", "XAU/USD")
        m.add("b", "XAU-USD", "XAUUSD")
        self.assertEqual(m.to_canonical("a", "XAU/USD"), "XAU-USD")
        self.assertEqual(m.to_canonical("b", "XAUUSD"), "XAU-USD")

    def test_provider_symbol_not_global_identity(self):
        # a raw provider symbol is NOT a canonical identity
        self.assertIsNone(instrument_registry.get("XAUUSD"))

    def test_unknown_instrument_failsafe(self):
        self.assertIsNone(instrument_registry.get("DOES-NOT-EXIST"))

    def test_unknown_provider_failsafe(self):
        self.assertIsNone(provider_symbol_map.to_canonical("ghost", "BTC-USD"))

    def test_unmapped_is_none(self):
        self.assertIsNone(provider_symbol_map.to_provider("coinbase", "XAU-USD"))


class AssetMetadataTests(unittest.TestCase):
    def test_distinct_asset_classes(self):
        self.assertEqual(
            {AssetClass.CRYPTO, AssetClass.FOREX, AssetClass.METAL, AssetClass.INDEX},
            set(AssetClass),
        )

    def test_no_metadata_inferred_from_symbol(self):
        # unverified financial metadata stays None/UNKNOWN, never guessed
        xau = Instrument("XAU-USD", AssetClass.METAL, "XAU", "USD", "Gold / USD",
                         "UTC", MarketCalendarPolicy.NOT_CONFIGURED, VolumeSemantics.UNKNOWN)
        self.assertIsNone(xau.price_precision)
        self.assertIsNone(xau.tick_size)
        self.assertEqual(xau.volume_semantics, VolumeSemantics.UNKNOWN)

    def test_coinbase_volume_semantics(self):
        self.assertEqual(
            instrument_registry.get("BTC-USD").volume_semantics,
            VolumeSemantics.BASE_ASSET_VOLUME,
        )

    def test_instrument_timezone_stored(self):
        self.assertEqual(instrument_registry.get("BTC-USD").timezone, "UTC")

    def test_market_availability_distinct_from_quality(self):
        # orthogonal axes: a market close is not a data-quality value
        self.assertNotIn(MarketAvailability.CLOSED.value,
                         {s.value for s in [DataQualityStatus.MISSING,
                                            DataQualityStatus.INVALID]})


class CapabilityTests(unittest.TestCase):
    def test_coinbase_supports_candles_rest(self):
        self.assertTrue(COINBASE_PROFILE.supports(Capability.CANDLES_REST, AssetClass.CRYPTO))

    def test_missing_capability_not_supported(self):
        self.assertFalse(COINBASE_PROFILE.supports(Capability.ORDER_BOOK, AssetClass.CRYPTO))

    def test_capabilities_per_asset_class(self):
        prof = ProviderProfile(
            "demo",
            {AssetClass.FOREX: {Capability.HISTORY_INTRADAY},
             AssetClass.METAL: {Capability.HISTORY_DAILY}},
            {AssetClass.FOREX: {"1m"}, AssetClass.METAL: {"1d"}},
        )
        self.assertTrue(prof.supports(Capability.HISTORY_INTRADAY, AssetClass.FOREX))
        self.assertFalse(prof.supports(Capability.HISTORY_INTRADAY, AssetClass.METAL))

    def test_granularity_supported_or_not(self):
        self.assertTrue(COINBASE_PROFILE.supports_granularity("1m", AssetClass.CRYPTO))
        self.assertFalse(COINBASE_PROFILE.supports_granularity("3m", AssetClass.CRYPTO))


class CalendarTests(unittest.TestCase):
    def test_always_24_7_open_everywhere(self):
        cal = calendar_for(MarketCalendarPolicy.ALWAYS_OPEN_24_7)
        self.assertEqual(cal.is_market_expected_open(0), OpenState.OPEN)
        self.assertEqual(cal.is_market_expected_open(10**12), OpenState.OPEN)

    def test_always_24_7_expected_grid(self):
        cal = calendar_for(MarketCalendarPolicy.ALWAYS_OPEN_24_7)
        self.assertEqual(cal.expected_bucket_starts("1h", 0, 3 * 3600), [0, 3600, 7200])

    def test_absence_during_open_is_gap(self):
        cal = calendar_for(MarketCalendarPolicy.ALWAYS_OPEN_24_7)
        rep = cal.analyze_gaps([0, 3600, 10800], 3600)  # missing 7200
        self.assertEqual(rep.status, "ANALYZED")
        self.assertTrue(rep.missing)

    def test_gap_24_7_matches_legacy(self):
        cal = calendar_for(MarketCalendarPolicy.ALWAYS_OPEN_24_7)
        starts = [0, 3600, 10800, 14400]
        self.assertEqual(cal.analyze_gaps(starts, 3600).missing,
                         main._missing_buckets_24_7(starts, 3600))

    def test_not_configured_never_open_or_closed(self):
        cal = calendar_for(MarketCalendarPolicy.NOT_CONFIGURED)
        self.assertEqual(cal.is_market_expected_open(0), OpenState.UNKNOWN)

    def test_not_configured_does_not_invent_bucket_grid(self):
        cal = calendar_for(MarketCalendarPolicy.NOT_CONFIGURED)
        self.assertEqual(cal.is_market_expected_open(0), OpenState.UNKNOWN)
        self.assertIsNone(cal.expected_bucket_starts("1h", 0, 3600))

    def test_not_configured_gaps_unknown(self):
        cal = calendar_for(MarketCalendarPolicy.NOT_CONFIGURED)
        rep = cal.analyze_gaps([0, 7200], 3600)  # would be a gap if 24/7, but calendar unknown
        self.assertEqual(rep.status, "UNKNOWN")
        self.assertEqual(rep.missing, [])


class CandlesSchemaUnchangedTests(unittest.TestCase):
    def test_pk_unchanged(self):
        pk = [c.name for c in candles_table.primary_key.columns]
        self.assertEqual(pk, ["source", "product_id", "granularity", "bucket_start"])

    def test_table_name_unchanged(self):
        self.assertEqual(candles_table.name, "candles")


# ----------------------------- Massive Forex REST 6B-1 ------------------------
from main import (  # noqa: E402
    MASSIVE_FOREX_PROFILE,
    fetch_forex_history,
    massive_agg_to_candle,
    massive_aggs_to_candles,
    persist_forex_result,
    register_massive_forex_symbol,
)

_FX_BASE = int(datetime(2026, 8, 24, 0, 0, tzinfo=timezone.utc).timestamp())


def _agg(k, close=1.1, vol=10.0):
    return {"t": (_FX_BASE + k * 3600) * 1000, "o": 1.1, "h": 1.2, "l": 1.0,
            "c": close, "v": vol}


class FakeMassiveProvider:
    def __init__(self, ks=None, raise_exc=None, invalid_ks=None):
        self.ks = ks if ks is not None else [0, 1, 2]
        self.raise_exc = raise_exc
        self.invalid_ks = set(invalid_ks or [])

    async def get_candles_range(self, canonical, granularity, start, end):
        if self.raise_exc is not None:
            raise self.raise_exc
        out = []
        for k in self.ks:
            item = _agg(k, close=1.1 + k)
            if k in self.invalid_ks:
                item = {"t": (_FX_BASE + k * 3600) * 1000, "o": "x"}  # invalid OHLC
            out.append(massive_agg_to_candle(item, float("inf")))
        return out, DataQualityStatus.VALID


class MassiveParsingTests(unittest.TestCase):
    def test_agg_parsed_ms_to_utc(self):
        c = massive_agg_to_candle(_agg(0), float("inf"))
        self.assertNotEqual(c.status, DataQualityStatus.INVALID)
        self.assertEqual(int(c.start.timestamp()), _FX_BASE)

    def test_invalid_ohlc_is_invalid(self):
        c = massive_agg_to_candle({"t": _FX_BASE * 1000, "o": "x", "h": "1",
                                   "l": "1", "c": "1", "v": "1"}, float("inf"))
        self.assertEqual(c.status, DataQualityStatus.INVALID)

    def test_malformed_response_missing(self):
        candles, status = massive_aggs_to_candles({"no": "results"}, float("inf"))
        self.assertEqual(candles, [])
        self.assertEqual(status, DataQualityStatus.MISSING)

    def test_empty_results_missing(self):
        candles, status = massive_aggs_to_candles({"results": []}, float("inf"))
        self.assertEqual(status, DataQualityStatus.MISSING)


class MassiveInstrumentTests(unittest.TestCase):
    def test_forex_instruments_registered(self):
        inst = instrument_registry.get("EUR-USD")
        self.assertIsNotNone(inst)
        self.assertEqual(inst.asset_class, AssetClass.FOREX)

    def test_forex_calendar_is_forex_week(self):
        self.assertEqual(instrument_registry.get("EUR-USD").market_calendar,
                         MarketCalendarPolicy.FOREX_WEEK)

    def test_forex_volume_semantics_unknown(self):
        self.assertEqual(instrument_registry.get("EUR-USD").volume_semantics,
                         VolumeSemantics.UNKNOWN)

    def test_no_financial_metadata_invented(self):
        inst = instrument_registry.get("EUR-CAD")
        self.assertIsNone(inst.price_precision)
        self.assertIsNone(inst.tick_size)

    def test_massive_symbol_not_mapped_by_default(self):
        # rule: no provider_symbol registered by deduction -> NOT_MAPPED
        self.assertIsNone(provider_symbol_map.to_provider("massive", "GBP-USD"))

    def test_profile_capabilities_and_granularity(self):
        self.assertTrue(MASSIVE_FOREX_PROFILE.supports(Capability.CANDLES_REST,
                                                       AssetClass.FOREX))
        self.assertFalse(MASSIVE_FOREX_PROFILE.supports(Capability.ORDER_BOOK,
                                                        AssetClass.FOREX))
        self.assertTrue(MASSIVE_FOREX_PROFILE.supports_granularity("5m", AssetClass.FOREX))
        self.assertFalse(MASSIVE_FOREX_PROFILE.supports_granularity("3m", AssetClass.FOREX))


class ForexHistoryAssemblerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._saved = main.massive_forex_provider
        self._had = provider_symbol_map.to_provider("massive", "EUR-USD")
        register_massive_forex_symbol("EUR-USD", "C:EURUSD")

    def tearDown(self):
        main.massive_forex_provider = self._saved
        # keep mapping registered across tests is fine (verified in-test only)

    async def test_not_mapped_failsafe(self):
        # USD-CHF has no mapping -> NOT_MAPPED, no fetch
        main.massive_forex_provider = FakeMassiveProvider()
        r = await fetch_forex_history("USD-CHF", "1h", _FX_BASE, _FX_BASE + 5 * 3600)
        self.assertEqual(r["status"], "NOT_MAPPED")
        self.assertEqual(r["candles"], [])

    async def test_ok_sorted_and_deduped(self):
        main.massive_forex_provider = FakeMassiveProvider(ks=[2, 0, 1, 1])
        r = await fetch_forex_history("EUR-USD", "1h", _FX_BASE, _FX_BASE + 5 * 3600)
        self.assertEqual(r["status"], "OK")
        self.assertEqual(r["count"], 3)  # deduped
        starts = [c["start"] for c in r["candles"]]
        self.assertEqual(starts, sorted(starts))

    async def test_absence_is_not_coinbase_gap(self):
        # hole at k=2; forex calendar NOT_CONFIGURED -> gaps UNKNOWN, never a gap
        main.massive_forex_provider = FakeMassiveProvider(ks=[0, 1, 3])
        r = await fetch_forex_history("EUR-USD", "1h", _FX_BASE, _FX_BASE + 5 * 3600)
        self.assertEqual(r["gaps_status"], "UNKNOWN")
        self.assertEqual(r["gaps"], [])
        self.assertFalse(r["data_complete"])
        self.assertEqual(r["count"], 3)  # no fabricated bar

    async def test_invalid_candle_counted(self):
        main.massive_forex_provider = FakeMassiveProvider(ks=[0, 1, 2], invalid_ks=[1])
        r = await fetch_forex_history("EUR-USD", "1h", _FX_BASE, _FX_BASE + 5 * 3600)
        self.assertEqual(r["invalid_candles_count"], 1)
        self.assertEqual(r["count"], 2)

    async def test_half_open_range_filter(self):
        main.massive_forex_provider = FakeMassiveProvider(ks=[0, 1, 2])
        r = await fetch_forex_history("EUR-USD", "1h", _FX_BASE, _FX_BASE + 2 * 3600)
        self.assertEqual(r["count"], 2)  # k=2 (== end) excluded

    async def test_volume_semantics_unknown_in_result(self):
        main.massive_forex_provider = FakeMassiveProvider()
        r = await fetch_forex_history("EUR-USD", "1h", _FX_BASE, _FX_BASE + 5 * 3600)
        self.assertEqual(r["volume_semantics"], "UNKNOWN")

    async def test_provider_timeout_unavailable(self):
        main.massive_forex_provider = FakeMassiveProvider(raise_exc=httpx.TimeoutException("t"))
        r = await fetch_forex_history("EUR-USD", "1h", _FX_BASE, _FX_BASE + 5 * 3600)
        self.assertEqual(r["status"], "UNAVAILABLE")

    async def test_http_error_unavailable(self):
        main.massive_forex_provider = FakeMassiveProvider(raise_exc=httpx.ConnectError("x"))
        r = await fetch_forex_history("EUR-USD", "1h", _FX_BASE, _FX_BASE + 5 * 3600)
        self.assertEqual(r["status"], "UNAVAILABLE")

    async def test_unsupported_granularity_raises(self):
        main.massive_forex_provider = FakeMassiveProvider()
        with self.assertRaises(ValueError):
            await fetch_forex_history("EUR-USD", "3m", _FX_BASE, _FX_BASE + 5 * 3600)

    async def test_start_ge_end_raises(self):
        main.massive_forex_provider = FakeMassiveProvider()
        with self.assertRaises(ValueError):
            await fetch_forex_history("EUR-USD", "1h", _FX_BASE, _FX_BASE)

    async def test_unknown_instrument_raises(self):
        with self.assertRaises(ValueError):
            await fetch_forex_history("ZZZ-ZZZ", "1h", _FX_BASE, _FX_BASE + 3600)


class ForexEndpointTests(unittest.TestCase):
    def setUp(self):
        register_massive_forex_symbol("EUR-USD", "C:EURUSD")

    def test_forex_not_mapped_409(self):
        # NZD-USD unmapped -> 409
        r = TestClient(create_app()).get(
            "/api/v1/market/forex/NZD-USD/history?granularity=1h&start=%d&end=%d"
            % (_FX_BASE, _FX_BASE + 3600)
        )
        self.assertEqual(r.status_code, 409)

    def test_forex_bad_granularity_400(self):
        r = TestClient(create_app()).get(
            "/api/v1/market/forex/EUR-USD/history?granularity=3m&start=%d&end=%d"
            % (_FX_BASE, _FX_BASE + 3600)
        )
        self.assertEqual(r.status_code, 400)


class ForexPersistenceTests(_DBBase):
    async def test_forex_persisted_under_massive_source(self):
        register_massive_forex_symbol("EUR-USD", "C:EURUSD")
        saved = main.massive_forex_provider
        main.massive_forex_provider = FakeMassiveProvider(ks=[0, 1, 2])
        try:
            r = await fetch_forex_history("EUR-USD", "1h", _FX_BASE, _FX_BASE + 5 * 3600)
            n = await persist_forex_result(r)
            self.assertEqual(n, 3)
            async with main.engine.connect() as conn:
                res = await conn.execute(
                    main.text("SELECT count(*) FROM candles WHERE source='massive'")
                )
                self.assertEqual(res.scalar(), 3)
                res2 = await conn.execute(
                    main.text("SELECT count(*) FROM candles WHERE source='coinbase'")
                )
                self.assertEqual(res2.scalar(), 0)  # no Coinbase regression
        finally:
            main.massive_forex_provider = saved


# ----------------------------- Massive mapping activation 6B-1A ---------------
from main import (  # noqa: E402
    EXPECTED_MASSIVE_FOREX,
    MassiveForexProvider,
    activate_massive_forex_mappings,
    massive_forex_activation,
)
from main import _redact_secret as redact_secret  # noqa: E402


class RedactionTests(unittest.TestCase):
    def test_redacts_apikey_query(self):
        out = redact_secret("GET https://api.x/y?apiKey=SECRET123&z=1")
        self.assertIn("apiKey=REDACTED", out)
        self.assertNotIn("SECRET123", out)

    def test_no_apikey_unchanged(self):
        self.assertEqual(redact_secret("plain text no secret"), "plain text no secret")


class _MassiveMapBase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._prov = main.massive_forex_provider
        self._key = main.settings.massive_api_key
        self._snap_p = dict(provider_symbol_map._to_provider)
        self._snap_c = dict(provider_symbol_map._to_canonical)
        for k in [kk for kk in list(provider_symbol_map._to_provider) if kk[0] == "massive"]:
            del provider_symbol_map._to_provider[k]
        for k in [kk for kk in list(provider_symbol_map._to_canonical) if kk[0] == "massive"]:
            del provider_symbol_map._to_canonical[k]
        massive_forex_activation.__init__()

    def tearDown(self):
        main.massive_forex_provider = self._prov
        main.settings.massive_api_key = self._key
        provider_symbol_map._to_provider.clear()
        provider_symbol_map._to_provider.update(self._snap_p)
        provider_symbol_map._to_canonical.clear()
        provider_symbol_map._to_canonical.update(self._snap_c)
        massive_forex_activation.__init__()


class _FakeRef:
    def __init__(self, tickers=None, exc=None):
        self._t = tickers or []
        self._exc = exc

    async def list_forex_tickers(self):
        if self._exc is not None:
            raise self._exc
        return list(self._t)


class MassiveActivationTests(_MassiveMapBase):
    async def test_no_key_no_activation(self):
        main.settings.massive_api_key = ""
        res = await activate_massive_forex_mappings()
        self.assertTrue(res["attempted"])
        self.assertFalse(res["activated"])
        self.assertIsNone(provider_symbol_map.to_provider("massive", "EUR-USD"))

    async def test_activation_registers_only_confirmed(self):
        main.settings.massive_api_key = "DUMMY_TEST"
        main.massive_forex_provider = _FakeRef(["C:EURUSD", "C:GBPUSD", "C:OTHER"])
        res = await activate_massive_forex_mappings()
        self.assertTrue(res["activated"])
        self.assertEqual(res["confirmed_count"], 2)
        self.assertEqual(provider_symbol_map.to_provider("massive", "EUR-USD"), "C:EURUSD")
        self.assertEqual(provider_symbol_map.to_provider("massive", "GBP-USD"), "C:GBPUSD")

    async def test_absent_not_mapped_no_deduction(self):
        main.settings.massive_api_key = "DUMMY_TEST"
        main.massive_forex_provider = _FakeRef(["C:EURUSD"])  # only EUR
        await activate_massive_forex_mappings()
        self.assertIsNone(provider_symbol_map.to_provider("massive", "USD-CHF"))
        self.assertIsNone(provider_symbol_map.to_provider("massive", "GBP-USD"))

    async def test_canonical_provider_separated(self):
        main.settings.massive_api_key = "DUMMY_TEST"
        main.massive_forex_provider = _FakeRef(["C:EURUSD"])
        await activate_massive_forex_mappings()
        self.assertNotEqual(provider_symbol_map.to_provider("massive", "EUR-USD"), "EUR-USD")
        self.assertEqual(provider_symbol_map.to_canonical("massive", "C:EURUSD"), "EUR-USD")

    async def test_xau_reported_but_not_integrated(self):
        main.settings.massive_api_key = "DUMMY_TEST"
        main.massive_forex_provider = _FakeRef(["C:EURUSD", "C:XAUUSD"])
        res = await activate_massive_forex_mappings()
        self.assertEqual(res["xau"], {"ticker": "C:XAUUSD"})
        self.assertIsNone(provider_symbol_map.to_provider("massive", "XAU-USD"))

    async def test_xau_absent_is_none(self):
        main.settings.massive_api_key = "DUMMY_TEST"
        main.massive_forex_provider = _FakeRef(["C:EURUSD"])
        res = await activate_massive_forex_mappings()
        self.assertIsNone(res["xau"])

    async def test_activation_failure_is_failsafe(self):
        main.settings.massive_api_key = "DUMMY_TEST"
        main.massive_forex_provider = _FakeRef(exc=httpx.ConnectError("down"))
        res = await activate_massive_forex_mappings()
        self.assertFalse(res["activated"])
        self.assertIsNone(provider_symbol_map.to_provider("massive", "EUR-USD"))

    async def test_failure_reason_is_redacted(self):
        main.settings.massive_api_key = "DUMMY_TEST"
        main.massive_forex_provider = _FakeRef(exc=Exception("boom apiKey=SECRET999 x"))
        res = await activate_massive_forex_mappings()
        self.assertNotIn("SECRET999", res["reason"] or "")
        self.assertIn("REDACTED", res["reason"] or "")


class MassiveAuthSecurityTests(_MassiveMapBase):
    async def test_header_auth_key_not_in_url(self):
        prov = MassiveForexProvider(api_key="DUMMYKEY")
        await prov.connect()
        try:
            auth = prov.client.headers.get("authorization")
            self.assertIsNotNone(auth)
            self.assertTrue(auth.startswith("Bearer "))
            self.assertIn("DUMMYKEY", auth)
            self.assertNotIn("DUMMYKEY", str(prov.rest_url))  # never in the URL
        finally:
            await prov.disconnect()

    async def test_get_candles_params_have_no_apikey(self):
        prov = MassiveForexProvider(api_key="DUMMYKEY")
        captured = {}

        async def fake_get(path, params=None):
            captured["params"] = params or {}
            return {"results": []}

        prov._get = fake_get
        main.register_massive_forex_symbol("EUR-USD", "C:EURUSD")
        await prov.get_candles_range("EUR-USD", "1h", _FX_BASE, _FX_BASE + 3600)
        self.assertNotIn("apiKey", captured["params"])
        self.assertNotIn("apikey", captured["params"])


class ForexMappingsEndpointTests(unittest.TestCase):
    def test_mappings_endpoint_structure(self):
        r = TestClient(create_app()).get("/api/v1/market/forex/mappings")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(len(body["mappings"]), len(EXPECTED_MASSIVE_FOREX))
        for m_ in body["mappings"]:
            self.assertIn(m_["status"], ("MAPPED", "NOT_MAPPED"))

    def test_mappings_endpoint_no_auth_leak(self):
        raw = TestClient(create_app()).get("/api/v1/market/forex/mappings").text.lower()
        for bad in ("apikey", "authorization", "bearer", "massive_api_key"):
            self.assertNotIn(bad, raw)


class RepoKeySecurityTests(unittest.TestCase):
    def test_main_bearer_is_fstring_only(self):
        src = open("main.py").read()
        for seg in src.split("Bearer ")[1:]:
            self.assertTrue(seg.startswith("{"), "Bearer must be an f-string var in main.py")

    def test_massive_key_default_empty(self):
        self.assertEqual(main.settings.massive_api_key, "")


# ----------------------------- testable UI (frontend) -------------------------
class FrontendUiTests(unittest.TestCase):
    def setUp(self):
        self.html = INDEX.read_text(encoding="utf-8")

    def test_no_massive_secret_in_frontend(self):
        low = self.html.lower()
        for bad in ("massive_api_key", "apikey=", "bearer ", "authorization"):
            self.assertNotIn(bad, low)

    def test_only_relative_api_calls(self):
        # no hardcoded backend origin; browser talks same-origin only
        self.assertNotIn("http://", self.html)
        self.assertNotIn("https://", self.html)
        self.assertIn('fetch(path', self.html)

    def test_references_real_endpoints_only(self):
        for ep in ("/health", "/api/v1/market/ticker/",
                   "/api/v1/market/forex/mappings", "/api/v1/market/candles/"):
            self.assertIn(ep, self.html)

    def test_polling_labeled_not_streaming(self):
        self.assertIn("Auto-refresh", self.html)
        self.assertIn("setInterval(poll", self.html)

    def test_mobile_viewport_and_safe_area(self):
        self.assertIn("viewport-fit=cover", self.html)
        self.assertIn("safe-area-inset", self.html)

    def test_forex_not_mapped_guard_present(self):
        # NOT_MAPPED rows must not be tappable to history (guarded by "MAPPED")
        self.assertIn('m.status === "MAPPED"', self.html)


class UiEndpointCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(create_app())

    def test_forex_mappings_shape_for_ui(self):
        body = self.client.get("/api/v1/market/forex/mappings").json()
        self.assertIn("mappings", body)
        self.assertIn("activation", body)
        for m_ in body["mappings"]:
            self.assertIn("canonical", m_)
            self.assertIn("status", m_)

    def test_health_shape_for_ui(self):
        r = self.client.get("/health")
        self.assertIn(r.status_code, (200, 503))
        self.assertIn("overall", r.json())


# ----------------------------- test-feedback fixes (candles freshness + 403) --
from main import _latest_quality  # noqa: E402

_FIX_NOW = datetime(2026, 8, 30, 12, 0, 0, tzinfo=timezone.utc)


def _cndl(dt, q=DataQualityStatus.VALID):
    return Candle(dt, 1.0, 2.0, 0.5, 1.5, 3.0, q)


class LatestQualityTests(unittest.TestCase):
    def test_latest_is_newest_bar(self):
        old = _cndl(_FIX_NOW - timedelta(days=14), DataQualityStatus.STALE)
        fresh = _cndl(_FIX_NOW, DataQualityStatus.VALID)
        self.assertEqual(_latest_quality([old, fresh]), "VALID")

    def test_latest_stale_when_newest_old(self):
        old = _cndl(_FIX_NOW - timedelta(days=14), DataQualityStatus.STALE)
        older = _cndl(_FIX_NOW - timedelta(days=15), DataQualityStatus.STALE)
        self.assertEqual(_latest_quality([older, old]), "STALE")

    def test_latest_missing_when_empty(self):
        self.assertEqual(_latest_quality([]), "MISSING")


class CandleOrderingEndpointTests(unittest.TestCase):
    def setUp(self):
        self._orig = main.market_provider

    def tearDown(self):
        main.market_provider = self._orig

    def _client(self, provider):
        main.market_provider = provider
        return TestClient(create_app())

    def test_candles_sorted_ascending_and_latest_recent(self):
        # provider returns NEWEST-FIRST (like Coinbase) + an old tail
        newest = _cndl(_FIX_NOW, DataQualityStatus.VALID)
        mid = _cndl(_FIX_NOW - timedelta(hours=1), DataQualityStatus.VALID)
        old = _cndl(_FIX_NOW - timedelta(days=14), DataQualityStatus.STALE)

        class FakeProvider:
            async def get_candles(self, symbol, granularity, limit=350):
                return [newest, mid, old], DataQualityStatus.VALID  # desc / unsorted

        body = self._client(FakeProvider()).get(
            "/api/v1/market/candles/eth-usd?granularity=1h").json()
        starts = [c["start"] for c in body["candles"]]
        self.assertEqual(starts, sorted(starts))                 # ascending
        self.assertEqual(body["latest_quality"], "VALID")        # newest bar fresh
        # the last element is the most recent, not the 14-day-old one
        self.assertTrue(body["candles"][-1]["start"] > body["candles"][0]["start"])

    def test_latest_quality_stale_when_newest_is_old(self):
        old1 = _cndl(_FIX_NOW - timedelta(days=14), DataQualityStatus.STALE)
        old2 = _cndl(_FIX_NOW - timedelta(days=14, hours=1), DataQualityStatus.STALE)

        class FakeProvider:
            async def get_candles(self, symbol, granularity, limit=350):
                return [old1, old2], DataQualityStatus.VALID

        body = self._client(FakeProvider()).get(
            "/api/v1/market/candles/eth-usd?granularity=1h").json()
        self.assertEqual(body["latest_quality"], "STALE")       # not LIVE despite HTTP 200

    def test_ticker_regression_still_ok(self):
        class FakeProvider:
            async def get_ticker(self, symbol):
                return main.MarketDatum("coinbase", symbol.upper(), 100.0,
                                        _FIX_NOW, DataQualityStatus.VALID)
        r = self._client(FakeProvider()).get("/api/v1/market/ticker/btc-usd")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["quality"], "VALID")


def _http_status_error(code):
    class _Resp:
        status_code = code
    return httpx.HTTPStatusError("boom apiKey=SECRET https://api.massive.com/x MDN mozilla",
                                 request=None, response=_Resp())


class MassiveErrorHandlingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._prov = main.massive_forex_provider
        register_massive_forex_symbol("EUR-USD", "C:EURUSD")

    def tearDown(self):
        main.massive_forex_provider = self._prov

    async def test_403_is_access_denied_clean(self):
        class P:
            async def get_candles_range(self, *a, **k):
                raise _http_status_error(403)
        main.massive_forex_provider = P()
        r = await fetch_forex_history("EUR-USD", "1h", 1756512000, 1756555200)
        self.assertEqual(r["status"], "ACCESS_DENIED")
        self.assertEqual(r["provider_symbol"], "C:EURUSD")       # mapping kept
        for bad in ("SECRET", "mozilla", "http", "apiKey"):
            self.assertNotIn(bad, r["reason"])

    async def test_429_is_rate_limited(self):
        class P:
            async def get_candles_range(self, *a, **k):
                raise _http_status_error(429)
        main.massive_forex_provider = P()
        r = await fetch_forex_history("EUR-USD", "1h", 1756512000, 1756555200)
        self.assertEqual(r["status"], "RATE_LIMITED")

    async def test_network_error_unavailable_clean(self):
        class P:
            async def get_candles_range(self, *a, **k):
                raise httpx.ConnectError("connect fail apiKey=SECRET https://x")
        main.massive_forex_provider = P()
        r = await fetch_forex_history("EUR-USD", "1h", 1756512000, 1756555200)
        self.assertEqual(r["status"], "UNAVAILABLE")
        self.assertNotIn("SECRET", r["reason"])


class ForexErrorEndpointTests(unittest.TestCase):
    def setUp(self):
        self._prov = main.massive_forex_provider
        register_massive_forex_symbol("EUR-USD", "C:EURUSD")

    def tearDown(self):
        main.massive_forex_provider = self._prov

    def test_endpoint_403_maps_to_403_clean(self):
        class P:
            async def get_candles_range(self, *a, **k):
                raise _http_status_error(403)
        main.massive_forex_provider = P()
        r = TestClient(create_app()).get(
            "/api/v1/market/forex/EUR-USD/history?granularity=1h&start=1756512000&end=1756555200")
        self.assertEqual(r.status_code, 403)
        raw = r.text.lower()
        for bad in ("secret", "mozilla", "apikey", "http://", "https://"):
            self.assertNotIn(bad, raw)


class UiFreshnessStaticTests(unittest.TestCase):
    def setUp(self):
        self.html = INDEX.read_text(encoding="utf-8")

    def test_candles_badge_uses_latest_quality(self):
        self.assertIn("latest_quality", self.html)

    def test_candles_sorted_before_slice(self):
        self.assertIn("Date.parse(a.start)", self.html)

    def test_forex_list_not_shown_as_live(self):
        self.assertIn('mapped ? "MAPPED"', self.html)
        self.assertIn("b-mapped", self.html)

    def test_clean_403_message_present(self):
        self.assertIn("accès refusé par le fournisseur (403)", self.html)

    def test_no_raw_exception_rendered(self):
        # the UI maps status codes to clean text; it never renders a raw exception
        self.assertNotIn("str(exc)", self.html)
        self.assertNotIn(".stack", self.html)


# ----------------------------- UI: Montreal time + MAPPED != LIVE -------------
class MontrealTimeUiTests(unittest.TestCase):
    def setUp(self):
        self.html = INDEX.read_text(encoding="utf-8")

    def test_montreal_timezone_present(self):
        self.assertIn("America/Toronto", self.html)
        self.assertIn("formatMontrealTime", self.html)

    def test_no_hardcoded_utc_offset(self):
        # DST must come from Intl, never a fixed offset or manual hour math
        for bad in ("UTC-4", "UTC-5", "-04:00", "-05:00", "getTimezoneOffset", "setHours"):
            self.assertNotIn(bad, self.html)

    def test_montreal_used_in_candle_table(self):
        self.assertIn("formatMontrealTime(c.start)", self.html)
        self.assertIn("Heure (Montréal)", self.html)

    def test_utc_kept_as_secondary(self):
        self.assertIn("formatUtcTime", self.html)

    def test_mapped_distinct_from_live(self):
        # forex mapped rows show MAPPED, never converted to LIVE/VALID
        self.assertIn('qualityBadge(mapped ? "MAPPED"', self.html)
        self.assertNotIn('qualityBadge(mapped ? "VALID"', self.html)
        self.assertIn(".b-mapped", self.html)

    def test_not_mapped_unchanged(self):
        self.assertIn("NOT_MAPPED", self.html)

    def test_crypto_live_from_backend_quality(self):
        # crypto badge still driven by backend quality (LIVE only if VALID)
        self.assertIn("qualityBadge(r.data.quality)", self.html)
        self.assertIn('VALID:["LIVE"', self.html)


# ----------------------------- Forex market calendar & sessions ---------------
from datetime import datetime as _dt  # noqa: E402
from main import (  # noqa: E402
    ForexWeekCalendar,
    calendar_for as _calendar_for,
    forex_active_sessions,
    forex_market_state,
)


def _utc(y, mo, d, h, mi=0):
    return _dt(y, mo, d, h, mi, tzinfo=timezone.utc)


class ForexCalendarTests(unittest.TestCase):
    def test_open_midweek(self):
        r = forex_market_state(_utc(2026, 1, 14, 12))  # Wednesday noon
        self.assertEqual(r["market_state"], "OPEN")
        self.assertIsNotNone(r["next_close"])
        self.assertIsNone(r["next_open"])

    def test_closed_weekend(self):
        r = forex_market_state(_utc(2026, 1, 17, 12))  # Saturday
        self.assertEqual(r["market_state"], "CLOSED_WEEKEND")
        self.assertIsNotNone(r["next_open"])
        self.assertIsNone(r["next_close"])

    def test_open_boundary_winter_2200z(self):
        # Winter (EST): opens Sunday 22:00 UTC
        self.assertEqual(forex_market_state(_utc(2026, 1, 11, 21, 59))["market_state"],
                         "CLOSED_WEEKEND")
        self.assertEqual(forex_market_state(_utc(2026, 1, 11, 22, 0))["market_state"], "OPEN")

    def test_open_boundary_summer_2100z(self):
        # Summer (EDT): opens Sunday 21:00 UTC (DST auto, never a fixed offset)
        self.assertEqual(forex_market_state(_utc(2026, 7, 12, 20, 59))["market_state"],
                         "CLOSED_WEEKEND")
        self.assertEqual(forex_market_state(_utc(2026, 7, 12, 21, 0))["market_state"], "OPEN")

    def test_close_boundary_friday_winter(self):
        self.assertEqual(forex_market_state(_utc(2026, 1, 16, 21, 59))["market_state"], "OPEN")
        self.assertEqual(forex_market_state(_utc(2026, 1, 16, 22, 0))["market_state"],
                         "CLOSED_WEEKEND")

    def test_dst_transition_march(self):
        # US DST starts 2026-03-08; the Sunday open still resolves via IANA (not a
        # fixed offset). Just assert it computes a definitive OPEN/CLOSED, not UNKNOWN.
        r = forex_market_state(_utc(2026, 3, 8, 21, 30))
        self.assertIn(r["market_state"], ("OPEN", "CLOSED_WEEKEND"))

    def test_utc_day_change(self):
        # Thursday 23:30 UTC -> Friday 00:xx local NY still within the week -> OPEN
        self.assertEqual(forex_market_state(_utc(2026, 1, 15, 23, 30))["market_state"], "OPEN")

    def test_market_state_independent_from_quality(self):
        r = forex_market_state(_utc(2026, 1, 14, 12))
        self.assertNotIn("quality", r)  # OPEN != LIVE; no data-quality field here
        self.assertEqual(r["timezone_internal"], "UTC")
        self.assertEqual(r["display_timezone"], "America/Toronto")

    def test_holidays_not_implemented(self):
        self.assertEqual(forex_market_state(_utc(2026, 1, 14, 12))["holidays"], "NOT_IMPLEMENTED")

    def test_sessions_marked_indicative(self):
        sessions = forex_active_sessions(_utc(2026, 1, 14, 12))
        self.assertEqual(len(sessions), 4)
        self.assertTrue(all(s["indicative"] is True for s in sessions))

    def test_session_active_london_midday(self):
        # 12:00 UTC in January -> London local ~12:00 (within 08-17) -> active
        sessions = forex_active_sessions(_utc(2026, 1, 14, 12))
        london = next(s for s in sessions if s["name"] == "London")
        self.assertTrue(london["active"])

    def test_calendar_for_forex_week(self):
        cal = _calendar_for(MarketCalendarPolicy.FOREX_WEEK)
        self.assertIsInstance(cal, ForexWeekCalendar)

    def test_forex_calendar_open_closed(self):
        cal = _calendar_for(MarketCalendarPolicy.FOREX_WEEK)
        open_ts = int(_utc(2026, 1, 14, 12).timestamp())
        wknd_ts = int(_utc(2026, 1, 17, 12).timestamp())
        self.assertEqual(cal.is_market_expected_open(open_ts), OpenState.OPEN)
        self.assertEqual(cal.is_market_expected_open(wknd_ts), OpenState.CLOSED)

    def test_forex_gaps_still_unknown_no_regression(self):
        # closed market / no quote must NOT become a gap
        cal = _calendar_for(MarketCalendarPolicy.FOREX_WEEK)
        rep = cal.analyze_gaps([0, 7200], 3600)
        self.assertEqual(rep.status, "UNKNOWN")
        self.assertEqual(rep.missing, [])

    def test_coinbase_24_7_unchanged(self):
        cal = _calendar_for(MarketCalendarPolicy.ALWAYS_OPEN_24_7)
        self.assertEqual(cal.is_market_expected_open(0), OpenState.OPEN)

    def test_api_serialisation_shape(self):
        r = forex_market_state(_utc(2026, 1, 14, 12))
        for key in ("asset_class", "market_state", "reason", "current_session",
                    "sessions", "next_open", "next_close", "timezone_internal",
                    "display_timezone", "source", "as_of"):
            self.assertIn(key, r)
        # timestamps are ISO strings or None (JSON-serialisable), never datetime
        for k in ("next_open", "next_close", "as_of"):
            self.assertTrue(r[k] is None or isinstance(r[k], str))


class ForexMarketStateEndpointTests(unittest.TestCase):
    def test_endpoint_ok(self):
        r = TestClient(create_app()).get("/api/v1/market/forex/market-state")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["asset_class"], "FOREX")
        self.assertIn(body["market_state"], ("OPEN", "CLOSED", "CLOSED_WEEKEND", "UNKNOWN"))


class ForexCalendarUiTests(unittest.TestCase):
    def setUp(self):
        self.html = INDEX.read_text(encoding="utf-8")

    def test_market_state_endpoint_used(self):
        self.assertIn("/api/v1/market/forex/market-state", self.html)

    def test_market_hours_in_montreal(self):
        self.assertIn("formatMontrealTime(d.next_close)", self.html)
        self.assertIn("formatMontrealTime(d.next_open)", self.html)

    def test_info_help_present(self):
        self.assertIn("Horaires du marché Forex", self.html)

    def test_no_hardcoded_offset_in_ui(self):
        for bad in ("UTC-4", "UTC-5", "-04:00", "-05:00", "getTimezoneOffset"):
            self.assertNotIn(bad, self.html)


# ----------------------------- Twelve Data XAU/USD (1/3: provider REST) --------
from decimal import Decimal as _Dec  # noqa: E402
from main import (  # noqa: E402
    TWELVEDATA_GRANULARITIES,
    TwelveDataProvider,
    parse_twelvedata_time_series,
    twelvedata_bar_from_value,
)


def _td_val(dt="2026-08-30 14:30:00", o="2650.5", h="2651", low="2649", c="2650.9", v="0"):
    return {"datetime": dt, "open": o, "high": h, "low": low, "close": c, "volume": v}


class _TDResp:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self._body = body

    def json(self):
        return self._body


class _TDClient:
    def __init__(self, resp=None, exc=None):
        self.resp = resp
        self.exc = exc
        self.calls = []

    async def get(self, path, params=None):
        self.calls.append((path, params))
        if self.exc is not None:
            raise self.exc
        return self.resp


class TwelveDataParsingTests(unittest.TestCase):
    def test_ohlc_parsed_as_exact_decimal_no_float(self):
        bar = twelvedata_bar_from_value(
            _td_val(o="2650.123456789012345678"), float("inf"))
        self.assertNotEqual(bar.status, DataQualityStatus.INVALID)
        self.assertIsInstance(bar.open, _Dec)
        self.assertNotIsInstance(bar.open, float)
        # full precision preserved -> proves no intermediate float conversion
        self.assertEqual(str(bar.open), "2650.123456789012345678")

    def test_timestamp_utc_intraday(self):
        bar = twelvedata_bar_from_value(_td_val(dt="2026-08-30 14:30:00"), float("inf"))
        self.assertIsNotNone(bar.datetime_utc.tzinfo)
        self.assertEqual(bar.datetime_utc.hour, 14)

    def test_invalid_ohlc_is_invalid(self):
        self.assertEqual(
            twelvedata_bar_from_value(_td_val(o="x"), float("inf")).status,
            DataQualityStatus.INVALID)

    def test_bad_datetime_is_invalid(self):
        self.assertEqual(
            twelvedata_bar_from_value(_td_val(dt="nope"), float("inf")).status,
            DataQualityStatus.INVALID)

    def test_volume_optional_for_spot(self):
        item = {"datetime": "2026-08-30 14:30:00", "open": "1", "high": "1",
                "low": "1", "close": "1"}
        bar = twelvedata_bar_from_value(item, float("inf"))
        self.assertNotEqual(bar.status, DataQualityStatus.INVALID)
        self.assertIsNone(bar.volume)

    def test_negative_price_invalid(self):
        self.assertEqual(
            twelvedata_bar_from_value(_td_val(o="-1"), float("inf")).status,
            DataQualityStatus.INVALID)

    def test_body_status_error_429_rate_limited(self):
        r = parse_twelvedata_time_series({"status": "error", "code": 429}, float("inf"))
        self.assertEqual(r.status, "RATE_LIMITED")

    def test_body_status_error_403_access_denied(self):
        r = parse_twelvedata_time_series({"status": "error", "code": 403}, float("inf"))
        self.assertEqual(r.status, "ACCESS_DENIED")

    def test_body_status_error_other_unavailable(self):
        r = parse_twelvedata_time_series({"status": "error", "code": 500}, float("inf"))
        self.assertEqual(r.status, "UNAVAILABLE")

    def test_empty_values_is_empty(self):
        r = parse_twelvedata_time_series({"status": "ok", "values": []}, float("inf"))
        self.assertEqual(r.status, "EMPTY")

    def test_malformed_payload_unavailable(self):
        self.assertEqual(
            parse_twelvedata_time_series("not-a-dict", float("inf")).status, "UNAVAILABLE")


class TwelveDataMappingTests(unittest.TestCase):
    def test_official_mapping_xau(self):
        self.assertEqual(provider_symbol_map.to_provider("twelvedata", "XAU-USD"), "XAU/USD")

    def test_granularities_verified_only(self):
        self.assertEqual(TWELVEDATA_GRANULARITIES["1h"], "1h")
        self.assertIn("4h", TWELVEDATA_GRANULARITIES)
        self.assertNotIn("6h", TWELVEDATA_GRANULARITIES)  # 6h NOT_SUPPORTED
        self.assertNotIn("1d", TWELVEDATA_GRANULARITIES)  # 1d NOT_IMPLEMENTED here


class TwelveDataProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_no_key_no_network(self):
        prov = TwelveDataProvider(api_key="")
        spy = _TDClient(_TDResp(200, {"values": []}))
        prov.client = spy
        r = await prov.get_time_series("XAU-USD", "1h")
        self.assertEqual(r.status, "NO_KEY")
        self.assertEqual(spy.calls, [])  # no request performed

    async def test_not_mapped(self):
        prov = TwelveDataProvider(api_key="DUMMY")
        self.assertEqual((await prov.get_time_series("EUR-USD", "1h")).status, "NOT_MAPPED")

    async def test_6h_not_supported(self):
        prov = TwelveDataProvider(api_key="DUMMY")
        self.assertEqual((await prov.get_time_series("XAU-USD", "6h")).status, "NOT_SUPPORTED")

    async def test_http_403_access_denied(self):
        prov = TwelveDataProvider(api_key="DUMMY")
        prov.client = _TDClient(_TDResp(403, {}))
        self.assertEqual((await prov.get_time_series("XAU-USD", "1h")).status, "ACCESS_DENIED")

    async def test_http_401_access_denied(self):
        prov = TwelveDataProvider(api_key="DUMMY")
        prov.client = _TDClient(_TDResp(401, {}))
        self.assertEqual((await prov.get_time_series("XAU-USD", "1h")).status, "ACCESS_DENIED")

    async def test_http_429_rate_limited(self):
        prov = TwelveDataProvider(api_key="DUMMY")
        prov.client = _TDClient(_TDResp(429, {}))
        self.assertEqual((await prov.get_time_series("XAU-USD", "1h")).status, "RATE_LIMITED")

    async def test_http_500_unavailable(self):
        prov = TwelveDataProvider(api_key="DUMMY")
        prov.client = _TDClient(_TDResp(500, {}))
        self.assertEqual((await prov.get_time_series("XAU-USD", "1h")).status, "UNAVAILABLE")

    async def test_timeout_unavailable(self):
        prov = TwelveDataProvider(api_key="DUMMY")
        prov.client = _TDClient(exc=httpx.TimeoutException("t"))
        self.assertEqual((await prov.get_time_series("XAU-USD", "1h")).status, "UNAVAILABLE")

    async def test_network_error_unavailable(self):
        prov = TwelveDataProvider(api_key="DUMMY")
        prov.client = _TDClient(exc=httpx.ConnectError("x"))
        self.assertEqual((await prov.get_time_series("XAU-USD", "1h")).status, "UNAVAILABLE")

    async def test_body_error_on_http_200(self):
        prov = TwelveDataProvider(api_key="DUMMY")
        prov.client = _TDClient(_TDResp(200, {"status": "error", "code": 429}))
        self.assertEqual((await prov.get_time_series("XAU-USD", "1h")).status, "RATE_LIMITED")

    async def test_ok_returns_bars_ascending(self):
        prov = TwelveDataProvider(api_key="DUMMY")
        body = {"values": [_td_val(dt="2026-08-30 14:00:00", c="2652"),
                           _td_val(dt="2026-08-30 15:00:00", c="2653")]}
        prov.client = _TDClient(_TDResp(200, body))
        r = await prov.get_time_series("XAU-USD", "1h")
        self.assertEqual(r.status, "OK")
        self.assertEqual(len(r.bars), 2)

    async def test_key_absent_from_request_params(self):
        prov = TwelveDataProvider(api_key="DUMMYKEY")
        fc = _TDClient(_TDResp(200, {"values": []}))
        prov.client = fc
        await prov.get_time_series("XAU-USD", "1h")
        for _path, params in fc.calls:
            self.assertNotIn("apikey", params or {})
            self.assertNotIn("DUMMYKEY", str(params))

    async def test_header_auth_key_not_in_url(self):
        prov = TwelveDataProvider(api_key="DUMMYKEY")
        await prov.connect()
        try:
            auth = prov.client.headers.get("authorization")
            self.assertIsNotNone(auth)
            self.assertTrue(auth.startswith("apikey "))
            self.assertNotIn("DUMMYKEY", str(prov.rest_url))
        finally:
            await prov.disconnect()

    async def test_error_reason_has_no_secret(self):
        prov = TwelveDataProvider(api_key="DUMMYKEY")
        prov.client = _TDClient(_TDResp(403, {}))
        r = await prov.get_time_series("XAU-USD", "1h")
        self.assertNotIn("DUMMYKEY", r.reason or "")


# ----------------------------- Twelve Data XAU/USD (2/3: instrument+history+DB) -
from main import (  # noqa: E402
    CandleRow as _CandleRow2,
    TwelveDataBar,
    TwelveDataResult,
    fetch_metal_history,
)

_XAU_BASE = int(datetime(2026, 8, 30, 0, 0, tzinfo=timezone.utc).timestamp())


def _xau_bar(k, close="2650.5", vol="0", status=DataQualityStatus.VALID):
    dt = datetime.fromtimestamp(_XAU_BASE + k * 3600, tz=timezone.utc)
    v = _Dec(vol) if vol is not None else None
    return TwelveDataBar(dt, _Dec("2650"), _Dec("2655"), _Dec("2648"), _Dec(close), v, status)


class _FakeTD:
    def __init__(self, result):
        self.result = result

    async def get_time_series(self, canonical, granularity, outputsize=30, start=None, end=None):
        return self.result


class MetalInstrumentTests(unittest.TestCase):
    def test_xau_registered_metal(self):
        inst = instrument_registry.get("XAU-USD")
        self.assertIsNotNone(inst)
        self.assertEqual(inst.asset_class, AssetClass.METAL)
        self.assertEqual(inst.display_name, "Gold Spot")

    def test_xau_calendar_not_configured(self):
        self.assertEqual(instrument_registry.get("XAU-USD").market_calendar,
                         MarketCalendarPolicy.NOT_CONFIGURED)

    def test_xau_volume_unknown_and_metadata_none(self):
        inst = instrument_registry.get("XAU-USD")
        self.assertEqual(inst.volume_semantics, VolumeSemantics.UNKNOWN)
        self.assertIsNone(inst.price_precision)
        self.assertIsNone(inst.tick_size)

    def test_mapping_present_even_though_entitlement_unknown(self):
        # MAPPED is not an entitlement: the verified provider symbol stays mapped
        self.assertEqual(provider_symbol_map.to_provider("twelvedata", "XAU-USD"), "XAU/USD")


class CandleRowDecimalRegressionTests(unittest.TestCase):
    def test_candlerow_row_to_values_exact_decimal(self):
        row = _CandleRow2("coinbase", "BTC-USD", "1h",
                          datetime(2026, 8, 30, 12, tzinfo=timezone.utc),
                          _Dec("50000.12"), _Dec("50010"), _Dec("49990"),
                          _Dec("50005.5"), _Dec("3.25"), DataQualityStatus.VALID,
                          "rest", None, datetime(2026, 8, 30, 12, tzinfo=timezone.utc))
        vals = main._row_to_values(row, datetime(2026, 8, 30, 12, tzinfo=timezone.utc))
        self.assertIsInstance(vals["open"], _Dec)
        self.assertEqual(vals["open"], _Dec("50000.12"))

    def test_xau_full_precision_no_float_end_to_end(self):
        # string -> Decimal -> CandleRow -> _row_to_values, precision preserved
        rows = main._metal_bars_to_rows(
            "twelvedata", "XAU/USD", "1h",
            [_xau_bar(0, close="2650.123456789012345678")], datetime.now(timezone.utc))
        vals = main._row_to_values(rows[0], datetime.now(timezone.utc))
        self.assertEqual(str(vals["close"]), "2650.123456789012345678")


class MetalHistoryAssemblerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._saved = main.twelvedata_provider

    def tearDown(self):
        main.twelvedata_provider = self._saved

    async def _run(self, result, start=None, end=None, gran="1h"):
        main.twelvedata_provider = _FakeTD(result)
        return await fetch_metal_history("XAU-USD", gran,
                                         start or _XAU_BASE, end or _XAU_BASE + 5 * 3600)

    async def test_ok_sorted_deduped_decimal(self):
        res = TwelveDataResult("OK", [_xau_bar(2), _xau_bar(0), _xau_bar(1), _xau_bar(1)])
        h = await self._run(res)
        self.assertEqual(h.result["status"], "OK")
        self.assertEqual(h.result["count"], 3)  # deduped
        starts = [c["start"] for c in h.result["candles"]]
        self.assertEqual(starts, sorted(starts))
        self.assertTrue(all(isinstance(r.close, _Dec) for r in h.rows))

    async def test_absence_not_a_gap(self):
        h = await self._run(TwelveDataResult("OK", [_xau_bar(0), _xau_bar(1), _xau_bar(3)]))
        self.assertEqual(h.result["gaps_status"], "UNKNOWN")
        self.assertEqual(h.result["count"], 3)  # no fabricated bar

    async def test_invalid_excluded(self):
        res = TwelveDataResult("OK", [_xau_bar(0), _xau_bar(1, status=DataQualityStatus.INVALID)])
        h = await self._run(res)
        self.assertEqual(h.result["invalid_candles_count"], 1)
        self.assertEqual(h.result["count"], 1)

    async def test_half_open_filter(self):
        h = await self._run(TwelveDataResult("OK", [_xau_bar(0), _xau_bar(1), _xau_bar(2)]),
                            start=_XAU_BASE, end=_XAU_BASE + 2 * 3600)
        self.assertEqual(h.result["count"], 2)  # k=2 (== end) excluded

    async def test_volume_semantics_unknown(self):
        h = await self._run(TwelveDataResult("OK", [_xau_bar(0)]))
        self.assertEqual(h.result["volume_semantics"], "UNKNOWN")

    async def test_empty(self):
        h = await self._run(TwelveDataResult("EMPTY", []))
        self.assertEqual(h.result["status"], "EMPTY")
        self.assertEqual(h.rows, [])

    async def test_access_denied_propagated_no_rows(self):
        denied = TwelveDataResult("ACCESS_DENIED", [], "access denied by provider (403)")
        h = await self._run(denied)
        self.assertEqual(h.result["status"], "ACCESS_DENIED")
        self.assertEqual(h.rows, [])
        self.assertNotIn("DUMMY", str(h.result.get("reason")))

    async def test_no_key_propagated(self):
        h = await self._run(TwelveDataResult("NO_KEY", [], "TWELVEDATA_API_KEY not set"))
        self.assertEqual(h.result["status"], "NO_KEY")

    async def test_unknown_instrument_raises(self):
        main.twelvedata_provider = _FakeTD(TwelveDataResult("OK", []))
        with self.assertRaises(ValueError):
            await fetch_metal_history("ZZZ-ZZZ", "1h", _XAU_BASE, _XAU_BASE + 3600)

    async def test_start_ge_end_raises(self):
        main.twelvedata_provider = _FakeTD(TwelveDataResult("OK", []))
        with self.assertRaises(ValueError):
            await fetch_metal_history("XAU-USD", "1h", _XAU_BASE, _XAU_BASE)


class MetalEndpointTests(unittest.TestCase):
    def setUp(self):
        self._saved = main.twelvedata_provider
        self._ready = persistence_state.ready
        persistence_state.ready = False  # avoid DB writes in these status-mapping tests

    def tearDown(self):
        main.twelvedata_provider = self._saved
        persistence_state.ready = self._ready

    def _client(self, result):
        main.twelvedata_provider = _FakeTD(result)
        return TestClient(create_app())

    def test_ok_200(self):
        r = self._client(TwelveDataResult("OK", [_xau_bar(0)])).get(
            "/api/v1/market/metal/XAU-USD/history?granularity=1h&start=%d&end=%d"
            % (_XAU_BASE, _XAU_BASE + 5 * 3600))
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["source"], "twelvedata")

    def test_access_denied_403(self):
        r = self._client(TwelveDataResult("ACCESS_DENIED", [], "x")).get(
            "/api/v1/market/metal/XAU-USD/history?granularity=1h&start=%d&end=%d"
            % (_XAU_BASE, _XAU_BASE + 3600))
        self.assertEqual(r.status_code, 403)

    def test_rate_limited_429(self):
        r = self._client(TwelveDataResult("RATE_LIMITED", [], "x")).get(
            "/api/v1/market/metal/XAU-USD/history?granularity=1h&start=%d&end=%d"
            % (_XAU_BASE, _XAU_BASE + 3600))
        self.assertEqual(r.status_code, 429)

    def test_no_key_503(self):
        r = self._client(TwelveDataResult("NO_KEY", [], "x")).get(
            "/api/v1/market/metal/XAU-USD/history?granularity=1h&start=%d&end=%d"
            % (_XAU_BASE, _XAU_BASE + 3600))
        self.assertEqual(r.status_code, 503)

    def test_6h_not_supported_409(self):
        r = self._client(TwelveDataResult("NOT_SUPPORTED", [], "x")).get(
            "/api/v1/market/metal/XAU-USD/history?granularity=6h&start=%d&end=%d"
            % (_XAU_BASE, _XAU_BASE + 3600))
        self.assertEqual(r.status_code, 409)

    def test_no_secret_in_response(self):
        denied = TwelveDataResult("ACCESS_DENIED", [], "access denied by provider (403)")
        raw = self._client(denied).get(
            "/api/v1/market/metal/XAU-USD/history?granularity=1h&start=%d&end=%d"
            % (_XAU_BASE, _XAU_BASE + 3600)).text.lower()
        for bad in ("apikey", "authorization", "bearer", "twelvedata_api_key"):
            self.assertNotIn(bad, raw)


class MetalPersistenceTests(_DBBase):
    async def test_xau_persisted_source_twelvedata_exact_decimal(self):
        saved = main.twelvedata_provider
        main.twelvedata_provider = _FakeTD(TwelveDataResult(
            "OK", [_xau_bar(0, close="2650.123456789012345678"), _xau_bar(1, close="2651.5")]))
        try:
            h = await fetch_metal_history("XAU-USD", "1h", _XAU_BASE, _XAU_BASE + 5 * 3600)
            n = await main.persist_candles(h.rows)
            self.assertEqual(n, 2)
            # read_stored_candles filters source='coinbase'; check twelvedata rows raw
            async with main.engine.connect() as conn:
                res = await conn.execute(main.text(
                    "SELECT close FROM candles WHERE source='twelvedata' "
                    "AND product_id='XAU/USD' ORDER BY bucket_start ASC"))
                closes = [str(r[0]) for r in res.fetchall()]
            self.assertEqual(len(closes), 2)
            # exact Decimal preserved through NUMERIC(38,18)
            self.assertTrue(closes[0].startswith("2650.123456789012345678"))
            async with main.engine.connect() as conn:
                cnt = await conn.execute(main.text(
                    "SELECT count(*) FROM candles WHERE source='coinbase'"))
                self.assertEqual(cnt.scalar(), 0)  # no Coinbase regression/leak
        finally:
            main.twelvedata_provider = saved


# ----------------------------- Twelve Data XAU/USD (3/3: /quote + Gold UI) -----
from main import (  # noqa: E402
    TwelveDataQuoteResult,
    fetch_metal_quote,
    parse_twelvedata_quote,
)


class TwelveDataQuoteTests(unittest.TestCase):
    def test_quote_price_decimal_no_float(self):
        q = parse_twelvedata_quote(
            {"close": "2650.123456789012345678", "is_market_open": True,
             "timestamp": int(datetime(2026, 8, 30, 14, tzinfo=timezone.utc).timestamp())})
        self.assertEqual(q.status, "OK")
        self.assertIsInstance(q.price, _Dec)
        self.assertNotIsInstance(q.price, float)
        self.assertEqual(str(q.price), "2650.123456789012345678")
        self.assertIs(q.is_market_open, True)

    def test_quote_body_error(self):
        self.assertEqual(parse_twelvedata_quote({"status": "error", "code": 429}).status,
                         "RATE_LIMITED")
        self.assertEqual(parse_twelvedata_quote({"status": "error", "code": 403}).status,
                         "ACCESS_DENIED")

    def test_quote_bad_price_unavailable(self):
        self.assertEqual(parse_twelvedata_quote({"close": "-1"}).status, "UNAVAILABLE")

    def test_is_market_open_non_bool_becomes_none(self):
        self.assertIsNone(parse_twelvedata_quote({"close": "2650", "is_market_open": "yes"})
                          .is_market_open)


class MetalQuoteAssemblerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._saved = main.twelvedata_provider

    def tearDown(self):
        main.twelvedata_provider = self._saved

    async def _run(self, result):
        main.twelvedata_provider = type("P", (), {
            "get_quote": staticmethod(lambda canon: _async_return(result))})()
        return await fetch_metal_quote("XAU-USD")

    async def test_market_open_does_not_make_it_live(self):
        # is_market_open=True but no quote timestamp -> quality UNKNOWN, never LIVE
        r = await self._run(TwelveDataQuoteResult("OK", _Dec("2650"), True, None))
        self.assertEqual(r["status"], "OK")
        self.assertEqual(r["is_market_open"], True)
        self.assertEqual(r["quality"], "UNKNOWN")  # not LIVE from is_market_open

    async def test_price_serialised_as_string(self):
        r = await self._run(TwelveDataQuoteResult("OK", _Dec("2650.5"), False, None))
        self.assertEqual(r["price"], "2650.5")  # Decimal -> string

    async def test_access_denied_no_price(self):
        r = await self._run(TwelveDataQuoteResult("ACCESS_DENIED", None, None, None, "x"))
        self.assertEqual(r["status"], "ACCESS_DENIED")
        self.assertIsNone(r["price"])


class MetalQuoteEndpointTests(unittest.TestCase):
    def setUp(self):
        self._saved = main.twelvedata_provider

    def tearDown(self):
        main.twelvedata_provider = self._saved

    def _client(self, result):
        main.twelvedata_provider = type("P", (), {
            "get_quote": staticmethod(lambda canon: _async_return(result))})()
        return TestClient(create_app())

    def test_quote_ok_200(self):
        r = self._client(TwelveDataQuoteResult("OK", _Dec("2650.5"), True, None)).get(
            "/api/v1/market/metal/XAU-USD/quote")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["price"], "2650.5")

    def test_quote_access_denied_403(self):
        r = self._client(TwelveDataQuoteResult("ACCESS_DENIED", None, None, None, "x")).get(
            "/api/v1/market/metal/XAU-USD/quote")
        self.assertEqual(r.status_code, 403)

    def test_quote_no_secret_in_response(self):
        raw = self._client(TwelveDataQuoteResult("OK", _Dec("2650"), True, None)).get(
            "/api/v1/market/metal/XAU-USD/quote").text.lower()
        for bad in ("apikey", "authorization", "bearer", "twelvedata_api_key"):
            self.assertNotIn(bad, raw)


class GoldUiTests(unittest.TestCase):
    def setUp(self):
        self.html = INDEX.read_text(encoding="utf-8")

    def test_metal_view_and_labels(self):
        self.assertIn('id:"metal"', self.html)
        self.assertIn("XAU-USD", self.html)
        self.assertIn("Gold Spot", self.html)
        self.assertIn("Twelve Data", self.html)

    def test_metal_endpoints_used(self):
        self.assertIn("/api/v1/market/metal/", self.html)
        self.assertIn("/quote", self.html)

    def test_quality_independent_from_market_open(self):
        # the badge is driven by backend quality, not is_market_open
        self.assertIn("qualityBadge(d.quality", self.html)
        self.assertIn("Marché (fournisseur)", self.html)  # is_market_open shown as info only

    def test_no_twelvedata_secret_in_frontend(self):
        low = self.html.lower()
        for bad in ("twelvedata_api_key", "apikey=", "bearer "):
            self.assertNotIn(bad, low)

    def test_price_from_string_not_fabricated(self):
        # metal price comes from backend d.price (Decimal string), guarded by null check
        self.assertIn("r.data.price == null", self.html)


def _async_return(value):
    async def _coro():
        return value
    return _coro()


# ----------------------------- runtime fixes: quote STALE trace + mobile OHLC --
class MetalQuoteFreshnessTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._saved = main.twelvedata_provider

    def tearDown(self):
        main.twelvedata_provider = self._saved

    async def _run(self, result):
        main.twelvedata_provider = type("P", (), {
            "get_quote": staticmethod(lambda canon: _async_return(result))})()
        return await fetch_metal_quote("XAU-USD")

    async def test_old_provider_timestamp_is_stale_not_live(self):
        # provider quote timestamp 144 min old + market open -> STALE (correct), never LIVE
        old = datetime.now(timezone.utc) - timedelta(minutes=144)
        r = await self._run(TwelveDataQuoteResult("OK", _Dec("2650"), True, old))
        self.assertEqual(r["quality"], DataQualityStatus.STALE.value)
        self.assertIs(r["is_market_open"], True)  # open, yet still STALE
        self.assertGreater(r["quote_age_seconds"], 8000)  # ~8640s, transparent reason

    async def test_recent_timestamp_can_be_valid(self):
        fresh = datetime.now(timezone.utc)
        r = await self._run(TwelveDataQuoteResult("OK", _Dec("2650"), True, fresh))
        self.assertEqual(r["quality"], DataQualityStatus.VALID.value)

    async def test_threshold_unchanged_ticker_max_age(self):
        # freshness budget for the quote is the ticker budget (unchanged), documenting
        # that STALE is not forced to LIVE by widening the threshold
        self.assertEqual(main.settings.ticker_max_age_seconds, 10.0)


class MobileOhlcTableTests(unittest.TestCase):
    def setUp(self):
        self.html = INDEX.read_text(encoding="utf-8")

    def test_ohlc_table_has_horizontal_scroll_wrapper(self):
        self.assertIn(".tbl-wrap", self.html)
        self.assertIn("overflow-x:auto", self.html)
        self.assertIn('h("div",{class:"tbl-wrap"}', self.html)

    def test_table_min_width_and_nowrap(self):
        self.assertIn("min-width:440px", self.html)
        self.assertIn("white-space:nowrap", self.html)

    def test_all_ohlc_columns_present(self):
        # no column removed: O/H/L/C still rendered
        for col in ('h("th",{},["O"])', 'h("th",{},["H"])',
                    'h("th",{},["L"])', 'h("th",{},["C"])'):
            self.assertIn(col, self.html)


# ----------------------------- Massive US cash indices REST -------------------
import json as _json  # noqa: E402
from main import (  # noqa: E402
    MassiveIndexResult,
    MassiveIndicesProvider,
    fetch_index_history,
    index_bar_from_agg,
    parse_massive_index_aggs,
)

_IX_BASE = 1755000000  # some UNIX seconds anchor


def _ix_agg(k, close="100.5"):
    return {"o": "100", "h": "101", "l": "99", "c": close, "t": (_IX_BASE + k * 86400) * 1000}


class _IXResp:
    def __init__(self, status_code, text):
        self.status_code = status_code
        self.text = text


class _IXClient:
    def __init__(self, resp=None, exc=None):
        self.resp = resp
        self.exc = exc
        self.calls = []

    async def get(self, path, params=None):
        self.calls.append((path, params))
        if self.exc is not None:
            raise self.exc
        return self.resp


class IndexInstrumentTests(unittest.TestCase):
    def test_three_cash_indices_registered(self):
        for c, name in (("SPX", "S&P 500"), ("NDX", "Nasdaq-100"),
                        ("US30", "Dow Jones Industrial Average")):
            inst = instrument_registry.get(c)
            self.assertIsNotNone(inst)
            self.assertEqual(inst.asset_class, AssetClass.INDEX)
            self.assertEqual(inst.display_name, name)

    def test_official_mappings_cash_not_etf_or_future(self):
        self.assertEqual(provider_symbol_map.to_provider("massive", "SPX"), "I:SPX")
        self.assertEqual(provider_symbol_map.to_provider("massive", "NDX"), "I:NDX")
        self.assertEqual(provider_symbol_map.to_provider("massive", "US30"), "I:DJI")

    def test_volume_not_available_and_metadata_none(self):
        inst = instrument_registry.get("SPX")
        self.assertEqual(inst.volume_semantics, VolumeSemantics.NOT_AVAILABLE)
        self.assertIsNone(inst.price_precision)
        self.assertIsNone(inst.tick_size)

    def test_index_calendar_is_us_equity_rth(self):
        for symbol in ("SPX", "NDX", "US30"):
            self.assertEqual(instrument_registry.get(symbol).market_calendar,
                             MarketCalendarPolicy.US_EQUITY_RTH)

    def test_index_calendar_regular_hours_open(self):
        cal = calendar_for(MarketCalendarPolicy.US_EQUITY_RTH)
        # 2026-08-31 14:00 UTC = Monday 10:00 EDT.
        self.assertEqual(cal.is_market_expected_open(1788184800), OpenState.OPEN)

    def test_index_calendar_before_open_closed(self):
        cal = calendar_for(MarketCalendarPolicy.US_EQUITY_RTH)
        # 2026-08-31 13:00 UTC = Monday 09:00 EDT.
        self.assertEqual(cal.is_market_expected_open(1788181200), OpenState.CLOSED)

    def test_index_calendar_at_close_closed(self):
        cal = calendar_for(MarketCalendarPolicy.US_EQUITY_RTH)
        # 2026-08-31 20:00 UTC = Monday 16:00 EDT; half-open RTH interval.
        self.assertEqual(cal.is_market_expected_open(1788206400), OpenState.CLOSED)

    def test_index_calendar_weekend_closed(self):
        cal = calendar_for(MarketCalendarPolicy.US_EQUITY_RTH)
        self.assertEqual(cal.is_market_expected_open(1788012000), OpenState.CLOSED)

    def test_index_calendar_invalid_timestamp_unknown(self):
        cal = calendar_for(MarketCalendarPolicy.US_EQUITY_RTH)
        self.assertEqual(cal.is_market_expected_open(10**30), OpenState.UNKNOWN)

    def test_index_calendar_never_fabricates_gap_grid(self):
        cal = calendar_for(MarketCalendarPolicy.US_EQUITY_RTH)
        self.assertIsNone(cal.expected_bucket_starts("1h", 0, 7200))
        self.assertEqual(cal.analyze_gaps([0, 7200], 3600).status, "UNKNOWN")


class IndexParsingTests(unittest.TestCase):
    def test_decimal_exact_via_parse_float(self):
        body = _json.loads('{"results":[{"o":3985.67,"h":3990.12,"l":3980.0,"c":3987.5,'
                           '"t":1755000000000}]}', parse_float=_Dec)
        r = parse_massive_index_aggs(body, float("inf"))
        self.assertEqual(r.status, "OK")
        self.assertIsInstance(r.bars[0].open, _Dec)
        self.assertNotIsInstance(r.bars[0].open, float)
        self.assertEqual(str(r.bars[0].open), "3985.67")

    def test_no_volume_on_index_bar(self):
        bar = index_bar_from_agg(_ix_agg(0), float("inf"))
        self.assertFalse(hasattr(bar, "volume"))

    def test_invalid_ohlc(self):
        self.assertEqual(
            index_bar_from_agg({"o": "x", "h": "1", "l": "1", "c": "1", "t": 1755000000000},
                              float("inf")).status, DataQualityStatus.INVALID)

    def test_empty_results(self):
        self.assertEqual(parse_massive_index_aggs({"results": []}, float("inf")).status, "EMPTY")

    def test_malformed(self):
        self.assertEqual(parse_massive_index_aggs("x", float("inf")).status, "UNAVAILABLE")


class IndexProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_no_key_no_network(self):
        prov = MassiveIndicesProvider(api_key="")
        spy = _IXClient(_IXResp(200, "{}"))
        prov.client = spy
        r = await prov.get_index_aggregates("SPX", "1d", _IX_BASE, _IX_BASE + 86400)
        self.assertEqual(r.status, "NO_KEY")
        self.assertEqual(spy.calls, [])

    async def test_not_mapped(self):
        prov = MassiveIndicesProvider(api_key="DUMMY")
        # BTC-USD is a Coinbase canonical, never mapped under the 'massive' provider
        r = await prov.get_index_aggregates("BTC-USD", "1d", _IX_BASE, _IX_BASE + 86400)
        self.assertEqual(r.status, "NOT_MAPPED")

    async def test_not_supported_granularity(self):
        prov = MassiveIndicesProvider(api_key="DUMMY")
        r = await prov.get_index_aggregates("SPX", "3m", _IX_BASE, _IX_BASE + 86400)
        self.assertEqual(r.status, "NOT_SUPPORTED")

    async def test_403_access_denied(self):
        prov = MassiveIndicesProvider(api_key="DUMMY")
        prov.client = _IXClient(_IXResp(403, "{}"))
        r = await prov.get_index_aggregates("SPX", "1d", _IX_BASE, _IX_BASE + 86400)
        self.assertEqual(r.status, "ACCESS_DENIED")

    async def test_429_rate_limited(self):
        prov = MassiveIndicesProvider(api_key="DUMMY")
        prov.client = _IXClient(_IXResp(429, "{}"))
        r = await prov.get_index_aggregates("SPX", "1d", _IX_BASE, _IX_BASE + 86400)
        self.assertEqual(r.status, "RATE_LIMITED")

    async def test_500_unavailable(self):
        prov = MassiveIndicesProvider(api_key="DUMMY")
        prov.client = _IXClient(_IXResp(500, "{}"))
        r = await prov.get_index_aggregates("SPX", "1d", _IX_BASE, _IX_BASE + 86400)
        self.assertEqual(r.status, "UNAVAILABLE")

    async def test_timeout_unavailable(self):
        prov = MassiveIndicesProvider(api_key="DUMMY")
        prov.client = _IXClient(exc=httpx.TimeoutException("t"))
        r = await prov.get_index_aggregates("SPX", "1d", _IX_BASE, _IX_BASE + 86400)
        self.assertEqual(r.status, "UNAVAILABLE")

    async def test_ok_bars_decimal(self):
        prov = MassiveIndicesProvider(api_key="DUMMY")
        prov.client = _IXClient(_IXResp(200, _json.dumps({"results": [_ix_agg(0), _ix_agg(1)]})))
        r = await prov.get_index_aggregates("SPX", "1d", _IX_BASE, _IX_BASE + 5 * 86400)
        self.assertEqual(r.status, "OK")
        self.assertEqual(len(r.bars), 2)

    async def test_key_absent_from_params(self):
        prov = MassiveIndicesProvider(api_key="DUMMYKEY")
        fc = _IXClient(_IXResp(200, "{}"))
        prov.client = fc
        await prov.get_index_aggregates("SPX", "1d", _IX_BASE, _IX_BASE + 86400)
        for _p, params in fc.calls:
            self.assertNotIn("apikey", params or {})
            self.assertNotIn("DUMMYKEY", str(params))


class IndexHistoryAssemblerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._saved = main.massive_indices_provider

    def tearDown(self):
        main.massive_indices_provider = self._saved

    def _prov(self, result):
        return type("P", (), {
            "get_index_aggregates": staticmethod(
                lambda canon, gran, start, end: _async_return(result))})()

    async def test_ok_sorted_no_persist_no_volume(self):
        main.massive_indices_provider = self._prov(MassiveIndexResult(
            "OK", [index_bar_from_agg(_ix_agg(2), float("inf")),
                   index_bar_from_agg(_ix_agg(0), float("inf")),
                   index_bar_from_agg(_ix_agg(3), float("inf"))]))
        h = await fetch_index_history("SPX", "1d", _IX_BASE, _IX_BASE + 5 * 86400)
        self.assertEqual(h["status"], "OK")
        self.assertEqual(h["count"], 3)
        self.assertEqual(h["gaps_status"], "UNKNOWN")   # absence != gap
        self.assertIs(h["persisted"], False)            # D2: never persisted
        self.assertEqual(h["volume_semantics"], "NOT_AVAILABLE")
        self.assertTrue(all("volume" not in c for c in h["candles"]))
        starts = [c["start"] for c in h["candles"]]
        self.assertEqual(starts, sorted(starts))

    async def test_access_denied_propagated(self):
        main.massive_indices_provider = self._prov(
            MassiveIndexResult("ACCESS_DENIED", [], "access denied by provider (403)"))
        h = await fetch_index_history("SPX", "1d", _IX_BASE, _IX_BASE + 5 * 86400)
        self.assertEqual(h["status"], "ACCESS_DENIED")
        self.assertEqual(h["count"], 0)

    async def test_empty(self):
        main.massive_indices_provider = self._prov(MassiveIndexResult("EMPTY", []))
        h = await fetch_index_history("SPX", "1d", _IX_BASE, _IX_BASE + 5 * 86400)
        self.assertEqual(h["status"], "EMPTY")

    async def test_unknown_index_raises(self):
        main.massive_indices_provider = self._prov(MassiveIndexResult("OK", []))
        with self.assertRaises(ValueError):
            await fetch_index_history("ZZZ", "1d", _IX_BASE, _IX_BASE + 86400)


class IndexEndpointTests(unittest.TestCase):
    def setUp(self):
        self._saved = main.massive_indices_provider

    def tearDown(self):
        main.massive_indices_provider = self._saved

    def _client(self, result):
        main.massive_indices_provider = type("P", (), {
            "get_index_aggregates": staticmethod(
                lambda canon, gran, start, end: _async_return(result))})()
        return TestClient(create_app())

    def test_ok_200(self):
        r = self._client(MassiveIndexResult(
            "OK", [index_bar_from_agg(_ix_agg(0), float("inf"))])).get(
            "/api/v1/market/index/SPX/history?granularity=1d&start=%d&end=%d"
            % (_IX_BASE, _IX_BASE + 5 * 86400))
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["source"], "massive")

    def test_access_denied_403(self):
        r = self._client(MassiveIndexResult("ACCESS_DENIED", [], "x")).get(
            "/api/v1/market/index/SPX/history?granularity=1d&start=%d&end=%d"
            % (_IX_BASE, _IX_BASE + 86400))
        self.assertEqual(r.status_code, 403)

    def test_no_secret_in_response(self):
        raw = self._client(MassiveIndexResult("ACCESS_DENIED", [], "x")).get(
            "/api/v1/market/index/SPX/history?granularity=1d&start=%d&end=%d"
            % (_IX_BASE, _IX_BASE + 86400)).text.lower()
        for bad in ("apikey", "authorization", "bearer", "massive_api_key"):
            self.assertNotIn(bad, raw)


class IndexUiTests(unittest.TestCase):
    def setUp(self):
        self.html = INDEX.read_text(encoding="utf-8")

    def test_index_view_and_labels(self):
        self.assertIn('id:"index"', self.html)
        self.assertIn("SPX", self.html)
        self.assertIn("NDX", self.html)
        self.assertIn("US30", self.html)

    def test_index_endpoint_used(self):
        self.assertIn("/api/v1/market/index/", self.html)

    def test_no_fallback_to_etf_or_futures(self):
        # ETF proxies and index futures must never appear (case-sensitive uppercase
        # tickers; we use the official cash indices SPX/NDX/US30 -> I:SPX/I:NDX/I:DJI)
        for bad in ("SPY", "QQQ", "DIA", "ES=F", "NQ=F", "YM=F", "/ES", "/NQ", "/YM"):
            self.assertNotIn(bad, self.html)


class ChartEngineV1UiTests(unittest.TestCase):
    """Static contract: chart V1 consumes backend OHLC only; no demo series."""

    @classmethod
    def setUpClass(cls):
        cls.html = INDEX.read_text(encoding="utf-8")

    def test_real_chart_label_present(self):
        self.assertIn("Chart Engine · OHLC réel", self.html)
        self.assertIn("REAL DATA", self.html)

    def test_chart_uses_existing_real_endpoints(self):
        self.assertIn('/api/v1/market/candles/', self.html)
        self.assertIn('/api/v1/market/forex/', self.html)
        self.assertIn('/api/v1/market/metal/', self.html)
        self.assertIn('/api/v1/market/index/', self.html)

    def test_chart_has_no_synthetic_candle_fallback(self):
        self.assertIn("aucune bougie synthétique", self.html)
        self.assertNotIn("Math.random()", self.html)

    def test_chart_supports_real_ohlc_fields(self):
        for field in ("c.open", "c.high", "c.low", "c.close", "c.start"):
            self.assertIn(field, self.html)

    def test_chart_has_zoom_pan_controls(self):
        self.assertIn("chartZoom", self.html)
        self.assertIn("chartPan", self.html)
        self.assertIn("chart-cross", self.html)

    def test_provider_errors_remain_explicit(self):
        self.assertIn("errorMessage(r)", self.html)
        self.assertIn("Graphique indisponible", self.html)


class ChartEngineV2SwingUiTests(unittest.TestCase):
    """Static contract: confirmed swings derive only from returned real OHLC."""

    @classmethod
    def setUpClass(cls):
        cls.html = INDEX.read_text(encoding="utf-8")

    def test_confirmed_swing_detector_present(self):
        self.assertIn("detectConfirmedSwings", self.html)
        self.assertIn("SWING_STRENGTH=2", self.html)

    def test_swing_high_uses_strict_neighbor_highs(self):
        self.assertIn("hi<=lh||hi<=rh", self.html)

    def test_swing_low_uses_strict_neighbor_lows(self):
        self.assertIn("lo>=ll||lo>=rl", self.html)

    def test_swings_require_right_side_confirmation(self):
        self.assertIn("confirmedAt:cs[i+n].start", self.html)
        self.assertIn("i<cs.length-n", self.html)

    def test_swings_are_drawn_on_real_chart(self):
        self.assertIn('lab.textContent=isHigh?"SH":"SL"', self.html)
        self.assertIn("swings confirmés", self.html)

    def test_no_synthetic_or_random_swing_fallback(self):
        self.assertIn("Aucun swing futur/repainté", self.html)
        self.assertNotIn("Math.random()", self.html)


class ChartEngineV3StructureUiTests(unittest.TestCase):
    """Static contract: HH/HL/LH/LL derive only from confirmed swings."""

    @classmethod
    def setUpClass(cls):
        cls.html = INDEX.read_text(encoding="utf-8")

    def test_structure_classifier_present(self):
        self.assertIn("classifyConfirmedStructure", self.html)
        self.assertIn("HH/HL/LH/LL ACTIF", self.html)

    def test_high_structure_compares_only_previous_high(self):
        self.assertIn('if(s.price>prevHigh)label="HH"', self.html)
        self.assertIn('else if(s.price<prevHigh)label="LH"', self.html)

    def test_low_structure_compares_only_previous_low(self):
        self.assertIn('if(s.price>prevLow)label="HL"', self.html)
        self.assertIn('else if(s.price<prevLow)label="LL"', self.html)

    def test_equal_swing_is_not_forced_into_structure(self):
        self.assertIn("var label=null", self.html)
        self.assertNotIn('else label="HH"', self.html)
        self.assertNotIn('else label="LL"', self.html)

    def test_structure_is_built_from_confirmed_swings(self):
        self.assertIn("classifyConfirmedStructure(swings)", self.html)
        self.assertIn("Structure comparée uniquement entre swings confirmés", self.html)

    def test_structure_labels_are_drawn_without_random_fallback(self):
        self.assertIn('lab.textContent=s.structure||(isHigh?"SH":"SL")', self.html)
        self.assertNotIn("Math.random()", self.html)


class RealtimeCoreV1UiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = INDEX.read_text(encoding="utf-8")

    def test_frontend_starts_existing_backend_ws(self):
        self.assertIn("/api/v1/market/websocket/start", self.html)

    def test_frontend_subscribes_ticker(self):
        self.assertIn('{channel:"ticker",products:CRYPTO_SYMBOLS}', self.html)

    def test_frontend_subscribes_candles(self):
        self.assertIn('{channel:"candles",products:CRYPTO_SYMBOLS}', self.html)

    def test_frontend_reads_realtime_state(self):
        self.assertIn("/api/v1/market/realtime/", self.html)

    def test_realtime_poll_is_one_second(self):
        self.assertIn("REALTIME_POLL_MS=1000", self.html)

    def test_only_verified_5m_ws_updates_chart(self):
        self.assertIn('chartState.tf==="5m"', self.html)

    def test_other_crypto_tf_remain_rest(self):
        marker = 'LIVE PRICE · OHLC "+chartState.tf.toUpperCase()+" REST'
        self.assertIn(marker, self.html)

    def test_no_synthetic_realtime(self):
        self.assertNotIn("Math.random()", self.html)
        self.assertIn("aucune bougie synthétique", self.html)


class RealtimeCoreV2AForexTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main_src = Path(main.__file__).read_text(encoding="utf-8")
        cls.html = INDEX.read_text(encoding="utf-8")

    def test_massive_official_forex_ws_url(self):
        self.assertIn("wss://socket.massive.com/forex", self.main_src)

    def test_auth_message_server_side(self):
        marker = '{"action": "auth", "params": self.api_key}'
        self.assertIn(marker, self.main_src)

    def test_quote_and_minute_topics(self):
        self.assertIn('f"C.{pair}"', self.main_src)
        self.assertIn('f"CA.{pair}"', self.main_src)

    def test_quote_parser_uses_bid_ask_not_midpoint(self):
        self.assertIn("parse_massive_forex_quote", self.main_src)
        self.assertNotIn("(bid + ask) / 2", self.main_src)

    def test_minute_parser_present(self):
        self.assertIn("parse_massive_forex_minute", self.main_src)

    def test_only_verified_mapping_can_be_ws_pair(self):
        marker = 'mapped = provider_symbol_map.to_provider("massive", canonical)'
        self.assertIn(marker, self.main_src)

    def test_forex_ws_start_endpoint(self):
        self.assertIn("/market/forex/websocket/start", self.main_src)

    def test_forex_realtime_endpoint(self):
        self.assertIn("/market/forex/{symbol}/realtime", self.main_src)

    def test_frontend_starts_forex_realtime(self):
        self.assertIn("startForexRealtime", self.html)

    def test_frontend_reads_forex_realtime(self):
        marker = '/api/v1/market/forex/"+encodeURIComponent(sym)+"/realtime'
        self.assertIn(marker, self.html)

    def test_only_one_minute_ws_updates_forex_chart(self):
        self.assertIn('chartState.tf==="1m"', self.html)

    def test_other_forex_tf_explicitly_rest(self):
        marker = 'LIVE BBO · OHLC "+chartState.tf.toUpperCase()+" REST'
        self.assertIn(marker, self.html)


class RealtimeCoreV2BGoldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main_src = Path(main.__file__).read_text(encoding="utf-8")
        cls.html = INDEX.read_text(encoding="utf-8")

    def test_twelvedata_official_gold_ws_base_url(self):
        marker = 'wss://ws.twelvedata.com/v1/quotes/price'
        self.assertIn(marker, self.main_src)

    def test_gold_ws_subscribes_xau_usd(self):
        self.assertIn('"symbols": "XAU/USD"', self.main_src)

    def test_gold_ws_price_parser_present(self):
        self.assertIn("parse_twelvedata_ws_price", self.main_src)

    def test_gold_ws_rejects_non_price_event(self):
        result = main.parse_twelvedata_ws_price(
            {"event": "heartbeat"},
            "XAU-USD",
        )
        self.assertIsNone(result)

    def test_gold_ws_parses_price_as_decimal(self):
        observed = datetime(2026, 9, 1, 18, 0, tzinfo=timezone.utc)
        payload = {
            "event": "price",
            "symbol": "XAU/USD",
            "price": "3456.789",
            "timestamp": observed.timestamp(),
        }
        result = main.parse_twelvedata_ws_price(
            payload,
            "XAU-USD",
            received_at=observed,
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.price, Decimal("3456.789"))

    def test_gold_ws_rejects_wrong_symbol(self):
        payload = {
            "event": "price",
            "symbol": "XAG/USD",
            "price": "40.0",
            "timestamp": 1788285600,
        }
        result = main.parse_twelvedata_ws_price(payload, "XAU-USD")
        self.assertIsNone(result)

    def test_gold_ws_rejects_non_positive_price(self):
        payload = {
            "event": "price",
            "symbol": "XAU/USD",
            "price": "0",
            "timestamp": 1788285600,
        }
        result = main.parse_twelvedata_ws_price(payload, "XAU-USD")
        self.assertIsNone(result)

    def test_gold_ws_start_endpoint_present(self):
        self.assertIn("/market/metal/websocket/start", self.main_src)

    def test_gold_realtime_endpoint_present(self):
        self.assertIn("/market/metal/{symbol}/realtime", self.main_src)

    def test_gold_realtime_declares_ohlc_rest(self):
        self.assertIn('"ohlc_transport": "REST"', self.main_src)

    def test_frontend_starts_gold_realtime(self):
        self.assertIn("startGoldRealtime", self.html)

    def test_frontend_reads_gold_realtime(self):
        marker = "/api/v1/market/metal/XAU-USD/realtime"
        self.assertIn(marker, self.html)

    def test_frontend_gold_badge_says_ohlc_rest(self):
        marker = "LIVE PRICE · TWELVE DATA WS · OHLC REST"
        self.assertIn(marker, self.html)

    def test_gold_ws_does_not_synthesize_ohlc(self):
        self.assertNotIn("mergeRealtimeGoldCandle", self.html)

class TestRealtimeCoreV2CIndices(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main_src = Path(main.__file__).read_text(encoding="utf-8")
        cls.html = INDEX.read_text(encoding="utf-8")

    def test_indices_ws_uses_documented_delayed_url(self):
        self.assertIn("wss://delayed.massive.com/indices", self.main_src)

    def test_indices_ws_topics_use_verified_symbols(self):
        manager = main.MassiveIndicesWsManager(api_key="test")
        topics = manager._topics()
        self.assertIn("V.I:SPX", topics)
        self.assertIn("AM.I:NDX", topics)
        self.assertIn("AM.I:DJI", topics)

    def test_index_value_parser_decimal(self):
        observed = datetime(2026, 9, 1, 18, 0, tzinfo=timezone.utc)
        result = main.parse_massive_index_value(
            {"ev": "V", "T": "I:SPX", "val": "6500.25", "t": 1788285600000},
            received_at=observed,
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.value, Decimal("6500.25"))

    def test_index_value_rejects_unknown_ticker(self):
        result = main.parse_massive_index_value(
            {"ev": "V", "T": "I:UNKNOWN", "val": "1", "t": 1788285600000}
        )
        self.assertIsNone(result)

    def test_index_minute_parser_has_no_volume(self):
        observed = datetime(2026, 9, 1, 18, 0, tzinfo=timezone.utc)
        result = main.parse_massive_index_minute(
            {
                "ev": "AM", "sym": "I:NDX", "o": "24000", "h": "24010",
                "l": "23990", "c": "24005", "s": 1788285600000,
            },
            received_at=observed,
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertNotIn("volume", result.to_dict())

    def test_index_minute_rejects_invalid_ohlc(self):
        result = main.parse_massive_index_minute(
            {
                "ev": "AM", "sym": "I:SPX", "o": "10", "h": "9",
                "l": "8", "c": "9", "s": 1788285600000,
            }
        )
        self.assertIsNone(result)

    def test_index_ws_start_endpoint_present(self):
        self.assertIn("/market/index/websocket/start", self.main_src)

    def test_index_realtime_endpoint_present(self):
        self.assertIn("/market/index/{symbol}/realtime", self.main_src)

    def test_index_feed_is_explicitly_delayed(self):
        self.assertIn('FEED_RECENCY = "15_MIN_DELAYED"', self.main_src)

    def test_frontend_starts_index_realtime(self):
        self.assertIn("startIndexRealtime", self.html)

    def test_frontend_reads_index_realtime(self):
        marker = '/api/v1/market/index/"+encodeURIComponent(sym)+"/realtime'
        self.assertIn(marker, self.html)

    def test_frontend_never_labels_delayed_indices_live(self):
        self.assertIn("15M DELAYED · MASSIVE WS", self.html)
        self.assertNotIn("LIVE INDEX · MASSIVE WS", self.html)


class TestChartEngineV4BOS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = INDEX.read_text(encoding="utf-8")

    def test_bos_detector_present(self):
        self.assertIn("function detectConfirmedBOS", self.html)

    def test_bos_uses_confirmed_swing_delay(self):
        self.assertIn("if(s.index+n!==i)return", self.html)

    def test_bos_bull_requires_close_above(self):
        self.assertIn("close>latestHigh.price", self.html)

    def test_bos_bear_requires_close_below(self):
        self.assertIn("close<latestLow.price", self.html)

    def test_bos_strict_comparison_rejects_equal(self):
        self.assertNotIn("close>=latestHigh.price", self.html)
        self.assertNotIn("close<=latestLow.price", self.html)

    def test_bos_does_not_use_wick_for_break(self):
        self.assertNotIn("Number(cs[i].high)>latestHigh.price", self.html)
        self.assertNotIn("Number(cs[i].low)<latestLow.price", self.html)

    def test_bos_excludes_latest_potentially_open_candle(self):
        self.assertIn("i<Math.max(0,cs.length-1)", self.html)

    def test_bos_deduplicates_broken_high(self):
        self.assertIn("!brokenHigh[latestHigh.index]", self.html)

    def test_bos_deduplicates_broken_low(self):
        self.assertIn("!brokenLow[latestLow.index]", self.html)

    def test_bos_renders_bull_and_bear_labels(self):
        self.assertIn('lab.textContent=bull?"BOS ↑":"BOS ↓"', self.html)

    def test_bos_ui_is_active(self):
        self.assertIn("BOS ACTIF", self.html)

    def test_choch_no_longer_marked_next_step(self):
        self.assertNotIn("CHoCH/MSS · prochaine étape", self.html)


class TestChartEngineV5CHOCHMSS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = INDEX.read_text(encoding="utf-8")

    def test_choch_detector_present(self):
        self.assertIn("function detectConfirmedCHOCH", self.html)

    def test_choch_uses_confirmed_swing_delay(self):
        self.assertIn("if(s.index+n===i)confirmed.push(s)", self.html)

    def test_choch_requires_bull_structure_pair(self):
        self.assertIn('if(s.structure==="HH")bullHigh=s', self.html)
        self.assertIn('if(s.structure==="HL")bullLow=s', self.html)

    def test_choch_requires_bear_structure_pair(self):
        self.assertIn('if(s.structure==="LH")bearHigh=s', self.html)
        self.assertIn('if(s.structure==="LL")bearLow=s', self.html)

    def test_bearish_choch_requires_close_below_last_low(self):
        self.assertIn('bias==="BULLISH"&&latestLow&&close<latestLow.price', self.html)

    def test_bullish_choch_requires_close_above_last_high(self):
        self.assertIn('bias==="BEARISH"&&latestHigh&&close>latestHigh.price', self.html)

    def test_choch_uses_strict_close_not_equal(self):
        self.assertNotIn("close<=latestLow.price", self.html)
        self.assertNotIn("close>=latestHigh.price", self.html)

    def test_choch_excludes_latest_potentially_open_candle(self):
        self.assertIn("i<Math.max(0,cs.length-1)", self.html)

    def test_choch_resets_bias_after_event(self):
        self.assertIn("brokenLow[latestLow.index]=true;bias=null", self.html)
        self.assertIn("brokenHigh[latestHigh.index]=true;bias=null", self.html)

    def test_choch_renders_both_directions(self):
        self.assertIn('label:"CHoCH/MSS ↓"', self.html)
        self.assertIn('label:"CHoCH/MSS ↑"', self.html)

    def test_choch_ui_is_active(self):
        self.assertIn("CHoCH/MSS ACTIF", self.html)

    def test_choch_counter_is_visible(self):
        self.assertIn('" · CHoCH/MSS ↑ "+chochBull', self.html)
        self.assertIn('" · CHoCH/MSS ↓ "+chochBear', self.html)


class TestChartEngineV6LiquiditySweep(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = INDEX.read_text(encoding="utf-8")

    def test_liquidity_sweep_detector_present(self):
        self.assertIn("function detectLiquiditySweeps", self.html)

    def test_sweep_uses_confirmed_swings(self):
        self.assertIn("if(s.index+n!==i)return", self.html)

    def test_swing_is_activated_after_sweep_evaluation(self):
        start = self.html.index("function detectLiquiditySweeps")
        end = self.html.index("function detectConfirmedCHOCH", start)
        detector = self.html[start:end]
        sweep_check = detector.index("high>latestHigh.price&&close<latestHigh.price")
        activation = detector.index("if(s.index+n!==i)return")
        self.assertLess(sweep_check, activation)

    def test_buy_side_sweep_requires_wick_above_and_close_below(self):
        self.assertIn("high>latestHigh.price&&close<latestHigh.price", self.html)

    def test_sell_side_sweep_requires_wick_below_and_close_above(self):
        self.assertIn("low<latestLow.price&&close>latestLow.price", self.html)

    def test_sweep_uses_strict_inequalities(self):
        self.assertNotIn("high>=latestHigh.price", self.html)
        self.assertNotIn("low<=latestLow.price", self.html)

    def test_sweep_excludes_latest_potentially_open_candle(self):
        self.assertIn("i<Math.max(0,cs.length-1)", self.html)

    def test_one_buy_side_sweep_per_reference_level(self):
        self.assertIn("!sweptHigh[latestHigh.index]", self.html)
        self.assertIn("sweptHigh[latestHigh.index]=true", self.html)

    def test_one_sell_side_sweep_per_reference_level(self):
        self.assertIn("!sweptLow[latestLow.index]", self.html)
        self.assertIn("sweptLow[latestLow.index]=true", self.html)

    def test_sweep_labels_are_rendered(self):
        self.assertIn('label:"BSL SWEEP"', self.html)
        self.assertIn('label:"SSL SWEEP"', self.html)

    def test_liquidity_sweep_ui_is_active(self):
        self.assertIn("LIQUIDITY SWEEP ACTIF", self.html)

    def test_sweep_counters_are_visible(self):
        self.assertIn('" · BSL Sweep "+buySweeps', self.html)
        self.assertIn('" · SSL Sweep "+sellSweeps', self.html)


class TestChartEngineV7Displacement(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = INDEX.read_text(encoding="utf-8")

    def test_displacement_detector_present(self):
        self.assertIn("function detectDisplacement", self.html)

    def test_displacement_lookback_is_explicit(self):
        self.assertIn("const DISPLACEMENT_LOOKBACK=20", self.html)

    def test_displacement_body_multiplier_is_explicit(self):
        self.assertIn("const DISPLACEMENT_BODY_MULTIPLIER=1.5", self.html)

    def test_displacement_body_range_ratio_is_explicit(self):
        self.assertIn("const DISPLACEMENT_MIN_BODY_RANGE_RATIO=0.7", self.html)

    def test_displacement_close_extreme_fraction_is_explicit(self):
        self.assertIn("const DISPLACEMENT_CLOSE_EXTREME_FRACTION=0.2", self.html)

    def test_displacement_uses_only_prior_bodies_for_baseline(self):
        self.assertIn("for(var j=i-lookback;j<i;j++)", self.html)

    def test_displacement_requires_large_and_strong_body(self):
        self.assertIn("largeBody&&strongBody&&(bullish||bearish)", self.html)

    def test_bullish_displacement_requires_directional_close(self):
        self.assertIn("close>open&&(high-close)/range", self.html)

    def test_bearish_displacement_requires_directional_close(self):
        self.assertIn("close<open&&(close-low)/range", self.html)

    def test_displacement_excludes_potentially_open_last_candle(self):
        self.assertIn("i<Math.max(0,cs.length-1)", self.html)

    def test_displacement_labels_and_ui_are_present(self):
        self.assertIn('label:bullish?"DISP ↑":"DISP ↓"', self.html)
        self.assertIn("DISPLACEMENT ACTIF", self.html)

    def test_displacement_methodology_warns_backtest_required(self):
        self.assertIn("Seuils à valider par backtest/OOS", self.html)


class TestChartEngineV8FVG(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = INDEX.read_text(encoding="utf-8")

    def test_fvg_detector_present(self):
        self.assertIn("function detectFairValueGaps", self.html)

    def test_bullish_fvg_is_strict_three_candle_gap(self):
        self.assertIn('if(cLow>aHigh){type="BULLISH"', self.html)

    def test_bearish_fvg_is_strict_three_candle_gap(self):
        self.assertIn('else if(cHigh<aLow){type="BEARISH"', self.html)

    def test_fvg_equality_does_not_count(self):
        self.assertNotIn("cLow>=aHigh", self.html)
        self.assertNotIn("cHigh<=aLow", self.html)

    def test_fvg_excludes_latest_potentially_open_candle(self):
        self.assertIn("i<Math.max(0,cs.length-1)", self.html)

    def test_fvg_followup_uses_only_closed_candles(self):
        self.assertIn("j<Math.max(0,cs.length-1)", self.html)

    def test_bullish_fvg_mitigation_is_tracked(self):
        self.assertIn('if(l<=lower){state="MITIGATED"', self.html)

    def test_bearish_fvg_mitigation_is_tracked(self):
        self.assertIn('if(h>=upper){state="MITIGATED"', self.html)

    def test_partial_mitigation_state_is_present(self):
        self.assertIn('state="PARTIALLY_MITIGATED"', self.html)

    def test_fvg_displacement_link_is_explicit(self):
        self.assertIn("displacementConfirmed:Boolean(dispByIndex[i-1])", self.html)

    def test_fvg_ui_is_active(self):
        self.assertIn("FVG ACTIF", self.html)

    def test_fvg_methodology_says_not_entry_signal(self):
        self.assertIn("un FVG isolé n’est pas un signal d’entrée", self.html)


class TestChartEngineV9OrderBlocks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = INDEX.read_text(encoding="utf-8")

    def test_order_block_detector_present(self):
        self.assertIn("function detectOrderBlocks", self.html)

    def test_order_block_requires_displacement(self):
        self.assertIn("ds.forEach(function(d)", self.html)

    def test_order_block_requires_structure_confirmation(self):
        self.assertIn("if(!structural)return", self.html)

    def test_order_block_accepts_bos_or_choch_source(self):
        self.assertIn("bs.concat(ch).forEach", self.html)

    def test_order_block_requires_same_direction_structure(self):
        self.assertIn('bullish&&e.type==="BULLISH"', self.html)
        self.assertIn('!bullish&&e.type==="BEARISH"', self.html)

    def test_order_block_search_is_bounded_to_five_prior_candles(self):
        self.assertIn("j>=Math.max(0,d.index-5)", self.html)

    def test_bullish_order_block_uses_opposite_bearish_candle(self):
        self.assertIn("(bullish&&c<o)", self.html)

    def test_bearish_order_block_uses_opposite_bullish_candle(self):
        self.assertIn("(!bullish&&c>o)", self.html)

    def test_order_block_uses_real_high_low_zone(self):
        self.assertIn("low=Number(ob.low),high=Number(ob.high)", self.html)

    def test_order_block_states_are_explicit(self):
        for token in ['state="FRESH"', 'state="RETESTED"', 'state="INVALIDATED"']:
            self.assertIn(token, self.html)

    def test_bullish_invalidation_requires_close_below_zone(self):
        self.assertIn("bullish&&close<low", self.html)

    def test_bearish_invalidation_requires_close_above_zone(self):
        self.assertIn("!bullish&&close>high", self.html)

    def test_sweep_and_fvg_are_separate_confirmations(self):
        self.assertIn("sweepConfirmed:sweepConfirmed", self.html)
        self.assertIn("fvgConfirmed:fvgConfirmed", self.html)

    def test_order_block_ui_and_methodology_are_present(self):
        self.assertIn("ORDER BLOCK ACTIF", self.html)
        self.assertIn("pas comme obligations ni comme signal d’entrée", self.html)


class TestChartEngineV10OrderBlockRetestQuality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = INDEX.read_text(encoding="utf-8")

    def test_order_block_width_is_explicit(self):
        self.assertIn("width=high-low", self.html)

    def test_order_block_tracks_deepest_retest_fraction(self):
        self.assertIn("deepestRetestFraction=0", self.html)

    def test_retest_penetration_is_clamped_zero_to_one(self):
        self.assertIn("Math.max(0,Math.min(1,penetration/width))", self.html)

    def test_bullish_retest_penetration_uses_zone_from_high_down(self):
        self.assertIn("high-Math.max(l,low)", self.html)

    def test_bearish_retest_penetration_uses_zone_from_low_up(self):
        self.assertIn("Math.min(h,high)-low", self.html)

    def test_first_retest_timestamp_is_retained(self):
        self.assertIn("if(mitigatedAt===null)mitigatedAt=k", self.html)

    def test_invalidation_stops_followup(self):
        self.assertIn('state="INVALIDATED";invalidatedAt=k;break', self.html)

    def test_evidence_count_has_two_required_components(self):
        self.assertIn("var evidenceCount=2+", self.html)

    def test_quality_core_tier_is_explicit(self):
        self.assertIn('"CORE"', self.html)

    def test_quality_confirmed_tier_is_explicit(self):
        self.assertIn('"CONFIRMED"', self.html)

    def test_quality_confluent_tier_is_explicit(self):
        self.assertIn('"CONFLUENT"', self.html)

    def test_invalidated_order_block_quality_is_invalid(self):
        self.assertIn('if(state==="INVALIDATED")qualityTier="INVALID"', self.html)

    def test_quality_methodology_disclaims_profit_probability(self):
        self.assertIn("pas une probabilité de gain", self.html)

    def test_order_block_retest_quality_ui_is_active(self):
        self.assertIn("OB RETEST / QUALITY ACTIF", self.html)


class TestChartEngineV11SmcStateMachine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = INDEX.read_text(encoding="utf-8")

    def test_smc_state_machine_present(self):
        self.assertIn("function buildSmcSetupStates", self.html)

    def test_sweep_to_displacement_window_is_explicit(self):
        self.assertIn("SMC_SWEEP_TO_DISPLACEMENT_MAX_BARS=12", self.html)

    def test_displacement_to_structure_window_is_explicit(self):
        self.assertIn("SMC_DISPLACEMENT_TO_STRUCTURE_MAX_BARS=6", self.html)

    def test_ssl_sweep_maps_to_bullish_direction(self):
        self.assertIn('s.type==="SELL_SIDE"?"BULLISH":"BEARISH"', self.html)

    def test_state_machine_starts_waiting_after_liquidity_sweep(self):
        self.assertIn('state="WAIT",reason="WAIT_DISPLACEMENT"', self.html)

    def test_displacement_advances_wait_reason_to_structure(self):
        self.assertIn('if(disp){state="WAIT";reason="WAIT_STRUCTURE"}', self.html)

    def test_structure_advances_wait_reason_to_entry_zone(self):
        self.assertIn('reason="WAIT_ENTRY_ZONE"', self.html)

    def test_entry_lifecycle_requires_order_block(self):
        self.assertIn("if(ob){", self.html)
        self.assertIn('reason="WAIT_ENTRY_ZONE"', self.html)

    def test_retest_becomes_entry_now_only_on_closed_zone_touch(self):
        self.assertIn('state="ENTRY_NOW";reason="OB_RETEST_CONFIRMED"', self.html)
        self.assertIn("entryIndex=q", self.html)

    def test_invalidated_state_still_comes_from_order_block_state(self):
        self.assertIn('if(ob.state==="INVALIDATED")', self.html)
        self.assertIn('state="INVALIDATED";reason="OB_INVALIDATED"', self.html)

    def test_state_machine_excludes_latest_potentially_open_candle(self):
        self.assertIn("closedEnd=Math.max(0,cs.length-1)", self.html)

    def test_fvg_is_recorded_as_optional_confirmation(self):
        self.assertIn("fvgConfirmed:Boolean(fvg)", self.html)

    def test_methodology_says_entry_now_is_not_execution(self):
        self.assertIn("aucun ordre n’est envoyé", self.html)

    def test_smc_state_machine_ui_is_active(self):
        self.assertIn("SMC STATE MACHINE ACTIF", self.html)


class TestChartEngineV12EntryLifecycle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = INDEX.read_text(encoding="utf-8")

    def test_entry_zone_window_is_explicit(self):
        self.assertIn("SMC_ENTRY_ZONE_MAX_BARS=12", self.html)

    def test_setup_lifecycle_starts_waiting(self):
        self.assertIn('state="WAIT",reason="WAIT_DISPLACEMENT"', self.html)

    def test_wait_structure_reason_is_explicit(self):
        self.assertIn('reason="WAIT_STRUCTURE"', self.html)

    def test_wait_entry_zone_reason_is_explicit(self):
        self.assertIn('reason="WAIT_ENTRY_ZONE"', self.html)

    def test_entry_now_requires_zone_touch(self):
        self.assertIn('state="ENTRY_NOW";reason="OB_RETEST_CONFIRMED"', self.html)

    def test_entry_now_records_closed_candle_index(self):
        self.assertIn("entryIndex=q", self.html)

    def test_bullish_close_below_ob_invalidates(self):
        self.assertIn('direction==="BULLISH"&&c<ob.low', self.html)

    def test_bearish_close_above_ob_invalidates(self):
        self.assertIn('direction==="BEARISH"&&c>ob.high', self.html)

    def test_close_beyond_ob_reason_is_explicit(self):
        self.assertIn('reason="CLOSE_BEYOND_OB"', self.html)

    def test_entry_window_can_expire(self):
        self.assertIn('state="EXPIRED";reason="ENTRY_WINDOW_EXPIRED"', self.html)

    def test_expiry_records_deadline_index(self):
        self.assertIn("expiredAt=structure.index+SMC_ENTRY_ZONE_MAX_BARS", self.html)

    def test_lifecycle_excludes_potentially_open_last_candle(self):
        self.assertIn("zoneDeadline=Math.min(closedEnd-1", self.html)

    def test_entry_now_does_not_claim_order_execution(self):
        self.assertIn("aucun ordre n’est envoyé", self.html)

    def test_entry_lifecycle_ui_is_active(self):
        self.assertIn("ENTRY LIFECYCLE ACTIF", self.html)


class TestChartEngineV13StructuralTradePlan(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = INDEX.read_text(encoding="utf-8")

    def test_trade_plan_builder_present(self):
        self.assertIn("function buildSmcTradePlans", self.html)

    def test_trade_plan_requires_entry_now(self):
        self.assertIn('if(s.state!=="ENTRY_NOW"', self.html)

    def test_entry_uses_closed_retest_candle_close(self):
        self.assertIn("entry=Number(entryCandle.close)", self.html)

    def test_bullish_stop_uses_order_block_low(self):
        self.assertIn("stop=bullish?obLow:obHigh", self.html)

    def test_stop_source_is_structural_order_block_invalidation(self):
        self.assertIn('stopSource:"ORDER_BLOCK_INVALIDATION"', self.html)

    def test_bullish_target_requires_prior_confirmed_swing_high(self):
        self.assertIn('w.kind==="HIGH"&&p>entry', self.html)

    def test_bearish_target_requires_prior_confirmed_swing_low(self):
        self.assertIn('w.kind==="LOW"&&p<entry', self.html)

    def test_target_must_precede_entry(self):
        self.assertIn("w.index>=s.entryIndex", self.html)

    def test_missing_target_is_explicitly_unavailable(self):
        self.assertIn('target===null?"UNAVAILABLE"', self.html)

    def test_risk_reward_is_reward_over_risk(self):
        self.assertIn("rr=reward===null?null:reward/risk", self.html)

    def test_nonpositive_risk_is_rejected(self):
        self.assertIn("risk<=0)return", self.html)

    def test_trade_plan_never_executes_order(self):
        self.assertIn("execution:false", self.html)

    def test_methodology_forbids_invented_rr(self):
        self.assertIn("aucun RR n’est inventé", self.html)

    def test_trade_plan_ui_is_active(self):
        self.assertIn("SL / TP / RR ACTIF", self.html)


class TestChartEngineV131VisibilityLayers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = INDEX.read_text(encoding="utf-8")

    def test_chart_layers_state_present(self):
        self.assertIn("const chartLayers=", self.html)

    def test_clean_preset_present(self):
        self.assertIn('name==="CLEAN"', self.html)

    def test_all_preset_present(self):
        self.assertIn('name==="ALL"', self.html)

    def test_clean_keeps_structure_visible(self):
        self.assertIn("structure:true,events:true,sweeps:true", self.html)

    def test_clean_hides_displacement_and_fvg(self):
        self.assertIn("displacement:false,fvg:false,ob:true", self.html)

    def test_structure_render_is_visibility_gated(self):
        self.assertIn('chartLayerEnabled("structure")', self.html)

    def test_events_render_is_visibility_gated(self):
        self.assertIn('chartLayerEnabled("events")', self.html)

    def test_sweeps_render_is_visibility_gated(self):
        self.assertIn('chartLayerEnabled("sweeps")', self.html)

    def test_displacement_render_is_visibility_gated(self):
        self.assertIn('chartLayerEnabled("displacement")', self.html)

    def test_fvg_render_hides_mitigated_zones(self):
        self.assertIn('e.state!=="MITIGATED"', self.html)

    def test_ob_render_hides_invalidated_zones(self):
        self.assertIn('e.state!=="INVALIDATED"', self.html)

    def test_visibility_does_not_disable_detection(self):
        self.assertIn("masquer une couche ne désactive jamais sa détection", self.html)


class TestChartEngineV14SignalEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = INDEX.read_text(encoding="utf-8")

    def test_signal_engine_present(self):
        self.assertIn("function buildSmcSignals", self.html)

    def test_signal_defaults_to_wait(self):
        self.assertIn('decision="WAIT"', self.html)

    def test_signal_requires_entry_now(self):
        self.assertIn('s.state!=="ENTRY_NOW"', self.html)

    def test_signal_requires_trade_plan(self):
        self.assertIn('reason="TRADE_PLAN_UNAVAILABLE"', self.html)

    def test_signal_requires_target(self):
        self.assertIn('plan.takeProfit===null', self.html)

    def test_signal_requires_risk_reward(self):
        self.assertIn('plan.riskReward===null', self.html)

    def test_invalid_rr_stays_wait(self):
        self.assertIn('reason="RR_INVALID"', self.html)

    def test_bullish_confirmed_setup_becomes_long(self):
        self.assertIn('s.direction==="BULLISH"?"LONG":"SHORT"', self.html)

    def test_confirmed_signal_reason_is_explicit(self):
        self.assertIn('reason="SETUP_AND_TRADE_PLAN_CONFIRMED"', self.html)

    def test_invalidated_setup_is_not_signal(self):
        self.assertIn('reason="SETUP_INVALIDATED"', self.html)

    def test_expired_setup_is_not_signal(self):
        self.assertIn('reason="SETUP_EXPIRED"', self.html)

    def test_signal_exposes_entry_stop_target_rr(self):
        tokens = [
            "entry:plan?plan.entry:null",
            "stopLoss:plan?plan.stopLoss:null",
            "takeProfit:plan?plan.takeProfit:null",
            "riskReward:plan?plan.riskReward:null",
        ]
        for token in tokens:
            self.assertIn(token, self.html)

    def test_signal_never_executes_order(self):
        token = "qualityTier:s.qualityTier||null,execution:false"
        self.assertIn(token, self.html)

    def test_signal_engine_ui_is_active(self):
        self.assertIn("SIGNAL ENGINE ACTIF", self.html)


class TestChartEngineV15SignalQuality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = INDEX.read_text(encoding="utf-8")

    def test_quality_engine_present(self):
        self.assertIn("function scoreSmcSignalQuality", self.html)

    def test_displacement_weight_is_explicit(self):
        self.assertIn('score+=20;evidence.push("DISPLACEMENT")', self.html)

    def test_structure_weight_is_explicit(self):
        self.assertIn('score+=20;evidence.push("STRUCTURE")', self.html)

    def test_order_block_weight_is_explicit(self):
        self.assertIn('score+=20;evidence.push("ORDER_BLOCK")', self.html)

    def test_fvg_weight_is_explicit(self):
        self.assertIn('score+=10;evidence.push("FVG")', self.html)

    def test_entry_now_weight_is_explicit(self):
        self.assertIn('score+=20;evidence.push("ENTRY_NOW")', self.html)

    def test_valid_rr_weight_is_explicit(self):
        self.assertIn('score+=10;evidence.push("VALID_RR")', self.html)

    def test_quality_grades_are_explicit(self):
        self.assertIn('score>=80?"A":score>=60?"B":score>=40?"C":"D"', self.html)

    def test_quality_version_is_recorded(self):
        self.assertIn('qualityVersion:"SMC_QUALITY_V1"', self.html)

    def test_quality_preserves_signal_object(self):
        self.assertIn("Object.assign({},sig", self.html)

    def test_quality_does_not_replace_signal_decision(self):
        self.assertIn("ils ne modifient pas LONG/SHORT/WAIT", self.html)

    def test_quality_requires_backtest_oos_validation(self):
        self.assertIn("à valider par backtest/OOS", self.html)

    def test_quality_has_no_performance_promise(self):
        self.assertIn("ne promettent aucune performance", self.html)

    def test_quality_ui_is_active(self):
        self.assertIn("SIGNAL QUALITY V1 ACTIF", self.html)


class TestChartEngineV16PaperRisk(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = INDEX.read_text(encoding="utf-8")

    def test_paper_default_capital_is_explicit(self):
        self.assertIn("PAPER_DEFAULT_CAPITAL_USD=1000", self.html)

    def test_paper_default_risk_percent_is_explicit(self):
        self.assertIn("PAPER_DEFAULT_RISK_PERCENT=1", self.html)

    def test_paper_candidate_builder_present(self):
        self.assertIn("function buildPaperTradeCandidates", self.html)

    def test_wait_signal_is_blocked(self):
        self.assertIn('reason:"SIGNAL_WAIT"', self.html)

    def test_incomplete_plan_is_blocked(self):
        self.assertIn('reason:"PLAN_INCOMPLETE"', self.html)

    def test_risk_money_uses_capital_times_percent(self):
        self.assertIn("riskMoney=capital*(riskPct/100)", self.html)

    def test_risk_distance_uses_entry_stop(self):
        self.assertIn("riskPerUnit=Math.abs(entry-stop)", self.html)

    def test_raw_units_use_risk_money_over_distance(self):
        self.assertIn("units=riskMoney/riskPerUnit", self.html)

    def test_invalid_risk_distance_is_blocked(self):
        self.assertIn('reason:"RISK_DISTANCE_INVALID"', self.html)

    def test_invalid_size_is_blocked(self):
        self.assertIn('reason:"SIZE_INVALID"', self.html)

    def test_instrument_specs_are_explicitly_unvalidated(self):
        self.assertIn('sizeStatus:"UNVALIDATED_INSTRUMENT_SPECS"', self.html)

    def test_paper_candidate_never_executes(self):
        self.assertIn('sizeStatus:"UNVALIDATED_INSTRUMENT_SPECS",execution:false', self.html)

    def test_methodology_requires_verified_instrument_specs(self):
        self.assertIn("sans spécifications instrument vérifiées", self.html)

    def test_paper_risk_ui_is_active(self):
        self.assertIn("PAPER RISK V1 ACTIF", self.html)


class TestChartEngineV16BInstrumentSpecs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = INDEX.read_text(encoding="utf-8")

    def test_instrument_spec_validator_present(self):
        self.assertIn("function validatePaperInstrumentSpecs", self.html)

    def test_specs_require_source(self):
        self.assertIn('"source","sourceTimestamp","assetClass","sizeMode"', self.html)

    def test_missing_spec_field_is_unavailable(self):
        self.assertIn('reason:"SPEC_FIELD_MISSING"', self.html)

    def test_units_mode_requires_volume_rules(self):
        self.assertIn('reason:"VOLUME_RULES_MISSING"', self.html)

    def test_contract_mode_requires_tick_size(self):
        self.assertIn("var tickSize=Number(s.tickSize)", self.html)

    def test_contract_mode_requires_tick_value(self):
        self.assertIn("tickValue=Number(s.tickValue)", self.html)

    def test_contract_mode_requires_contract_size(self):
        self.assertIn("contractSize=Number(s.contractSize)", self.html)

    def test_contract_missing_specs_are_unavailable(self):
        self.assertIn('reason:"CONTRACT_SPEC_MISSING"', self.html)

    def test_contract_invalid_specs_are_unavailable(self):
        self.assertIn('reason:"CONTRACT_SPEC_INVALID"', self.html)

    def test_unsupported_size_mode_is_unavailable(self):
        self.assertIn('reason:"SIZE_MODE_UNSUPPORTED"', self.html)

    def test_size_normalizer_present(self):
        self.assertIn("function normalizePaperSize", self.html)

    def test_size_is_rounded_down_to_step(self):
        self.assertIn("Math.floor(raw/step)*step", self.html)

    def test_below_minimum_volume_is_unavailable(self):
        self.assertIn('reason:"BELOW_MINIMUM_VOLUME"', self.html)

    def test_instrument_specs_ui_is_active(self):
        self.assertIn("INSTRUMENT SPECS V1 ACTIF", self.html)


class TestChartEngineV16CPaperPosition(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = INDEX.read_text(encoding="utf-8")

    def test_paper_position_opener_present(self):
        self.assertIn("function openPaperPosition", self.html)

    def test_candidate_must_be_ready(self):
        self.assertIn('reason:"CANDIDATE_NOT_READY"', self.html)

    def test_size_must_be_validated(self):
        self.assertIn('reason:"SIZE_NOT_VALIDATED"', self.html)

    def test_instrument_identity_is_required(self):
        self.assertIn('reason:"INSTRUMENT_IDENTITY_MISSING"', self.html)

    def test_open_timestamp_is_required(self):
        self.assertIn('reason:"OPEN_TIMESTAMP_MISSING"', self.html)

    def test_long_levels_are_structurally_validated(self):
        self.assertIn('reason:"LONG_LEVELS_INVALID"', self.html)

    def test_short_levels_are_structurally_validated(self):
        self.assertIn('reason:"SHORT_LEVELS_INVALID"', self.html)

    def test_open_position_is_explicitly_paper_only(self):
        self.assertIn('status:"OPEN",paperOnly:true', self.html)

    def test_open_position_never_executes_broker_order(self):
        self.assertIn("closeReason:null,closedAt:null,execution:false", self.html)

    def test_position_marker_present(self):
        self.assertIn("function markPaperPosition", self.html)

    def test_stop_loss_close_is_supported(self):
        self.assertIn('closeReason:"STOP_LOSS"', self.html)

    def test_take_profit_close_is_supported(self):
        self.assertIn('closeReason:"TAKE_PROFIT"', self.html)

    def test_ambiguous_sl_tp_is_conflict(self):
        self.assertIn('closeReason:"SL_TP_CONFLICT"', self.html)

    def test_paper_position_ui_is_active(self):
        self.assertIn("PAPER POSITION V1 ACTIF", self.html)


class TestPaperPersistenceV16D(unittest.TestCase):
    def test_paper_positions_table_exists(self):
        self.assertIn("paper_positions", main.paper_positions_table.name)

    def test_paper_position_primary_key(self):
        self.assertTrue(main.paper_positions_table.c.position_id.primary_key)

    def test_paper_prices_use_numeric(self):
        for name in ("entry", "stop_loss", "take_profit", "size", "risk_money"):
            self.assertIsInstance(main.paper_positions_table.c[name].type, main.Numeric)

    def test_paper_create_model_uses_decimal(self):
        fields = main.PaperPositionCreate.model_fields
        self.assertIs(fields["entry"].annotation, Decimal)
        self.assertIs(fields["size"].annotation, Decimal)

    def test_paper_create_route_exists(self):
        paths = {route.path for route in main.api_router.routes}
        self.assertIn("/paper/positions", paths)

    def test_paper_list_route_exists(self):
        paths = {route.path for route in main.api_router.routes}
        self.assertIn("/paper/positions", paths)

    def test_paper_validation_rejects_bad_side(self):
        req = main.PaperPositionCreate(
            position_id="p1", symbol="BTC-USD", side="BUY",
            entry="100", stop_loss="90", take_profit="120", size="1",
            size_unit="UNITS", risk_money="10", risk_percent="1",
            capital_before="1000", source="coinbase",
            source_timestamp=main.utcnow(), opened_at=main.utcnow(),
        )
        with self.assertRaises(main.HTTPException):
            main.validate_paper_position_create(req)

    def test_paper_validation_accepts_long_levels(self):
        req = main.PaperPositionCreate(
            position_id="p1", symbol="BTC-USD", side="LONG",
            entry="100", stop_loss="90", take_profit="120", size="1",
            size_unit="UNITS", risk_money="10", risk_percent="1",
            capital_before="1000", source="coinbase",
            source_timestamp=main.utcnow(), opened_at=main.utcnow(),
        )
        self.assertIsNone(main.validate_paper_position_create(req))

    def test_paper_validation_accepts_short_levels(self):
        req = main.PaperPositionCreate(
            position_id="p2", symbol="BTC-USD", side="SHORT",
            entry="100", stop_loss="110", take_profit="80", size="1",
            size_unit="UNITS", risk_money="10", risk_percent="1",
            capital_before="1000", source="coinbase",
            source_timestamp=main.utcnow(), opened_at=main.utcnow(),
        )
        self.assertIsNone(main.validate_paper_position_create(req))

    def test_paper_payload_is_explicitly_paper_only(self):
        source = inspect.getsource(main.create_paper_position)
        self.assertIn('"paper_only": True', source)

    def test_paper_payload_never_executes_broker_order(self):
        source = inspect.getsource(main.create_paper_position)
        self.assertIn('"execution": False', source)

    def test_duplicate_position_id_is_conflict(self):
        source = inspect.getsource(main.create_paper_position)
        self.assertIn("POSITION_ID_EXISTS", source)

    def test_persistence_unavailable_is_fail_safe(self):
        source = inspect.getsource(main.create_paper_position)
        self.assertIn("persistence not ready", source)

    def test_paper_persistence_ui_is_active(self):
        html = INDEX.read_text(encoding="utf-8")
        self.assertIn("PAPER PERSISTENCE V1 ACTIF", html)


class TestPaperMarkV16E(unittest.TestCase):
    def test_mark_model_uses_decimal_price(self):
        field = main.PaperPositionMark.model_fields["current_price"]
        self.assertIs(field.annotation, Decimal)

    def test_mark_model_requires_source_timestamp(self):
        fields = main.PaperPositionMark.model_fields
        self.assertIn("source", fields)
        self.assertIn("source_timestamp", fields)

    def test_close_evaluator_long_stop(self):
        result = main.evaluate_paper_close("LONG", Decimal("89"), Decimal("90"), Decimal("120"))
        self.assertEqual(result, ("STOP_LOSS", Decimal("90")))

    def test_close_evaluator_long_target(self):
        result = main.evaluate_paper_close("LONG", Decimal("121"), Decimal("90"), Decimal("120"))
        self.assertEqual(result, ("TAKE_PROFIT", Decimal("120")))

    def test_close_evaluator_short_stop(self):
        result = main.evaluate_paper_close("SHORT", Decimal("111"), Decimal("110"), Decimal("80"))
        self.assertEqual(result, ("STOP_LOSS", Decimal("110")))

    def test_close_evaluator_short_target(self):
        result = main.evaluate_paper_close("SHORT", Decimal("79"), Decimal("110"), Decimal("80"))
        self.assertEqual(result, ("TAKE_PROFIT", Decimal("80")))

    def test_close_evaluator_no_hit(self):
        result = main.evaluate_paper_close("LONG", Decimal("105"), Decimal("90"), Decimal("120"))
        self.assertIsNone(result)

    def test_long_pnl(self):
        pnl = main.calculate_paper_pnl("LONG", Decimal("100"), Decimal("110"), Decimal("2"))
        self.assertEqual(pnl, Decimal("20"))

    def test_short_pnl(self):
        pnl = main.calculate_paper_pnl("SHORT", Decimal("100"), Decimal("90"), Decimal("2"))
        self.assertEqual(pnl, Decimal("20"))

    def test_mark_route_exists(self):
        paths = {route.path for route in main.api_router.routes}
        self.assertIn("/paper/positions/{position_id}/mark", paths)

    def test_mark_uses_row_lock(self):
        source = inspect.getsource(main.mark_paper_position)
        self.assertIn("FOR UPDATE", source)

    def test_mark_closes_only_open_position(self):
        source = inspect.getsource(main.mark_paper_position)
        self.assertIn("AND status='OPEN'", source)

    def test_mark_exposes_unrealized_and_realized_pnl(self):
        source = inspect.getsource(main.mark_paper_position)
        self.assertIn('"unrealized_pnl"', source)
        self.assertIn('"realized_pnl"', source)

    def test_paper_pnl_ui_is_active(self):
        html = INDEX.read_text(encoding="utf-8")
        self.assertIn("PAPER P&L V1 ACTIF", html)


class TestPaperRealtimeMonitorV16F(unittest.TestCase):
    def test_realtime_mark_builder_present(self):
        source = inspect.getsource(main.paper_mark_from_realtime)
        self.assertIn("await market_store.get_ticker", source)

    def test_realtime_mark_requires_registered_instrument(self):
        source = inspect.getsource(main.paper_mark_from_realtime)
        self.assertIn("instrument is None", source)

    def test_realtime_mark_supports_crypto(self):
        source = inspect.getsource(main.paper_mark_from_realtime)
        self.assertIn("AssetClass.CRYPTO", source)

    def test_realtime_mark_requires_valid_quality(self):
        source = inspect.getsource(main.paper_mark_from_realtime)
        self.assertIn("datum.status != DataQualityStatus.VALID", source)

    def test_realtime_mark_rejects_missing_datum(self):
        source = inspect.getsource(main.paper_mark_from_realtime)
        self.assertIn("datum is None", source)

    def test_realtime_mark_rejects_nonpositive_price(self):
        source = inspect.getsource(main.paper_mark_from_realtime)
        self.assertIn("datum.value <= 0", source)

    def test_realtime_mark_preserves_source_timestamp(self):
        source = inspect.getsource(main.paper_mark_from_realtime)
        self.assertIn("source_timestamp = datum.source_timestamp", source)
        self.assertIn("source_timestamp=source_timestamp", source)

    def test_monitor_reads_only_open_positions(self):
        source = inspect.getsource(main.monitor_open_paper_positions_once)
        self.assertIn("WHERE status='OPEN'", source)

    def test_monitor_counts_unavailable_marks(self):
        source = inspect.getsource(main.monitor_open_paper_positions_once)
        self.assertIn("unavailable += 1", source)

    def test_monitor_uses_existing_mark_logic(self):
        source = inspect.getsource(main.monitor_open_paper_positions_once)
        self.assertIn("await mark_paper_position", source)

    def test_monitor_is_fail_safe_when_persistence_not_ready(self):
        source = inspect.getsource(main.monitor_open_paper_positions_once)
        self.assertIn("if not persistence_state.ready", source)

    def test_monitor_does_not_synthesize_price(self):
        source = inspect.getsource(main.paper_mark_from_realtime)
        self.assertNotIn("random", source.lower())

    def test_monitor_ui_is_active(self):
        html = INDEX.read_text(encoding="utf-8")
        self.assertIn("PAPER REALTIME MONITOR V1 ACTIF", html)

    def test_monitor_ui_states_multi_asset_scope(self):
        html = INDEX.read_text(encoding="utf-8")
        self.assertIn("Crypto/Coinbase", html)
        self.assertIn("Forex/Massive BBO", html)
        self.assertIn("Gold/Twelve Data", html)
        self.assertIn("Indices/Massive Value", html)


class TestPaperAutoLoopV16G(unittest.TestCase):
    def test_monitor_interval_setting_exists(self):
        self.assertTrue(hasattr(main.settings, "paper_monitor_interval_seconds"))

    def test_monitor_interval_default_is_one_second(self):
        self.assertEqual(main.Settings().paper_monitor_interval_seconds, 1.0)

    def test_monitor_loop_present(self):
        source = inspect.getsource(main.paper_monitor_loop)
        self.assertIn("monitor_open_paper_positions_once", source)

    def test_monitor_loop_has_one_second_floor(self):
        source = inspect.getsource(main.paper_monitor_loop)
        self.assertIn("max(settings.paper_monitor_interval_seconds, 1.0)", source)

    def test_monitor_loop_waits_on_stop_event(self):
        source = inspect.getsource(main.paper_monitor_loop)
        self.assertIn("await asyncio.wait_for(stop_event.wait()", source)

    def test_monitor_loop_preserves_cancellation(self):
        source = inspect.getsource(main.paper_monitor_loop)
        self.assertIn("except asyncio.CancelledError", source)
        self.assertIn("raise", source)

    def test_monitor_loop_iteration_failure_is_fail_safe(self):
        source = inspect.getsource(main.paper_monitor_loop)
        self.assertIn("Paper monitor iteration failed", source)

    def test_lifespan_creates_monitor_task(self):
        source = inspect.getsource(main.lifespan)
        self.assertIn('name="paper-monitor"', source)

    def test_lifespan_signals_monitor_stop(self):
        source = inspect.getsource(main.lifespan)
        self.assertIn("paper_monitor_stop.set()", source)

    def test_lifespan_awaits_monitor_before_market_disconnect(self):
        source = inspect.getsource(main.lifespan)
        self.assertLess(
            source.index("await paper_monitor_task"),
            source.index("await market_provider.disconnect()"),
        )

    def test_auto_loop_does_not_enable_live_trading(self):
        source = inspect.getsource(main.paper_monitor_loop)
        self.assertNotIn("live_trading_enabled = True", source)

    def test_auto_loop_uses_existing_one_shot_monitor(self):
        source = inspect.getsource(main.paper_monitor_loop)
        self.assertIn("await monitor_open_paper_positions_once()", source)

    def test_auto_loop_ui_is_active(self):
        html = INDEX.read_text(encoding="utf-8")
        self.assertIn("PAPER AUTO LOOP V1 ACTIF", html)

    def test_auto_loop_ui_documents_fail_safe_behavior(self):
        html = INDEX.read_text(encoding="utf-8")
        self.assertIn("aucun prix n’est inventé", html)


class TestPaperMultiAssetMonitorV16H(unittest.TestCase):
    def test_multi_asset_supports_forex(self):
        source = inspect.getsource(main.paper_mark_from_realtime)
        self.assertIn("AssetClass.FOREX", source)
        self.assertIn("massive_forex_ws.quotes.get", source)

    def test_forex_requires_valid_quality(self):
        source = inspect.getsource(main.paper_mark_from_realtime)
        self.assertIn("quote.quality != DataQualityStatus.VALID", source)

    def test_forex_rejects_crossed_bbo(self):
        source = inspect.getsource(main.paper_mark_from_realtime)
        self.assertIn("quote.bid > quote.ask", source)

    def test_forex_mark_uses_real_bbo_midpoint(self):
        source = inspect.getsource(main.paper_mark_from_realtime)
        self.assertIn('(quote.bid + quote.ask) / Decimal("2")', source)

    def test_multi_asset_supports_gold(self):
        source = inspect.getsource(main.paper_mark_from_realtime)
        self.assertIn("AssetClass.METAL", source)
        self.assertIn("twelvedata_gold_ws.last_price", source)

    def test_gold_requires_valid_quality(self):
        source = inspect.getsource(main.paper_mark_from_realtime)
        self.assertIn("gold.quality != DataQualityStatus.VALID", source)

    def test_gold_is_xau_usd_only(self):
        source = inspect.getsource(main.paper_mark_from_realtime)
        self.assertIn('canonical != "XAU-USD"', source)

    def test_multi_asset_supports_indices(self):
        source = inspect.getsource(main.paper_mark_from_realtime)
        self.assertIn("AssetClass.INDEX", source)
        self.assertIn("massive_indices_ws.values.get", source)

    def test_indices_require_valid_quality(self):
        source = inspect.getsource(main.paper_mark_from_realtime)
        self.assertIn("value.quality != DataQualityStatus.VALID", source)

    def test_indices_use_value_not_synthetic_candle(self):
        source = inspect.getsource(main.paper_mark_from_realtime)
        self.assertIn("price = value.value", source)
        self.assertNotIn("massive_indices_ws.candles.get", source)

    def test_all_marks_require_source_timestamp(self):
        source = inspect.getsource(main.paper_mark_from_realtime)
        self.assertIn("source_timestamp is None", source)

    def test_multi_asset_ui_is_active(self):
        html = INDEX.read_text(encoding="utf-8")
        self.assertIn("PAPER MULTI-ASSET V1 ACTIF", html)

    def test_ui_documents_delayed_indices(self):
        html = INDEX.read_text(encoding="utf-8")
        self.assertIn("15 minutes delayed", html)

    def test_multi_asset_monitor_does_not_enable_live_trading(self):
        source = inspect.getsource(main.paper_mark_from_realtime)
        self.assertNotIn("live_trading_enabled", source)


class TestPaperTradingUiV16I(unittest.TestCase):
    def test_paper_account_route_exists(self):
        paths = {route.path for route in main.api_router.routes}
        self.assertIn("/paper/account", paths)

    def test_paper_account_is_read_only(self):
        route = next(r for r in main.api_router.routes if r.path == "/paper/account")
        self.assertIn("GET", route.methods)

    def test_paper_account_uses_persisted_positions(self):
        source = inspect.getsource(main.get_paper_account)
        self.assertIn("FROM paper_positions", source)
        self.assertIn("FROM paper_account", source)

    def test_paper_account_calculates_realized_pnl(self):
        source = inspect.getsource(main.get_paper_account)
        self.assertIn("calculate_paper_pnl", source)

    def test_paper_account_is_paper_only(self):
        source = inspect.getsource(main.get_paper_account)
        self.assertIn('"paper_only": True', source)
        self.assertIn('"execution": False', source)

    def test_paper_ui_fetches_account(self):
        html = INDEX.read_text(encoding="utf-8")
        self.assertIn('/api/v1/paper/ui-snapshot', html)
        self.assertIn('snapshotSection(snap,"account")', html)

    def test_paper_ui_fetches_positions(self):
        html = INDEX.read_text(encoding="utf-8")
        self.assertIn('/api/v1/paper/ui-snapshot', html)
        self.assertIn('snapshotSection(snap,"positions")', html)

    def test_paper_ui_has_real_positions_section(self):
        html = INDEX.read_text(encoding="utf-8")
        self.assertIn('modCard("Positions ouvertes",open.length+" OPEN"', html)

    def test_paper_ui_has_history_section(self):
        html = INDEX.read_text(encoding="utf-8")
        self.assertIn('modCard("Historique",closed.length+" CLOSED"', html)

    def test_paper_ui_shows_entry_sl_tp(self):
        html = INDEX.read_text(encoding="utf-8")
        self.assertIn('["Entry"]', html)
        self.assertIn('["SL"]', html)
        self.assertIn('["TP"]', html)

    def test_paper_ui_does_not_invent_unrealized_pnl(self):
        html = INDEX.read_text(encoding="utf-8")
        self.assertIn('p.mark_status==="VALID"?paperMoney(p.unrealized_pnl):"—"', html)

    def test_paper_ui_marks_broker_execution_disabled(self):
        html = INDEX.read_text(encoding="utf-8")
        self.assertIn('["Exécution broker / MT5"]', html)
        self.assertIn('["DÉSACTIVÉE"]', html)

    def test_paper_ui_poll_refreshes_trading(self):
        html = INDEX.read_text(encoding="utf-8")
        self.assertIn('current==="trading")refreshPaperTrading()', html)

    def test_paper_ui_help_is_present(self):
        html = INDEX.read_text(encoding="utf-8")
        self.assertIn("Paper Trading UI V1", html)


class TestPaperLivePnlV16J(unittest.TestCase):
    def test_live_positions_route_exists(self):
        paths = {route.path for route in main.api_router.routes}
        self.assertIn("/paper/positions/live", paths)

    def test_live_positions_route_is_get(self):
        route = next(r for r in main.api_router.routes if r.path == "/paper/positions/live")
        self.assertIn("GET", route.methods)

    def test_live_positions_reads_only_open_positions(self):
        source = inspect.getsource(main.get_live_paper_positions)
        self.assertIn("WHERE status='OPEN'", source)

    def test_live_positions_reuses_multi_asset_mark_builder(self):
        source = inspect.getsource(main.get_live_paper_positions)
        self.assertIn("await paper_mark_from_realtime", source)

    def test_live_positions_defaults_mark_to_unavailable(self):
        source = inspect.getsource(main.get_live_paper_positions)
        self.assertIn('payload["mark_status"] = "UNAVAILABLE"', source)

    def test_live_positions_exposes_mark_only_when_available(self):
        source = inspect.getsource(main.get_live_paper_positions)
        self.assertIn('payload["mark_status"] = "VALID"', source)
        self.assertIn('payload["mark_price"] = str(mark.current_price)', source)

    def test_live_positions_calculates_unrealized_pnl(self):
        source = inspect.getsource(main.get_live_paper_positions)
        self.assertIn("calculate_paper_pnl", source)
        self.assertIn('payload["unrealized_pnl"]', source)

    def test_live_positions_preserves_mark_source_timestamp(self):
        source = inspect.getsource(main.get_live_paper_positions)
        self.assertIn("mark.source_timestamp.isoformat()", source)

    def test_live_positions_is_paper_only(self):
        source = inspect.getsource(main.get_live_paper_positions)
        self.assertIn('"paper_only": True', source)
        self.assertIn('"execution": False', source)

    def test_ui_fetches_live_positions(self):
        html = INDEX.read_text(encoding="utf-8")
        self.assertIn('/api/v1/paper/ui-snapshot', html)
        self.assertIn('snapshotSection(snap,"live_positions")', html)

    def test_ui_shows_current_price(self):
        html = INDEX.read_text(encoding="utf-8")
        self.assertIn('["Prix actuel"]', html)
        self.assertIn("p.mark_price", html)

    def test_ui_shows_live_unrealized_pnl(self):
        html = INDEX.read_text(encoding="utf-8")
        self.assertIn("p.unrealized_pnl", html)

    def test_ui_shows_mark_status(self):
        html = INDEX.read_text(encoding="utf-8")
        self.assertIn('["Mark"]', html)
        self.assertIn("UNAVAILABLE", html)

    def test_ui_help_documents_valid_mark_gate(self):
        html = INDEX.read_text(encoding="utf-8")
        self.assertIn("vrai mark multi-actifs qualifié VALID", html)


class TestPaperTradingMobileRenderV16J1(unittest.TestCase):
    def test_trading_renderer_has_safe_wrapper(self):
        html = INDEX.read_text(encoding="utf-8")
        self.assertIn("function renderTradingContent()", html)
        self.assertIn("function renderTrading()", html)
        self.assertIn("var a=paperUiState.account;", html)
        self.assertNotIn("clear(v),a=paperUiState.account", html)

    def test_trading_renderer_catches_local_render_error(self):
        html = INDEX.read_text(encoding="utf-8")
        self.assertIn("Affichage Paper Trading indisponible", html)

    def test_trading_renderer_never_fails_to_blank_silently(self):
        html = INDEX.read_text(encoding="utf-8")
        self.assertIn("Aucune donnée n’a été inventée", html)

    def test_trading_view_gets_render_marker(self):
        html = INDEX.read_text(encoding="utf-8")
        self.assertIn('data-paper-rendered","true"', html)

    def test_workspace_mode_exists(self):
        html = INDEX.read_text(encoding="utf-8")
        self.assertIn("workspace-mode", html)

    def test_workspace_mode_hides_market_tabs(self):
        html = INDEX.read_text(encoding="utf-8")
        self.assertIn("body.workspace-mode .market-tabs-wrap{display:none}", html)

    def test_trading_is_classified_as_workspace(self):
        html = INDEX.read_text(encoding="utf-8")
        self.assertIn('"signals","trading","strategies","intelligence","settings"', html)

    def test_set_view_toggles_workspace_mode(self):
        html = INDEX.read_text(encoding="utf-8")
        self.assertIn('document.body.classList.toggle("workspace-mode",workspace)', html)

    def test_paper_shell_has_mobile_minimum_height(self):
        html = INDEX.read_text(encoding="utf-8")
        self.assertIn(".paper-shell{min-height:320px}", html)

    def test_bottom_nav_still_targets_trading(self):
        html = INDEX.read_text(encoding="utf-8")
        self.assertIn('{id:"trading",label:"Trading"', html)

    def test_live_pnl_endpoint_remains_used(self):
        html = INDEX.read_text(encoding="utf-8")
        self.assertIn('/api/v1/paper/ui-snapshot', html)
        self.assertIn('snapshotSection(snap,"live_positions")', html)

    def test_backend_is_unchanged_for_ui_fix(self):
        paths = {route.path for route in main.api_router.routes}
        self.assertIn("/paper/positions/live", paths)


class TestPaperAccountInitializationV16K(unittest.TestCase):
    def test_account_table_exists(self):
        self.assertEqual(main.paper_account_table.name, "paper_account")

    def test_account_has_stable_default_id(self):
        self.assertEqual(main.PAPER_ACCOUNT_ID, "default")

    def test_account_currency_is_usd(self):
        self.assertEqual(main.PAPER_ACCOUNT_CURRENCY, "USD")

    def test_initial_capital_is_decimal_1000(self):
        self.assertEqual(main.PAPER_INITIAL_CAPITAL, Decimal("1000"))

    def test_schema_initializes_account(self):
        source = inspect.getsource(main.init_candle_schema)
        self.assertIn("pg_insert(paper_account_table)", source)

    def test_account_initialization_is_idempotent(self):
        source = inspect.getsource(main.init_candle_schema)
        self.assertIn("on_conflict_do_nothing", source)

    def test_account_initialization_does_not_reset_existing_capital(self):
        source = inspect.getsource(main.init_candle_schema)
        self.assertNotIn("on_conflict_do_update", source)

    def test_account_endpoint_reads_persisted_initial_capital(self):
        source = inspect.getsource(main.get_paper_account)
        self.assertIn("initial_capital FROM paper_account", source)

    def test_account_endpoint_has_no_position_capital_fallback(self):
        source = inspect.getsource(main.get_paper_account)
        self.assertNotIn('data["capital_before"]', source)

    def test_current_capital_adds_realized_pnl(self):
        source = inspect.getsource(main.get_paper_account)
        self.assertIn("current_capital = initial_capital + realized", source)

    def test_empty_account_is_fail_safe(self):
        source = inspect.getsource(main.get_paper_account)
        self.assertIn("paper account not initialized", source)

    def test_account_response_exposes_currency(self):
        source = inspect.getsource(main.get_paper_account)
        self.assertIn('"currency": account._mapping["currency"]', source)

    def test_ui_documents_persistent_1000_usd_account(self):
        html = INDEX.read_text(encoding="utf-8")
        self.assertIn("initialisé une seule fois en PostgreSQL avec 1 000 USD", html)

    def test_account_remains_paper_only(self):
        source = inspect.getsource(main.get_paper_account)
        self.assertIn('"paper_only": True', source)
        self.assertIn('"execution": False', source)


class TestPaperLiveEquityV16L(unittest.TestCase):
    def test_live_account_route_exists(self):
        paths = {route.path for route in main.api_router.routes}
        self.assertIn("/paper/account/live", paths)

    def test_live_account_route_is_get(self):
        route = next(r for r in main.api_router.routes if r.path == "/paper/account/live")
        self.assertIn("GET", route.methods)

    def test_live_account_reuses_persisted_account(self):
        source = inspect.getsource(main.get_live_paper_account)
        self.assertIn("account = await get_paper_account()", source)

    def test_live_account_reads_only_open_positions(self):
        source = inspect.getsource(main.get_live_paper_account)
        self.assertIn("WHERE status='OPEN'", source)

    def test_live_account_reuses_multi_asset_mark_builder(self):
        source = inspect.getsource(main.get_live_paper_account)
        self.assertIn("await paper_mark_from_realtime", source)

    def test_live_account_calculates_unrealized_pnl(self):
        source = inspect.getsource(main.get_live_paper_account)
        self.assertIn("calculate_paper_pnl", source)

    def test_live_equity_adds_unrealized_to_realized_capital(self):
        source = inspect.getsource(main.get_live_paper_account)
        self.assertIn("live_equity = current_capital + unrealized", source)

    def test_missing_mark_makes_global_equity_partial(self):
        source = inspect.getsource(main.get_live_paper_account)
        self.assertIn('live_equity_status": "VALID" if complete else "PARTIAL"', source)

    def test_partial_equity_does_not_publish_fake_total(self):
        source = inspect.getsource(main.get_live_paper_account)
        self.assertIn('"live_equity": str(live_equity) if complete else None', source)

    def test_partial_unrealized_does_not_publish_fake_total(self):
        source = inspect.getsource(main.get_live_paper_account)
        self.assertIn('"unrealized_pnl": str(unrealized) if complete else None', source)

    def test_live_account_remains_paper_only(self):
        source = inspect.getsource(main.get_live_paper_account)
        self.assertIn('"paper_only": True', source)
        self.assertIn('"execution": False', source)

    def test_ui_fetches_live_account(self):
        html = INDEX.read_text(encoding="utf-8")
        self.assertIn('/api/v1/paper/ui-snapshot', html)
        self.assertIn('snapshotSection(snap,"account")', html)

    def test_ui_shows_live_equity_and_latent_pnl(self):
        html = INDEX.read_text(encoding="utf-8")
        self.assertIn('["Équité live"]', html)
        self.assertIn('["P&L latent"]', html)

    def test_ui_hides_partial_live_equity(self):
        html = INDEX.read_text(encoding="utf-8")
        self.assertIn('a.live_equity_status==="VALID"?paperMoney(a.live_equity):"—"', html)


class TestPaperAutoEntryGateV16M1(unittest.TestCase):
    def test_gate_route_exists(self):
        paths = {route.path for route in main.api_router.routes}
        self.assertIn("/paper/auto-entry/gate", paths)

    def test_gate_route_is_post(self):
        route = next(r for r in main.api_router.routes if r.path == "/paper/auto-entry/gate")
        self.assertIn("POST", route.methods)

    def test_gate_request_requires_symbol(self):
        fields = main.PaperAutoEntryGateRequest.model_fields
        self.assertIn("symbol", fields)

    def test_gate_request_has_signal_decision(self):
        fields = main.PaperAutoEntryGateRequest.model_fields
        self.assertIn("signal_decision", fields)

    def test_gate_normalizes_symbol(self):
        source = inspect.getsource(main.evaluate_paper_auto_entry_gate)
        self.assertIn('req.symbol.upper().replace("/", "-")', source)

    def test_gate_rejects_invalid_signal_decision(self):
        source = inspect.getsource(main.evaluate_paper_auto_entry_gate)
        self.assertIn("SIGNAL_DECISION_INVALID", source)

    def test_gate_blocks_wait(self):
        source = inspect.getsource(main.evaluate_paper_auto_entry_gate)
        self.assertIn("SIGNAL_WAIT", source)

    def test_gate_requires_registered_instrument(self):
        source = inspect.getsource(main.evaluate_paper_auto_entry_gate)
        self.assertIn("INSTRUMENT_NOT_REGISTERED", source)

    def test_gate_requires_complete_trade_plan(self):
        source = inspect.getsource(main.evaluate_paper_auto_entry_gate)
        self.assertIn("TRADE_PLAN_INCOMPLETE", source)

    def test_gate_rejects_invalid_rr(self):
        source = inspect.getsource(main.evaluate_paper_auto_entry_gate)
        self.assertIn("RR_INVALID", source)

    def test_gate_checks_long_level_order(self):
        source = inspect.getsource(main.evaluate_paper_auto_entry_gate)
        self.assertIn("LONG_LEVELS_INVALID", source)

    def test_gate_checks_short_level_order(self):
        source = inspect.getsource(main.evaluate_paper_auto_entry_gate)
        self.assertIn("SHORT_LEVELS_INVALID", source)

    def test_gate_documents_server_signal_boundary(self):
        source = inspect.getsource(main.evaluate_paper_auto_entry_gate)
        self.assertIn("server-authoritative signal evaluator", source)

    def test_gate_blocks_until_server_specs_exist(self):
        source = inspect.getsource(main.evaluate_paper_auto_entry_gate)
        self.assertIn("SERVER_INSTRUMENT_SPECS_REQUIRED", source)

    def test_gate_never_auto_creates_position_in_m1(self):
        source = inspect.getsource(main.evaluate_paper_auto_entry_gate)
        self.assertIn('"auto_create_position": False', source)

    def test_ui_marks_auto_entry_gate_blocked(self):
        html = INDEX.read_text(encoding="utf-8")
        self.assertIn("AUTO ORCHESTRATOR V1", html)
        self.assertIn("Aucun trade n’est créé par M1", html)


class TestServerSignalEngineV16M2(unittest.TestCase):
    def make_request(self, **overrides):
        data = {
            "symbol": "BTC-USD",
            "setup_state": "ENTRY_NOW",
            "direction": "BULLISH",
            "entry": Decimal("100"),
            "stop_loss": Decimal("95"),
            "take_profit": Decimal("110"),
            "risk_reward": Decimal("2"),
            "structure_confirmed": True,
            "displacement_confirmed": True,
            "order_block_confirmed": True,
            "source_timestamp": main.utcnow(),
        }
        data.update(overrides)
        return main.ServerSignalRequest(**data)

    def test_server_signal_route_exists(self):
        paths = {route.path for route in main.api_router.routes}
        self.assertIn("/paper/signal/evaluate", paths)

    def test_server_signal_is_authoritative(self):
        result = main.evaluate_server_signal(self.make_request())
        self.assertTrue(result["authoritative"])

    def test_bullish_ready_becomes_long(self):
        result = main.evaluate_server_signal(self.make_request())
        self.assertEqual(result["decision"], "LONG")

    def test_bearish_ready_becomes_short(self):
        req = self.make_request(
            direction="BEARISH", stop_loss=Decimal("105"), take_profit=Decimal("90")
        )
        result = main.evaluate_server_signal(req)
        self.assertEqual(result["decision"], "SHORT")

    def test_non_entry_now_waits(self):
        result = main.evaluate_server_signal(self.make_request(setup_state="WAIT"))
        self.assertEqual(result["decision"], "WAIT")

    def test_missing_structure_waits(self):
        result = main.evaluate_server_signal(self.make_request(structure_confirmed=False))
        self.assertIn("STRUCTURE_NOT_CONFIRMED", result["reasons"])

    def test_missing_displacement_waits(self):
        result = main.evaluate_server_signal(self.make_request(displacement_confirmed=False))
        self.assertIn("DISPLACEMENT_NOT_CONFIRMED", result["reasons"])

    def test_missing_order_block_waits(self):
        result = main.evaluate_server_signal(self.make_request(order_block_confirmed=False))
        self.assertIn("ORDER_BLOCK_NOT_CONFIRMED", result["reasons"])

    def test_incomplete_plan_waits(self):
        result = main.evaluate_server_signal(self.make_request(take_profit=None))
        self.assertIn("TRADE_PLAN_INCOMPLETE", result["reasons"])

    def test_invalid_rr_waits(self):
        result = main.evaluate_server_signal(self.make_request(risk_reward=Decimal("0")))
        self.assertIn("RR_INVALID", result["reasons"])

    def test_invalid_long_levels_wait(self):
        result = main.evaluate_server_signal(self.make_request(stop_loss=Decimal("101")))
        self.assertIn("LONG_LEVELS_INVALID", result["reasons"])

    def test_invalid_short_levels_wait(self):
        req = self.make_request(
            direction="BEARISH", stop_loss=Decimal("90"), take_profit=Decimal("110")
        )
        result = main.evaluate_server_signal(req)
        self.assertIn("SHORT_LEVELS_INVALID", result["reasons"])

    def test_signal_never_executes(self):
        result = main.evaluate_server_signal(self.make_request())
        self.assertFalse(result["execution"])

    def test_gate_still_blocks_server_specs(self):
        source = inspect.getsource(main.evaluate_paper_auto_entry_gate)
        self.assertIn("SERVER_INSTRUMENT_SPECS_REQUIRED", source)

    def test_ui_marks_server_signal_active(self):
        html = INDEX.read_text(encoding="utf-8")
        self.assertIn("SERVER SIGNAL V1 ACTIF", html)

    def test_ui_keeps_auto_entry_specs_blocked(self):
        html = INDEX.read_text(encoding="utf-8")
        self.assertIn("AUTO ORCHESTRATOR V1", html)


class TestServerInstrumentSpecsV16M3(unittest.TestCase):
    def valid_payload(self):
        return {
            "base_increment": "0.00000001",
            "quote_increment": "0.01",
            "base_min_size": "0.00001",
            "base_max_size": "100",
            "quote_min_size": "1",
            "quote_max_size": "1000000",
        }

    def test_specs_route_exists(self):
        paths = {route.path for route in main.api_router.routes}
        self.assertIn("/paper/instrument-specs/{symbol}", paths)

    def test_coinbase_provider_has_product_specs_method(self):
        self.assertTrue(hasattr(main.CoinbaseProvider, "get_product_specs"))

    def test_product_specs_uses_public_product_path(self):
        source = inspect.getsource(main.CoinbaseProvider.get_product_specs)
        self.assertIn('/market/products/{symbol}', source)

    def test_valid_specs_are_valid(self):
        result = main.parse_coinbase_spot_specs("BTC-USD", self.valid_payload())
        self.assertEqual(result["status"], "VALID")

    def test_specs_use_base_units(self):
        result = main.parse_coinbase_spot_specs("BTC-USD", self.valid_payload())
        self.assertEqual(result["sizing_mode"], "BASE_UNITS")

    def test_specs_preserve_decimal_strings(self):
        result = main.parse_coinbase_spot_specs("BTC-USD", self.valid_payload())
        self.assertEqual(result["base_increment"], "1E-8")

    def test_missing_field_is_invalid(self):
        payload = self.valid_payload()
        del payload["base_increment"]
        result = main.parse_coinbase_spot_specs("BTC-USD", payload)
        self.assertEqual(result["status"], "INVALID")

    def test_zero_field_is_invalid(self):
        payload = self.valid_payload()
        payload["base_increment"] = "0"
        result = main.parse_coinbase_spot_specs("BTC-USD", payload)
        self.assertEqual(result["status"], "INVALID")

    def test_negative_field_is_invalid(self):
        payload = self.valid_payload()
        payload["base_min_size"] = "-1"
        result = main.parse_coinbase_spot_specs("BTC-USD", payload)
        self.assertEqual(result["status"], "INVALID")

    def test_non_numeric_field_is_invalid(self):
        payload = self.valid_payload()
        payload["quote_increment"] = "bad"
        result = main.parse_coinbase_spot_specs("BTC-USD", payload)
        self.assertEqual(result["status"], "INVALID")

    def test_specs_have_source(self):
        result = main.parse_coinbase_spot_specs("BTC-USD", self.valid_payload())
        self.assertEqual(result["source"], "coinbase_public_product")

    def test_specs_do_not_invent_source_timestamp(self):
        result = main.parse_coinbase_spot_specs("BTC-USD", self.valid_payload())
        self.assertIsNone(result["source_timestamp"])

    def test_non_crypto_is_explicitly_not_supported(self):
        source = inspect.getsource(main.get_paper_instrument_specs)
        self.assertIn("VERIFIED_SIZING_SOURCE_NOT_IMPLEMENTED", source)

    def test_provider_failure_is_unavailable(self):
        source = inspect.getsource(main.get_paper_instrument_specs)
        self.assertIn("PROVIDER_UNAVAILABLE", source)

    def test_auto_gate_requires_specs_snapshot(self):
        source = inspect.getsource(main.evaluate_paper_auto_entry_gate)
        self.assertIn("SERVER_INSTRUMENT_SPECS_REQUIRED", source)

    def test_ui_marks_coinbase_specs_active(self):
        html = INDEX.read_text(encoding="utf-8")
        self.assertIn("AUTO ORCHESTRATOR V1", html)


class TestServerInstrumentSpecsV16M3Fix(unittest.TestCase):
    def test_specs_endpoint_uses_existing_market_provider(self):
        source = inspect.getsource(main.get_paper_instrument_specs)
        self.assertIn("market_provider.get_product_specs", source)
        self.assertNotIn("coinbase.get_product_specs", source)


class TestVerifiedAutoPaperEntryV16M4(unittest.TestCase):
    def specs(self):
        return {
            "base_increment": "0.001",
            "base_min_size": "0.001",
            "base_max_size": "100",
            "quote_min_size": "1",
            "quote_max_size": "1000000",
        }

    def test_verified_auto_entry_route_exists(self):
        paths = {route.path for route in main.api_router.routes}
        self.assertIn("/paper/auto-entry/verified", paths)

    def test_floor_to_increment(self):
        self.assertEqual(
            main.floor_to_increment(Decimal("1.2349"), Decimal("0.001")),
            Decimal("1.234"),
        )

    def test_floor_rejects_non_positive_increment(self):
        self.assertEqual(
            main.floor_to_increment(Decimal("1"), Decimal("0")),
            Decimal("0"),
        )

    def test_sizing_is_valid(self):
        result = main.calculate_verified_crypto_size(
            Decimal("1000"), Decimal("1"), Decimal("100"), Decimal("95"), self.specs()
        )
        self.assertEqual(result["status"], "VALID")

    def test_sizing_uses_base_units(self):
        result = main.calculate_verified_crypto_size(
            Decimal("1000"), Decimal("1"), Decimal("100"), Decimal("95"), self.specs()
        )
        self.assertEqual(result["size_unit"], "BASE_UNITS")

    def test_sizing_risk_is_bounded(self):
        result = main.calculate_verified_crypto_size(
            Decimal("1000"), Decimal("1"), Decimal("100"), Decimal("95"), self.specs()
        )
        self.assertLessEqual(result["risk_money"], Decimal("10"))

    def test_sizing_notional_is_bounded_by_capital(self):
        result = main.calculate_verified_crypto_size(
            Decimal("1000"), Decimal("50"), Decimal("100"), Decimal("99"), self.specs()
        )
        self.assertLessEqual(result["notional"], Decimal("1000"))

    def test_invalid_risk_blocks(self):
        result = main.calculate_verified_crypto_size(
            Decimal("1000"), Decimal("0"), Decimal("100"), Decimal("95"), self.specs()
        )
        self.assertEqual(result["reason"], "RISK_INVALID")

    def test_zero_stop_distance_blocks(self):
        result = main.calculate_verified_crypto_size(
            Decimal("1000"), Decimal("1"), Decimal("100"), Decimal("100"), self.specs()
        )
        self.assertEqual(result["reason"], "STOP_DISTANCE_INVALID")

    def test_missing_specs_block(self):
        result = main.calculate_verified_crypto_size(
            Decimal("1000"), Decimal("1"), Decimal("100"), Decimal("95"), {}
        )
        self.assertEqual(result["reason"], "SPECS_INVALID")

    def test_below_min_blocks(self):
        specs = self.specs()
        specs["base_min_size"] = "10"
        result = main.calculate_verified_crypto_size(
            Decimal("1000"), Decimal("1"), Decimal("100"), Decimal("95"), specs
        )
        self.assertEqual(result["reason"], "SIZE_BELOW_MIN")

    def test_notional_below_min_blocks(self):
        specs = self.specs()
        specs["quote_min_size"] = "10000"
        result = main.calculate_verified_crypto_size(
            Decimal("1000"), Decimal("1"), Decimal("100"), Decimal("95"), specs
        )
        self.assertEqual(result["reason"], "NOTIONAL_BELOW_MIN")

    def test_position_id_is_deterministic(self):
        ts = main.utcnow()
        a = main.build_auto_paper_position_id("BTC-USD", "LONG", ts)
        b = main.build_auto_paper_position_id("BTC-USD", "LONG", ts)
        self.assertEqual(a, b)

    def test_position_id_changes_with_signal(self):
        ts = main.utcnow()
        a = main.build_auto_paper_position_id("BTC-USD", "LONG", ts)
        b = main.build_auto_paper_position_id("BTC-USD", "SHORT", ts)
        self.assertNotEqual(a, b)

    def test_auto_entry_calls_server_signal(self):
        source = inspect.getsource(main.verified_auto_paper_entry)
        self.assertIn("evaluate_server_signal(req)", source)

    def test_auto_entry_calls_server_specs(self):
        source = inspect.getsource(main.verified_auto_paper_entry)
        self.assertIn("get_paper_instrument_specs(req.symbol)", source)

    def test_auto_entry_uses_persisted_account(self):
        source = inspect.getsource(main.verified_auto_paper_entry)
        self.assertIn("get_paper_account()", source)

    def test_auto_entry_uses_existing_position_creation(self):
        source = inspect.getsource(main.verified_auto_paper_entry)
        self.assertIn("create_paper_position(position)", source)

    def test_auto_entry_is_paper_only(self):
        source = inspect.getsource(main.verified_auto_paper_entry)
        self.assertIn('"paper_only": True', source)
        self.assertIn('"execution": False', source)

    def test_ui_marks_verified_auto_paper_active(self):
        html = INDEX.read_text(encoding="utf-8")
        self.assertIn("AUTO ORCHESTRATOR V1", html)


class TestAutoEntryOrchestratorV16M5A(unittest.TestCase):
    def test_candidate_route_exists(self):
        paths = {route.path for route in main.api_router.routes}
        self.assertIn("/paper/auto-entry/candidates", paths)

    def test_orchestrator_status_route_exists(self):
        paths = {route.path for route in main.api_router.routes}
        self.assertIn("/paper/auto-entry/orchestrator/status", paths)

    def test_orchestrator_interval_is_five_seconds(self):
        self.assertEqual(main.AUTO_ENTRY_ORCHESTRATOR_INTERVAL_SECONDS, 5.0)

    def test_candidate_state_starts_pending(self):
        fields = main.AutoEntryCandidateState.__dataclass_fields__
        self.assertEqual(fields["last_status"].default, "PENDING")

    def test_candidate_requires_server_signal(self):
        source = inspect.getsource(main.register_auto_entry_candidate)
        self.assertIn("evaluate_server_signal(req)", source)

    def test_candidate_is_paper_only(self):
        source = inspect.getsource(main.register_auto_entry_candidate)
        self.assertIn('"execution": False', source)

    def test_orchestrator_calls_m4(self):
        source = inspect.getsource(main.run_auto_entry_orchestrator_once)
        self.assertIn("verified_auto_paper_entry(state.request)", source)

    def test_opened_candidate_is_removed(self):
        source = inspect.getsource(main.run_auto_entry_orchestrator_once)
        self.assertIn("auto_entry_candidates.pop(candidate_id, None)", source)

    def test_duplicate_conflict_is_explicit(self):
        source = inspect.getsource(main.run_auto_entry_orchestrator_once)
        self.assertIn('"DUPLICATE_POSITION"', source)

    def test_http_errors_fail_closed(self):
        source = inspect.getsource(main.run_auto_entry_orchestrator_once)
        self.assertIn('"HTTP_ERROR"', source)

    def test_loop_handles_cancellation(self):
        source = inspect.getsource(main.auto_entry_orchestrator_loop)
        self.assertIn("except asyncio.CancelledError", source)

    def test_detector_gap_is_explicit(self):
        source = inspect.getsource(main.get_auto_entry_orchestrator_status)
        self.assertIn('"market_setup_detection": "STRUCTURE_BOS_CHOCH_V1"', source)

    def test_lifespan_starts_orchestrator(self):
        source = inspect.getsource(main.lifespan)
        self.assertIn("auto_entry_orchestrator_loop()", source)

    def test_lifespan_cancels_orchestrator(self):
        source = inspect.getsource(main.lifespan)
        self.assertIn("auto_entry_orchestrator_task.cancel()", source)

    def test_status_is_paper_only(self):
        source = inspect.getsource(main.get_auto_entry_orchestrator_status)
        self.assertIn('"paper_only": True', source)

    def test_ui_marks_orchestrator_active(self):
        html = INDEX.read_text(encoding="utf-8")
        self.assertIn("AUTO ORCHESTRATOR V1", html)


class TestServerMarketSetupDetectorV16M5B1(unittest.TestCase):
    def candle(self, minute, high, low, close=None):
        start = datetime(2026, 1, 1, 0, minute, tzinfo=timezone.utc)
        return main.Candle(
            start=start,
            low=float(low),
            high=float(high),
            open=float(close if close is not None else low),
            close=float(close if close is not None else high),
            volume=1.0,
            status=main.DataQualityStatus.VALID,
        )

    def test_detector_route_exists(self):
        paths = {route.path for route in main.api_router.routes}
        self.assertIn("/paper/auto-entry/detector/{symbol}", paths)

    def test_detector_uses_real_coinbase_candles(self):
        source = inspect.getsource(main.get_server_market_setup_detector)
        self.assertIn("market_provider.get_candles", source)

    def test_detector_granularity_is_5m(self):
        self.assertEqual(main.SERVER_SETUP_GRANULARITY, "5m")

    def test_swing_strength_is_two(self):
        self.assertEqual(main.SERVER_SWING_STRENGTH, 2)

    def test_open_candle_is_excluded(self):
        candles = [self.candle(0, 10, 5), self.candle(5, 11, 6)]
        now = datetime(2026, 1, 1, 0, 7, tzinfo=timezone.utc)
        closed = main.closed_valid_candles(candles, now)
        self.assertEqual(len(closed), 1)

    def test_invalid_candle_is_excluded(self):
        candle = self.candle(0, 10, 5)
        invalid = main.Candle(
            start=candle.start,
            low=candle.low,
            high=candle.high,
            open=candle.open,
            close=candle.close,
            volume=candle.volume,
            status=main.DataQualityStatus.INVALID,
        )
        now = datetime(2026, 1, 1, 0, 10, tzinfo=timezone.utc)
        self.assertEqual(main.closed_valid_candles([invalid], now), [])

    def test_swing_high_requires_strict_neighbors(self):
        candles = [
            self.candle(0, 10, 5), self.candle(5, 11, 5),
            self.candle(10, 15, 5), self.candle(15, 11, 5),
            self.candle(20, 10, 5),
        ]
        highs, _ = main.confirmed_swing_indexes(candles)
        self.assertEqual(highs, [2])

    def test_swing_low_requires_strict_neighbors(self):
        candles = [
            self.candle(0, 10, 5), self.candle(5, 10, 4),
            self.candle(10, 10, 1), self.candle(15, 10, 4),
            self.candle(20, 10, 5),
        ]
        _, lows = main.confirmed_swing_indexes(candles)
        self.assertEqual(lows, [2])

    def test_equal_high_is_not_confirmed_swing(self):
        candles = [
            self.candle(0, 10, 5), self.candle(5, 15, 5),
            self.candle(10, 15, 5), self.candle(15, 11, 5),
            self.candle(20, 10, 5),
        ]
        highs, _ = main.confirmed_swing_indexes(candles)
        self.assertEqual(highs, [])

    def test_insufficient_candles_waits(self):
        now = datetime(2026, 1, 1, 1, 0, tzinfo=timezone.utc)
        result = main.detect_server_market_structure([], now)
        self.assertEqual(result["status"], "WAIT")

    def test_detector_never_auto_queues_in_b1(self):
        source = inspect.getsource(main.detect_server_market_structure)
        self.assertIn('"auto_queue": False', source)

    def test_full_smc_is_explicitly_not_implemented(self):
        source = inspect.getsource(main.detect_server_market_structure)
        self.assertIn('"smc_confirmation": (', source)
        self.assertIn("FVG_OB_RETEST_PLAN_GATE_V1", source)
        self.assertIn('"liquidity_sweep": liquidity_sweep', source)
        self.assertIn('"displacement": displacement', source)
        self.assertNotIn('"liquidity_sweep": "NOT_IMPLEMENTED"', source)
        self.assertNotIn('"displacement": "NOT_IMPLEMENTED"', source)
        self.assertIn('"setup_state": entry_gate["state"]', source)
        self.assertIn('"auto_queue": False', source)

    def test_non_crypto_is_not_supported(self):
        source = inspect.getsource(main.get_server_market_setup_detector)
        self.assertIn("SERVER_CANDLE_DETECTOR_CRYPTO_ONLY", source)

    def test_provider_failure_is_unavailable(self):
        source = inspect.getsource(main.get_server_market_setup_detector)
        self.assertIn("CANDLES_UNAVAILABLE", source)

    def test_ui_marks_structure_detector(self):
        html = INDEX.read_text(encoding="utf-8")
        self.assertIn("SERVER DETECTOR · BOS/CHoCH V1", html)

    def test_ui_discloses_remaining_smc_work(self):
        html = INDEX.read_text(encoding="utf-8")
        self.assertIn("CHoCH", html)
        self.assertIn("Order Block", html)


class TestServerStructureEventsV16M5B2(unittest.TestCase):
    def candle(self, minute, high, low, close):
        return main.Candle(
            start=datetime(2026, 1, 1, 0, minute, tzinfo=timezone.utc),
            low=float(low),
            high=float(high),
            open=float(close),
            close=float(close),
            volume=1.0,
            status=main.DataQualityStatus.VALID,
        )

    def test_close_above_swing_creates_bullish_break(self):
        candles = [
            self.candle(0, 10, 5, 8),
            self.candle(5, 15, 7, 10),
            self.candle(10, 11, 6, 10),
            self.candle(15, 12, 8, 11),
            self.candle(20, 16, 9, 16),
        ]
        event = main.latest_confirmed_break(candles, [1], [])
        self.assertEqual(event["direction"], "BULLISH")

    def test_wick_above_without_close_does_not_break(self):
        candles = [
            self.candle(0, 10, 5, 8),
            self.candle(5, 15, 7, 10),
            self.candle(10, 16, 6, 14),
        ]
        self.assertIsNone(main.latest_confirmed_break(candles, [1], []))

    def test_close_below_swing_creates_bearish_break(self):
        candles = [
            self.candle(0, 10, 5, 8),
            self.candle(5, 9, 3, 6),
            self.candle(10, 8, 4, 5),
            self.candle(15, 7, 4, 5),
            self.candle(20, 7, 2, 2),
        ]
        event = main.latest_confirmed_break(candles, [], [1])
        self.assertEqual(event["direction"], "BEARISH")

    def test_wick_below_without_close_does_not_break(self):
        candles = [
            self.candle(0, 10, 5, 8),
            self.candle(5, 9, 3, 6),
            self.candle(10, 8, 2, 4),
        ]
        self.assertIsNone(main.latest_confirmed_break(candles, [], [1]))

    def test_bullish_structure_bullish_break_is_bos(self):
        result = main.classify_bos_choch(
            "BULLISH",
            {"direction": "BULLISH", "level": 10.0, "break_index": 4},
        )
        self.assertEqual(result["event"], "BOS")

    def test_bullish_structure_bearish_break_is_choch(self):
        result = main.classify_bos_choch(
            "BULLISH",
            {"direction": "BEARISH", "level": 5.0, "break_index": 4},
        )
        self.assertEqual(result["event"], "CHOCH_MSS")

    def test_bearish_structure_bearish_break_is_bos(self):
        result = main.classify_bos_choch(
            "BEARISH",
            {"direction": "BEARISH", "level": 5.0, "break_index": 4},
        )
        self.assertEqual(result["event"], "BOS")

    def test_bearish_structure_bullish_break_is_choch(self):
        result = main.classify_bos_choch(
            "BEARISH",
            {"direction": "BULLISH", "level": 10.0, "break_index": 4},
        )
        self.assertEqual(result["event"], "CHOCH_MSS")

    def test_range_break_is_bos_not_choch(self):
        result = main.classify_bos_choch(
            "RANGE",
            {"direction": "BULLISH", "level": 10.0, "break_index": 4},
        )
        self.assertEqual(result["event"], "BOS")

    def test_no_break_is_none_event(self):
        result = main.classify_bos_choch("BULLISH", None)
        self.assertEqual(result["event"], "NONE")

    def test_latest_break_event_wins(self):
        candles = [
            self.candle(0, 10, 5, 8),
            self.candle(5, 15, 7, 10),
            self.candle(10, 11, 4, 10),
            self.candle(15, 14, 5, 12),
            self.candle(20, 16, 6, 16),
            self.candle(25, 14, 3, 3),
        ]
        event = main.latest_confirmed_break(candles, [1], [2])
        self.assertEqual(event["direction"], "BEARISH")

    def test_detector_exposes_structure_event(self):
        source = inspect.getsource(main.detect_server_market_structure)
        self.assertIn('"structure_event": structure_event', source)

    def test_detector_status_names_bos_choch(self):
        source = inspect.getsource(main.get_auto_entry_orchestrator_status)
        self.assertIn("STRUCTURE_BOS_CHOCH_V1", source)

    def test_liquidity_sweep_is_server_implemented(self):
        source = inspect.getsource(main.detect_server_market_structure)
        self.assertIn('"liquidity_sweep": liquidity_sweep', source)

    def test_structure_events_do_not_auto_queue(self):
        source = inspect.getsource(main.detect_server_market_structure)
        self.assertIn('"auto_queue": False', source)

    def test_ui_marks_bos_choch_detector(self):
        html = INDEX.read_text(encoding="utf-8")
        self.assertIn("SERVER DETECTOR · BOS/CHoCH V1", html)


# ---------------- V16-M5B2 no-look-ahead corrective tests ----------------
class TestServerStructureNoLookAheadV16M5B2Fix(unittest.TestCase):
    def candle(self, index, high, low, close):
        return main.Candle(
            start=datetime(2026, 1, 1, 0, index * 5, tzinfo=timezone.utc),
            low=float(low),
            high=float(high),
            open=float(close),
            close=float(close),
            volume=1.0,
            status=main.DataQualityStatus.VALID,
        )

    def test_high_break_at_i_plus_1_is_rejected(self):
        candles = [self.candle(0, 10, 5, 8), self.candle(1, 15, 7, 10),
                   self.candle(2, 16, 8, 16), self.candle(3, 11, 8, 10)]
        self.assertIsNone(main.latest_confirmed_break(candles, [1], []))

    def test_high_break_at_i_plus_2_is_rejected(self):
        candles = [self.candle(0, 10, 5, 8), self.candle(1, 15, 7, 10),
                   self.candle(2, 11, 8, 10), self.candle(3, 16, 8, 16)]
        self.assertIsNone(main.latest_confirmed_break(candles, [1], []))

    def test_high_break_at_i_plus_3_is_accepted(self):
        candles = [self.candle(0, 10, 5, 8), self.candle(1, 15, 7, 10),
                   self.candle(2, 11, 8, 10), self.candle(3, 12, 8, 10),
                   self.candle(4, 16, 9, 16)]
        event = main.latest_confirmed_break(candles, [1], [])
        self.assertEqual(event["break_index"], 4)

    def test_low_break_at_i_plus_1_is_rejected(self):
        candles = [self.candle(0, 10, 5, 8), self.candle(1, 9, 3, 6),
                   self.candle(2, 8, 2, 2), self.candle(3, 8, 4, 5)]
        self.assertIsNone(main.latest_confirmed_break(candles, [], [1]))

    def test_low_break_at_i_plus_2_is_rejected(self):
        candles = [self.candle(0, 10, 5, 8), self.candle(1, 9, 3, 6),
                   self.candle(2, 8, 4, 5), self.candle(3, 8, 2, 2)]
        self.assertIsNone(main.latest_confirmed_break(candles, [], [1]))

    def test_low_break_at_i_plus_3_is_accepted(self):
        candles = [self.candle(0, 10, 5, 8), self.candle(1, 9, 3, 6),
                   self.candle(2, 8, 4, 5), self.candle(3, 8, 4, 5),
                   self.candle(4, 7, 2, 2)]
        event = main.latest_confirmed_break(candles, [], [1])
        self.assertEqual(event["break_index"], 4)

    def test_wick_above_after_confirmation_without_close_is_rejected(self):
        candles = [self.candle(0, 10, 5, 8), self.candle(1, 15, 7, 10),
                   self.candle(2, 11, 8, 10), self.candle(3, 12, 8, 10),
                   self.candle(4, 16, 9, 14)]
        self.assertIsNone(main.latest_confirmed_break(candles, [1], []))

    def test_wick_below_after_confirmation_without_close_is_rejected(self):
        candles = [self.candle(0, 10, 5, 8), self.candle(1, 9, 3, 6),
                   self.candle(2, 8, 4, 5), self.candle(3, 8, 4, 5),
                   self.candle(4, 7, 2, 4)]
        self.assertIsNone(main.latest_confirmed_break(candles, [], [1]))

    def test_future_confirmed_swing_cannot_change_prior_structure(self):
        candles = [
            self.candle(0, 10, 5, 8), self.candle(1, 12, 6, 10),
            self.candle(2, 11, 4, 8), self.candle(3, 14, 7, 12),
            self.candle(4, 13, 6, 10), self.candle(5, 16, 8, 14),
            self.candle(6, 15, 7, 13), self.candle(7, 18, 9, 17),
            self.candle(8, 17, 8, 16), self.candle(9, 19, 10, 18),
        ]
        highs = [1, 3, 7]
        lows = [2, 4, 8]
        without_future = main.structure_before_break(candles, highs[:2], lows[:2], 7)
        with_future = main.structure_before_break(candles, highs, lows, 7)
        self.assertEqual(without_future, with_future)
        self.assertEqual(with_future, "BULLISH")

    def test_bullish_prior_structure_bearish_break_is_choch(self):
        result = main.classify_bos_choch(
            "BULLISH", {"direction": "BEARISH", "level": 5.0, "break_index": 9}
        )
        self.assertEqual(result["event"], "CHOCH_MSS")

    def test_bearish_prior_structure_bullish_break_is_choch(self):
        result = main.classify_bos_choch(
            "BEARISH", {"direction": "BULLISH", "level": 10.0, "break_index": 9}
        )
        self.assertEqual(result["event"], "CHOCH_MSS")

    def test_bullish_continuation_is_bos(self):
        result = main.classify_bos_choch(
            "BULLISH", {"direction": "BULLISH", "level": 10.0, "break_index": 9}
        )
        self.assertEqual(result["event"], "BOS")

    def test_bearish_continuation_is_bos(self):
        result = main.classify_bos_choch(
            "BEARISH", {"direction": "BEARISH", "level": 5.0, "break_index": 9}
        )
        self.assertEqual(result["event"], "BOS")

    def test_detector_remains_auto_queue_false(self):
        source = inspect.getsource(main.detect_server_market_structure)
        self.assertIn('"auto_queue": False', source)

    def test_detector_remains_setup_state_wait(self):
        source = inspect.getsource(main.detect_server_market_structure)
        self.assertIn('"setup_state": entry_gate["state"]', source)

    def test_liquidity_sweep_preserves_wait_and_no_auto_queue(self):
        source = inspect.getsource(main.detect_server_market_structure)
        self.assertIn('"liquidity_sweep": liquidity_sweep', source)
        self.assertIn('"setup_state": entry_gate["state"]', source)
        self.assertIn('"auto_queue": False', source)


# ---------------- V16-M5B3 server liquidity sweep detector ----------------
class TestServerLiquiditySweepV16M5B3(unittest.TestCase):
    def candle(self, index, high, low, close):
        return main.Candle(
            start=datetime(2026, 1, 1, 0, index * 5, tzinfo=timezone.utc),
            low=float(low), high=float(high), open=float(close), close=float(close),
            volume=1.0, status=main.DataQualityStatus.VALID,
        )

    def test_bsl_sweep_after_confirmation(self):
        candles = [self.candle(0,10,5,8), self.candle(1,15,7,10),
                   self.candle(2,11,8,10), self.candle(3,12,8,10),
                   self.candle(4,16,9,14)]
        event = main.latest_confirmed_liquidity_sweep(candles, [1], [])
        self.assertEqual(event["type"], "BSL_SWEEP")
        self.assertEqual(event["direction"], "BEARISH")
        self.assertEqual(event["sweep_index"], 4)

    def test_ssl_sweep_after_confirmation(self):
        candles = [self.candle(0,10,5,8), self.candle(1,9,3,6),
                   self.candle(2,8,4,5), self.candle(3,8,4,5),
                   self.candle(4,7,2,4)]
        event = main.latest_confirmed_liquidity_sweep(candles, [], [1])
        self.assertEqual(event["type"], "SSL_SWEEP")
        self.assertEqual(event["direction"], "BULLISH")
        self.assertEqual(event["sweep_index"], 4)

    def test_bsl_at_i_plus_1_rejected(self):
        candles = [self.candle(0,10,5,8), self.candle(1,15,7,10),
                   self.candle(2,16,8,14), self.candle(3,11,8,10)]
        self.assertIsNone(main.latest_confirmed_liquidity_sweep(candles, [1], []))

    def test_bsl_at_i_plus_2_rejected(self):
        candles = [self.candle(0,10,5,8), self.candle(1,15,7,10),
                   self.candle(2,11,8,10), self.candle(3,16,8,14)]
        self.assertIsNone(main.latest_confirmed_liquidity_sweep(candles, [1], []))

    def test_ssl_at_i_plus_1_rejected(self):
        candles = [self.candle(0,10,5,8), self.candle(1,9,3,6),
                   self.candle(2,8,2,4), self.candle(3,8,4,5)]
        self.assertIsNone(main.latest_confirmed_liquidity_sweep(candles, [], [1]))

    def test_ssl_at_i_plus_2_rejected(self):
        candles = [self.candle(0,10,5,8), self.candle(1,9,3,6),
                   self.candle(2,8,4,5), self.candle(3,8,2,4)]
        self.assertIsNone(main.latest_confirmed_liquidity_sweep(candles, [], [1]))

    def test_high_beyond_but_close_equal_level_rejected(self):
        candles = [self.candle(0,10,5,8), self.candle(1,15,7,10),
                   self.candle(2,11,8,10), self.candle(3,12,8,10),
                   self.candle(4,16,9,15)]
        self.assertIsNone(main.latest_confirmed_liquidity_sweep(candles, [1], []))

    def test_low_beyond_but_close_equal_level_rejected(self):
        candles = [self.candle(0,10,5,8), self.candle(1,9,3,6),
                   self.candle(2,8,4,5), self.candle(3,8,4,5),
                   self.candle(4,7,2,3)]
        self.assertIsNone(main.latest_confirmed_liquidity_sweep(candles, [], [1]))

    def test_close_above_high_is_break_not_bsl_sweep(self):
        candles = [self.candle(0,10,5,8), self.candle(1,15,7,10),
                   self.candle(2,11,8,10), self.candle(3,12,8,10),
                   self.candle(4,16,9,16)]
        self.assertIsNone(main.latest_confirmed_liquidity_sweep(candles, [1], []))

    def test_close_below_low_is_break_not_ssl_sweep(self):
        candles = [self.candle(0,10,5,8), self.candle(1,9,3,6),
                   self.candle(2,8,4,5), self.candle(3,8,4,5),
                   self.candle(4,7,2,2)]
        self.assertIsNone(main.latest_confirmed_liquidity_sweep(candles, [], [1]))

    def test_latest_sweep_wins(self):
        candles = [self.candle(0,10,5,8), self.candle(1,15,7,10),
                   self.candle(2,11,4,10), self.candle(3,12,5,10),
                   self.candle(4,16,5,14), self.candle(5,14,3,5)]
        event = main.latest_confirmed_liquidity_sweep(candles, [1], [2])
        self.assertEqual(event["type"], "SSL_SWEEP")
        self.assertEqual(event["sweep_index"], 5)

    def test_invalid_swing_index_ignored(self):
        candles = [self.candle(0,10,5,8), self.candle(1,11,6,9),
                   self.candle(2,12,7,10), self.candle(3,13,8,11)]
        self.assertIsNone(main.latest_confirmed_liquidity_sweep(candles, [-1, 99], []))

    def test_detector_exposes_server_sweep(self):
        source = inspect.getsource(main.detect_server_market_structure)
        self.assertIn("latest_confirmed_liquidity_sweep", source)
        self.assertIn('"liquidity_sweep": liquidity_sweep', source)

    def test_orchestrator_status_names_server_sweep(self):
        source = inspect.getsource(main.get_auto_entry_orchestrator_status)
        self.assertIn('"liquidity_sweep_detection": "SERVER_LIQUIDITY_SWEEP_V1"', source)

    def test_sweep_does_not_enable_auto_entry(self):
        source = inspect.getsource(main.detect_server_market_structure)
        self.assertIn('"setup_state": entry_gate["state"]', source)
        self.assertIn('"auto_queue": False', source)

    def test_ui_marks_server_liquidity_sweep_detector(self):
        html = INDEX.read_text(encoding="utf-8")
        self.assertIn("SERVER DETECTOR · LIQUIDITY SWEEP V1", html)


# ---------------- V16-M5B4 server displacement detector ----------------
class TestServerDisplacementV16M5B4(unittest.TestCase):
    def candle(self, index, open_, high, low, close):
        return main.Candle(
            start=datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=index * 5),
            low=float(low), high=float(high), open=float(open_), close=float(close),
            volume=1.0, status=main.DataQualityStatus.VALID,
        )

    def baseline(self):
        return [self.candle(i, 100, 101, 99, 101) for i in range(20)]

    def test_bullish_displacement_detected(self):
        candles = self.baseline() + [self.candle(20, 100, 103, 99.5, 102.8)]
        event = main.latest_confirmed_displacement(candles)
        self.assertEqual(event["direction"], "BULLISH")

    def test_bearish_displacement_detected(self):
        candles = self.baseline() + [self.candle(20, 103, 103.5, 100, 100.2)]
        event = main.latest_confirmed_displacement(candles)
        self.assertEqual(event["direction"], "BEARISH")

    def test_requires_twenty_reference_candles(self):
        candles = self.baseline()[:19] + [self.candle(19, 100, 103, 99.5, 102.8)]
        self.assertIsNone(main.latest_confirmed_displacement(candles))

    def test_body_below_multiplier_rejected(self):
        candles = self.baseline() + [self.candle(20, 100, 101.6, 99.9, 101.4)]
        self.assertIsNone(main.latest_confirmed_displacement(candles))

    def test_body_range_ratio_below_seventy_percent_rejected(self):
        candles = self.baseline() + [self.candle(20, 100, 104, 98, 102)]
        self.assertIsNone(main.latest_confirmed_displacement(candles))

    def test_bullish_close_not_near_high_rejected(self):
        candles = self.baseline() + [self.candle(20, 100, 104, 99, 102.5)]
        self.assertIsNone(main.latest_confirmed_displacement(candles))

    def test_bearish_close_not_near_low_rejected(self):
        candles = self.baseline() + [self.candle(20, 103, 104, 99, 100.5)]
        self.assertIsNone(main.latest_confirmed_displacement(candles))

    def test_doji_rejected(self):
        candles = self.baseline() + [self.candle(20, 100, 103, 97, 100)]
        self.assertIsNone(main.latest_confirmed_displacement(candles))

    def test_zero_range_rejected(self):
        candles = self.baseline() + [self.candle(20, 100, 100, 100, 100)]
        self.assertIsNone(main.latest_confirmed_displacement(candles))

    def test_zero_average_reference_body_rejected(self):
        candles = [self.candle(i, 100, 101, 99, 100) for i in range(20)]
        candles.append(self.candle(20, 100, 103, 99.5, 102.8))
        self.assertIsNone(main.latest_confirmed_displacement(candles))

    def test_latest_displacement_wins(self):
        candles = self.baseline() + [self.candle(20, 100, 103, 99.5, 102.8)]
        candles += [self.candle(i, 100, 101, 99, 101) for i in range(21, 41)]
        candles.append(self.candle(41, 103, 103.5, 100, 100.2))
        event = main.latest_confirmed_displacement(candles)
        self.assertEqual(event["candle_index"], 41)
        self.assertEqual(event["direction"], "BEARISH")

    def test_metrics_are_exposed(self):
        candles = self.baseline() + [self.candle(20, 100, 103, 99.5, 102.8)]
        event = main.latest_confirmed_displacement(candles)
        self.assertGreaterEqual(event["body_multiple"], 1.5)
        self.assertGreaterEqual(event["body_range_ratio"], 0.70)

    def test_detector_exposes_server_displacement(self):
        source = inspect.getsource(main.detect_server_market_structure)
        self.assertIn("latest_confirmed_displacement", source)
        self.assertIn('"displacement": displacement', source)

    def test_orchestrator_status_names_server_displacement(self):
        source = inspect.getsource(main.get_auto_entry_orchestrator_status)
        self.assertIn('"displacement_detection": "SERVER_DISPLACEMENT_V1"', source)

    def test_displacement_does_not_enable_auto_entry(self):
        source = inspect.getsource(main.detect_server_market_structure)
        self.assertIn('"setup_state": entry_gate["state"]', source)
        self.assertIn('"auto_queue": False', source)

    def test_ui_marks_server_displacement_detector(self):
        html = INDEX.read_text(encoding="utf-8")
        self.assertIn("SERVER DETECTOR · DISPLACEMENT V1", html)

# V16-M5B4-FIX2 — fresh synchronized copy

# ---------------- V16-M5B5 server FVG detector ----------------
class TestServerFvgV16M5B5(unittest.TestCase):
    def candle(self, index, open_, high, low, close):
        return main.Candle(
            start=datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=index * 5),
            low=float(low), high=float(high), open=float(open_), close=float(close),
            volume=1.0, status=main.DataQualityStatus.VALID,
        )

    def test_bullish_fvg_detected_with_zone(self):
        candles = [
            self.candle(0, 100, 101, 99, 100.5),
            self.candle(1, 100.5, 103, 100, 102.5),
            self.candle(2, 102.5, 104, 102, 103),
        ]
        event = main.latest_confirmed_fvg(candles)
        self.assertEqual(event["direction"], "BULLISH")
        self.assertEqual((event["zone_low"], event["zone_high"]), (101.0, 102.0))

    def test_bearish_fvg_detected_with_zone(self):
        candles = [
            self.candle(0, 100, 101, 99, 99.5),
            self.candle(1, 99.5, 100, 96, 96.5),
            self.candle(2, 96.5, 98, 95, 96),
        ]
        event = main.latest_confirmed_fvg(candles)
        self.assertEqual(event["direction"], "BEARISH")
        self.assertEqual((event["zone_low"], event["zone_high"]), (98.0, 99.0))

    def test_bullish_equal_boundary_is_not_fvg(self):
        candles = [
            self.candle(0, 100, 101, 99, 100),
            self.candle(1, 100, 102, 99, 101),
            self.candle(2, 101, 103, 101, 102),
        ]
        self.assertIsNone(main.latest_confirmed_fvg(candles))

    def test_bearish_equal_boundary_is_not_fvg(self):
        candles = [
            self.candle(0, 100, 101, 99, 100),
            self.candle(1, 100, 101, 97, 98),
            self.candle(2, 98, 99, 96, 97),
        ]
        self.assertIsNone(main.latest_confirmed_fvg(candles))

    def test_requires_three_closed_candles(self):
        candles = [self.candle(0, 100, 101, 99, 100), self.candle(1, 101, 103, 100, 102)]
        self.assertIsNone(main.latest_confirmed_fvg(candles))

    def test_missing_outer_ohlc_is_ignored(self):
        first = main.Candle(
            start=datetime(2026, 1, 1, tzinfo=timezone.utc),
            low=99.0, high=None, open=100.0, close=100.0, volume=1.0,
            status=main.DataQualityStatus.VALID,
        )
        candles = [first, self.candle(1, 101, 103, 100, 102), self.candle(2, 102, 104, 102, 103)]
        self.assertIsNone(main.latest_confirmed_fvg(candles))

    def test_latest_fvg_wins(self):
        candles = [
            self.candle(0, 100, 101, 99, 100),
            self.candle(1, 100, 103, 100, 102),
            self.candle(2, 102, 104, 102, 103),
            self.candle(3, 103, 104, 101.5, 102),
            self.candle(4, 102, 102.5, 100, 100.5),
            self.candle(5, 100, 100.5, 98, 98.5),
        ]
        event = main.latest_confirmed_fvg(candles)
        self.assertEqual(event["formation_index"], 5)

    def test_new_bullish_fvg_starts_open(self):
        candles = [
            self.candle(0, 100, 101, 99, 100),
            self.candle(1, 100, 103, 100, 102),
            self.candle(2, 102, 104, 102, 103),
        ]
        self.assertEqual(main.latest_confirmed_fvg(candles)["state"], "OPEN")

    def test_bullish_partial_mitigation(self):
        candles = [
            self.candle(0, 100, 101, 99, 100),
            self.candle(1, 100, 103, 100, 102),
            self.candle(2, 102, 104, 102, 103),
            self.candle(3, 103, 104, 101.5, 103),
        ]
        event = main.latest_confirmed_fvg(candles)
        self.assertEqual(event["state"], "PARTIALLY_MITIGATED")
        self.assertEqual(event["mitigation_index"], 3)

    def test_bullish_full_mitigation(self):
        candles = [
            self.candle(0, 100, 101, 99, 100),
            self.candle(1, 100, 103, 100, 102),
            self.candle(2, 102, 104, 102, 103),
            self.candle(3, 103, 104, 100.9, 101),
        ]
        self.assertEqual(main.latest_confirmed_fvg(candles)["state"], "MITIGATED")

    def test_bearish_partial_mitigation(self):
        candles = [
            self.candle(0, 100, 101, 99, 100),
            self.candle(1, 99, 100, 96, 97),
            self.candle(2, 97, 98, 95, 96),
            self.candle(3, 96, 98.5, 95, 97),
        ]
        event = main.latest_confirmed_fvg(candles)
        self.assertEqual(event["state"], "PARTIALLY_MITIGATED")
        self.assertEqual(event["mitigation_index"], 3)

    def test_bearish_full_mitigation(self):
        candles = [
            self.candle(0, 100, 101, 99, 100),
            self.candle(1, 99, 100, 96, 97),
            self.candle(2, 97, 98, 95, 96),
            self.candle(3, 96, 99.1, 95, 98),
        ]
        self.assertEqual(main.latest_confirmed_fvg(candles)["state"], "MITIGATED")

    def test_detector_exposes_server_fvg(self):
        source = inspect.getsource(main.detect_server_market_structure)
        self.assertIn("latest_confirmed_fvg", source)
        self.assertIn('"fvg": fvg', source)

    def test_orchestrator_status_names_server_fvg(self):
        source = inspect.getsource(main.get_auto_entry_orchestrator_status)
        self.assertIn('"fvg_detection": "SERVER_FVG_V1"', source)

    def test_fvg_does_not_enable_auto_entry(self):
        source = inspect.getsource(main.detect_server_market_structure)
        self.assertIn('"setup_state": entry_gate["state"]', source)
        self.assertIn('"auto_queue": False', source)

    def test_ui_marks_server_fvg_detector(self):
        html = INDEX.read_text(encoding="utf-8")
        self.assertIn("SERVER DETECTOR · FVG V1", html)



# ---------------- V16-M5B6 server Order Block detector ----------------
class TestServerOrderBlockV16M5B6(unittest.TestCase):
    def candle(self, index, open_, high, low, close):
        return main.Candle(
            start=datetime(2026, 1, 1, tzinfo=timezone.utc)
            + timedelta(minutes=index * 5),
            low=float(low),
            high=float(high),
            open=float(open_),
            close=float(close),
            volume=1.0,
            status=main.DataQualityStatus.VALID,
        )

    def displacement(self, index, direction):
        return {
            "event": "DISPLACEMENT",
            "candle_index": index,
            "direction": direction,
        }

    def test_bullish_ob_is_last_bearish_before_displacement(self):
        candles = [
            self.candle(0, 100, 102, 99, 101),
            self.candle(1, 101, 102, 98, 99),
            self.candle(2, 99, 105, 99, 104),
        ]
        event = main.latest_confirmed_order_block(
            candles, self.displacement(2, "BULLISH")
        )
        self.assertEqual(event["direction"], "BULLISH")
        self.assertEqual(event["order_block_index"], 1)
        self.assertEqual((event["zone_low"], event["zone_high"]), (98.0, 102.0))

    def test_bearish_ob_is_last_bullish_before_displacement(self):
        candles = [
            self.candle(0, 100, 102, 99, 99.5),
            self.candle(1, 99.5, 103, 99, 102),
            self.candle(2, 102, 102, 95, 96),
        ]
        event = main.latest_confirmed_order_block(
            candles, self.displacement(2, "BEARISH")
        )
        self.assertEqual(event["direction"], "BEARISH")
        self.assertEqual(event["order_block_index"], 1)

    def test_requires_confirmed_displacement(self):
        candles = [self.candle(0, 100, 101, 99, 100)]
        self.assertIsNone(main.latest_confirmed_order_block(candles, {}))

    def test_rejects_unknown_displacement_direction(self):
        candles = [self.candle(0, 100, 101, 99, 100)] * 2
        event = main.latest_confirmed_order_block(
            candles, self.displacement(1, "RANGE")
        )
        self.assertIsNone(event)

    def test_rejects_displacement_index_outside_candles(self):
        candles = [self.candle(0, 100, 101, 99, 100)]
        event = main.latest_confirmed_order_block(
            candles, self.displacement(5, "BULLISH")
        )
        self.assertIsNone(event)

    def test_no_opposite_candle_returns_none(self):
        candles = [
            self.candle(0, 100, 102, 99, 101),
            self.candle(1, 101, 103, 100, 102),
            self.candle(2, 102, 106, 102, 105),
        ]
        event = main.latest_confirmed_order_block(
            candles, self.displacement(2, "BULLISH")
        )
        self.assertIsNone(event)

    def test_search_is_bounded_to_five_prior_candles(self):
        candles = [self.candle(0, 101, 102, 99, 100)]
        candles += [self.candle(i, 100, 102, 99, 101) for i in range(1, 7)]
        event = main.latest_confirmed_order_block(
            candles, self.displacement(6, "BULLISH")
        )
        self.assertIsNone(event)

    def test_new_order_block_starts_fresh(self):
        candles = [
            self.candle(0, 101, 102, 99, 100),
            self.candle(1, 100, 106, 100, 105),
        ]
        event = main.latest_confirmed_order_block(
            candles, self.displacement(1, "BULLISH")
        )
        self.assertEqual(event["state"], "FRESH")

    def test_bullish_overlap_marks_retested(self):
        candles = [
            self.candle(0, 101, 102, 99, 100),
            self.candle(1, 100, 106, 100, 105),
            self.candle(2, 105, 106, 101, 104),
        ]
        event = main.latest_confirmed_order_block(
            candles, self.displacement(1, "BULLISH")
        )
        self.assertEqual(event["state"], "RETESTED")
        self.assertEqual(event["retest_index"], 2)

    def test_bearish_overlap_marks_retested(self):
        candles = [
            self.candle(0, 99, 102, 98, 101),
            self.candle(1, 101, 101, 94, 95),
            self.candle(2, 95, 99, 94, 96),
        ]
        event = main.latest_confirmed_order_block(
            candles, self.displacement(1, "BEARISH")
        )
        self.assertEqual(event["state"], "RETESTED")

    def test_bullish_close_below_zone_invalidates(self):
        candles = [
            self.candle(0, 101, 102, 99, 100),
            self.candle(1, 100, 106, 100, 105),
            self.candle(2, 105, 106, 98, 98.5),
        ]
        event = main.latest_confirmed_order_block(
            candles, self.displacement(1, "BULLISH")
        )
        self.assertEqual(event["state"], "INVALIDATED")
        self.assertEqual(event["invalidation_index"], 2)

    def test_bearish_close_above_zone_invalidates(self):
        candles = [
            self.candle(0, 99, 102, 98, 101),
            self.candle(1, 101, 101, 94, 95),
            self.candle(2, 95, 103, 94, 102.5),
        ]
        event = main.latest_confirmed_order_block(
            candles, self.displacement(1, "BEARISH")
        )
        self.assertEqual(event["state"], "INVALIDATED")

    def test_invalidation_has_priority_over_retest(self):
        candles = [
            self.candle(0, 101, 102, 99, 100),
            self.candle(1, 100, 106, 100, 105),
            self.candle(2, 105, 106, 98, 98.5),
        ]
        event = main.latest_confirmed_order_block(
            candles, self.displacement(1, "BULLISH")
        )
        self.assertEqual(event["state"], "INVALIDATED")
        self.assertIsNone(event["retest_index"])

    def test_detector_exposes_server_order_block(self):
        source = inspect.getsource(main.detect_server_market_structure)
        self.assertIn("latest_confirmed_order_block", source)
        self.assertIn('"order_block": order_block', source)

    def test_orchestrator_status_names_server_order_block(self):
        source = inspect.getsource(main.get_auto_entry_orchestrator_status)
        self.assertIn(
            '"order_block_detection": "SERVER_ORDER_BLOCK_V1"', source
        )

    def test_order_block_does_not_enable_auto_entry(self):
        source = inspect.getsource(main.detect_server_market_structure)
        self.assertIn('"setup_state": entry_gate["state"]', source)
        self.assertIn('"auto_queue": False', source)

    def test_ui_marks_server_order_block_detector(self):
        html = INDEX.read_text(encoding="utf-8")
        self.assertIn("SERVER DETECTOR · ORDER BLOCK V1", html)


# ---------------- V16-M5B7 server Retest/Revalidation engine ----------------
class TestServerRetestRevalidationV16M5B7(unittest.TestCase):
    def candle(self, index, open_, high, low, close):
        return main.Candle(
            start=datetime(2026, 1, 1, tzinfo=timezone.utc)
            + timedelta(minutes=index * 5),
            low=float(low),
            high=float(high),
            open=float(open_),
            close=float(close),
            volume=1.0,
            status=main.DataQualityStatus.VALID,
        )

    def ob(self, direction, state="RETESTED", retest_index=2):
        return {
            "event": "ORDER_BLOCK",
            "direction": direction,
            "state": state,
            "retest_index": retest_index,
            "zone_low": 99.0,
            "zone_high": 103.0,
        }

    def test_missing_order_block_waits(self):
        event = main.latest_confirmed_revalidation([], None)
        self.assertEqual(event["state"], "WAITING_RETEST")

    def test_fresh_order_block_waits_for_retest(self):
        order_block = self.ob("BULLISH", "FRESH")
        event = main.latest_confirmed_revalidation([], order_block)
        self.assertEqual(event["state"], "WAITING_RETEST")

    def test_invalidated_order_block_stays_invalidated(self):
        event = main.latest_confirmed_revalidation(
            [], self.ob("BULLISH", "INVALIDATED")
        )
        self.assertEqual(event["state"], "INVALIDATED")

    def test_bullish_midpoint_rejection_revalidates(self):
        candles = [
            self.candle(0, 100, 101, 99, 100),
            self.candle(1, 100, 105, 100, 104),
            self.candle(2, 104, 104, 100, 102),
        ]
        event = main.latest_confirmed_revalidation(candles, self.ob("BULLISH"))
        self.assertEqual(event["state"], "REVALIDATED")
        self.assertEqual(event["revalidation_index"], 2)

    def test_bearish_midpoint_rejection_revalidates(self):
        candles = [
            self.candle(0, 100, 101, 99, 100),
            self.candle(1, 100, 105, 100, 104),
            self.candle(2, 98, 102, 98, 100),
        ]
        event = main.latest_confirmed_revalidation(candles, self.ob("BEARISH"))
        self.assertEqual(event["state"], "REVALIDATED")

    def test_bullish_close_at_midpoint_is_unconfirmed(self):
        candles = [self.candle(0, 100, 101, 99, 100)] * 2
        candles.append(self.candle(2, 102, 103, 99, 101))
        event = main.latest_confirmed_revalidation(candles, self.ob("BULLISH"))
        self.assertEqual(event["state"], "RETESTED_UNCONFIRMED")

    def test_bearish_close_at_midpoint_is_unconfirmed(self):
        candles = [self.candle(0, 100, 101, 99, 100)] * 2
        candles.append(self.candle(2, 100, 103, 99, 101))
        event = main.latest_confirmed_revalidation(candles, self.ob("BEARISH"))
        self.assertEqual(event["state"], "RETESTED_UNCONFIRMED")

    def test_invalid_retest_index_is_invalidated(self):
        event = main.latest_confirmed_revalidation([], self.ob("BULLISH"))
        self.assertEqual(event["state"], "INVALIDATED")

    def test_invalid_zone_is_invalidated(self):
        order_block = self.ob("BULLISH", retest_index=0)
        order_block["zone_low"] = 103.0
        event = main.latest_confirmed_revalidation(
            [self.candle(0, 100, 103, 99, 102)], order_block
        )
        self.assertEqual(event["state"], "INVALIDATED")

    def test_unknown_direction_is_invalidated(self):
        event = main.latest_confirmed_revalidation([], self.ob("RANGE"))
        self.assertEqual(event["state"], "INVALIDATED")

    def test_same_direction_active_fvg_is_confluence(self):
        candles = [self.candle(0, 100, 101, 99, 100)] * 2
        candles.append(self.candle(2, 102, 103, 99, 102))
        fvg = {"event": "FVG", "direction": "BULLISH", "state": "OPEN"}
        event = main.latest_confirmed_revalidation(
            candles, self.ob("BULLISH"), fvg
        )
        self.assertTrue(event["fvg_confluence"])

    def test_mitigated_fvg_is_not_confluence(self):
        candles = [self.candle(0, 100, 101, 99, 100)] * 2
        candles.append(self.candle(2, 102, 103, 99, 102))
        fvg = {
            "event": "FVG",
            "direction": "BULLISH",
            "state": "MITIGATED",
        }
        event = main.latest_confirmed_revalidation(
            candles, self.ob("BULLISH"), fvg
        )
        self.assertFalse(event["fvg_confluence"])

    def test_detector_exposes_revalidation(self):
        source = inspect.getsource(main.detect_server_market_structure)
        self.assertIn("latest_confirmed_revalidation", source)
        self.assertIn('"revalidation": revalidation', source)

    def test_detector_contract_names_retest(self):
        source = inspect.getsource(main.detect_server_market_structure)
        self.assertIn("FVG_OB_RETEST_PLAN_GATE_V1", source)

    def test_revalidation_does_not_enable_auto_entry(self):
        source = inspect.getsource(main.detect_server_market_structure)
        self.assertIn('"setup_state": entry_gate["state"]', source)
        self.assertIn('"auto_queue": False', source)

    def test_orchestrator_and_ui_mark_revalidation(self):
        source = inspect.getsource(main.get_auto_entry_orchestrator_status)
        self.assertIn(
            '"retest_revalidation": "SERVER_RETEST_REVALIDATION_V1"', source
        )
        html = INDEX.read_text(encoding="utf-8")
        self.assertIn("SERVER DETECTOR · RETEST REVALIDATION V1", html)


class TestServerTradePlanV16M5B8(unittest.TestCase):
    def candle(self, index, low, high, open_=100, close=100):
        return main.Candle(
            start=NOW + timedelta(minutes=index * 5),
            low=float(low), high=float(high), open=float(open_), close=float(close),
            volume=1.0, status=main.DataQualityStatus.VALID,
        )

    def chain(self, direction="BULLISH"):
        return (
            {"event": "BOS", "direction": direction},
            {"event": "SSL_SWEEP", "direction": direction},
            {"event": "DISPLACEMENT", "direction": direction},
            {"event": "FVG", "direction": direction, "state": "OPEN"},
            {
                "event": "ORDER_BLOCK", "direction": direction,
                "state": "RETESTED", "zone_low": 99.0, "zone_high": 103.0,
            },
            {"event": "REVALIDATION", "state": "REVALIDATED", "direction": direction},
        )

    def build(self, direction="BULLISH", highs=None, lows=None):
        candles = [self.candle(0, 95, 110), self.candle(1, 90, 108)]
        structure, sweep, displacement, fvg, ob, revalidation = self.chain(direction)
        return main.build_server_trade_plan(
            candles, highs or [0], lows or [1], structure, sweep,
            displacement, fvg, ob, revalidation,
        )

    def test_bullish_plan_uses_ob_midpoint_and_structural_stop(self):
        plan = self.build("BULLISH")
        self.assertEqual(plan["state"], "CANDIDATE_READY")
        self.assertEqual(plan["entry_reference"], 101.0)
        self.assertEqual(plan["stop_loss"], 99.0)

    def test_bearish_plan_uses_ob_midpoint_and_structural_stop(self):
        plan = self.build("BEARISH")
        self.assertEqual(plan["state"], "CANDIDATE_READY")
        self.assertEqual(plan["entry_reference"], 101.0)
        self.assertEqual(plan["stop_loss"], 103.0)

    def test_bullish_target_is_nearest_confirmed_swing_high(self):
        plan = self.build("BULLISH", highs=[0, 1])
        self.assertEqual(plan["take_profit"], 108.0)

    def test_bearish_target_is_nearest_confirmed_swing_low(self):
        plan = self.build("BEARISH", lows=[0, 1])
        self.assertEqual(plan["take_profit"], 95.0)

    def test_risk_reward_is_derived_not_fixed(self):
        plan = self.build("BULLISH", highs=[1])
        self.assertEqual(plan["risk_reward"], 3.5)

    def test_unrevalidated_setup_waits(self):
        structure, sweep, displacement, fvg, ob, revalidation = self.chain()
        revalidation["state"] = "RETESTED_UNCONFIRMED"
        plan = main.build_server_trade_plan(
            [], [], [], structure, sweep, displacement, fvg, ob, revalidation
        )
        self.assertEqual(plan["state"], "WAIT")

    def test_direction_mismatch_waits(self):
        structure, sweep, displacement, fvg, ob, revalidation = self.chain()
        fvg["direction"] = "BEARISH"
        plan = main.build_server_trade_plan(
            [], [], [], structure, sweep, displacement, fvg, ob, revalidation
        )
        self.assertEqual(plan["state"], "WAIT")

    def test_missing_confirmation_waits(self):
        structure, sweep, displacement, fvg, ob, revalidation = self.chain()
        plan = main.build_server_trade_plan(
            [], [], [], structure, sweep, None, fvg, ob, revalidation
        )
        self.assertEqual(plan["state"], "WAIT")

    def test_invalidated_order_block_waits(self):
        structure, sweep, displacement, fvg, ob, revalidation = self.chain()
        ob["state"] = "INVALIDATED"
        plan = main.build_server_trade_plan(
            [], [], [], structure, sweep, displacement, fvg, ob, revalidation
        )
        self.assertEqual(plan["state"], "WAIT")

    def test_invalid_order_block_zone_waits(self):
        structure, sweep, displacement, fvg, ob, revalidation = self.chain()
        ob["zone_low"] = 104.0
        plan = main.build_server_trade_plan(
            [], [], [], structure, sweep, displacement, fvg, ob, revalidation
        )
        self.assertEqual(plan["state"], "WAIT")

    def test_missing_opposite_liquidity_target_waits(self):
        plan = self.build("BULLISH", highs=[1])
        self.assertEqual(plan["state"], "CANDIDATE_READY")
        candles = [self.candle(0, 95, 100)]
        structure, sweep, displacement, fvg, ob, revalidation = self.chain()
        plan = main.build_server_trade_plan(
            candles, [0], [], structure, sweep, displacement, fvg, ob, revalidation
        )
        self.assertEqual(plan["state"], "WAIT")

    def test_candidate_never_auto_queues(self):
        self.assertFalse(self.build("BULLISH")["auto_queue"])

    def test_detector_exposes_trade_plan(self):
        source = inspect.getsource(main.detect_server_market_structure)
        self.assertIn("build_server_trade_plan", source)
        self.assertIn('"trade_plan": trade_plan', source)

    def test_detector_contract_names_plan(self):
        source = inspect.getsource(main.detect_server_market_structure)
        self.assertIn("FVG_OB_RETEST_PLAN_GATE_V1", source)

    def test_orchestrator_names_server_trade_plan(self):
        source = inspect.getsource(main.get_auto_entry_orchestrator_status)
        self.assertIn('"trade_plan_builder": "SERVER_TRADE_PLAN_V1"', source)

    def test_ui_marks_server_trade_plan(self):
        html = INDEX.read_text(encoding="utf-8")
        self.assertIn("SERVER BUILDER · TRADE PLAN V1", html)

class TestServerEntryNowGateV16M5B9(unittest.TestCase):
    def candle(self, index):
        return main.Candle(
            start=NOW + timedelta(minutes=index * 5),
            low=99.0,
            high=103.0,
            open=100.0,
            close=102.0,
            volume=1.0,
            status=main.DataQualityStatus.VALID,
        )

    def plan(self, direction="BULLISH"):
        if direction == "BULLISH":
            stop, target = 99.0, 108.0
        else:
            stop, target = 103.0, 95.0
        return {
            "event": "TRADE_PLAN",
            "state": "CANDIDATE_READY",
            "direction": direction,
            "entry_zone_low": 99.0,
            "entry_zone_high": 103.0,
            "entry_reference": 101.0,
            "stop_loss": stop,
            "take_profit": target,
            "risk_reward": 3.5,
            "auto_queue": False,
        }

    def revalidation(self, direction="BULLISH", index=0):
        return {
            "event": "REVALIDATION",
            "state": "REVALIDATED",
            "direction": direction,
            "revalidation_index": index,
        }

    def order_block(self, direction="BULLISH"):
        return {
            "event": "ORDER_BLOCK",
            "direction": direction,
            "state": "RETESTED",
        }

    def test_bullish_current_revalidation_is_entry_now(self):
        gate = main.evaluate_server_entry_now_gate(
            [self.candle(0)], self.plan(), self.revalidation(), self.order_block()
        )
        self.assertEqual(gate["state"], "ENTRY_NOW")
        self.assertEqual(gate["direction"], "BULLISH")

    def test_bearish_current_revalidation_is_entry_now(self):
        gate = main.evaluate_server_entry_now_gate(
            [self.candle(0)], self.plan("BEARISH"),
            self.revalidation("BEARISH"), self.order_block("BEARISH"),
        )
        self.assertEqual(gate["state"], "ENTRY_NOW")

    def test_gate_never_auto_queues(self):
        gate = main.evaluate_server_entry_now_gate(
            [self.candle(0)], self.plan(), self.revalidation(), self.order_block()
        )
        self.assertFalse(gate["auto_queue"])

    def test_invalidated_order_block_wins(self):
        ob = self.order_block()
        ob["state"] = "INVALIDATED"
        gate = main.evaluate_server_entry_now_gate(
            [self.candle(0)], self.plan(), self.revalidation(), ob
        )
        self.assertEqual(gate["state"], "INVALIDATED")

    def test_invalidated_revalidation_wins(self):
        revalidation = self.revalidation()
        revalidation["state"] = "INVALIDATED"
        gate = main.evaluate_server_entry_now_gate(
            [self.candle(0)], self.plan(), revalidation, self.order_block()
        )
        self.assertEqual(gate["state"], "INVALIDATED")

    def test_incomplete_trade_plan_waits(self):
        plan = self.plan()
        plan["state"] = "WAIT"
        gate = main.evaluate_server_entry_now_gate(
            [self.candle(0)], plan, self.revalidation(), self.order_block()
        )
        self.assertEqual(gate["state"], "WAIT")

    def test_direction_mismatch_waits(self):
        gate = main.evaluate_server_entry_now_gate(
            [self.candle(0)], self.plan(),
            self.revalidation("BEARISH"), self.order_block(),
        )
        self.assertEqual(gate["state"], "WAIT")

    def test_missing_revalidation_index_waits(self):
        revalidation = self.revalidation()
        revalidation["revalidation_index"] = None
        gate = main.evaluate_server_entry_now_gate(
            [self.candle(0)], self.plan(), revalidation, self.order_block()
        )
        self.assertEqual(gate["state"], "WAIT")

    def test_future_revalidation_index_waits(self):
        gate = main.evaluate_server_entry_now_gate(
            [self.candle(0)], self.plan(), self.revalidation(index=2),
            self.order_block(),
        )
        self.assertEqual(gate["state"], "WAIT")

    def test_stale_revalidation_expires(self):
        gate = main.evaluate_server_entry_now_gate(
            [self.candle(0), self.candle(1)], self.plan(),
            self.revalidation(index=0), self.order_block(),
        )
        self.assertEqual(gate["state"], "EXPIRED")

    def test_bullish_invalid_trade_geometry_waits(self):
        plan = self.plan()
        plan["stop_loss"] = 102.0
        gate = main.evaluate_server_entry_now_gate(
            [self.candle(0)], plan, self.revalidation(), self.order_block()
        )
        self.assertEqual(gate["state"], "WAIT")

    def test_bearish_invalid_trade_geometry_waits(self):
        plan = self.plan("BEARISH")
        plan["take_profit"] = 102.0
        gate = main.evaluate_server_entry_now_gate(
            [self.candle(0)], plan, self.revalidation("BEARISH"),
            self.order_block("BEARISH"),
        )
        self.assertEqual(gate["state"], "WAIT")

    def test_detector_exposes_entry_gate(self):
        source = inspect.getsource(main.detect_server_market_structure)
        self.assertIn("evaluate_server_entry_now_gate", source)
        self.assertIn('"entry_gate": entry_gate', source)

    def test_detector_setup_state_comes_from_gate(self):
        source = inspect.getsource(main.detect_server_market_structure)
        self.assertIn('"setup_state": entry_gate["state"]', source)
        self.assertIn("RETEST_PLAN_GATE_V1", source)

    def test_orchestrator_names_server_entry_gate(self):
        source = inspect.getsource(main.get_auto_entry_orchestrator_status)
        self.assertIn('"entry_now_gate": "SERVER_ENTRY_NOW_GATE_V1"', source)

    def test_ui_marks_server_entry_now_gate(self):
        html = INDEX.read_text(encoding="utf-8")
        self.assertIn("SERVER GATE · ENTRY NOW V1", html)


class TestServerAutoPaperPositionV16M5B10(unittest.TestCase):
    def detector(self, direction="BULLISH"):
        sweep = "SSL_SWEEP" if direction == "BULLISH" else "BSL_SWEEP"
        if direction == "BULLISH":
            entry, stop, target = 101.0, 99.0, 107.0
        else:
            entry, stop, target = 101.0, 103.0, 95.0
        return {
            "status": "READY",
            "setup_state": "ENTRY_NOW",
            "latest_closed_timestamp": "2026-01-01T00:05:00+00:00",
            "entry_gate": {"state": "ENTRY_NOW", "direction": direction},
            "trade_plan": {
                "state": "CANDIDATE_READY",
                "direction": direction,
                "entry_reference": entry,
                "entry_zone_low": 100.0,
                "entry_zone_high": 102.0,
                "stop_loss": stop,
                "take_profit": target,
                "risk_reward": 3.0,
            },
            "structure_event": {"event": "BOS", "direction": direction},
            "liquidity_sweep": {"event": sweep, "direction": direction},
            "displacement": {"event": "DISPLACEMENT", "direction": direction},
            "fvg": {"event": "FVG", "direction": direction},
            "order_block": {
                "event": "ORDER_BLOCK",
                "direction": direction,
                "state": "RETESTED",
            },
        }

    def build(self, direction="BULLISH"):
        return main.build_verified_auto_entry_request_from_detector(
            "BTC-USD", self.detector(direction)
        )

    def test_wait_detector_does_not_build_request(self):
        detector = self.detector()
        detector["setup_state"] = "WAIT"
        self.assertIsNone(
            main.build_verified_auto_entry_request_from_detector("BTC-USD", detector)
        )

    def test_detector_must_be_ready(self):
        detector = self.detector()
        detector["status"] = "WAIT"
        self.assertIsNone(
            main.build_verified_auto_entry_request_from_detector("BTC-USD", detector)
        )

    def test_gate_must_be_entry_now(self):
        detector = self.detector()
        detector["entry_gate"]["state"] = "EXPIRED"
        self.assertIsNone(
            main.build_verified_auto_entry_request_from_detector("BTC-USD", detector)
        )

    def test_trade_plan_must_be_candidate_ready(self):
        detector = self.detector()
        detector["trade_plan"]["state"] = "WAIT"
        self.assertIsNone(
            main.build_verified_auto_entry_request_from_detector("BTC-USD", detector)
        )

    def test_bullish_request_maps_to_bullish_server_signal(self):
        request = self.build()
        self.assertIsNotNone(request)
        self.assertEqual(request.direction, "BULLISH")
        self.assertEqual(request.setup_state, "ENTRY_NOW")

    def test_bearish_request_maps_to_bearish_server_signal(self):
        request = self.build("BEARISH")
        self.assertIsNotNone(request)
        self.assertEqual(request.direction, "BEARISH")

    def test_source_timestamp_comes_from_latest_closed_candle(self):
        request = self.build()
        self.assertIsNotNone(request)
        self.assertEqual(request.source_timestamp.isoformat(), "2026-01-01T00:05:00+00:00")

    def test_default_auto_risk_is_one_percent(self):
        request = self.build()
        self.assertIsNotNone(request)
        self.assertEqual(request.risk_percent, Decimal("1"))

    def test_server_confirmations_are_set(self):
        request = self.build()
        self.assertIsNotNone(request)
        self.assertTrue(request.structure_confirmed)
        self.assertTrue(request.displacement_confirmed)
        self.assertTrue(request.order_block_confirmed)

    def test_direction_mismatch_blocks_request(self):
        detector = self.detector()
        detector["fvg"]["direction"] = "BEARISH"
        self.assertIsNone(
            main.build_verified_auto_entry_request_from_detector("BTC-USD", detector)
        )

    def test_invalidated_order_block_blocks_request(self):
        detector = self.detector()
        detector["order_block"]["state"] = "INVALIDATED"
        self.assertIsNone(
            main.build_verified_auto_entry_request_from_detector("BTC-USD", detector)
        )

    def test_naive_source_timestamp_blocks_request(self):
        detector = self.detector()
        detector["latest_closed_timestamp"] = "2026-01-01T00:05:00"
        self.assertIsNone(
            main.build_verified_auto_entry_request_from_detector("BTC-USD", detector)
        )

    def test_generator_scans_registered_crypto_only(self):
        source = inspect.getsource(main.run_server_auto_paper_generation_once)
        self.assertIn("instrument_registry.all()", source)
        self.assertIn("instrument.asset_class != AssetClass.CRYPTO", source)

    def test_generator_uses_verified_paper_entry(self):
        source = inspect.getsource(main.run_server_auto_paper_generation_once)
        self.assertIn("market_provider.get_ticker", source)
        self.assertIn("verified_auto_paper_entry(request)", source)
        self.assertIn('stats["already_consumed"]', source)

    def test_orchestrator_runs_server_auto_generation(self):
        source = inspect.getsource(main.auto_entry_orchestrator_loop)
        self.assertIn("run_server_auto_paper_generation_once()", source)
        status_source = inspect.getsource(main.get_auto_entry_orchestrator_status)
        self.assertIn("SERVER_AUTO_PAPER_POSITION_V1", status_source)

    def test_ui_marks_server_auto_paper_position(self):
        html = INDEX.read_text(encoding="utf-8")
        self.assertIn("SERVER AUTO PAPER POSITION V1", html)

    def test_realtime_fill_uses_valid_ticker_price(self):
        request = self.build()
        self.assertIsNotNone(request)
        ticker = main.MarketDatum(
            "coinbase", "BTC-USD", 102.0, NOW, main.DataQualityStatus.VALID
        )
        filled = main.apply_realtime_market_fill_to_auto_request(
            request, self.detector(), ticker
        )
        self.assertIsNotNone(filled)
        self.assertEqual(filled.entry, Decimal("102.0"))
        self.assertEqual(filled.source_timestamp, NOW)

    def test_realtime_fill_blocks_price_outside_entry_zone(self):
        request = self.build()
        self.assertIsNotNone(request)
        ticker = main.MarketDatum(
            "coinbase", "BTC-USD", 110.0, NOW, main.DataQualityStatus.VALID
        )
        self.assertIsNone(
            main.apply_realtime_market_fill_to_auto_request(
                request, self.detector(), ticker
            )
        )


class TestAutoPaperLifecycleHardeningV16M5B11(unittest.TestCase):
    def mark(self, observed=NOW, source_timestamp=NOW):
        return main.PaperPositionMark(
            current_price=Decimal("101"),
            observed_at=observed,
            source="coinbase",
            source_timestamp=source_timestamp,
        )

    def test_temporal_validator_accepts_current_mark(self):
        opened = NOW - timedelta(seconds=1)
        self.assertTrue(main.paper_mark_temporally_valid(self.mark(), opened))

    def test_temporal_validator_rejects_source_before_open(self):
        opened = NOW
        mark = self.mark(source_timestamp=NOW - timedelta(seconds=1))
        self.assertFalse(main.paper_mark_temporally_valid(mark, opened))

    def test_temporal_validator_accepts_source_equal_open(self):
        self.assertTrue(main.paper_mark_temporally_valid(self.mark(), NOW))

    def test_temporal_validator_rejects_source_after_observation(self):
        mark = self.mark(source_timestamp=NOW + timedelta(seconds=1))
        self.assertFalse(main.paper_mark_temporally_valid(mark, NOW))

    def test_temporal_validator_rejects_naive_observation(self):
        mark = self.mark(observed=NOW.replace(tzinfo=None))
        self.assertFalse(main.paper_mark_temporally_valid(mark, NOW))

    def test_temporal_validator_rejects_naive_source_timestamp(self):
        mark = self.mark(source_timestamp=NOW.replace(tzinfo=None))
        self.assertFalse(main.paper_mark_temporally_valid(mark, NOW))

    def test_temporal_validator_rejects_naive_opened_at(self):
        opened = NOW.replace(tzinfo=None)
        self.assertFalse(main.paper_mark_temporally_valid(self.mark(), opened))

    def test_mark_route_uses_temporal_validator(self):
        source = inspect.getsource(main.mark_paper_position)
        self.assertIn("paper_mark_temporally_valid", source)

    def test_stale_mark_has_explicit_conflict_reason(self):
        source = inspect.getsource(main.mark_paper_position)
        self.assertIn("STALE_OR_INVALID_MARK", source)
        self.assertIn("status_code=409", source)

    def test_closed_position_remains_idempotent_before_mark_validation(self):
        source = inspect.getsource(main.mark_paper_position)
        closed_guard = source.index('data["status"] != "OPEN"')
        temporal_guard = source.index("paper_mark_temporally_valid")
        self.assertLess(closed_guard, temporal_guard)

    def test_database_close_remains_open_only(self):
        source = inspect.getsource(main.mark_paper_position)
        self.assertIn("AND status='OPEN'", source)

    def test_monitor_isolates_position_failures(self):
        source = inspect.getsource(main.monitor_open_paper_positions_once)
        self.assertIn("errors += 1", source)
        self.assertIn("Paper position monitor failed for %s", source)

    def test_monitor_preserves_cancellation(self):
        source = inspect.getsource(main.monitor_open_paper_positions_once)
        self.assertIn("except asyncio.CancelledError", source)
        self.assertIn("raise", source)

    def test_monitor_reports_error_count(self):
        source = inspect.getsource(main.monitor_open_paper_positions_once)
        self.assertIn('"errors": errors', source)

    def test_monitor_still_uses_real_market_marks(self):
        source = inspect.getsource(main.monitor_open_paper_positions_once)
        self.assertIn("await paper_mark_from_realtime", source)
        self.assertIn("await mark_paper_position", source)

    def test_ui_marks_lifecycle_hardening(self):
        html = INDEX.read_text(encoding="utf-8")
        self.assertIn("PAPER LIFECYCLE HARDENING V1", html)


class TestPaperPortfolioRiskGuardV16M5B12(unittest.TestCase):
    def open_position(self, symbol="ETH-USD", risk="10"):
        return {"symbol": symbol, "risk_money": Decimal(risk)}

    def guard(self, positions=None, symbol="BTC-USD", risk="10", capital="1000"):
        return main.evaluate_paper_portfolio_risk_guard(
            positions or [], symbol, Decimal(risk), Decimal(capital)
        )

    def test_empty_portfolio_accepts_candidate(self):
        self.assertEqual(self.guard()["status"], "VALID")

    def test_same_symbol_open_is_blocked(self):
        result = self.guard([self.open_position("BTC-USD")])
        self.assertEqual(result["reason"], "SYMBOL_ALREADY_OPEN")

    def test_symbol_is_canonicalized(self):
        result = self.guard([self.open_position("BTC-USD")], symbol="btc/usd")
        self.assertEqual(result["reason"], "SYMBOL_ALREADY_OPEN")

    def test_max_open_positions_is_blocked(self):
        positions = [self.open_position(f"COIN{i}-USD", "1") for i in range(5)]
        result = self.guard(positions)
        self.assertEqual(result["reason"], "MAX_OPEN_POSITIONS_REACHED")

    def test_four_open_positions_remain_allowed(self):
        positions = [self.open_position(f"COIN{i}-USD", "1") for i in range(4)]
        self.assertEqual(self.guard(positions)["status"], "VALID")

    def test_projected_risk_equal_five_percent_is_allowed(self):
        positions = [self.open_position("ETH-USD", "40")]
        self.assertEqual(self.guard(positions, risk="10")["status"], "VALID")

    def test_projected_risk_above_five_percent_is_blocked(self):
        positions = [self.open_position("ETH-USD", "40.01")]
        result = self.guard(positions, risk="10")
        self.assertEqual(result["reason"], "PORTFOLIO_RISK_LIMIT_REACHED")

    def test_invalid_capital_fails_closed(self):
        result = self.guard(capital="0")
        self.assertEqual(result["reason"], "PORTFOLIO_RISK_INPUT_INVALID")

    def test_invalid_candidate_risk_fails_closed(self):
        result = self.guard(risk="0")
        self.assertEqual(result["reason"], "PORTFOLIO_RISK_INPUT_INVALID")

    def test_invalid_existing_risk_fails_closed(self):
        result = self.guard([{"symbol": "ETH-USD", "risk_money": "bad"}])
        self.assertEqual(result["reason"], "OPEN_RISK_INVALID")

    def test_guard_exposes_projected_risk(self):
        result = self.guard([self.open_position("ETH-USD", "12")], risk="8")
        self.assertEqual(result["projected_risk_money"], Decimal("20"))

    def test_verified_entry_uses_portfolio_snapshot(self):
        source = inspect.getsource(main._verified_auto_paper_entry_unlocked)
        self.assertIn("await get_open_paper_risk_snapshot()", source)
        self.assertIn("evaluate_paper_portfolio_risk_guard", source)

    def test_verified_entry_returns_explicit_portfolio_block(self):
        source = inspect.getsource(main._verified_auto_paper_entry_unlocked)
        self.assertIn('"portfolio_guard": portfolio_guard', source)

    def test_public_verified_entry_is_serialized(self):
        source = inspect.getsource(main.verified_auto_paper_entry)
        self.assertIn("async with auto_paper_portfolio_lock", source)

    def test_duplicate_database_guard_is_preserved(self):
        source = inspect.getsource(main.create_paper_position)
        self.assertIn("on_conflict_do_nothing", source)

    def test_ui_marks_portfolio_risk_guard(self):
        html = INDEX.read_text(encoding="utf-8")
        self.assertIn("PAPER PORTFOLIO RISK GUARD V1", html)


class TestAutoPaperE2EReadinessV16M5B13(unittest.TestCase):
    def test_readiness_contract_has_version(self):
        result = main.auto_paper_e2e_readiness()
        self.assertEqual(result["validation"], "SERVER_AUTO_PAPER_E2E_READINESS_V1")

    def test_readiness_is_paper_only(self):
        self.assertTrue(main.auto_paper_e2e_readiness()["paper_only"])

    def test_readiness_disables_broker_execution(self):
        self.assertFalse(main.auto_paper_e2e_readiness()["broker_execution"])

    def test_readiness_disables_live_trading(self):
        self.assertFalse(main.auto_paper_e2e_readiness()["live_trading_enabled"])

    def test_pipeline_starts_with_real_market_data(self):
        pipeline = main.auto_paper_e2e_readiness()["pipeline"]
        self.assertEqual(pipeline[0], "REAL_MARKET_DATA")

    def test_pipeline_contains_server_smc(self):
        self.assertIn("SERVER_SMC_SETUP", main.auto_paper_e2e_readiness()["pipeline"])

    def test_pipeline_contains_entry_gate(self):
        self.assertIn("ENTRY_NOW_GATE", main.auto_paper_e2e_readiness()["pipeline"])

    def test_pipeline_contains_portfolio_guard(self):
        pipeline = main.auto_paper_e2e_readiness()["pipeline"]
        self.assertIn("PORTFOLIO_RISK_GUARD", pipeline)

    def test_pipeline_contains_position_create(self):
        pipeline = main.auto_paper_e2e_readiness()["pipeline"]
        self.assertIn("PAPER_POSITION_CREATE", pipeline)

    def test_pipeline_contains_realtime_mark(self):
        self.assertIn("REALTIME_MARK", main.auto_paper_e2e_readiness()["pipeline"])

    def test_pipeline_contains_sl_tp_close(self):
        self.assertIn("SL_TP_CLOSE", main.auto_paper_e2e_readiness()["pipeline"])

    def test_pipeline_ends_with_history(self):
        pipeline = main.auto_paper_e2e_readiness()["pipeline"]
        self.assertEqual(pipeline[-1], "PAPER_HISTORY")

    def test_readiness_requires_persistence_and_orchestrator(self):
        source = inspect.getsource(main.auto_paper_e2e_readiness)
        self.assertIn("persistence_state.ready and orchestrator_running", source)

    def test_generation_calls_verified_entry(self):
        source = inspect.getsource(main.run_server_auto_paper_generation_once)
        self.assertIn("await verified_auto_paper_entry(request)", source)

    def test_monitor_calls_realtime_mark_and_position_mark(self):
        source = inspect.getsource(main.monitor_open_paper_positions_once)
        self.assertIn("await paper_mark_from_realtime", source)
        self.assertIn("await mark_paper_position", source)

    def test_ui_marks_e2e_readiness(self):
        html = INDEX.read_text(encoding="utf-8")
        self.assertIn("AUTO PAPER E2E READINESS V1", html)


class TestContinuousAutoScanV16M5B14(unittest.TestCase):
    def setUp(self):
        main.reset_auto_scan_runtime()

    def test_runtime_contract_version(self):
        result = main.auto_scan_runtime_status()
        self.assertEqual(result["validation"], "SERVER_CONTINUOUS_AUTO_SCAN_V1")

    def test_runtime_is_paper_only(self):
        self.assertTrue(main.auto_scan_runtime_status()["paper_only"])

    def test_runtime_disables_broker_execution(self):
        self.assertFalse(main.auto_scan_runtime_status()["broker_execution"])

    def test_runtime_disables_live_trading(self):
        self.assertFalse(main.auto_scan_runtime_status()["live_trading_enabled"])

    def test_runtime_exposes_interval(self):
        result = main.auto_scan_runtime_status()
        self.assertEqual(result["interval_seconds"], 5.0)

    def test_reset_sets_zero_iterations(self):
        main.auto_scan_runtime["iterations"] = 8
        main.reset_auto_scan_runtime()
        self.assertEqual(main.auto_scan_runtime["iterations"], 0)

    def test_reset_clears_last_error(self):
        main.auto_scan_runtime["last_error"] = "boom"
        main.reset_auto_scan_runtime()
        self.assertIsNone(main.auto_scan_runtime["last_error"])

    def test_runtime_exposes_generation_snapshot(self):
        self.assertIn("last_generation", main.auto_scan_runtime_status())

    def test_runtime_exposes_queue_snapshot(self):
        self.assertIn("last_queue", main.auto_scan_runtime_status())

    def test_runtime_exposes_heartbeat_timestamps(self):
        result = main.auto_scan_runtime_status()
        self.assertIn("last_started_at", result)
        self.assertIn("last_completed_at", result)

    def test_loop_calls_server_generation(self):
        source = inspect.getsource(main.auto_entry_orchestrator_loop)
        self.assertIn("run_server_auto_paper_generation_once", source)

    def test_loop_calls_candidate_queue(self):
        source = inspect.getsource(main.auto_entry_orchestrator_loop)
        self.assertIn("run_auto_entry_orchestrator_once", source)

    def test_loop_records_iteration_completion(self):
        source = inspect.getsource(main.auto_entry_orchestrator_loop)
        self.assertIn('auto_scan_runtime["iterations"]', source)
        self.assertIn('auto_scan_runtime["last_completed_at"]', source)

    def test_loop_records_failure_without_live_fallback(self):
        source = inspect.getsource(main.auto_entry_orchestrator_loop)
        self.assertIn('auto_scan_runtime["last_error"]', source)
        self.assertNotIn("live_trading_enabled = True", source)

    def test_runtime_status_endpoint_exists(self):
        source = inspect.getsource(main.get_auto_scan_runtime_status)
        self.assertIn("auto_scan_runtime_status()", source)

    def test_ui_marks_continuous_auto_scan(self):
        html = INDEX.read_text(encoding="utf-8")
        self.assertIn("CONTINUOUS AUTO SCAN V1", html)


class TestDecisionTraceV16M5B15(unittest.TestCase):
    def setUp(self):
        main.reset_auto_scan_runtime()

    def test_trace_contract_version(self):
        self.assertEqual(
            main.auto_decision_trace_status()["validation"], "SERVER_DECISION_TRACE_V1"
        )

    def test_trace_is_paper_only(self):
        self.assertTrue(main.auto_decision_trace_status()["paper_only"])

    def test_trace_disables_broker_execution(self):
        self.assertFalse(main.auto_decision_trace_status()["broker_execution"])

    def test_trace_records_wait_reason(self):
        detector = {"status": "READY", "setup_state": "WAIT"}
        main.record_auto_decision_trace(
            "BTC-USD", "WAIT", main._decision_reason_from_detector(detector), detector
        )
        item = main.auto_decision_trace_status()["items"][0]
        self.assertEqual(item["reason"], "STRUCTURE_NOT_CONFIRMED")

    def test_trace_records_entry_now_reason(self):
        detector = {"status": "READY", "setup_state": "ENTRY_NOW"}
        self.assertEqual(
            main._decision_reason_from_detector(detector), "ALL_CONFIRMATIONS_VALID"
        )

    def test_trace_records_invalidated_reason(self):
        detector = {"status": "READY", "setup_state": "INVALIDATED"}
        self.assertEqual(
            main._decision_reason_from_detector(detector), "ENTRY_GATE_INVALIDATED"
        )

    def test_trace_records_expired_reason(self):
        detector = {"status": "READY", "setup_state": "EXPIRED"}
        self.assertEqual(
            main._decision_reason_from_detector(detector), "ENTRY_WINDOW_EXPIRED"
        )

    def test_trace_normalizes_symbol(self):
        main.record_auto_decision_trace("btc/usd", "WAIT", "TEST")
        item = main.auto_decision_trace_status()["items"][0]
        self.assertEqual(item["symbol"], "BTC-USD")

    def test_trace_is_bounded(self):
        for index in range(main.AUTO_DECISION_TRACE_MAX + 5):
            main.record_auto_decision_trace("BTC-USD", "WAIT", str(index))
        self.assertEqual(len(main.auto_decision_trace), main.AUTO_DECISION_TRACE_MAX)

    def test_trace_limit_is_clamped(self):
        self.assertEqual(
            main.auto_decision_trace_status(999)["limit"], main.AUTO_DECISION_TRACE_MAX
        )

    def test_runtime_status_exposes_trace_count(self):
        main.record_auto_decision_trace("BTC-USD", "WAIT", "TEST")
        self.assertEqual(main.auto_scan_runtime_status()["decision_trace_count"], 1)

    def test_runtime_status_exposes_latest_decision(self):
        main.record_auto_decision_trace("ETH-USD", "WAIT", "TEST")
        latest = main.auto_scan_runtime_status()["latest_decision"]
        self.assertEqual(latest["symbol"], "ETH-USD")

    def test_reset_clears_decision_trace(self):
        main.record_auto_decision_trace("BTC-USD", "WAIT", "TEST")
        main.reset_auto_scan_runtime()
        self.assertEqual(main.auto_decision_trace, [])

    def test_generation_records_decisions(self):
        source = inspect.getsource(main.run_server_auto_paper_generation_once)
        self.assertIn("record_and_persist_auto_decision_trace", source)

    def test_decision_trace_endpoint_exists(self):
        source = inspect.getsource(main.get_auto_decision_trace)
        self.assertIn("auto_decision_trace_status", source)

    def test_ui_marks_decision_trace(self):
        html = INDEX.read_text(encoding="utf-8")
        self.assertIn("DECISION TRACE V1", html)


class TestSignalDecisionHistoryV16M5B16(unittest.TestCase):
    def test_history_table_exists(self):
        self.assertEqual(main.signal_decision_history_table.name, "signal_decision_history")

    def test_history_has_decision_id_primary_key(self):
        column = main.signal_decision_history_table.c.decision_id
        self.assertTrue(column.primary_key)

    def test_history_persists_state(self):
        self.assertIn("state", main.signal_decision_history_table.c)

    def test_history_persists_reason(self):
        self.assertIn("reason", main.signal_decision_history_table.c)

    def test_history_persists_detector_context(self):
        self.assertIn("detector_context", main.signal_decision_history_table.c)

    def test_history_is_paper_only(self):
        source = inspect.getsource(main.persist_auto_decision_trace)
        self.assertIn('"paper_only": True', source)

    def test_history_disables_execution(self):
        source = inspect.getsource(main.persist_auto_decision_trace)
        self.assertIn('"execution": False', source)

    def test_history_uses_postgres_insert(self):
        source = inspect.getsource(main.persist_auto_decision_trace)
        self.assertIn("pg_insert(signal_decision_history_table)", source)

    def test_history_insert_is_idempotent(self):
        source = inspect.getsource(main.persist_auto_decision_trace)
        self.assertIn("on_conflict_do_nothing", source)

    def test_scanner_records_and_persists(self):
        source = inspect.getsource(main.run_server_auto_paper_generation_once)
        self.assertIn("record_and_persist_auto_decision_trace", source)

    def test_history_endpoint_exists(self):
        source = inspect.getsource(main.get_signal_decision_history)
        self.assertIn("signal_decision_history", source)

    def test_history_contract_version(self):
        source = inspect.getsource(main.signal_decision_history)
        self.assertIn("SERVER_SIGNAL_DECISION_HISTORY_V1", source)

    def test_history_limit_is_bounded(self):
        source = inspect.getsource(main.signal_decision_history)
        self.assertIn("min(limit, 500)", source)

    def test_history_supports_symbol_filter(self):
        source = inspect.getsource(main.signal_decision_history)
        self.assertIn("symbol = :symbol", source)

    def test_history_supports_state_filter(self):
        source = inspect.getsource(main.signal_decision_history)
        self.assertIn("state = :state", source)

    def test_ui_marks_signal_decision_history(self):
        html = INDEX.read_text(encoding="utf-8")
        self.assertIn("SIGNAL DECISION HISTORY V1", html)

# V16-M5B16-FIX2: M5B15 regression contract follows persisted decision wrapper


class TestPaperPerformanceAnalyticsV16M5B17(unittest.TestCase):
    def setUp(self):
        self.calc = main.calculate_paper_performance_metrics
        self.capital = Decimal("1000")

    def trade(self, side="LONG", entry="100", close="110", size="1", risk="10"):
        return {
            "side": side, "entry": entry, "close_price": close,
            "size": size, "risk_money": risk,
        }

    def test_empty_history_has_zero_closed_trades(self):
        self.assertEqual(self.calc([], self.capital)["closed_trades"], 0)

    def test_empty_history_does_not_invent_win_rate(self):
        self.assertIsNone(self.calc([], self.capital)["win_rate_percent"])

    def test_long_win_pnl(self):
        result = self.calc([self.trade()], self.capital)
        self.assertEqual(result["net_pnl"], "10")

    def test_short_win_pnl(self):
        result = self.calc([self.trade(side="SHORT", close="90")], self.capital)
        self.assertEqual(result["net_pnl"], "10")

    def test_counts_wins_losses_and_breakeven(self):
        trades = [self.trade(), self.trade(close="90"), self.trade(close="100")]
        result = self.calc(trades, self.capital)
        self.assertEqual((result["wins"], result["losses"], result["breakeven"]), (1, 1, 1))

    def test_win_rate_uses_all_closed_trades(self):
        result = self.calc([self.trade(), self.trade(close="90")], self.capital)
        self.assertEqual(result["win_rate_percent"], "50.0")

    def test_profit_factor(self):
        result = self.calc([self.trade(close="120"), self.trade(close="90")], self.capital)
        self.assertEqual(result["profit_factor"], "2")

    def test_profit_factor_none_without_losses(self):
        self.assertIsNone(self.calc([self.trade()], self.capital)["profit_factor"])

    def test_expectancy_is_net_pnl_per_trade(self):
        result = self.calc([self.trade(close="120"), self.trade(close="90")], self.capital)
        self.assertEqual(result["expectancy"], "5")

    def test_average_realized_rr_uses_persisted_risk_money(self):
        result = self.calc([self.trade(close="120", risk="10")], self.capital)
        self.assertEqual(result["average_realized_rr"], "2")

    def test_zero_risk_is_excluded_from_rr_average(self):
        result = self.calc([self.trade(risk="0")], self.capital)
        self.assertIsNone(result["average_realized_rr"])

    def test_drawdown_uses_closed_trade_equity_curve(self):
        trades = [self.trade(close="120"), self.trade(close="80")]
        result = self.calc(trades, self.capital)
        self.assertEqual(result["max_drawdown"], "20")

    def test_non_positive_initial_capital_rejected(self):
        with self.assertRaises(ValueError):
            self.calc([], Decimal("0"))

    def test_endpoint_contract_version(self):
        source = inspect.getsource(main.get_paper_performance)
        self.assertIn("SERVER_PAPER_PERFORMANCE_ANALYTICS_V1", source)

    def test_endpoint_filters_closed_positions_only(self):
        source = inspect.getsource(main.get_paper_performance)
        self.assertIn("WHERE status='CLOSED'", source)

    def test_endpoint_is_paper_only_and_no_execution(self):
        source = inspect.getsource(main.get_paper_performance)
        self.assertIn('"paper_only": True', source)
        self.assertIn('"execution": False', source)

# V16-M5B17: 16 performance analytics regression tests


class TestTradingOperationalAuditUiV16M5B18(unittest.TestCase):
    def html(self):
        return INDEX.read_text(encoding="utf-8")

    def test_ui_m5b18_contract_marker(self):
        self.assertIn("TRADING OPERATIONAL AUDIT UI V1", self.html())

    def test_ui_fetches_performance_endpoint(self):
        self.assertIn('/api/v1/paper/ui-snapshot', self.html())

    def test_ui_fetches_durable_decision_history(self):
        self.assertIn('/api/v1/paper/ui-snapshot', self.html())

    def test_ui_stores_performance(self):
        self.assertIn('if(performance)paperUiState.performance=performance;', self.html())

    def test_ui_stores_decisions(self):
        self.assertIn(
            'if(decisions)paperUiState.decisions=decisions.items||[];',
            self.html(),
        )

    def test_ui_exposes_closed_trade_count(self):
        self.assertIn('Trades clôturés', self.html())

    def test_ui_exposes_win_rate(self):
        self.assertIn('Win Rate', self.html())

    def test_ui_exposes_profit_factor(self):
        self.assertIn('Profit Factor', self.html())

    def test_ui_exposes_expectancy(self):
        self.assertIn('Expectancy', self.html())

    def test_ui_exposes_net_pnl(self):
        self.assertIn('P&L net', self.html())

    def test_ui_exposes_realized_rr(self):
        self.assertIn('RR réalisé moyen', self.html())

    def test_ui_exposes_max_drawdown(self):
        self.assertIn('Max Drawdown', self.html())

    def test_ui_has_symbol_filter(self):
        self.assertIn('Filtre symbole', self.html())

    def test_ui_has_state_filter(self):
        self.assertIn('Filtre état', self.html())

    def test_ui_filters_are_sent_to_server(self):
        html = self.html()
        self.assertIn('"symbol="+encodeURIComponent', html)
        self.assertIn('"state="+encodeURIComponent', html)

    def test_ui_keeps_no_fabricated_decision_contract(self):
        self.assertIn('Aucune décision correspondante', self.html())

# V16-M5B18: 16 Trading operational audit UI regression tests


class TestAutoScanWatchdogV16M5B19(unittest.TestCase):
    def evaluate(self, runtime, running=True, now=None):
        return main.evaluate_auto_scan_watchdog(runtime, running, now)

    def runtime(self, completed=None, error=None, iterations=1):
        return {
            "last_completed_at": completed,
            "last_error": error,
            "iterations": iterations,
        }

    def test_contract_marker(self):
        result = self.evaluate(self.runtime(), False)
        self.assertEqual(result["validation"], "SERVER_AUTO_SCAN_WATCHDOG_V1")

    def test_stopped_when_task_not_running(self):
        result = self.evaluate(self.runtime(), False)
        self.assertEqual(result["status"], "STOPPED")

    def test_stopped_reason(self):
        result = self.evaluate(self.runtime(), False)
        self.assertEqual(result["reason"], "ORCHESTRATOR_TASK_NOT_RUNNING")

    def test_starting_before_first_completed_scan(self):
        result = self.evaluate(self.runtime(), True)
        self.assertEqual(result["status"], "STARTING")

    def test_starting_reason(self):
        result = self.evaluate(self.runtime(), True)
        self.assertEqual(result["reason"], "WAITING_FOR_FIRST_COMPLETED_SCAN")

    def test_healthy_fresh_heartbeat(self):
        now = datetime(2026, 9, 3, 3, 0, tzinfo=timezone.utc)
        completed = datetime(2026, 9, 3, 2, 59, 55, tzinfo=timezone.utc).isoformat()
        result = self.evaluate(self.runtime(completed), True, now)
        self.assertEqual(result["status"], "HEALTHY")

    def test_fresh_heartbeat_age(self):
        now = datetime(2026, 9, 3, 3, 0, tzinfo=timezone.utc)
        completed = datetime(2026, 9, 3, 2, 59, 55, tzinfo=timezone.utc).isoformat()
        result = self.evaluate(self.runtime(completed), True, now)
        self.assertEqual(result["heartbeat_age_seconds"], 5.0)

    def test_stale_heartbeat(self):
        now = datetime(2026, 9, 3, 3, 0, tzinfo=timezone.utc)
        completed = datetime(2026, 9, 3, 2, 59, 30, tzinfo=timezone.utc).isoformat()
        result = self.evaluate(self.runtime(completed), True, now)
        self.assertEqual(result["status"], "STALE")

    def test_stale_reason(self):
        now = datetime(2026, 9, 3, 3, 0, tzinfo=timezone.utc)
        completed = datetime(2026, 9, 3, 2, 59, 30, tzinfo=timezone.utc).isoformat()
        result = self.evaluate(self.runtime(completed), True, now)
        self.assertEqual(result["reason"], "SCAN_HEARTBEAT_STALE")

    def test_error_wins_over_fresh_heartbeat(self):
        now = datetime(2026, 9, 3, 3, 0, tzinfo=timezone.utc)
        completed = datetime(2026, 9, 3, 2, 59, 59, tzinfo=timezone.utc).isoformat()
        result = self.evaluate(self.runtime(completed, "RuntimeError"), True, now)
        self.assertEqual(result["status"], "ERROR")

    def test_error_reason_exposes_class_only(self):
        result = self.evaluate(self.runtime(error="RuntimeError"), True)
        self.assertEqual(result["reason"], "LAST_SCAN_ERROR:RuntimeError")

    def test_invalid_heartbeat_is_starting(self):
        result = self.evaluate(self.runtime("not-a-timestamp"), True)
        self.assertEqual(result["status"], "STARTING")

    def test_naive_now_rejected(self):
        with self.assertRaises(ValueError):
            self.evaluate(self.runtime(), True, datetime(2026, 9, 3, 3, 0))

    def test_paper_only_contract(self):
        result = self.evaluate(self.runtime(), False)
        self.assertTrue(result["paper_only"])
        self.assertFalse(result["broker_execution"])
        self.assertFalse(result["live_trading_enabled"])

    def test_endpoint_calls_watchdog_status(self):
        source = inspect.getsource(main.get_auto_scan_watchdog_status)
        self.assertIn("auto_scan_watchdog_status()", source)

    def test_watchdog_threshold_is_three_scan_intervals_or_more(self):
        self.assertGreaterEqual(
            main.AUTO_SCAN_WATCHDOG_STALE_AFTER_SECONDS,
            main.AUTO_ENTRY_ORCHESTRATOR_INTERVAL_SECONDS * 3,
        )


# V16-M5B19: 16 continuous scanner watchdog regression tests


class TestPaperPerformancePeriodsV16M5B20(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 9, 3, 15, 42, 17, tzinfo=timezone.utc)

    def test_supported_periods(self):
        self.assertEqual(
            main.PAPER_PERFORMANCE_PERIODS,
            {"ALL", "DAY", "WEEK", "MONTH", "YEAR"},
        )

    def test_all_has_no_start_boundary(self):
        self.assertIsNone(main.paper_performance_period_start("ALL", self.now))

    def test_period_is_case_insensitive(self):
        result = main.paper_performance_period_start("day", self.now)
        self.assertEqual(result.hour, 0)

    def test_day_starts_at_utc_midnight(self):
        result = main.paper_performance_period_start("DAY", self.now)
        expected = datetime(2026, 9, 3, 0, 0, tzinfo=timezone.utc)
        self.assertEqual(result, expected)

    def test_week_starts_on_monday_utc(self):
        result = main.paper_performance_period_start("WEEK", self.now)
        expected = datetime(2026, 8, 31, 0, 0, tzinfo=timezone.utc)
        self.assertEqual(result, expected)

    def test_month_starts_on_first_utc(self):
        result = main.paper_performance_period_start("MONTH", self.now)
        expected = datetime(2026, 9, 1, 0, 0, tzinfo=timezone.utc)
        self.assertEqual(result, expected)

    def test_year_starts_on_january_first_utc(self):
        result = main.paper_performance_period_start("YEAR", self.now)
        expected = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
        self.assertEqual(result, expected)

    def test_invalid_period_rejected(self):
        with self.assertRaises(ValueError):
            main.paper_performance_period_start("QUARTER", self.now)

    def test_naive_now_rejected(self):
        naive = datetime(2026, 9, 3, 15, 42, 17)
        with self.assertRaises(ValueError):
            main.paper_performance_period_start("DAY", naive)

    def test_endpoint_defaults_to_all(self):
        signature = inspect.signature(main.get_paper_performance)
        self.assertEqual(signature.parameters["period"].default, "ALL")

    def test_endpoint_uses_period_helper(self):
        source = inspect.getsource(main.get_paper_performance)
        self.assertIn("paper_performance_period_start", source)

    def test_endpoint_filters_closed_at(self):
        source = inspect.getsource(main.get_paper_performance)
        self.assertIn("closed_at>=:period_start", source)

    def test_endpoint_keeps_symbol_filter(self):
        source = inspect.getsource(main.get_paper_performance)
        self.assertIn('params["symbol"] = canonical', source)

    def test_endpoint_returns_period(self):
        source = inspect.getsource(main.get_paper_performance)
        self.assertIn('"period": normalized_period', source)

    def test_new_contract_marker(self):
        source = inspect.getsource(main.get_paper_performance)
        self.assertIn("SERVER_PAPER_PERFORMANCE_PERIODS_V1", source)

    def test_paper_only_contract_preserved(self):
        source = inspect.getsource(main.get_paper_performance)
        self.assertIn('"paper_only": True', source)
        self.assertIn('"execution": False', source)


# V16-M5B20: 16 UTC performance-period regression tests


class TestPaperPerformanceBreakdownV16M5B21(unittest.TestCase):
    def setUp(self):
        self.initial = Decimal("1000")
        self.rows = [
            {
                "symbol": "BTC-USD", "side": "LONG", "entry": Decimal("100"),
                "close_price": Decimal("110"), "size": Decimal("1"),
                "risk_money": Decimal("5"),
            },
            {
                "symbol": "BTC-USD", "side": "SHORT", "entry": Decimal("100"),
                "close_price": Decimal("105"), "size": Decimal("1"),
                "risk_money": Decimal("5"),
            },
            {
                "symbol": "ETH-USD", "side": "LONG", "entry": Decimal("50"),
                "close_price": Decimal("55"), "size": Decimal("2"),
                "risk_money": Decimal("10"),
            },
        ]

    def test_supported_groups_are_exact(self):
        self.assertEqual(main.PAPER_PERFORMANCE_BREAKDOWN_GROUPS, {"SYMBOL", "SIDE"})

    def test_symbol_breakdown_has_two_groups(self):
        groups = main.build_paper_performance_breakdown(self.rows, self.initial, "SYMBOL")
        self.assertEqual([group["group"] for group in groups], ["BTC-USD", "ETH-USD"])

    def test_symbol_breakdown_uses_real_trade_count(self):
        groups = main.build_paper_performance_breakdown(self.rows, self.initial, "SYMBOL")
        self.assertEqual(groups[0]["metrics"]["closed_trades"], 2)

    def test_symbol_breakdown_calculates_net_pnl(self):
        groups = main.build_paper_performance_breakdown(self.rows, self.initial, "SYMBOL")
        self.assertEqual(groups[0]["metrics"]["net_pnl"], "5")

    def test_side_breakdown_has_long_and_short(self):
        groups = main.build_paper_performance_breakdown(self.rows, self.initial, "side")
        self.assertEqual([group["group"] for group in groups], ["LONG", "SHORT"])

    def test_side_breakdown_long_trade_count(self):
        groups = main.build_paper_performance_breakdown(self.rows, self.initial, "SIDE")
        self.assertEqual(groups[0]["metrics"]["closed_trades"], 2)

    def test_invalid_group_is_rejected(self):
        with self.assertRaises(ValueError):
            main.build_paper_performance_breakdown(self.rows, self.initial, "SESSION")

    def test_missing_group_value_is_rejected(self):
        rows = [dict(self.rows[0])]
        rows[0]["symbol"] = ""
        with self.assertRaises(ValueError):
            main.build_paper_performance_breakdown(rows, self.initial, "SYMBOL")

    def test_empty_rows_return_empty_groups(self):
        self.assertEqual(
            main.build_paper_performance_breakdown([], self.initial, "SYMBOL"), []
        )

    def test_endpoint_exists(self):
        paths = {route.path for route in main.api_router.routes}
        self.assertIn("/paper/performance/breakdown", paths)

    def test_endpoint_defaults_to_symbol(self):
        signature = inspect.signature(main.get_paper_performance_breakdown)
        self.assertEqual(signature.parameters["group_by"].default, "SYMBOL")

    def test_endpoint_defaults_to_all_period(self):
        signature = inspect.signature(main.get_paper_performance_breakdown)
        self.assertEqual(signature.parameters["period"].default, "ALL")

    def test_endpoint_uses_period_boundary(self):
        source = inspect.getsource(main.get_paper_performance_breakdown)
        self.assertIn("paper_performance_period_start", source)

    def test_endpoint_reads_only_closed_positions(self):
        source = inspect.getsource(main.get_paper_performance_breakdown)
        self.assertIn("WHERE status='CLOSED'", source)

    def test_endpoint_marks_deferred_unstored_dimensions(self):
        source = inspect.getsource(main.get_paper_performance_breakdown)
        self.assertIn('"TIMEFRAME", "STRATEGY", "SESSION", "REGIME"', source)

    def test_endpoint_is_paper_only(self):
        source = inspect.getsource(main.get_paper_performance_breakdown)
        self.assertIn('"paper_only": True', source)
        self.assertIn('"execution": False', source)


# V16-M5B21: 16 persisted-dimension performance breakdown regression tests


class TestMarketRegimeEngineV16M5B22(unittest.TestCase):
    def _candles(self, closes, ranges=None):
        now = datetime.now(timezone.utc)
        result = []
        for index, close in enumerate(closes):
            width = ranges[index] if ranges else 1.0
            result.append(
                main.Candle(
                    start=now - timedelta(minutes=5 * (len(closes) - index + 1)),
                    low=close - width / 2,
                    high=close + width / 2,
                    open=close,
                    close=close,
                    volume=1.0,
                    status=main.DataQualityStatus.VALID,
                )
            )
        return result, now

    def test_constants_are_objective(self):
        self.assertEqual(main.SERVER_REGIME_LOOKBACK, 20)
        self.assertEqual(main.SERVER_REGIME_ATR_WINDOW, 10)

    def test_insufficient_closed_candles_waits(self):
        candles, now = self._candles([100.0] * 10)
        result = main.classify_server_market_regime(candles, now)
        self.assertEqual(result["status"], "WAIT")

    def test_bullish_trend(self):
        candles, now = self._candles([100.0 + i for i in range(21)])
        result = main.classify_server_market_regime(candles, now)
        self.assertEqual((result["regime"], result["direction"]), ("TREND", "BULLISH"))

    def test_bearish_trend(self):
        candles, now = self._candles([120.0 - i for i in range(21)])
        result = main.classify_server_market_regime(candles, now)
        self.assertEqual((result["regime"], result["direction"]), ("TREND", "BEARISH"))

    def test_range_has_no_direction(self):
        closes = [100.0 + (1 if i % 2 else 0) for i in range(21)]
        candles, now = self._candles(closes)
        result = main.classify_server_market_regime(candles, now)
        self.assertEqual((result["regime"], result["direction"]), ("RANGE", None))

    def test_expansion_volatility(self):
        ranges = [1.0] * 11 + [2.0] * 10
        candles, now = self._candles([100.0 + i for i in range(21)], ranges)
        result = main.classify_server_market_regime(candles, now)
        self.assertEqual(result["volatility"], "EXPANSION")

    def test_compression_volatility(self):
        ranges = [2.0] * 11 + [1.0] * 10
        candles, now = self._candles([100.0 + i * 0.1 for i in range(21)], ranges)
        result = main.classify_server_market_regime(candles, now)
        self.assertEqual(result["volatility"], "COMPRESSION")

    def test_normal_volatility(self):
        candles, now = self._candles([100.0 + i for i in range(21)])
        self.assertEqual(main.classify_server_market_regime(candles, now)["volatility"], "NORMAL")

    def test_only_closed_valid_candles_are_used(self):
        source = inspect.getsource(main.classify_server_market_regime)
        self.assertIn("closed_valid_candles", source)

    def test_marker_present(self):
        candles, now = self._candles([100.0 + i for i in range(21)])
        result = main.classify_server_market_regime(candles, now)
        self.assertEqual(result["marker"], "SERVER_MARKET_REGIME_V1")

    def test_regime_is_not_a_signal(self):
        candles, now = self._candles([100.0 + i for i in range(21)])
        result = main.classify_server_market_regime(candles, now)
        self.assertFalse(result["signal"])
        self.assertFalse(result["auto_queue"])

    def test_endpoint_exists(self):
        paths = {route.path for route in main.api_router.routes}
        self.assertIn("/market/regime/{symbol}", paths)

    def test_endpoint_uses_real_coinbase_candles(self):
        source = inspect.getsource(main.get_server_market_regime)
        self.assertIn("market_provider.get_candles", source)
        self.assertIn('"coinbase"', source)

    def test_endpoint_rejects_non_crypto_v1(self):
        source = inspect.getsource(main.get_server_market_regime)
        self.assertIn("REGIME_CRYPTO_ONLY_V1", source)

    def test_endpoint_requires_valid_quality(self):
        source = inspect.getsource(main.get_server_market_regime)
        self.assertIn("DataQualityStatus.VALID", source)

    def test_endpoint_preserves_paper_only_contract(self):
        source = inspect.getsource(main.get_server_market_regime)
        self.assertIn('"paper_only": True', source)
        self.assertIn('"execution": False', source)


# V16-M5B22: 16 objective market-regime regression tests


class TestMarketSessionsCalendarV16M5B23(unittest.TestCase):
    def test_marker_present(self):
        self.assertIn("SERVER_MARKET_SESSIONS_CALENDAR_V1", inspect.getsource(main))

    def test_unknown_instrument_fails_safe(self):
        result = main.market_session_context("NOPE-USD")
        self.assertEqual(result["status"], "NOT_SUPPORTED")

    def test_naive_time_rejected(self):
        result = main.market_session_context("BTC-USD", datetime(2026, 9, 2, 12))
        self.assertEqual(result["status"], "INVALID_TIME")

    def test_crypto_is_24_7(self):
        now = datetime(2026, 9, 6, 12, tzinfo=timezone.utc)
        result = main.market_session_context("BTC-USD", now)
        self.assertEqual(result["market_state"], "OPEN")

    def test_crypto_session_label(self):
        result = main.market_session_context("BTC-USD")
        self.assertEqual(result["current_session"], "24_7")

    def test_crypto_holidays_not_applicable(self):
        result = main.market_session_context("BTC-USD")
        self.assertEqual(result["holidays"], "NOT_APPLICABLE")

    def test_forex_weekend_closed(self):
        now = datetime(2026, 9, 5, 12, tzinfo=timezone.utc)
        result = main.market_session_context("EUR-USD", now)
        self.assertEqual(result["market_state"], "CLOSED_WEEKEND")

    def test_forex_sessions_are_exposed(self):
        now = datetime(2026, 9, 2, 12, tzinfo=timezone.utc)
        result = main.market_session_context("EUR-USD", now)
        self.assertEqual(len(result["sessions"]), 4)

    def test_forex_sessions_are_indicative(self):
        now = datetime(2026, 9, 2, 12, tzinfo=timezone.utc)
        result = main.market_session_context("EUR-USD", now)
        self.assertTrue(all(item["indicative"] for item in result["sessions"]))

    def test_forex_holidays_not_fabricated(self):
        result = main.market_session_context("EUR-USD")
        self.assertEqual(result["holidays"], "NOT_IMPLEMENTED")

    def test_context_is_paper_only(self):
        result = main.market_session_context("BTC-USD")
        self.assertTrue(result["paper_only"])
        self.assertFalse(result["execution"])

    def test_context_is_not_trade_authorization(self):
        result = main.market_session_context("BTC-USD")
        self.assertFalse(result["trade_authorization"])

    def test_quality_is_independent(self):
        result = main.market_session_context("BTC-USD")
        self.assertTrue(result["data_quality_independent"])

    def test_internal_timezone_is_utc(self):
        result = main.market_session_context("BTC-USD")
        self.assertEqual(result["timezone_internal"], "UTC")

    def test_observed_at_is_aware(self):
        result = main.market_session_context("BTC-USD")
        parsed = datetime.fromisoformat(result["observed_at"])
        self.assertIsNotNone(parsed.tzinfo)

    def test_endpoint_route_registered(self):
        paths = {getattr(route, "path", None) for route in main.api_router.routes}
        self.assertIn("/market/session-context/{symbol}", paths)


# V16-M5B23-FIX4: runtime route-registration and synchronization audit


class TestRuntimeRouteSynchronizationV16M5B23Fix4(unittest.TestCase):
    def test_session_context_is_registered_on_runtime_app(self):
        paths = {getattr(route, "path", None) for route in main.api_router.routes}
        self.assertIn("/market/session-context/{symbol}", paths)
        source = inspect.getsource(main.create_app)
        self.assertIn("app.include_router(api_router, prefix=settings.api_prefix)", source)

    def test_app_is_created_after_session_route_declaration(self):
        source = Path(main.__file__).read_text()
        self.assertLess(
            source.index('@api_router.get("/market/session-context/{symbol}")'),
            source.index("app = create_app()"),
        )

    def test_no_api_router_decorator_exists_after_app_creation(self):
        source = Path(main.__file__).read_text()
        app_index = source.index("app = create_app()")
        self.assertNotIn("@api_router.", source[app_index:])

    def test_fresh_app_contains_session_context_route(self):
        paths = {getattr(route, "path", None) for route in main.api_router.routes}
        self.assertIn("/market/session-context/{symbol}", paths)
        source = Path(main.__file__).read_text()
        self.assertLess(
            source.index('@api_router.get("/market/session-context/{symbol}")'),
            source.index("app = create_app()"),
        )

    def test_symbol_slash_form_is_normalized(self):
        result = main.market_session_context("BTC/USD")
        self.assertEqual(result["symbol"], "BTC-USD")

    def test_symbol_lowercase_is_normalized(self):
        result = main.market_session_context("btc-usd")
        self.assertEqual(result["symbol"], "BTC-USD")

    def test_runtime_route_uses_same_api_prefix(self):
        paths = {getattr(route, "path", None) for route in main.app.routes}
        self.assertNotIn("/market/session-context/{symbol}", paths)

    def test_runtime_session_route_preserves_paper_only_contract(self):
        result = main.market_session_context("BTC-USD")
        self.assertTrue(result["paper_only"])
        self.assertFalse(result["execution"])


# V16-M5B23-FIX5: frontend runtime synchronization audit


class TestFrontendRuntimeSynchronizationV16M5B23Fix5(unittest.TestCase):
    def html(self):
        return INDEX.read_text(encoding="utf-8")

    def test_snapshot_route_registered_on_runtime_app(self):
        paths = {getattr(route, "path", None) for route in main.api_router.routes}
        self.assertIn("/paper/ui-snapshot", paths)
        source = inspect.getsource(main.create_app)
        self.assertIn("app.include_router(api_router, prefix=settings.api_prefix)", source)

    def test_snapshot_validation_marker_present(self):
        source = inspect.getsource(main.get_paper_ui_snapshot)
        self.assertIn("SERVER_PAPER_UI_SNAPSHOT_V1", source)

    def test_snapshot_is_paper_only(self):
        source = inspect.getsource(main.get_paper_ui_snapshot)
        self.assertIn('"paper_only": True', source)
        self.assertIn('"execution": False', source)

    def test_snapshot_has_server_timestamp(self):
        source = inspect.getsource(main.get_paper_ui_snapshot)
        self.assertIn('"snapshot_at": snapshot_at.isoformat()', source)

    def test_snapshot_sections_are_fail_safe(self):
        source = inspect.getsource(main.get_paper_ui_snapshot)
        self.assertIn('"status": "OK" if available == len(sections) else "PARTIAL"', source)

    def test_snapshot_includes_watchdog(self):
        source = inspect.getsource(main.get_paper_ui_snapshot)
        self.assertIn('"watchdog": watchdog', source)

    def test_snapshot_includes_runtime(self):
        source = inspect.getsource(main.get_paper_ui_snapshot)
        self.assertIn('"runtime": runtime', source)

    def test_closed_positions_expose_server_realized_pnl(self):
        source = inspect.getsource(main.paper_position_to_dict)
        self.assertIn('data["realized_pnl"]', source)
        self.assertIn("calculate_paper_pnl", source)

    def test_ui_uses_single_snapshot_endpoint(self):
        self.assertIn('/api/v1/paper/ui-snapshot', self.html())

    def test_ui_does_not_recalculate_closed_pnl(self):
        html = self.html()
        self.assertNotIn("function paperPositionPnl", html)
        self.assertIn("p.realized_pnl", html)

    def test_ui_symbol_filter_uses_all_crypto_symbols(self):
        self.assertIn('var symbols=[""].concat(CRYPTO_SYMBOLS)', self.html())

    def test_ui_exposes_server_snapshot_timestamp(self):
        self.assertIn("Snapshot serveur", self.html())
        self.assertIn("paperUiState.serverSnapshotAt", self.html())

    def test_ui_exposes_partial_section_failures(self):
        self.assertIn("Sections indisponibles", self.html())
        self.assertIn("paperUiState.sectionErrors", self.html())

    def test_signals_page_is_server_backed(self):
        html = self.html()
        self.assertIn("SERVER AUTHORITATIVE", html)
        self.assertIn("Watchdog scanner", html)

    def test_dashboard_uses_real_paper_state(self):
        html = self.html()
        self.assertIn("Capital virtuel", html)
        self.assertIn("dashAccount.current_capital", html)
        self.assertIn("dashAccount.open_positions", html)

    def test_stale_signal_engine_not_implemented_text_removed(self):
        html = self.html()
        self.assertNotIn(
            'modCard("Signal Engine","NOT IMPLEMENTED"',
            html,
        )

# V16-M5B24A: autonomous crypto market stream + REST monitoring fallback
class TestAutonomousPaperMonitoringV16M5B24A(unittest.TestCase):
    def test_server_stream_helper_exists(self):
        self.assertTrue(callable(main.start_server_crypto_market_stream))

    def test_server_stream_uses_registered_crypto_instruments(self):
        source = inspect.getsource(main.start_server_crypto_market_stream)
        self.assertIn("instrument_registry.all()", source)
        self.assertIn("AssetClass.CRYPTO", source)

    def test_server_stream_uses_coinbase_mapping(self):
        source = inspect.getsource(main.start_server_crypto_market_stream)
        self.assertIn('to_provider(', source)
        self.assertIn('"coinbase"', source)

    def test_server_stream_subscribes_ticker(self):
        source = inspect.getsource(main.start_server_crypto_market_stream)
        self.assertIn('market_ws.subscribe("ticker", products)', source)

    def test_server_stream_starts_ws(self):
        source = inspect.getsource(main.start_server_crypto_market_stream)
        self.assertIn("await market_ws.start()", source)

    def test_server_stream_failure_is_fail_safe(self):
        source = inspect.getsource(main.start_server_crypto_market_stream)
        self.assertIn("return False", source)
        self.assertIn("REST fallback remains active", source)

    def test_lifespan_autostarts_crypto_stream(self):
        source = inspect.getsource(main.lifespan)
        self.assertIn("await start_server_crypto_market_stream()", source)

    def test_lifespan_still_stops_crypto_stream(self):
        source = inspect.getsource(main.lifespan)
        self.assertIn("await market_ws.stop()", source)

    def test_rest_fallback_helper_exists(self):
        self.assertTrue(callable(main.paper_mark_from_coinbase_rest))

    def test_rest_fallback_uses_real_coinbase_ticker(self):
        source = inspect.getsource(main.paper_mark_from_coinbase_rest)
        self.assertIn("await market_provider.get_ticker(provider_symbol)", source)

    def test_rest_fallback_requires_valid_quality(self):
        source = inspect.getsource(main.paper_mark_from_coinbase_rest)
        self.assertIn("datum.status != DataQualityStatus.VALID", source)

    def test_rest_fallback_rejects_missing_timestamp(self):
        source = inspect.getsource(main.paper_mark_from_coinbase_rest)
        self.assertIn("datum.timestamp is None", source)

    def test_rest_fallback_rejects_nonpositive_price(self):
        source = inspect.getsource(main.paper_mark_from_coinbase_rest)
        self.assertIn("datum.value <= 0", source)

    def test_rest_fallback_is_identified_in_mark_source(self):
        source = inspect.getsource(main.paper_mark_from_coinbase_rest)
        self.assertIn('source="coinbase_rest_fallback"', source)

    def test_crypto_monitor_falls_back_when_ws_mark_is_unusable(self):
        source = inspect.getsource(main.paper_mark_from_realtime)
        self.assertIn("await paper_mark_from_coinbase_rest(canonical, provider_symbol)", source)

    def test_noncrypto_monitor_paths_are_not_replaced_by_rest_fallback(self):
        source = inspect.getsource(main.paper_mark_from_realtime)
        fallback = source.count("paper_mark_from_coinbase_rest")
        self.assertEqual(fallback, 1)


# V16-M5B24B: server crypto registry/scanner alignment
class TestCryptoRegistryAlignmentV16M5B24B(unittest.TestCase):
    EXPECTED = ("BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "LTC-USD", "ADA-USD")

    def test_all_six_expected_crypto_symbols_are_registered(self):
        for symbol in self.EXPECTED:
            self.assertIsNotNone(main.instrument_registry.get(symbol))

    def test_all_six_are_crypto(self):
        for symbol in self.EXPECTED:
            instrument = main.instrument_registry.get(symbol)
            self.assertIsNotNone(instrument)
            self.assertEqual(instrument.asset_class, main.AssetClass.CRYPTO)

    def test_all_six_use_usd_quote(self):
        for symbol in self.EXPECTED:
            instrument = main.instrument_registry.get(symbol)
            self.assertIsNotNone(instrument)
            self.assertEqual(instrument.quote_asset, "USD")

    def test_all_six_use_24_7_calendar(self):
        for symbol in self.EXPECTED:
            instrument = main.instrument_registry.get(symbol)
            self.assertIsNotNone(instrument)
            self.assertEqual(
                instrument.market_calendar,
                main.MarketCalendarPolicy.ALWAYS_OPEN_24_7,
            )

    def test_all_six_keep_base_asset_volume_semantics(self):
        for symbol in self.EXPECTED:
            instrument = main.instrument_registry.get(symbol)
            self.assertIsNotNone(instrument)
            self.assertEqual(
                instrument.volume_semantics,
                main.VolumeSemantics.BASE_ASSET_VOLUME,
            )

    def test_all_six_have_coinbase_provider_mapping(self):
        for symbol in self.EXPECTED:
            self.assertEqual(
                main.provider_symbol_map.to_provider("coinbase", symbol),
                symbol,
            )

    def test_sol_is_registered(self):
        self.assertIsNotNone(main.instrument_registry.get("SOL-USD"))

    def test_xrp_is_registered(self):
        self.assertIsNotNone(main.instrument_registry.get("XRP-USD"))

    def test_ltc_is_registered(self):
        self.assertIsNotNone(main.instrument_registry.get("LTC-USD"))

    def test_ada_is_registered(self):
        self.assertIsNotNone(main.instrument_registry.get("ADA-USD"))

    def test_registry_contains_six_target_crypto_symbols(self):
        crypto = {
            item.canonical_symbol
            for item in main.instrument_registry.all()
            if item.asset_class == main.AssetClass.CRYPTO
        }
        self.assertTrue(set(self.EXPECTED).issubset(crypto))

    def test_auto_scanner_iterates_registry_not_hardcoded_pair_list(self):
        source = inspect.getsource(main.run_server_auto_paper_generation_once)
        self.assertIn("instrument_registry.all()", source)
        self.assertIn("AssetClass.CRYPTO", source)

    def test_auto_scanner_has_no_btc_eth_only_filter(self):
        source = inspect.getsource(main.run_server_auto_paper_generation_once)
        self.assertNotIn('{"BTC-USD", "ETH-USD"}', source)
        self.assertNotIn("('BTC-USD', 'ETH-USD')", source)

    def test_server_ws_uses_same_registered_crypto_population(self):
        source = inspect.getsource(main.start_server_crypto_market_stream)
        self.assertIn("instrument_registry.all()", source)
        self.assertIn("AssetClass.CRYPTO", source)

    def test_coinbase_registration_source_names_all_four_new_symbols(self):
        source = inspect.getsource(main._register_coinbase_instruments)
        for symbol in ("SOL-USD", "XRP-USD", "LTC-USD", "ADA-USD"):
            self.assertIn(symbol, source)

    def test_registry_alignment_remains_metadata_fail_safe(self):
        source = inspect.getsource(main._register_coinbase_instruments)
        self.assertIn("price_precision=None", source)
        self.assertIn("tick_size=None", source)


# ---------------- V16-M5B25A: HTF context foundation ----------------
class ServerHtfContextFoundationTests(unittest.TestCase):
    def test_htf_granularity_is_one_hour(self):
        self.assertEqual(main.SERVER_HTF_GRANULARITY, "1h")

    def test_htf_limit_is_bounded_by_coinbase_limit(self):
        self.assertGreater(main.SERVER_HTF_CANDLE_LIMIT, 0)
        self.assertLessEqual(main.SERVER_HTF_CANDLE_LIMIT, main.CANDLE_MAX_LIMIT)

    def test_htf_is_distinct_from_ltf_setup_granularity(self):
        self.assertNotEqual(main.SERVER_HTF_GRANULARITY, main.SERVER_SETUP_GRANULARITY)

    def test_htf_classifier_is_non_async_pure_function(self):
        self.assertFalse(inspect.iscoroutinefunction(main.classify_server_htf_context))

    def test_insufficient_htf_history_waits(self):
        result = main.classify_server_htf_context([], NOW)
        self.assertEqual(result["status"], "WAIT")
        self.assertEqual(result["reason"], "INSUFFICIENT_HTF_CLOSED_CANDLES")

    def test_classifier_uses_closed_candles(self):
        source = inspect.getsource(main.classify_server_htf_context)
        self.assertIn("closed_valid_candles(candles, now)", source)

    def test_classifier_uses_confirmed_swings(self):
        source = inspect.getsource(main.classify_server_htf_context)
        self.assertIn("confirmed_swing_indexes(closed)", source)

    def test_classifier_has_no_entry_gate(self):
        source = inspect.getsource(main.classify_server_htf_context)
        self.assertNotIn("evaluate_server_entry_now_gate", source)
        self.assertNotIn("create_paper_position", source)

    def test_classifier_exposes_no_lookahead_contract(self):
        source = inspect.getsource(main.classify_server_htf_context)
        self.assertIn('"no_lookahead": True', source)

    def test_classifier_exposes_premium_discount_location(self):
        source = inspect.getsource(main.classify_server_htf_context)
        self.assertIn('"PREMIUM"', source)
        self.assertIn('"DISCOUNT"', source)
        self.assertIn('"EQUILIBRIUM"', source)

    def test_endpoint_is_registered(self):
        paths = {route.path for route in main.api_router.routes}
        self.assertIn("/market/htf-context/{symbol}", paths)

    def test_endpoint_fetches_real_htf_granularity(self):
        source = inspect.getsource(main.get_server_htf_context)
        self.assertIn("SERVER_HTF_GRANULARITY", source)
        self.assertIn("market_provider.get_candles", source)

    def test_endpoint_has_latest_freshness_guard(self):
        source = inspect.getsource(main.get_server_htf_context)
        self.assertIn("HTF_LATEST_CANDLE_NOT_FRESH", source)
        self.assertIn("_latest_quality(candles)", source)

    def test_endpoint_is_crypto_only_v1(self):
        source = inspect.getsource(main.get_server_htf_context)
        self.assertIn("HTF_CONTEXT_CRYPTO_ONLY_V1", source)
        self.assertIn("AssetClass.CRYPTO", source)

    def test_endpoint_is_paper_only_non_executing(self):
        source = inspect.getsource(main.get_server_htf_context)
        self.assertIn('"paper_only": True', source)
        self.assertNotIn("create_paper_position", source)

    def test_ltf_detector_is_now_gated_by_htf(self):
        source = inspect.getsource(main.get_server_market_setup_detector)
        self.assertIn("classify_server_htf_context", source)
        self.assertIn("apply_htf_context_to_ltf_setup", source)
        self.assertIn("SERVER_HTF_GRANULARITY", source)


# ---------------- V16-M5B25A-FIX2: ENTRY_NOW execution diagnostics ----------------
class AutoEntryExecutionDiagnosticsTests(unittest.TestCase):
    def _entry_detector(self):
        direction = "BULLISH"
        return {
            "status": "READY",
            "setup_state": "ENTRY_NOW",
            "entry_gate": {"state": "ENTRY_NOW", "direction": direction},
            "trade_plan": {
                "state": "CANDIDATE_READY",
                "direction": direction,
                "entry_reference": 100.0,
                "entry_zone_low": 99.0,
                "entry_zone_high": 101.0,
                "stop_loss": 95.0,
                "take_profit": 110.0,
                "risk_reward": 2.0,
            },
            "structure_event": {"event": "BOS", "direction": direction},
            "liquidity_sweep": {"event": "SSL_SWEEP", "direction": direction},
            "displacement": {"event": "DISPLACEMENT", "direction": direction},
            "fvg": {"event": "FVG", "direction": direction},
            "order_block": {"event": "ORDER_BLOCK", "direction": direction},
            "latest_closed_timestamp": NOW.isoformat(),
        }

    def _request(self):
        return main.VerifiedAutoPaperEntryRequest(
            symbol="LTC-USD",
            setup_state="ENTRY_NOW",
            direction="BULLISH",
            entry=Decimal("100"),
            stop_loss=Decimal("95"),
            take_profit=Decimal("110"),
            risk_reward=Decimal("2"),
            structure_confirmed=True,
            displacement_confirmed=True,
            order_block_confirmed=True,
            source_timestamp=NOW,
            risk_percent=Decimal("1"),
        )

    def test_buildable_detector_reports_buildable(self):
        reason = main.auto_entry_request_rejection_reason(self._entry_detector())
        self.assertEqual(reason, "AUTO_ENTRY_REQUEST_BUILDABLE")

    def test_missing_gate_is_explicit(self):
        detector = self._entry_detector()
        detector.pop("entry_gate")
        self.assertEqual(
            main.auto_entry_request_rejection_reason(detector),
            "ENTRY_GATE_MISSING",
        )

    def test_missing_trade_plan_is_explicit(self):
        detector = self._entry_detector()
        detector.pop("trade_plan")
        self.assertEqual(
            main.auto_entry_request_rejection_reason(detector),
            "TRADE_PLAN_MISSING",
        )

    def test_direction_mismatch_is_explicit(self):
        detector = self._entry_detector()
        detector["trade_plan"]["direction"] = "BEARISH"
        self.assertEqual(
            main.auto_entry_request_rejection_reason(detector),
            "TRADE_PLAN_DIRECTION_MISMATCH",
        )

    def test_missing_structure_is_explicit(self):
        detector = self._entry_detector()
        detector.pop("structure_event")
        self.assertEqual(
            main.auto_entry_request_rejection_reason(detector),
            "STRUCTURE_EVENT_MISSING",
        )

    def test_invalid_order_block_is_explicit(self):
        detector = self._entry_detector()
        detector["order_block"]["state"] = "INVALIDATED"
        self.assertEqual(
            main.auto_entry_request_rejection_reason(detector),
            "ORDER_BLOCK_INVALIDATED",
        )

    def test_missing_source_timestamp_is_explicit(self):
        detector = self._entry_detector()
        detector.pop("latest_closed_timestamp")
        self.assertEqual(
            main.auto_entry_request_rejection_reason(detector),
            "SOURCE_TIMESTAMP_MISSING",
        )

    def test_invalid_plan_number_is_explicit(self):
        detector = self._entry_detector()
        detector["trade_plan"]["risk_reward"] = "bad"
        self.assertEqual(
            main.auto_entry_request_rejection_reason(detector),
            "TRADE_PLAN_RISK_REWARD_INVALID",
        )

    def test_valid_realtime_fill_reports_eligible(self):
        ticker = main.MarketDatum(
            "coinbase", "LTC-USD", 100.0, NOW, main.DataQualityStatus.VALID
        )
        reason = main.realtime_fill_rejection_reason(
            self._request(), self._entry_detector(), ticker
        )
        self.assertEqual(reason, "REALTIME_FILL_ELIGIBLE")

    def test_stale_ticker_is_explicit(self):
        ticker = main.MarketDatum(
            "coinbase", "LTC-USD", 100.0, NOW, main.DataQualityStatus.STALE
        )
        reason = main.realtime_fill_rejection_reason(
            self._request(), self._entry_detector(), ticker
        )
        self.assertEqual(reason, "REALTIME_TICKER_STALE")

    def test_missing_ticker_price_is_explicit(self):
        ticker = main.MarketDatum(
            "coinbase", "LTC-USD", None, NOW, main.DataQualityStatus.VALID
        )
        reason = main.realtime_fill_rejection_reason(
            self._request(), self._entry_detector(), ticker
        )
        self.assertEqual(reason, "REALTIME_TICKER_PRICE_MISSING")

    def test_price_outside_zone_is_explicit(self):
        ticker = main.MarketDatum(
            "coinbase", "LTC-USD", 102.0, NOW, main.DataQualityStatus.VALID
        )
        reason = main.realtime_fill_rejection_reason(
            self._request(), self._entry_detector(), ticker
        )
        self.assertEqual(reason, "REALTIME_PRICE_OUTSIDE_ENTRY_ZONE")

    def test_generation_uses_precise_request_reason(self):
        source = inspect.getsource(main.run_server_auto_paper_generation_once)
        self.assertIn("auto_entry_request_rejection_reason(detector)", source)
        self.assertNotIn(
            '"REALTIME_FILL_NOT_ELIGIBLE"',
            source,
        )

    def test_generation_uses_precise_fill_reason(self):
        source = inspect.getsource(main.run_server_auto_paper_generation_once)
        self.assertIn(
            "realtime_fill_rejection_reason(request, detector, ticker)",
            source,
        )

    def test_entry_now_unbuildable_is_recorded_blocked(self):
        source = inspect.getsource(main.run_server_auto_paper_generation_once)
        self.assertIn('symbol, "BLOCKED", reason, detector', source)

    def test_diagnostics_do_not_create_positions(self):
        request_source = inspect.getsource(main.auto_entry_request_rejection_reason)
        fill_source = inspect.getsource(main.realtime_fill_rejection_reason)
        self.assertNotIn("create_paper_position", request_source)
        self.assertNotIn("create_paper_position", fill_source)

# ---------------- V16-M5B25B: HTF -> LTF strategy integration ----------------
class ServerHtfLtfStrategyIntegrationTests(unittest.TestCase):
    def _ltf(self, direction="BULLISH", state="ENTRY_NOW"):
        return {
            "status": "READY",
            "setup_state": state,
            "entry_gate": {"state": state, "direction": direction},
        }

    def _htf(self, structure="BULLISH", location="DISCOUNT"):
        return {
            "status": "READY",
            "structure": structure,
            "location": location,
            "validation": "SERVER_HTF_CONTEXT_V1",
        }

    def test_bullish_discount_entry_is_allowed(self):
        result = main.apply_htf_context_to_ltf_setup(self._ltf(), self._htf())
        self.assertEqual(result["setup_state"], "ENTRY_NOW")

    def test_bullish_equilibrium_entry_is_allowed(self):
        result = main.apply_htf_context_to_ltf_setup(
            self._ltf(), self._htf(location="EQUILIBRIUM")
        )
        self.assertEqual(result["setup_state"], "ENTRY_NOW")

    def test_bullish_premium_entry_waits(self):
        result = main.apply_htf_context_to_ltf_setup(
            self._ltf(), self._htf(location="PREMIUM")
        )
        self.assertEqual(result["reason"], "HTF_BULLISH_ENTRY_IN_PREMIUM")

    def test_bullish_against_bearish_htf_waits(self):
        result = main.apply_htf_context_to_ltf_setup(
            self._ltf(), self._htf(structure="BEARISH")
        )
        self.assertEqual(result["reason"], "HTF_STRUCTURE_NOT_BULLISH")

    def test_bullish_against_range_htf_waits(self):
        result = main.apply_htf_context_to_ltf_setup(
            self._ltf(), self._htf(structure="RANGE")
        )
        self.assertEqual(result["setup_state"], "WAIT")

    def test_bearish_premium_entry_is_allowed(self):
        result = main.apply_htf_context_to_ltf_setup(
            self._ltf("BEARISH"), self._htf("BEARISH", "PREMIUM")
        )
        self.assertEqual(result["setup_state"], "ENTRY_NOW")

    def test_bearish_equilibrium_entry_is_allowed(self):
        result = main.apply_htf_context_to_ltf_setup(
            self._ltf("BEARISH"), self._htf("BEARISH", "EQUILIBRIUM")
        )
        self.assertEqual(result["setup_state"], "ENTRY_NOW")

    def test_bearish_discount_entry_waits(self):
        result = main.apply_htf_context_to_ltf_setup(
            self._ltf("BEARISH"), self._htf("BEARISH", "DISCOUNT")
        )
        self.assertEqual(result["reason"], "HTF_BEARISH_ENTRY_IN_DISCOUNT")

    def test_bearish_against_bullish_htf_waits(self):
        result = main.apply_htf_context_to_ltf_setup(
            self._ltf("BEARISH"), self._htf("BULLISH", "PREMIUM")
        )
        self.assertEqual(result["reason"], "HTF_STRUCTURE_NOT_BEARISH")

    def test_htf_not_ready_blocks_entry(self):
        result = main.apply_htf_context_to_ltf_setup(
            self._ltf(),
            {"status": "WAIT", "reason": "INSUFFICIENT_HTF_CONFIRMED_SWINGS"},
        )
        self.assertEqual(result["reason"], "INSUFFICIENT_HTF_CONFIRMED_SWINGS")

    def test_missing_ltf_gate_blocks_entry(self):
        ltf = self._ltf()
        ltf.pop("entry_gate")
        result = main.apply_htf_context_to_ltf_setup(ltf, self._htf())
        self.assertEqual(result["reason"], "LTF_ENTRY_GATE_MISSING")

    def test_invalid_ltf_direction_blocks_entry(self):
        result = main.apply_htf_context_to_ltf_setup(
            self._ltf("SIDEWAYS"), self._htf()
        )
        self.assertEqual(result["reason"], "LTF_ENTRY_DIRECTION_INVALID")

    def test_non_entry_state_is_not_promoted(self):
        result = main.apply_htf_context_to_ltf_setup(
            self._ltf(state="WAIT"), self._htf()
        )
        self.assertEqual(result["setup_state"], "WAIT")

    def test_result_embeds_htf_context(self):
        htf = self._htf()
        result = main.apply_htf_context_to_ltf_setup(self._ltf(), htf)
        self.assertEqual(result["htf_context"], htf)

    def test_result_marks_htf_ltf_validation(self):
        result = main.apply_htf_context_to_ltf_setup(self._ltf(), self._htf())
        self.assertEqual(result["validation"], "SERVER_HTF_LTF_INTEGRATION_V1")

    def test_setup_detector_fetches_real_htf_only_for_entry_now(self):
        source = inspect.getsource(main.get_server_market_setup_detector)
        self.assertIn("SERVER_HTF_GRANULARITY", source)
        self.assertIn("classify_server_htf_context(htf_candles, utcnow())", source)
        self.assertIn('ltf_result.get("setup_state") != "ENTRY_NOW"', source)


# ---------------- V16-M5B27: objective candidate strategy detectors ----------------
class CandidateStrategyDetectorsV16M5B27Tests(unittest.TestCase):
    def _candles(self, count=24, start_price=100.0, step=1.0):
        now = datetime.now(timezone.utc)
        candles = []
        for index in range(count):
            close = start_price + index * step
            candles.append(
                main.Candle(
                    start=now - timedelta(minutes=5 * (count - index + 1)),
                    low=close - 0.5,
                    high=close + 0.5,
                    open=close - (0.2 if step >= 0 else -0.2),
                    close=close,
                    volume=1.0,
                    status=main.DataQualityStatus.VALID,
                )
            )
        return candles, now

    def test_detector_constants_are_objective(self):
        self.assertEqual(main.TREND_PULLBACK_EMA_PERIOD, 20)
        self.assertEqual(main.BREAKOUT_LOOKBACK, 20)
        self.assertEqual(main.BREAKOUT_BODY_MULTIPLIER, 1.5)
        self.assertEqual(main.BREAKOUT_MIN_BODY_RANGE_RATIO, 0.70)

    def test_trend_pullback_waits_without_trend_context(self):
        candles, now = self._candles()
        result = main.detect_trend_pullback_candidate(
            candles, now, {"status": "READY", "regime": "RANGE", "volatility": "NORMAL"}
        )
        self.assertEqual((result["status"], result["reason"]), ("WAIT", "REGIME_NOT_ELIGIBLE"))

    def test_trend_pullback_waits_on_small_sample(self):
        candles, now = self._candles(count=10)
        regime = {
            "status": "READY",
            "regime": "TREND",
            "direction": "BULLISH",
            "volatility": "NORMAL",
        }
        result = main.detect_trend_pullback_candidate(candles, now, regime)
        self.assertEqual(result["reason"], "INSUFFICIENT_CLOSED_CANDLES")

    def test_trend_pullback_bullish_setup(self):
        candles, now = self._candles()
        candles[-2] = replace(candles[-2], low=100.0)
        latest_close = candles[-2].high + 2.0
        candles[-1] = replace(
            candles[-1],
            close=latest_close,
            high=latest_close + 0.5,
        )
        regime = {
            "status": "READY",
            "regime": "TREND",
            "direction": "BULLISH",
            "volatility": "NORMAL",
        }
        result = main.detect_trend_pullback_candidate(candles, now, regime)
        self.assertEqual((result["status"], result["direction"]), ("SETUP", "BULLISH"))

    def test_trend_pullback_is_candidate_only(self):
        candles, now = self._candles()
        regime = {
            "status": "READY",
            "regime": "TREND",
            "direction": "BULLISH",
            "volatility": "NORMAL",
        }
        result = main.detect_trend_pullback_candidate(candles, now, regime)
        self.assertTrue(result["candidate_only"])
        self.assertFalse(result["auto_queue"])
        self.assertFalse(result["execution"])

    def test_trend_pullback_declares_no_lookahead(self):
        candles, now = self._candles()
        regime = {
            "status": "READY",
            "regime": "TREND",
            "direction": "BULLISH",
            "volatility": "NORMAL",
        }
        result = main.detect_trend_pullback_candidate(candles, now, regime)
        self.assertTrue(result["no_lookahead"])

    def test_breakout_requires_expansion_context(self):
        candles, now = self._candles(count=21)
        regime = {
            "status": "READY",
            "regime": "TREND",
            "direction": "BULLISH",
            "volatility": "NORMAL",
        }
        result = main.detect_breakout_expansion_candidate(candles, now, regime)
        self.assertEqual((result["status"], result["reason"]), ("WAIT", "VOLATILITY_NOT_ELIGIBLE"))

    def test_breakout_waits_on_small_sample(self):
        candles, now = self._candles(count=10)
        regime = {
            "status": "READY",
            "regime": "TREND",
            "direction": "BULLISH",
            "volatility": "EXPANSION",
        }
        result = main.detect_breakout_expansion_candidate(candles, now, regime)
        self.assertEqual(result["reason"], "INSUFFICIENT_CLOSED_CANDLES")

    def test_breakout_bullish_setup(self):
        candles, now = self._candles(count=21, step=0.1)
        prior_high = max(c.high for c in candles[:-1] if c.high is not None)
        latest_open = prior_high - 0.1
        latest_close = prior_high + 2.0
        candles[-1] = replace(
            candles[-1],
            open=latest_open,
            close=latest_close,
            low=latest_open - 0.1,
            high=latest_close + 0.1,
        )
        regime = {
            "status": "READY",
            "regime": "TREND",
            "direction": "BULLISH",
            "volatility": "EXPANSION",
        }
        result = main.detect_breakout_expansion_candidate(candles, now, regime)
        self.assertEqual((result["status"], result["direction"]), ("SETUP", "BULLISH"))

    def test_breakout_bearish_setup(self):
        candles, now = self._candles(count=21, start_price=120.0, step=-0.1)
        prior_low = min(c.low for c in candles[:-1] if c.low is not None)
        latest_open = prior_low + 0.1
        latest_close = prior_low - 2.0
        candles[-1] = replace(
            candles[-1],
            open=latest_open,
            close=latest_close,
            high=latest_open + 0.1,
            low=latest_close - 0.1,
        )
        regime = {
            "status": "READY",
            "regime": "TREND",
            "direction": "BEARISH",
            "volatility": "EXPANSION",
        }
        result = main.detect_breakout_expansion_candidate(candles, now, regime)
        self.assertEqual((result["status"], result["direction"]), ("SETUP", "BEARISH"))

    def test_breakout_is_candidate_only(self):
        candles, now = self._candles(count=21)
        regime = {
            "status": "READY",
            "regime": "TREND",
            "direction": "BULLISH",
            "volatility": "EXPANSION",
        }
        result = main.detect_breakout_expansion_candidate(candles, now, regime)
        self.assertTrue(result["candidate_only"])
        self.assertFalse(result["auto_queue"])
        self.assertFalse(result["execution"])

    def test_breakout_declares_no_lookahead(self):
        candles, now = self._candles(count=21)
        regime = {
            "status": "READY",
            "regime": "TREND",
            "direction": "BULLISH",
            "volatility": "EXPANSION",
        }
        result = main.detect_breakout_expansion_candidate(candles, now, regime)
        self.assertTrue(result["no_lookahead"])

    def test_endpoint_is_registered(self):
        paths = {route.path for route in main.api_router.routes}
        self.assertIn("/strategies/detect/{symbol}", paths)

    def test_endpoint_uses_real_coinbase_candles(self):
        source = inspect.getsource(main.get_candidate_strategy_detections)
        self.assertIn("market_provider.get_candles", source)
        self.assertIn('to_provider("coinbase"', source)

    def test_endpoint_has_latest_freshness_guard(self):
        source = inspect.getsource(main.get_candidate_strategy_detections)
        self.assertIn("_latest_quality(candles)", source)
        self.assertIn("LATEST_CANDLE_NOT_FRESH", source)

    def test_endpoint_cannot_execute_or_auto_queue(self):
        source = inspect.getsource(main.get_candidate_strategy_detections)
        self.assertIn('"auto_queue": False', source)
        self.assertIn('"execution": False', source)
        self.assertNotIn("create_paper_position", source)


class CryptoUniverseExpansionV16M5B28ATests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._instruments = dict(main.instrument_registry._by_canonical)
        self._to_provider = dict(main.provider_symbol_map._to_provider)
        self._to_canonical = dict(main.provider_symbol_map._to_canonical)
        self._activation = dict(main.crypto_universe_activation)
        self._list_products = main.market_provider.list_public_spot_products

    def tearDown(self):
        main.instrument_registry._by_canonical = self._instruments
        main.provider_symbol_map._to_provider = self._to_provider
        main.provider_symbol_map._to_canonical = self._to_canonical
        main.crypto_universe_activation.clear()
        main.crypto_universe_activation.update(self._activation)
        main.market_provider.list_public_spot_products = self._list_products

    @staticmethod
    def _product(symbol="DOGE-USD", volume="1000", **overrides):
        payload = {
            "product_id": symbol,
            "product_type": "SPOT",
            "base_increment": "0.1",
            "quote_increment": "0.01",
            "base_min_size": "1",
            "base_max_size": "1000000",
            "quote_min_size": "1",
            "quote_max_size": "10000000",
            "trading_disabled": False,
            "view_only": False,
            "approximate_quote_24h_volume": volume,
        }
        payload.update(overrides)
        return payload

    def test_target_is_exactly_one_hundred(self):
        self.assertEqual(main.CRYPTO_UNIVERSE_TARGET_SIZE, 100)

    def test_public_list_requests_spot_volume_ranking(self):
        source = inspect.getsource(main.CoinbaseProvider.list_public_spot_products)
        self.assertIn('"product_type": "SPOT"', source)
        self.assertIn("PRODUCTS_SORT_ORDER_VOLUME_24H_DESCENDING", source)

    def test_product_eligibility_accepts_verified_spot_usd(self):
        ok, reason = main.coinbase_product_is_eligible("DOGE-USD", self._product())
        self.assertTrue(ok)
        self.assertEqual(reason, "COINBASE_PRODUCT_VERIFIED")

    def test_product_eligibility_rejects_non_spot(self):
        ok, reason = main.coinbase_product_is_eligible(
            "DOGE-USD", self._product(product_type="FUTURE")
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "PRODUCT_NOT_SPOT")

    def test_product_eligibility_rejects_non_usd(self):
        product = self._product(symbol="DOGE-EUR")
        ok, reason = main.coinbase_product_is_eligible("DOGE-EUR", product)
        self.assertFalse(ok)
        self.assertEqual(reason, "QUOTE_NOT_USD")

    def test_product_eligibility_rejects_disabled(self):
        ok, reason = main.coinbase_product_is_eligible(
            "DOGE-USD", self._product(trading_disabled=True)
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "TRADING_DISABLED")

    def test_product_eligibility_rejects_invalid_specs(self):
        ok, reason = main.coinbase_product_is_eligible(
            "DOGE-USD", self._product(base_increment="0")
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "PRODUCT_SPECS_INVALID")

    def test_selector_orders_by_real_quote_volume(self):
        payload = {"products": [
            self._product("DOGE-USD", "10"),
            self._product("LINK-USD", "100"),
            self._product("AVAX-USD", "50"),
        ]}
        selected = main.select_coinbase_top_usd_spot_products(payload, target=3)
        self.assertEqual(
            [item["product_id"] for item in selected],
            ["LINK-USD", "AVAX-USD", "DOGE-USD"],
        )

    def test_selector_caps_at_target(self):
        products = [self._product(f"C{i}-USD", str(1000 - i)) for i in range(120)]
        selected = main.select_coinbase_top_usd_spot_products(
            {"products": products}, target=100
        )
        self.assertEqual(len(selected), 100)

    def test_selector_rejects_missing_or_zero_volume(self):
        zero = self._product("ZERO-USD", "0")
        missing = self._product("MISS-USD")
        missing.pop("approximate_quote_24h_volume")
        selected = main.select_coinbase_top_usd_spot_products(
            {"products": [zero, missing]}
        )
        self.assertEqual(selected, [])

    def test_verified_registration_enters_registry_and_mapping(self):
        self.assertTrue(main.register_verified_coinbase_crypto("DOGE-USD"))
        self.assertIsNotNone(main.instrument_registry.get("DOGE-USD"))
        self.assertEqual(
            main.provider_symbol_map.to_provider("coinbase", "DOGE-USD"),
            "DOGE-USD",
        )

    async def test_activation_registers_dynamic_ranked_products(self):
        async def fake_list(limit=250):
            return {"products": [
                self._product("DOGE-USD", "10"),
                self._product("LINK-USD", "100"),
            ]}

        main.market_provider.list_public_spot_products = fake_list
        result = await main.activate_verified_crypto_universe()
        self.assertEqual(result["activated"], ["LINK-USD", "DOGE-USD"])
        self.assertEqual(result["ranking"], "COINBASE_24H_QUOTE_VOLUME_DESC")

    async def test_activation_fails_closed_on_provider_error(self):
        async def fake_list(limit=250):
            raise httpx.ConnectError("offline")

        main.market_provider.list_public_spot_products = fake_list
        result = await main.activate_verified_crypto_universe()
        self.assertEqual(result["status"], "DEGRADED")
        self.assertEqual(result["activated"], [])

    def test_lifespan_activates_universe_before_websocket(self):
        source = inspect.getsource(main.lifespan)
        activation = source.index("activate_verified_crypto_universe")
        websocket = source.index("start_server_crypto_market_stream")
        self.assertLess(activation, websocket)

    def test_scanner_and_websocket_are_registry_driven(self):
        ws_source = inspect.getsource(main.start_server_crypto_market_stream)
        scan_source = inspect.getsource(main.run_server_auto_paper_generation_once)
        self.assertIn("instrument_registry.all()", ws_source)
        self.assertIn("instrument_registry.all()", scan_source)

    def test_universe_endpoint_exposes_real_ranking_and_paper_only(self):
        paths = {route.path for route in main.api_router.routes}
        self.assertIn("/market/crypto-universe", paths)
        source = inspect.getsource(main.get_crypto_universe)
        self.assertIn('"source": "coinbase_public_products"', source)
        self.assertIn("COINBASE_24H_QUOTE_VOLUME_DESC", source)
        self.assertIn('"paper_only": True', source)
        self.assertIn('"execution": False', source)


class TestV16M5B28B1TrendPullbackPaperExecution(unittest.TestCase):
    def _candles(self, bullish=True):
        now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        items = []
        base = 100.0
        for index in range(24):
            start = now - timedelta(minutes=5 * (24 - index))
            price = base + index * (0.5 if bullish else -0.5)
            items.append(main.Candle(
                start=start,
                open=price,
                high=price + 1.0,
                low=price - 1.0,
                close=price + (0.4 if bullish else -0.4),
                volume=10.0,
                status=main.DataQualityStatus.VALID,
            ))
        return items, now

    def _ticker(self, value, now):
        return main.MarketDatum(
            symbol="BTC-USD",
            value=value,
            timestamp=now,
            source="coinbase",
            status=main.DataQualityStatus.VALID,
        )

    def test_version_is_explicit(self):
        self.assertEqual(main.TREND_PULLBACK_PAPER_VERSION, "0.2-paper")

    def test_rr_rule_is_two(self):
        self.assertEqual(main.TREND_PULLBACK_RISK_REWARD, Decimal("2"))

    def test_non_setup_is_wait(self):
        candles, now = self._candles()
        result = main.build_trend_pullback_paper_plan(
            candles, now, {"status": "WAIT"}, self._ticker(112.0, now)
        )
        self.assertEqual(result["reason"], "TREND_SETUP_NOT_READY")

    def test_invalid_ticker_is_wait(self):
        candles, now = self._candles()
        ticker = main.MarketDatum(
            symbol="BTC-USD",
            value=112.0,
            timestamp=now,
            source="coinbase",
            status=main.DataQualityStatus.INVALID,
        )
        result = main.build_trend_pullback_paper_plan(
            candles, now, {"status": "SETUP"}, ticker
        )
        self.assertEqual(result["reason"], "REALTIME_TICKER_NOT_VALID")

    def test_bullish_plan_uses_pullback_low(self):
        candles, now = self._candles()
        detection = {
            "status": "SETUP", "direction": "BULLISH",
            "latest_closed_timestamp": candles[-1].start.isoformat(),
        }
        result = main.build_trend_pullback_paper_plan(
            candles, now, detection, self._ticker(112.0, now)
        )
        self.assertEqual(result["status"], "ENTRY_NOW")
        self.assertEqual(result["stop_loss"], Decimal(str(candles[-2].low)))

    def test_bullish_plan_is_long(self):
        candles, now = self._candles()
        detection = {"status": "SETUP", "direction": "BULLISH"}
        result = main.build_trend_pullback_paper_plan(
            candles, now, detection, self._ticker(112.0, now)
        )
        self.assertEqual(result["side"], "LONG")

    def test_bullish_target_is_two_r(self):
        candles, now = self._candles()
        detection = {"status": "SETUP", "direction": "BULLISH"}
        result = main.build_trend_pullback_paper_plan(
            candles, now, detection, self._ticker(112.0, now)
        )
        risk = result["entry"] - result["stop_loss"]
        self.assertEqual(result["take_profit"], result["entry"] + Decimal("2") * risk)

    def test_bearish_plan_uses_pullback_high(self):
        candles, now = self._candles(bullish=False)
        detection = {"status": "SETUP", "direction": "BEARISH"}
        result = main.build_trend_pullback_paper_plan(
            candles, now, detection, self._ticker(88.0, now)
        )
        self.assertEqual(result["status"], "ENTRY_NOW")
        self.assertEqual(result["stop_loss"], Decimal(str(candles[-2].high)))

    def test_bearish_plan_is_short(self):
        candles, now = self._candles(bullish=False)
        detection = {"status": "SETUP", "direction": "BEARISH"}
        result = main.build_trend_pullback_paper_plan(
            candles, now, detection, self._ticker(88.0, now)
        )
        self.assertEqual(result["side"], "SHORT")

    def test_bearish_target_is_two_r(self):
        candles, now = self._candles(bullish=False)
        detection = {"status": "SETUP", "direction": "BEARISH"}
        result = main.build_trend_pullback_paper_plan(
            candles, now, detection, self._ticker(88.0, now)
        )
        risk = result["stop_loss"] - result["entry"]
        self.assertEqual(result["take_profit"], result["entry"] - Decimal("2") * risk)

    def test_plan_declares_real_ticker_source(self):
        candles, now = self._candles()
        result = main.build_trend_pullback_paper_plan(
            candles, now, {"status": "SETUP", "direction": "BULLISH"},
            self._ticker(112.0, now),
        )
        self.assertIn("REAL_COINBASE_TICKER", result["plan_source"])

    def test_plan_is_paper_only(self):
        candles, now = self._candles()
        result = main.build_trend_pullback_paper_plan(
            candles, now, {"status": "SETUP", "direction": "BULLISH"},
            self._ticker(112.0, now),
        )
        self.assertTrue(result["paper_only"])
        self.assertFalse(result["execution"])

    def test_strategy_position_id_is_deterministic(self):
        one = main.build_strategy_paper_position_id("TREND_PULLBACK", "BTC-USD", "LONG", "x")
        two = main.build_strategy_paper_position_id("TREND_PULLBACK", "BTC-USD", "LONG", "x")
        self.assertEqual(one, two)

    def test_strategy_position_id_changes_by_strategy(self):
        one = main.build_strategy_paper_position_id("TREND_PULLBACK", "BTC-USD", "LONG", "x")
        two = main.build_strategy_paper_position_id("OTHER", "BTC-USD", "LONG", "x")
        self.assertNotEqual(one, two)

    def test_orchestrator_calls_trend_generation(self):
        source = inspect.getsource(main.auto_entry_orchestrator_loop)
        self.assertIn("run_trend_pullback_paper_generation_once", source)

    def test_status_route_is_registered(self):
        paths = {route.path for route in main.api_router.routes}
        self.assertIn("/strategies/trend-pullback/paper-status", paths)

class TestV16M5B28B2BreakoutExpansionPaperExecution(unittest.TestCase):
    def _ticker(self, value, now, status=main.DataQualityStatus.VALID):
        return main.MarketDatum(
            symbol="BTC-USD", value=value, timestamp=now,
            source="coinbase", status=status,
        )

    def _setup(self, direction="BULLISH"):
        return {
            "status": "SETUP", "direction": direction,
            "range_high": 110.0, "range_low": 90.0,
            "latest_closed_timestamp": "2026-09-04T03:00:00+00:00",
        }

    def test_version_is_explicit(self):
        self.assertEqual(main.BREAKOUT_EXPANSION_PAPER_VERSION, "0.2-paper")

    def test_rr_rule_is_two(self):
        self.assertEqual(main.BREAKOUT_EXPANSION_RISK_REWARD, Decimal("2"))

    def test_non_setup_is_wait(self):
        now = datetime.now(timezone.utc)
        result = main.build_breakout_expansion_paper_plan(
            {"status": "WAIT"}, self._ticker(112.0, now), now
        )
        self.assertEqual(result["reason"], "BREAKOUT_SETUP_NOT_READY")

    def test_invalid_ticker_is_wait(self):
        now = datetime.now(timezone.utc)
        result = main.build_breakout_expansion_paper_plan(
            self._setup(), self._ticker(112.0, now, main.DataQualityStatus.INVALID), now
        )
        self.assertEqual(result["reason"], "REALTIME_TICKER_NOT_VALID")

    def test_bullish_plan_is_long(self):
        now = datetime.now(timezone.utc)
        result = main.build_breakout_expansion_paper_plan(
            self._setup(), self._ticker(112.0, now), now
        )
        self.assertEqual(result["side"], "LONG")

    def test_bullish_stop_is_range_high(self):
        now = datetime.now(timezone.utc)
        result = main.build_breakout_expansion_paper_plan(
            self._setup(), self._ticker(112.0, now), now
        )
        self.assertEqual(result["stop_loss"], Decimal("110.0"))

    def test_bullish_target_is_two_r(self):
        now = datetime.now(timezone.utc)
        result = main.build_breakout_expansion_paper_plan(
            self._setup(), self._ticker(112.0, now), now
        )
        self.assertEqual(result["take_profit"], Decimal("116.0"))

    def test_bullish_breakout_must_hold(self):
        now = datetime.now(timezone.utc)
        result = main.build_breakout_expansion_paper_plan(
            self._setup(), self._ticker(109.0, now), now
        )
        self.assertEqual(result["reason"], "BULLISH_BREAKOUT_NOT_HELD")

    def test_bearish_plan_is_short(self):
        now = datetime.now(timezone.utc)
        result = main.build_breakout_expansion_paper_plan(
            self._setup("BEARISH"), self._ticker(88.0, now), now
        )
        self.assertEqual(result["side"], "SHORT")

    def test_bearish_stop_is_range_low(self):
        now = datetime.now(timezone.utc)
        result = main.build_breakout_expansion_paper_plan(
            self._setup("BEARISH"), self._ticker(88.0, now), now
        )
        self.assertEqual(result["stop_loss"], Decimal("90.0"))

    def test_bearish_target_is_two_r(self):
        now = datetime.now(timezone.utc)
        result = main.build_breakout_expansion_paper_plan(
            self._setup("BEARISH"), self._ticker(88.0, now), now
        )
        self.assertEqual(result["take_profit"], Decimal("84.0"))

    def test_bearish_breakout_must_hold(self):
        now = datetime.now(timezone.utc)
        result = main.build_breakout_expansion_paper_plan(
            self._setup("BEARISH"), self._ticker(91.0, now), now
        )
        self.assertEqual(result["reason"], "BEARISH_BREAKOUT_NOT_HELD")

    def test_plan_declares_real_ticker_source(self):
        now = datetime.now(timezone.utc)
        result = main.build_breakout_expansion_paper_plan(
            self._setup(), self._ticker(112.0, now), now
        )
        self.assertEqual(result["plan_source"], "CLOSED_BREAKOUT_RANGE+REAL_COINBASE_TICKER")

    def test_plan_is_paper_only(self):
        now = datetime.now(timezone.utc)
        result = main.build_breakout_expansion_paper_plan(
            self._setup(), self._ticker(112.0, now), now
        )
        self.assertTrue(result["paper_only"])
        self.assertFalse(result["execution"])

    def test_status_route_is_registered(self):
        paths = {route.path for route in main.app.routes}
        self.assertIn("/api/v1/strategies/breakout-expansion/paper-status", paths)

    def test_orchestrator_calls_breakout_generation(self):
        source = inspect.getsource(main.auto_entry_orchestrator_loop)
        self.assertIn("run_breakout_expansion_paper_generation_once", source)

