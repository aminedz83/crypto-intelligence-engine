"""All backend tests in one file (single-file layout).

38 tests, none skipped. Pure-logic tests (data quality, health aggregation) plus
integration tests (health endpoints, failure modes, frontend serving) that need
FastAPI + httpx. No conditional skip: a missing dependency fails the run rather
than skipping.
"""

import asyncio
import json
import unittest
from datetime import datetime, timedelta, timezone
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
VIEW_IDS = [
    "dashboard", "scanner", "signals", "positions", "history", "news", "macro",
    "geopolitical", "onchain", "strategies", "backtesting", "ai", "risk",
    "performance", "system", "settings",
]


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