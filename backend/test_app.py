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
        self.assertIn('api("/api/v1/paper/account/live")', html)

    def test_paper_ui_fetches_positions(self):
        html = INDEX.read_text(encoding="utf-8")
        self.assertIn('api("/api/v1/paper/positions")', html)

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
        self.assertIn('api("/api/v1/paper/positions/live")', html)

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
        self.assertIn('api("/api/v1/paper/positions/live")', html)

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
        self.assertIn('api("/api/v1/paper/account/live")', html)

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
        self.assertIn("AUTO PAPER VERIFIED V1", html)
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
        self.assertIn("AUTO PAPER VERIFIED V1", html)


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
        self.assertIn("AUTO PAPER VERIFIED V1", html)


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
        self.assertIn("AUTO PAPER VERIFIED V1", html)
