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
VIEW_IDS = ["markets", "forex", "system", "detail"]


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

    def test_not_configured_placeholder_not_invented(self):
        # a declared-but-not-implemented policy must NOT invent hours
        cal = calendar_for(MarketCalendarPolicy.FOREX_WEEK)
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

    def test_forex_calendar_not_configured(self):
        self.assertEqual(instrument_registry.get("EUR-USD").market_calendar,
                         MarketCalendarPolicy.NOT_CONFIGURED)

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