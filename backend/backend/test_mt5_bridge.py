
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

BRIDGE_PATH = Path(__file__).resolve().parents[1] / "mt5_bridge" / "mt5_readonly_bridge.py"
SPEC = importlib.util.spec_from_file_location("mt5_readonly_bridge", BRIDGE_PATH)
bridge = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(bridge)


class FakeMT5:
    SYMBOL_TRADE_MODE_DISABLED = 0
    SYMBOL_TRADE_MODE_LONGONLY = 1
    SYMBOL_TRADE_MODE_SHORTONLY = 2
    SYMBOL_TRADE_MODE_CLOSEONLY = 3
    SYMBOL_TRADE_MODE_FULL = 4

    def account_info(self):
        return SimpleNamespace(company="Broker Inc", server="Broker-Demo", currency="USD")

    def terminal_info(self):
        return SimpleNamespace(company="Terminal Co")

    def symbol_info_tick(self, name):
        return SimpleNamespace(bid=1.1, ask=1.2, last=1.15, time_msc=123456)

    def symbols_get(self):
        return [SimpleNamespace(
            name="EURUSD.a", description="Euro Dollar", path="Forex\\Majors",
            digits=5, point=0.00001, trade_tick_size=0.00001,
            trade_tick_value=1.0, trade_contract_size=100000,
            volume_min=0.01, volume_max=100.0, volume_step=0.01,
            currency_base="EUR", currency_profit="USD", currency_margin="EUR",
            trade_mode=4, visible=True, select=True, session_deals=0,
            trade_stops_level=10, trade_freeze_level=5,
        )]


class V17MT5Bridge1Tests(TestCase):
    def test_version_is_explicit(self):
        self.assertEqual(bridge.BRIDGE_VERSION, "V17_MT5_BRIDGE_READONLY_1")

    def test_meta_trader_import_is_lazy(self):
        source = BRIDGE_PATH.read_text(encoding="utf-8")
        self.assertNotIn("import MetaTrader5", source)

    def test_forbidden_execution_primitive_absent(self):
        source = BRIDGE_PATH.read_text(encoding="utf-8")
        forbidden = "order_" + "send"
        self.assertNotIn(forbidden, source)

    def test_mapping_is_exact_not_guessed(self):
        mt5 = FakeMT5()
        row = bridge.symbol_payload(mt5, mt5.symbols_get()[0], {}, bridge.utcnow_iso()
                                    if hasattr(bridge, "utcnow_iso")
                                    else bridge.utc_now_iso())
        self.assertIsNone(row["canonical_symbol"])

    def test_explicit_mapping_is_applied(self):
        mt5 = FakeMT5()
        row = bridge.symbol_payload(
            mt5, mt5.symbols_get()[0], {"EURUSD.a": "EUR-USD"}, bridge.utc_now_iso()
        )
        self.assertEqual(row["canonical_symbol"], "EUR-USD")

    def test_exact_broker_symbol_is_preserved(self):
        mt5 = FakeMT5()
        row = bridge.symbol_payload(mt5, mt5.symbols_get()[0], {}, bridge.utc_now_iso())
        self.assertEqual(row["broker_symbol"], "EURUSD.a")

    def test_critical_specs_are_collected(self):
        mt5 = FakeMT5()
        row = bridge.symbol_payload(mt5, mt5.symbols_get()[0], {}, bridge.utc_now_iso())
        for key in ("point", "tick_size", "tick_value", "contract_size",
                    "volume_min", "volume_max", "volume_step"):
            self.assertIsNotNone(row[key])

    def test_tick_is_informational_metadata(self):
        mt5 = FakeMT5()
        row = bridge.symbol_payload(mt5, mt5.symbols_get()[0], {}, bridge.utc_now_iso())
        self.assertEqual(row["metadata"]["bid"], "1.1")
        self.assertEqual(row["metadata"]["ask"], "1.2")

    def test_terminal_identity_uses_logged_in_account(self):
        identity = bridge.terminal_identity(FakeMT5())
        self.assertEqual(identity.broker_name, "Broker Inc")
        self.assertEqual(identity.server, "Broker-Demo")
        self.assertEqual(identity.account_currency, "USD")

    def test_snapshot_contains_full_catalogue(self):
        snapshot = bridge.build_snapshot(FakeMT5(), {})
        self.assertEqual(len(snapshot["symbols"]), 1)

    def test_sync_url_uses_existing_backend_route(self):
        url = bridge.sync_url("https://example.test/", "broker-1")
        self.assertEqual(url, "https://example.test/api/v1/mt5/brokers/broker-1/sync")

    def test_mapping_file_must_be_object(self):
        path = Path(__file__).with_name("_bad_mt5_mapping.json")
        try:
            path.write_text(json.dumps(["EURUSD"]), encoding="utf-8")
            with self.assertRaises(bridge.BridgeError):
                bridge.load_explicit_mapping(str(path))
        finally:
            path.unlink(missing_ok=True)

    def test_mapping_normalizes_canonical_only(self):
        path = Path(__file__).with_name("_mt5_mapping.json")
        try:
            path.write_text(json.dumps({"EURUSD.a": "eur/usd"}), encoding="utf-8")
            mapping = bridge.load_explicit_mapping(str(path))
            self.assertEqual(mapping, {"EURUSD.a": "EUR-USD"})
        finally:
            path.unlink(missing_ok=True)

    def test_dry_run_does_not_require_broker_id(self):
        fake = FakeMT5()
        fake.initialize = lambda **kwargs: True
        fake.shutdown = lambda: None
        with patch.object(bridge, "load_mt5_module", return_value=fake), \
             patch.dict(bridge.os.environ, {}, clear=True):
            self.assertEqual(bridge.run(["--dry-run"]), 0)

    def test_normal_run_requires_broker_id(self):
        fake = FakeMT5()
        fake.initialize = lambda **kwargs: True
        fake.shutdown = lambda: None
        with patch.object(bridge, "load_mt5_module", return_value=fake), \
             patch.dict(bridge.os.environ, {}, clear=True):
            with self.assertRaises(bridge.BridgeError):
                bridge.run([])

    def test_trade_mode_full_is_named(self):
        self.assertEqual(bridge.trade_mode_name(FakeMT5(), 4), "FULL")

    def test_bridge_declares_terminal_source(self):
        mt5 = FakeMT5()
        row = bridge.symbol_payload(mt5, mt5.symbols_get()[0], {}, bridge.utc_now_iso())
        self.assertEqual(row["source"], "MT5_TERMINAL")
