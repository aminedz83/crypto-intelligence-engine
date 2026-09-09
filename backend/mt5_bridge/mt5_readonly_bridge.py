"""
V17-MT5-BRIDGE-1 — Windows/VPS read-only MetaTrader 5 bridge.

Purpose:
- Read the exact symbol catalogue and broker specifications from a locally
  running/logged-in MT5 terminal on Windows/VPS.
- Preserve exact broker symbol names.
- Apply canonical mappings ONLY from an explicit local JSON mapping file.
- POST the read-only snapshot to Crypto Intelligence Engine's existing
  /api/v1/mt5/brokers/{broker_id}/sync endpoint.

This bridge never submits, modifies, or closes broker orders.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

BRIDGE_VERSION = "V17_MT5_BRIDGE_READONLY_1"
DEFAULT_TIMEOUT_SECONDS = 20.0


class BridgeError(RuntimeError):
    """Clean bridge failure with no credential disclosure."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_decimal(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        decimal = Decimal(str(value))
    except Exception:
        return None
    if not decimal.is_finite():
        return None
    return format(decimal, "f")


def clean_optional_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def load_explicit_mapping(path: Optional[str]) -> Dict[str, str]:
    """Load exact broker_symbol -> canonical_symbol mappings. No guessing."""
    if not path:
        return {}
    mapping_path = Path(path)
    try:
        raw = json.loads(mapping_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise BridgeError(f"Mapping file not found: {mapping_path}") from exc
    except json.JSONDecodeError as exc:
        raise BridgeError(f"Mapping file is not valid JSON: {mapping_path}") from exc
    if not isinstance(raw, dict):
        raise BridgeError("Mapping JSON must be an object of exact symbol mappings.")
    result: Dict[str, str] = {}
    for broker_symbol, canonical_symbol in raw.items():
        if not isinstance(broker_symbol, str) or not broker_symbol.strip():
            raise BridgeError("Mapping contains an invalid broker symbol.")
        if not isinstance(canonical_symbol, str) or not canonical_symbol.strip():
            raise BridgeError(f"Mapping for {broker_symbol!r} is invalid.")
        result[broker_symbol] = canonical_symbol.strip().upper().replace("/", "-")
    return result


def load_mt5_module() -> Any:
    """Import MetaTrader5 only on the Windows/VPS runtime, never at module import."""
    try:
        return importlib.import_module("MetaTrader5")
    except Exception as exc:
        raise BridgeError(
            "MetaTrader5 Python package is unavailable. Install it on the Windows/VPS "
            "that runs the MT5 terminal."
        ) from exc


def trade_mode_name(mt5: Any, value: Any) -> Optional[str]:
    pairs = (
        ("SYMBOL_TRADE_MODE_DISABLED", "DISABLED"),
        ("SYMBOL_TRADE_MODE_LONGONLY", "LONGONLY"),
        ("SYMBOL_TRADE_MODE_SHORTONLY", "SHORTONLY"),
        ("SYMBOL_TRADE_MODE_CLOSEONLY", "CLOSEONLY"),
        ("SYMBOL_TRADE_MODE_FULL", "FULL"),
    )
    for constant_name, label in pairs:
        if getattr(mt5, constant_name, object()) == value:
            return label
    return str(value) if value is not None else None


def asset_group_from_path(path: Optional[str]) -> Optional[str]:
    """Informational grouping from MT5's own path, not a canonical identity guess."""
    if not path:
        return None
    first = path.replace("/", "\\").split("\\", 1)[0].strip()
    return first or None


@dataclass(frozen=True)
class TerminalIdentity:
    broker_name: str
    server: Optional[str]
    account_currency: Optional[str]


def terminal_identity(mt5: Any) -> TerminalIdentity:
    account = mt5.account_info()
    terminal = mt5.terminal_info()
    if account is None:
        raise BridgeError("MT5 account_info() is unavailable. Log in to the terminal first.")
    company = clean_optional_text(getattr(account, "company", None))
    terminal_company = clean_optional_text(getattr(terminal, "company", None)) if terminal else None
    broker_name = company or terminal_company or "MT5 Broker"
    return TerminalIdentity(
        broker_name=broker_name,
        server=clean_optional_text(getattr(account, "server", None)),
        account_currency=clean_optional_text(getattr(account, "currency", None)),
    )


def symbol_payload(
    mt5: Any,
    info: Any,
    mapping: Mapping[str, str],
    observed_at: str,
) -> Dict[str, Any]:
    name = clean_optional_text(getattr(info, "name", None))
    if not name:
        raise BridgeError("MT5 returned a symbol with no exact name.")
    tick = mt5.symbol_info_tick(name)
    path = clean_optional_text(getattr(info, "path", None))
    metadata: Dict[str, Any] = {
        "bridge_version": BRIDGE_VERSION,
        "bid": safe_decimal(getattr(tick, "bid", None)) if tick is not None else None,
        "ask": safe_decimal(getattr(tick, "ask", None)) if tick is not None else None,
        "last": safe_decimal(getattr(tick, "last", None)) if tick is not None else None,
        "tick_time_msc": getattr(tick, "time_msc", None) if tick is not None else None,
        "select": bool(getattr(info, "select", False)),
        "session_deals": getattr(info, "session_deals", None),
    }
    return {
        "broker_symbol": name,
        "canonical_symbol": mapping.get(name),
        "description": clean_optional_text(getattr(info, "description", None)),
        "path": path,
        "asset_group": asset_group_from_path(path),
        "digits": getattr(info, "digits", None),
        "point": safe_decimal(getattr(info, "point", None)),
        "tick_size": safe_decimal(getattr(info, "trade_tick_size", None)),
        "tick_value": safe_decimal(getattr(info, "trade_tick_value", None)),
        "contract_size": safe_decimal(getattr(info, "trade_contract_size", None)),
        "volume_min": safe_decimal(getattr(info, "volume_min", None)),
        "volume_max": safe_decimal(getattr(info, "volume_max", None)),
        "volume_step": safe_decimal(getattr(info, "volume_step", None)),
        "currency_base": clean_optional_text(getattr(info, "currency_base", None)),
        "currency_profit": clean_optional_text(getattr(info, "currency_profit", None)),
        "currency_margin": clean_optional_text(getattr(info, "currency_margin", None)),
        "trade_mode": trade_mode_name(mt5, getattr(info, "trade_mode", None)),
        "visible": bool(getattr(info, "visible", False)),
        "stops_level": getattr(info, "trade_stops_level", None),
        "freeze_level": getattr(info, "trade_freeze_level", None),
        "observed_at": observed_at,
        "source": "MT5_TERMINAL",
        "metadata": metadata,
    }


def build_snapshot(mt5: Any, mapping: Mapping[str, str]) -> Dict[str, Any]:
    identity = terminal_identity(mt5)
    symbols = mt5.symbols_get()
    if symbols is None:
        raise BridgeError("MT5 symbols_get() failed.")
    observed_at = utc_now_iso()
    rows = [symbol_payload(mt5, info, mapping, observed_at) for info in symbols]
    return {
        "broker_name": identity.broker_name,
        "server": identity.server,
        "account_currency": identity.account_currency,
        "symbols": rows,
    }


def sync_url(api_base_url: str, broker_id: str) -> str:
    base = api_base_url.rstrip("/")
    safe_id = broker_id.strip()
    if not safe_id:
        raise BridgeError("MT5_BROKER_ID is required.")
    return f"{base}/api/v1/mt5/brokers/{safe_id}/sync"


def post_snapshot(
    api_base_url: str,
    broker_id: str,
    snapshot: Mapping[str, Any],
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    body = json.dumps(snapshot, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        sync_url(api_base_url, broker_id),
        data=body,
        headers={"Content-Type": "application/json", "User-Agent": BRIDGE_VERSION},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise BridgeError(f"Engine sync rejected with HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise BridgeError("Cannot reach Crypto Intelligence Engine sync endpoint.") from exc
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise BridgeError("Engine sync returned invalid JSON.") from exc
    if not isinstance(parsed, dict):
        raise BridgeError("Engine sync returned an unexpected response.")
    return parsed


def initialize_terminal(mt5: Any, terminal_path: Optional[str]) -> None:
    kwargs: Dict[str, Any] = {}
    if terminal_path:
        kwargs["path"] = terminal_path
    if not mt5.initialize(**kwargs):
        code, message = mt5.last_error()
        raise BridgeError(f"MT5 initialize() failed: code={code}, message={message}")


def shutdown_terminal(mt5: Any) -> None:
    try:
        mt5.shutdown()
    except Exception:
        pass


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only MT5 catalogue bridge")
    parser.add_argument("--dry-run", action="store_true", help="Read MT5 but do not sync")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    return parser.parse_args(list(argv) if argv is not None else None)


def run(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    api_base_url = os.getenv("CIE_API_BASE_URL", "http://127.0.0.1:8000").strip()
    broker_id = os.getenv("MT5_BROKER_ID", "").strip()
    mapping_file = os.getenv("MT5_MAPPING_FILE", "").strip() or None
    terminal_path = os.getenv("MT5_TERMINAL_PATH", "").strip() or None
    mapping = load_explicit_mapping(mapping_file)
    mt5 = load_mt5_module()
    initialize_terminal(mt5, terminal_path)
    try:
        snapshot = build_snapshot(mt5, mapping)
        if args.dry_run:
            result: Dict[str, Any] = {
                "bridge_version": BRIDGE_VERSION,
                "dry_run": True,
                "broker_name": snapshot["broker_name"],
                "server": snapshot["server"],
                "account_currency": snapshot["account_currency"],
                "symbol_count": len(snapshot["symbols"]),
                "mapped_count": sum(
                    1 for row in snapshot["symbols"] if row["canonical_symbol"] is not None
                ),
                "paper_only": True,
                "execution": False,
            }
        else:
            if not broker_id:
                raise BridgeError("MT5_BROKER_ID is required unless --dry-run is used.")
            result = post_snapshot(api_base_url, broker_id, snapshot)
        print(json.dumps(result, indent=2 if args.pretty else None, ensure_ascii=False))
        return 0
    finally:
        shutdown_terminal(mt5)


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except BridgeError as exc:
        print(f"MT5 bridge error: {exc}", file=sys.stderr)
        raise SystemExit(2)
