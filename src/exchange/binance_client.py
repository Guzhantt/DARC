"""Thin wrapper around ccxt.binanceusdm with Testnet support."""
from __future__ import annotations

from typing import Any

import ccxt
import pandas as pd

from src.config import Config


class BinanceFutures:
    """USDT-M Perpetual Futures client."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.exchange = ccxt.binanceusdm(
            {
                "apiKey": cfg.api_key,
                "secret": cfg.api_secret,
                "enableRateLimit": True,
                "options": {"defaultType": "future"},
            }
        )
        if cfg.use_testnet:
            self.exchange.set_sandbox_mode(True)

    # ---------- Market data ----------
    def load_markets(self) -> dict[str, Any]:
        return self.exchange.load_markets()

    def list_usdt_perpetuals(self) -> list[str]:
        """Return all tradable USDT-margined perpetual symbols, e.g. 'BTC/USDT:USDT'."""
        markets = self.exchange.load_markets()
        out: list[str] = []
        for sym, m in markets.items():
            if (
                m.get("swap")
                and m.get("linear")
                and m.get("quote") == "USDT"
                and m.get("active", True)
            ):
                out.append(sym)
        return sorted(out)

    def fetch_ohlcv(
        self, symbol: str, timeframe: str, since_ms: int | None = None, limit: int = 1500
    ) -> pd.DataFrame:
        raw = self.exchange.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=limit)
        df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        return df

    def fetch_top_volume_symbols(self, n: int = 30) -> list[str]:
        """Top n USDT perpetuals by 24h quote volume."""
        tickers = self.exchange.fetch_tickers()
        perps = set(self.list_usdt_perpetuals())
        ranked = sorted(
            (
                (s, t.get("quoteVolume") or 0.0)
                for s, t in tickers.items()
                if s in perps
            ),
            key=lambda x: x[1],
            reverse=True,
        )
        return [s for s, _ in ranked[:n]]

    # ---------- Account / trading ----------
    def fetch_balance_usdt(self) -> float:
        bal = self.exchange.fetch_balance()
        return float(bal["total"].get("USDT", 0.0))

    def set_leverage(self, symbol: str, leverage: int) -> None:
        try:
            self.exchange.set_leverage(leverage, symbol)
        except Exception:
            pass  # already set, or exchange doesn't allow change

    def fetch_position(self, symbol: str) -> dict[str, Any] | None:
        positions = self.exchange.fetch_positions([symbol])
        for p in positions:
            contracts = float(p.get("contracts") or 0)
            if contracts != 0:
                return p
        return None

    def create_market_order(
        self, symbol: str, side: str, amount: float, reduce_only: bool = False
    ) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if reduce_only:
            params["reduceOnly"] = True
        return self.exchange.create_order(symbol, "market", side, amount, None, params)

    def create_stop_market(
        self, symbol: str, side: str, amount: float, stop_price: float
    ) -> dict[str, Any]:
        params = {"stopPrice": stop_price, "reduceOnly": True}
        return self.exchange.create_order(symbol, "STOP_MARKET", side, amount, None, params)

    def cancel_all(self, symbol: str) -> None:
        try:
            self.exchange.cancel_all_orders(symbol)
        except Exception:
            pass
