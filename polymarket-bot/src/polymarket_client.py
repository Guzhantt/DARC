"""Polymarket data client.

Two public APIs are used:
  - Gamma API (gamma-api.polymarket.com) for market metadata + token IDs by slug.
  - CLOB API  (clob.polymarket.com)       for live orderbook prices.

Both endpoints are read-only and require no auth. For order placement you
need py-clob-client + a funded Polygon wallet — see executor.LiveExecutor.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

import requests

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"

log = logging.getLogger(__name__)


class PolymarketError(RuntimeError):
    """Raised when Polymarket API returns something we can't recover from."""


@dataclass
class MarketInfo:
    slug: str
    question: str
    yes_token_id: str
    no_token_id: str
    end_date_iso: Optional[str]
    closed: bool
    active: bool
    # Display labels from Polymarket — e.g. "Up" / "Down" for the BTC up/down
    # series, "Yes" / "No" for typical event markets. Internally we still call
    # them YES/NO (yes_token_id/no_token_id) but UIs should prefer these.
    yes_label: str = "YES"
    no_label: str = "NO"


@dataclass
class BookSide:
    best_price: float            # best ask for buys; we derive best bid from spread
    spread: float                # ask - bid
    depth_at_best_usd: float
    depth_within_1pct_usd: float
    has_quotes: bool


@dataclass
class MarketSnapshot:
    yes_price: float
    no_price: float
    remaining_sec: float
    fetched_at: datetime
    yes_book: Optional[BookSide] = None
    no_book: Optional[BookSide] = None


def current_aligned_unix(period_sec: int, now_sec: Optional[float] = None) -> int:
    """floor(now / period) * period.

    For Polymarket's recurring up/down series the slug timestamp is the
    market's START time, e.g. btc-updown-15m-1779634800 starts at 2026-05-24
    15:00 UTC and ends at 15:15 UTC.
    """
    if now_sec is None:
        now_sec = time.time()
    if period_sec <= 0:
        raise ValueError("period_sec must be positive")
    return (int(now_sec) // period_sec) * period_sec


class PolymarketClient:
    def __init__(self, timeout_sec: float = 10.0):
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "polymarket-bot/0.1"})
        self._timeout = timeout_sec

    # ---------- metadata ----------

    def get_current_recurring_market(
        self, slug_templates: List[str], period_sec: int,
    ) -> MarketInfo:
        """Cascade resolver for a recurring series with multiple symbol fallbacks.

        Tries each template in order at offsets {0, +1, -1} periods. First
        active (non-closed) match wins.

        Templates can be e.g. ['btc-updown-15m-{start_unix}',
        'eth-updown-15m-{start_unix}']: BTC tried first, ETH used if BTC fails.
        """
        if not slug_templates:
            raise PolymarketError("No slug templates configured")
        last_err: Optional[Exception] = None
        base = current_aligned_unix(period_sec)
        for tmpl in slug_templates:
            for offset in (0, 1, -1):
                start_unix = base + offset * period_sec
                slug = tmpl.format(start_unix=start_unix)
                try:
                    info = self.get_market_by_slug(slug)
                    if info.closed:
                        log.debug("Slug %s is closed, trying next candidate", slug)
                        continue
                    return info
                except PolymarketError as e:
                    last_err = e
                    log.debug("Recurring slug %s not found: %s", slug, e)
        raise PolymarketError(
            f"No active recurring market for any template; last error: {last_err}"
        )

    def get_market_by_slug(self, slug: str) -> MarketInfo:
        info = self._try_markets_endpoint(slug)
        if info is not None:
            return info
        info = self._try_events_endpoint(slug)
        if info is not None:
            return info
        raise PolymarketError(f"No market or event found for slug={slug!r}")

    def _try_markets_endpoint(self, slug: str) -> Optional[MarketInfo]:
        try:
            resp = self._session.get(f"{GAMMA_BASE}/markets", params={"slug": slug}, timeout=self._timeout)
        except requests.RequestException as e:
            raise PolymarketError(f"Gamma /markets request failed for slug={slug}: {e}") from e
        if resp.status_code == 404:
            return None
        try:
            resp.raise_for_status()
        except requests.RequestException as e:
            raise PolymarketError(f"Gamma /markets HTTP error for slug={slug}: {e}") from e
        data = resp.json()
        if not isinstance(data, list) or not data:
            return None
        return _market_from_gamma(data[0])

    def _try_events_endpoint(self, slug: str) -> Optional[MarketInfo]:
        try:
            resp = self._session.get(f"{GAMMA_BASE}/events", params={"slug": slug}, timeout=self._timeout)
        except requests.RequestException as e:
            raise PolymarketError(f"Gamma /events request failed for slug={slug}: {e}") from e
        if resp.status_code == 404:
            return None
        try:
            resp.raise_for_status()
        except requests.RequestException as e:
            raise PolymarketError(f"Gamma /events HTTP error for slug={slug}: {e}") from e
        data = resp.json()
        if not isinstance(data, list) or not data:
            return None
        markets = data[0].get("markets") or []
        if not markets:
            return None
        for m in markets:
            if m.get("active", True) and not m.get("closed", False):
                return _market_from_gamma(m)
        return _market_from_gamma(markets[0])

    # ---------- prices ----------

    def get_midpoint(self, token_id: str) -> float:
        url = f"{CLOB_BASE}/midpoint"
        try:
            resp = self._session.get(url, params={"token_id": token_id}, timeout=self._timeout)
            resp.raise_for_status()
        except requests.RequestException as e:
            raise PolymarketError(f"CLOB midpoint request failed for {token_id}: {e}") from e
        body = resp.json()
        mid = body.get("mid")
        if mid is None:
            raise PolymarketError(f"CLOB midpoint missing 'mid' field: {body!r}")
        try:
            return float(mid)
        except (TypeError, ValueError) as e:
            raise PolymarketError(f"CLOB midpoint not numeric: {mid!r}") from e

    def get_book(self, token_id: str) -> Optional[BookSide]:
        try:
            resp = self._session.get(
                f"{CLOB_BASE}/book", params={"token_id": token_id}, timeout=self._timeout,
            )
            resp.raise_for_status()
            body = resp.json()
        except requests.RequestException as e:
            log.debug("Book fetch failed for %s: %s", token_id, e)
            return None
        except ValueError:
            return None

        asks = _parse_book_levels(body.get("asks") or [])
        bids = _parse_book_levels(body.get("bids") or [])
        asks.sort(key=lambda x: x[0])
        bids.sort(key=lambda x: x[0], reverse=True)
        if not asks:
            return BookSide(0.0, 1.0, 0.0, 0.0, False)
        best_ask = asks[0][0]
        best_bid = bids[0][0] if bids else 0.0
        spread = max(0.0, best_ask - best_bid)
        depth_at_best = asks[0][0] * asks[0][1]
        cutoff = best_ask * 1.01
        depth_within_1pct = sum(p * s for p, s in asks if p <= cutoff)
        return BookSide(
            best_price=best_ask,
            spread=spread,
            depth_at_best_usd=depth_at_best,
            depth_within_1pct_usd=depth_within_1pct,
            has_quotes=True,
        )

    def get_snapshot(self, market: MarketInfo, fetch_books: bool = True) -> MarketSnapshot:
        yes_price = self.get_midpoint(market.yes_token_id)
        no_price = self.get_midpoint(market.no_token_id)
        now = datetime.now(timezone.utc)
        remaining = _remaining_seconds(market.end_date_iso, now)
        yes_book = self.get_book(market.yes_token_id) if fetch_books else None
        no_book = self.get_book(market.no_token_id) if fetch_books else None
        return MarketSnapshot(
            yes_price=yes_price, no_price=no_price,
            remaining_sec=remaining, fetched_at=now,
            yes_book=yes_book, no_book=no_book,
        )


def _market_from_gamma(m: dict) -> MarketInfo:
    raw_ids = m.get("clobTokenIds")
    if isinstance(raw_ids, str):
        try:
            token_ids = json.loads(raw_ids)
        except json.JSONDecodeError as e:
            raise PolymarketError(f"Cannot parse clobTokenIds: {raw_ids!r}") from e
    else:
        token_ids = raw_ids
    if not isinstance(token_ids, list) or len(token_ids) < 2:
        raise PolymarketError(f"Market has no YES/NO token pair: {token_ids!r}")

    outcomes = m.get("outcomes")
    if isinstance(outcomes, str):
        try:
            outcomes = json.loads(outcomes)
        except json.JSONDecodeError:
            outcomes = ["Yes", "No"]
    if not outcomes:
        outcomes = ["Yes", "No"]

    yes_synonyms = {"yes", "up", "true"}
    no_synonyms = {"no", "down", "false"}
    yes_idx = next((i for i, o in enumerate(outcomes) if str(o).strip().lower() in yes_synonyms), 0)
    if len(token_ids) == 2:
        no_idx = 1 - yes_idx
    else:
        no_idx = next((i for i, o in enumerate(outcomes) if str(o).strip().lower() in no_synonyms), 1)

    return MarketInfo(
        slug=m.get("slug", ""),
        question=m.get("question", ""),
        yes_token_id=str(token_ids[yes_idx]),
        no_token_id=str(token_ids[no_idx]),
        end_date_iso=m.get("endDate") or m.get("end_date_iso"),
        closed=bool(m.get("closed", False)),
        active=bool(m.get("active", True)),
        yes_label=str(outcomes[yes_idx]).strip() or "YES",
        no_label=str(outcomes[no_idx]).strip() or "NO",
    )


def _parse_book_levels(levels: list) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for lvl in levels:
        try:
            p = float(lvl["price"])
            s = float(lvl["size"])
        except (KeyError, TypeError, ValueError):
            continue
        if p <= 0 or s <= 0:
            continue
        out.append((p, s))
    return out


def _remaining_seconds(end_date_iso: Optional[str], now: datetime) -> float:
    if not end_date_iso:
        return float("inf")
    try:
        end = datetime.fromisoformat(end_date_iso.replace("Z", "+00:00"))
    except ValueError:
        log.warning("Could not parse end_date_iso=%r, treating as no deadline", end_date_iso)
        return float("inf")
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    return (end - now).total_seconds()
