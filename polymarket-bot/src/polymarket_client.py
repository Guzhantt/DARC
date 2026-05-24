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
from typing import Optional

import requests

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"

log = logging.getLogger(__name__)


class PolymarketError(RuntimeError):
    """Raised when Polymarket API returns something we can't recover from."""


@dataclass
class MarketInfo:
    """Subset of market metadata we actually use."""
    slug: str
    question: str
    yes_token_id: str
    no_token_id: str
    end_date_iso: Optional[str]   # ISO 8601, may be None for open-ended markets
    closed: bool
    active: bool


@dataclass
class MarketSnapshot:
    """One tick's worth of market data for the strategy."""
    yes_price: float
    no_price: float
    remaining_sec: float          # may be negative if past end_date
    fetched_at: datetime


def current_aligned_unix(period_sec: int, now_sec: Optional[float] = None) -> int:
    """Largest multiple of `period_sec` not exceeding `now`.

    For Polymarket's recurring up/down series the slug timestamp is the
    market's START time, e.g. btc-updown-15m-1779634800 starts at 2026-05-24
    15:00 UTC and ends at 15:15 UTC. So the *currently trading* slug is
    floor(now / period) * period. At an exact boundary we advance to the new
    market (the previous one is settling and no longer tradeable).
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

    def get_current_recurring_market(self, slug_template: str, period_sec: int) -> MarketInfo:
        """Resolve the active market for a recurring series like btc-updown-15m.

        Tries the current aligned start timestamp first. If that slug isn't
        listed yet (rare — Polymarket usually pre-lists), falls back to the
        next one and the previous one before giving up.
        """
        last_err: Optional[Exception] = None
        base = current_aligned_unix(period_sec)
        # Order: current → next (in case current isn't listed yet) → previous
        # (in case we just rolled over and the new one is delayed).
        for offset_periods in (0, 1, -1):
            start_unix = base + offset_periods * period_sec
            slug = slug_template.format(start_unix=start_unix)
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
            f"No active recurring market found for template={slug_template!r}; last error: {last_err}"
        )

    def get_market_by_slug(self, slug: str) -> MarketInfo:
        """Resolve a market slug to token IDs and end date via Gamma.

        Polymarket exposes both /markets and /events. Recurring up/down markets
        are usually accessible directly under /markets, but for /event/<slug>
        URLs we fall back to /events and pick the first child market.
        """
        # Try /markets first.
        info = self._try_markets_endpoint(slug)
        if info is not None:
            return info
        # Fall back to /events (the URL pattern /event/<slug> uses this).
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
        # Recurring up/down events have exactly one child market; pick the first
        # active, non-closed one.
        for m in markets:
            if m.get("active", True) and not m.get("closed", False):
                return _market_from_gamma(m)
        return _market_from_gamma(markets[0])

    # ---------- prices ----------

    def get_midpoint(self, token_id: str) -> float:
        """Midpoint price for a single CLOB token. Returns price in [0, 1]."""
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

    def get_snapshot(self, market: MarketInfo) -> MarketSnapshot:
        """Fetch YES + NO midpoint and compute remaining_sec from end_date_iso."""
        yes_price = self.get_midpoint(market.yes_token_id)
        no_price = self.get_midpoint(market.no_token_id)
        now = datetime.now(timezone.utc)
        remaining = _remaining_seconds(market.end_date_iso, now)
        return MarketSnapshot(
            yes_price=yes_price,
            no_price=no_price,
            remaining_sec=remaining,
            fetched_at=now,
        )


def _market_from_gamma(m: dict) -> MarketInfo:
    """Parse a market dict from either /markets or /events response."""
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

    # Map outcomes to token IDs by index. Polymarket convention varies — Up/Down,
    # Yes/No, True/False all show up — so we match defensively.
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
    )


def _remaining_seconds(end_date_iso: Optional[str], now: datetime) -> float:
    """Parse Polymarket's ISO end date and return seconds until it.

    Returns +inf when the market has no end date (open-ended), so time-based
    rules degrade gracefully — strategy can still gate on price/momentum.
    """
    if not end_date_iso:
        return float("inf")
    try:
        # Polymarket sometimes returns "2025-05-24T19:00:00Z" and sometimes with offset.
        end = datetime.fromisoformat(end_date_iso.replace("Z", "+00:00"))
    except ValueError:
        log.warning("Could not parse end_date_iso=%r, treating as no deadline", end_date_iso)
        return float("inf")
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    return (end - now).total_seconds()
