"""YAML config loader with light validation."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List

import yaml


@dataclass
class SessionCfg:
    enabled_hours_utc_start: int
    enabled_hours_utc_end: int
    enabled_weekdays: List[int]
    off_hours_poll_sec: int


@dataclass
class RecurringCfg:
    enabled: bool
    slug_templates: List[str]
    period_sec: int


@dataclass
class MarketCfg:
    recurring: RecurringCfg
    slug: str
    yes_token_id: str
    no_token_id: str


@dataclass
class SpotCfg:
    symbol: str
    short_lookback_minutes: int
    long_lookback_minutes: int
    alignment_threshold: float
    rsi_period: int
    volume_lookback_minutes: int


@dataclass
class StrategyCfg:
    tick_interval_sec: int
    endgame_tick_interval_sec: int
    scout_max_price: float
    scout_min_remaining_sec: int
    scout_size_usd: float
    arb_total_threshold: float
    arb_min_remaining_sec: int
    min_profit_after_fees: float
    endgame_remaining_sec: int
    endgame_hedge_ratio: float
    unbalanced_threshold_usd: float
    stop_loss_pct: float
    inverted_total_threshold: float
    inverted_spot_reversal_pct: float


@dataclass
class FeesCfg:
    per_side_fee_rate: float
    slippage_estimate: float


@dataclass
class ExecutionQualityCfg:
    max_spread: float
    min_depth_usd: float


@dataclass
class RiskCfg:
    daily_max_loss_pct: float
    daily_max_loss_usd: float
    min_cash_floor_usd: float
    max_single_trade_usd: float


@dataclass
class ExecutionCfg:
    mode: str
    paper_starting_balance_usd: float


@dataclass
class LoggingCfg:
    level: str
    summary_interval_sec: int


@dataclass
class Config:
    session: SessionCfg
    market: MarketCfg
    spot: SpotCfg
    strategy: StrategyCfg
    fees: FeesCfg
    execution_quality: ExecutionQualityCfg
    risk: RiskCfg
    execution: ExecutionCfg
    logging: LoggingCfg


def _section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    if key not in raw or not isinstance(raw[key], dict):
        raise ValueError(f"config.yaml missing required section: {key}")
    return raw[key]


def load_config(path: str | Path = "config.yaml") -> Config:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p.resolve()}")

    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    # session
    session_cfg = SessionCfg(**_section(raw, "session"))

    # market — recurring nested
    market_raw = _section(raw, "market")
    rec_raw = market_raw.pop("recurring", None) or {
        "enabled": False, "slug_templates": [], "period_sec": 900,
    }
    # Backward compat: accept singular `slug_template` and promote to list.
    if "slug_template" in rec_raw and "slug_templates" not in rec_raw:
        st = rec_raw.pop("slug_template")
        rec_raw["slug_templates"] = [st] if st else []
    market_cfg = MarketCfg(recurring=RecurringCfg(**rec_raw), **market_raw)

    cfg = Config(
        session=session_cfg,
        market=market_cfg,
        spot=SpotCfg(**_section(raw, "spot")),
        strategy=StrategyCfg(**_section(raw, "strategy")),
        fees=FeesCfg(**_section(raw, "fees")),
        execution_quality=ExecutionQualityCfg(**_section(raw, "execution_quality")),
        risk=RiskCfg(**_section(raw, "risk")),
        execution=ExecutionCfg(**_section(raw, "execution")),
        logging=LoggingCfg(**_section(raw, "logging")),
    )

    # ---------- validation ----------
    if cfg.execution.mode not in ("paper", "live"):
        raise ValueError(f"execution.mode must be 'paper' or 'live', got {cfg.execution.mode!r}")
    if not (0 < cfg.strategy.scout_max_price < 1):
        raise ValueError("strategy.scout_max_price must be in (0, 1)")
    if not (0 < cfg.strategy.arb_total_threshold <= 1):
        raise ValueError("strategy.arb_total_threshold must be in (0, 1]")
    if cfg.strategy.tick_interval_sec <= 0 or cfg.strategy.endgame_tick_interval_sec <= 0:
        raise ValueError("tick intervals must be positive")
    if cfg.market.recurring.enabled:
        if not cfg.market.recurring.slug_templates:
            raise ValueError("market.recurring.enabled=true but slug_templates is empty")
        for tmpl in cfg.market.recurring.slug_templates:
            if "{start_unix}" not in tmpl:
                raise ValueError(f"slug template {tmpl!r} must contain '{{start_unix}}'")
        if cfg.market.recurring.period_sec <= 0:
            raise ValueError("market.recurring.period_sec must be positive")
    elif not cfg.market.slug:
        raise ValueError("Either market.recurring.enabled=true or market.slug must be set")
    if cfg.risk.max_single_trade_usd <= 0:
        raise ValueError("risk.max_single_trade_usd must be > 0")
    if cfg.risk.daily_max_loss_pct <= 0 and cfg.risk.daily_max_loss_usd <= 0:
        raise ValueError("Set at least one of risk.daily_max_loss_pct / daily_max_loss_usd > 0")
    if cfg.spot.long_lookback_minutes < cfg.spot.short_lookback_minutes:
        raise ValueError("spot.long_lookback_minutes must be >= short_lookback_minutes")
    if cfg.strategy.stop_loss_pct < 0:
        raise ValueError("strategy.stop_loss_pct must be >= 0")
    if cfg.strategy.inverted_total_threshold < 0:
        raise ValueError("strategy.inverted_total_threshold must be >= 0")

    return cfg
