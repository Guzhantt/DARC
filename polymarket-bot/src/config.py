"""YAML config loader with light validation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class RecurringCfg:
    enabled: bool
    slug_template: str
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
    lookback_minutes: int
    alignment_threshold: float


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


@dataclass
class FeesCfg:
    per_side_fee_rate: float
    slippage_estimate: float


@dataclass
class ExecutionQualityCfg:
    """Hard guards against bad fills. Set max_spread / min_depth_usd to 0 to disable."""
    max_spread: float
    min_depth_usd: float


@dataclass
class RiskCfg:
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

    market_raw = _section(raw, "market")
    rec_raw = market_raw.pop("recurring", None) or {"enabled": False, "slug_template": "", "period_sec": 900}
    market_cfg = MarketCfg(recurring=RecurringCfg(**rec_raw), **market_raw)

    cfg = Config(
        market=market_cfg,
        spot=SpotCfg(**_section(raw, "spot")),
        strategy=StrategyCfg(**_section(raw, "strategy")),
        fees=FeesCfg(**_section(raw, "fees")),
        execution_quality=ExecutionQualityCfg(**_section(raw, "execution_quality")),
        risk=RiskCfg(**_section(raw, "risk")),
        execution=ExecutionCfg(**_section(raw, "execution")),
        logging=LoggingCfg(**_section(raw, "logging")),
    )

    # Sanity checks. Catching these here saves debugging mid-loop.
    if cfg.execution.mode not in ("paper", "live"):
        raise ValueError(f"execution.mode must be 'paper' or 'live', got {cfg.execution.mode!r}")
    if not (0 < cfg.strategy.scout_max_price < 1):
        raise ValueError("strategy.scout_max_price must be in (0, 1)")
    if not (0 < cfg.strategy.arb_total_threshold <= 1):
        raise ValueError("strategy.arb_total_threshold must be in (0, 1]")
    if cfg.strategy.tick_interval_sec <= 0 or cfg.strategy.endgame_tick_interval_sec <= 0:
        raise ValueError("tick intervals must be positive")
    if cfg.market.recurring.enabled:
        if "{start_unix}" not in cfg.market.recurring.slug_template:
            raise ValueError("market.recurring.slug_template must contain '{start_unix}' placeholder")
        if cfg.market.recurring.period_sec <= 0:
            raise ValueError("market.recurring.period_sec must be positive")
    elif not cfg.market.slug:
        raise ValueError("Either market.recurring.enabled=true or market.slug must be set")
    if cfg.risk.max_single_trade_usd <= 0:
        raise ValueError("risk.max_single_trade_usd must be > 0")
    if cfg.risk.daily_max_loss_usd <= 0:
        raise ValueError("risk.daily_max_loss_usd must be > 0 (kill switch must be configured)")

    return cfg
