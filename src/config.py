"""Central configuration loaded from .env."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _env_bool(key: str, default: bool) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y"}


def _env_float(key: str, default: float) -> float:
    val = os.getenv(key)
    return float(val) if val else default


def _env_int(key: str, default: int) -> int:
    val = os.getenv(key)
    return int(val) if val else default


@dataclass(frozen=True)
class Config:
    api_key: str
    api_secret: str
    use_testnet: bool
    risk_per_trade: float
    max_leverage: int
    max_daily_drawdown: float
    max_open_positions: int
    data_dir: Path
    logs_dir: Path


def load_config() -> Config:
    cfg = Config(
        api_key=os.getenv("BINANCE_API_KEY", ""),
        api_secret=os.getenv("BINANCE_API_SECRET", ""),
        use_testnet=_env_bool("USE_TESTNET", True),
        risk_per_trade=_env_float("RISK_PER_TRADE", 0.01),
        max_leverage=_env_int("MAX_LEVERAGE", 3),
        max_daily_drawdown=_env_float("MAX_DAILY_DRAWDOWN", 0.05),
        max_open_positions=_env_int("MAX_OPEN_POSITIONS", 3),
        data_dir=ROOT / "data" / "cache",
        logs_dir=ROOT / "logs",
    )
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    return cfg
