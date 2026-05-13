"""Dual moving-average crossover with ATR-based stop.

Long when fast SMA crosses above slow SMA, short when fast crosses below.
Stop = entry -/+ atr_mult * ATR.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_prev = (df["high"] - df["close"].shift()).abs()
    low_prev = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_prev, low_prev], axis=1).max(axis=1)
    return tr.rolling(period).mean()


class MACrossStrategy(Strategy):
    name = "ma_cross"

    def __init__(self, fast: int = 20, slow: int = 50, atr_period: int = 14, atr_mult: float = 2.0):
        if fast >= slow:
            raise ValueError("fast must be < slow")
        self.fast = fast
        self.slow = slow
        self.atr_period = atr_period
        self.atr_mult = atr_mult

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        fast = df["close"].rolling(self.fast).mean()
        slow = df["close"].rolling(self.slow).mean()
        atr = _atr(df, self.atr_period)

        # Use shifted values so decision at bar t uses info from bar t-1 close
        fast_s, slow_s, atr_s = fast.shift(1), slow.shift(1), atr.shift(1)
        close_s = df["close"].shift(1)

        side = np.where(fast_s > slow_s, "long", np.where(fast_s < slow_s, "short", "flat"))
        stop = np.where(
            side == "long",
            close_s - self.atr_mult * atr_s,
            np.where(side == "short", close_s + self.atr_mult * atr_s, np.nan),
        )
        out["side"] = side
        out["stop"] = stop
        return out
