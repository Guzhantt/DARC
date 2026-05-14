"""RSI mean-reversion strategy.

Go long when RSI crosses above ``oversold`` (typical: 30), short when RSI crosses
below ``overbought`` (typical: 70). Exit (flat) when RSI returns to neutral band.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy
from .ma_cross import _atr


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


class RSIStrategy(Strategy):
    name = "rsi"

    def __init__(
        self,
        period: int = 14,
        oversold: float = 30,
        overbought: float = 70,
        neutral_low: float = 45,
        neutral_high: float = 55,
        atr_period: int = 14,
        atr_mult: float = 2.5,
    ):
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self.neutral_low = neutral_low
        self.neutral_high = neutral_high
        self.atr_period = atr_period
        self.atr_mult = atr_mult

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        rsi = _rsi(df["close"], self.period).shift(1)
        rsi_prev = rsi.shift(1)
        atr = _atr(df, self.atr_period).shift(1)
        close_s = df["close"].shift(1)

        side = np.full(len(df), "flat", dtype=object)
        position = "flat"
        for i in range(len(df)):
            r, rp = rsi.iloc[i], rsi_prev.iloc[i]
            if np.isnan(r) or np.isnan(rp):
                side[i] = position
                continue
            # Entries
            if position == "flat":
                if rp < self.oversold and r >= self.oversold:
                    position = "long"
                elif rp > self.overbought and r <= self.overbought:
                    position = "short"
            # Exits to flat
            elif position == "long" and r >= self.neutral_high:
                position = "flat"
            elif position == "short" and r <= self.neutral_low:
                position = "flat"
            side[i] = position

        stop = np.where(
            side == "long",
            close_s - self.atr_mult * atr,
            np.where(side == "short", close_s + self.atr_mult * atr, np.nan),
        )
        return pd.DataFrame({"side": side, "stop": stop}, index=df.index)
