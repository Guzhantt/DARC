"""Donchian channel breakout (trend-following).

Entry long on a new N-bar high close; short on new N-bar low close.
Exit when price closes back inside the channel of exit_period length.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy
from .ma_cross import _atr


class DonchianStrategy(Strategy):
    name = "donchian"

    def __init__(
        self,
        entry_period: int = 20,
        exit_period: int = 10,
        atr_period: int = 14,
        atr_mult: float = 3.0,
    ):
        self.entry_period = entry_period
        self.exit_period = exit_period
        self.atr_period = atr_period
        self.atr_mult = atr_mult

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        # Use only past data (exclude current bar via shift)
        upper_entry = close.rolling(self.entry_period).max().shift(1)
        lower_entry = close.rolling(self.entry_period).min().shift(1)
        upper_exit = close.rolling(self.exit_period).max().shift(1)
        lower_exit = close.rolling(self.exit_period).min().shift(1)
        atr = _atr(df, self.atr_period).shift(1)
        close_s = close.shift(1)

        side = np.full(len(df), "flat", dtype=object)
        position = "flat"
        for i in range(len(df)):
            c = close_s.iloc[i]
            if np.isnan(c) or np.isnan(upper_entry.iloc[i]):
                side[i] = position
                continue
            if position == "flat":
                if c >= upper_entry.iloc[i]:
                    position = "long"
                elif c <= lower_entry.iloc[i]:
                    position = "short"
            elif position == "long" and c <= lower_exit.iloc[i]:
                position = "flat"
            elif position == "short" and c >= upper_exit.iloc[i]:
                position = "flat"
            side[i] = position

        stop = np.where(
            side == "long",
            close_s - self.atr_mult * atr,
            np.where(side == "short", close_s + self.atr_mult * atr, np.nan),
        )
        return pd.DataFrame({"side": side, "stop": stop}, index=df.index)
