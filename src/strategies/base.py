"""Strategy interface.

A strategy consumes an OHLCV DataFrame (with column 'ts','open','high','low','close','volume')
and returns a DataFrame of signals. Signals are emitted on BAR CLOSE and acted upon at
the NEXT BAR'S OPEN in the backtester (no look-ahead).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

Side = Literal["long", "short", "flat"]


@dataclass
class Signal:
    side: Side          # desired target position at next bar
    stop_price: float   # protective stop for the new position (absolute price)


class Strategy:
    name: str = "base"

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame indexed like ``df`` with columns:
        - 'side' in {'long', 'short', 'flat'}    : desired position on next bar
        - 'stop' (float)                          : stop price for that position
        Strategies must not look at future bars (use .shift() where needed).
        """
        raise NotImplementedError
