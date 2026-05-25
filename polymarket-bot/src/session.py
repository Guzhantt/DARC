"""Trading-session gate.

Refuses to trade outside configured hours / weekdays. Polymarket recurring
markets are 24/7, but liquidity is much thinner during Asian + European
overnight and on weekends. Default window is UTC 13:00 - 01:00 next day,
weekdays only — this captures US morning through US evening.

Expressed as half-open [start_hour, end_hour) where end_hour > 24 wraps
into the next day. So `end_hour=25` means 01:00 the following day.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence


@dataclass(frozen=True)
class SessionWindow:
    start_hour_utc: int           # 0-23
    end_hour_utc: int             # 1-48 (>24 wraps into next day)
    enabled_weekdays: tuple[int, ...]   # 0=Mon..6=Sun

    def is_open(self, now: datetime | None = None) -> bool:
        now = now or datetime.now(timezone.utc)
        # Hour-of-week mod 24 with day check; window can wrap past midnight.
        weekday = now.weekday()
        hour = now.hour + now.minute / 60.0
        end_h = self.end_hour_utc
        # Two cases: window stays in one calendar day, or it wraps past midnight.
        if end_h <= 24:
            in_hour_window = self.start_hour_utc <= hour < end_h
            return in_hour_window and weekday in self.enabled_weekdays
        # Wrapping window: e.g. start=13, end=25 (=01:00 next day).
        # Open if (today is enabled AND hour >= start) OR
        #         (yesterday was enabled AND hour < end - 24).
        wrap_end = end_h - 24
        if hour >= self.start_hour_utc:
            return weekday in self.enabled_weekdays
        if hour < wrap_end:
            yesterday = (weekday - 1) % 7
            return yesterday in self.enabled_weekdays
        return False

    def describe(self) -> str:
        wd = ",".join(_short_weekday(d) for d in sorted(self.enabled_weekdays))
        end_disp = self.end_hour_utc - 24 if self.end_hour_utc > 24 else self.end_hour_utc
        suffix = " (next day)" if self.end_hour_utc > 24 else ""
        return f"UTC {self.start_hour_utc:02d}:00-{end_disp:02d}:00{suffix} on {wd}"


def from_config(start: int, end: int, weekdays: Sequence[int]) -> SessionWindow:
    if not 0 <= start <= 23:
        raise ValueError(f"start_hour must be in 0..23, got {start}")
    if not 1 <= end <= 48:
        raise ValueError(f"end_hour must be in 1..48, got {end}")
    if end <= start:
        raise ValueError(f"end_hour ({end}) must be greater than start_hour ({start})")
    valid = tuple(d for d in weekdays if 0 <= d <= 6)
    if not valid:
        raise ValueError("enabled_weekdays must contain at least one of 0..6")
    return SessionWindow(start_hour_utc=start, end_hour_utc=end, enabled_weekdays=valid)


def _short_weekday(d: int) -> str:
    return ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")[d % 7]
