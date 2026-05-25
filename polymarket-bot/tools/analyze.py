#!/usr/bin/env python3
"""Trade performance analyzer.

Reads logs/trades.csv and prints a one-page summary:
  - Total trades, win rate, avg P/L per trade
  - Breakdown by action (BUY scout / LOCK_SPREAD / STOP_LOSS / etc.)
  - Max drawdown (peak-to-trough equity)
  - Equity curve (ASCII sparkline)
  - Best / worst single trades
  - Per-market-window stats

Usage:
    python tools/analyze.py                      # default: logs/trades.csv + logs/ticks.csv
    python tools/analyze.py --trades path.csv    # custom trades file
    python tools/analyze.py --ticks path.csv     # custom ticks file (for equity curve)

Designed for beginners: just run it after a paper session and read the output.
No extra dependencies beyond Python 3.11+ stdlib + the csv module.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class Trade:
    timestamp: str
    mode: str
    market_slug: str
    side: str
    action: str        # BUY or SELL
    shares: float
    fill_price: float
    fee_usd: float
    notional_usd: float
    cash_after: float
    yes_shares_after: float
    no_shares_after: float
    reason: str


@dataclass
class TickRow:
    timestamp: str
    market_slug: str
    equity_usd: float
    decision: str


@dataclass
class PairResult:
    """A matched BUY→SELL (or BUY→settlement) pair on one side."""
    side: str
    buy_price: float
    sell_price: float      # 1.0 if won at settlement, 0.0 if lost, or actual sell price
    shares: float
    fee_usd: float
    pnl_usd: float         # net profit/loss including fees
    reason_buy: str
    reason_sell: str


def load_trades(path: Path) -> List[Trade]:
    if not path.exists():
        print(f"ERROR: {path} not found. Run the bot first to generate trade data.")
        sys.exit(1)
    trades: List[Trade] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                trades.append(Trade(
                    timestamp=row.get("timestamp_utc", ""),
                    mode=row.get("mode", ""),
                    market_slug=row.get("market_slug", ""),
                    side=row.get("side", ""),
                    action=row.get("action", "BUY"),
                    shares=float(row.get("shares", 0)),
                    fill_price=float(row.get("fill_price", 0)),
                    fee_usd=float(row.get("fee_usd", 0)),
                    notional_usd=float(row.get("notional_usd", 0)),
                    cash_after=float(row.get("cash_after", 0)),
                    yes_shares_after=float(row.get("yes_shares_after", 0)),
                    no_shares_after=float(row.get("no_shares_after", 0)),
                    reason=row.get("reason", ""),
                ))
            except (ValueError, KeyError):
                continue
    return trades


def load_ticks(path: Path) -> List[TickRow]:
    if not path.exists():
        return []
    ticks: List[TickRow] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ticks.append(TickRow(
                    timestamp=row.get("timestamp_utc", ""),
                    market_slug=row.get("market_slug", ""),
                    equity_usd=float(row.get("equity_usd", 0)),
                    decision=row.get("decision", ""),
                ))
            except (ValueError, KeyError):
                continue
    return ticks


def compute_pairs(trades: List[Trade]) -> List[PairResult]:
    """Match BUY→SELL pairs per side. Unmatched BUYs at the end are treated
    as still-open (marked with sell_price=NaN, pnl=0) and excluded from P/L.
    """
    # Group by market slug + side
    open_buys: Dict[str, List[Trade]] = defaultdict(list)
    pairs: List[PairResult] = []

    for t in trades:
        key = f"{t.market_slug}:{t.side}"
        if t.action == "BUY":
            open_buys[key].append(t)
        elif t.action == "SELL":
            # Match against oldest open buy (FIFO).
            if open_buys[key]:
                buy = open_buys[key].pop(0)
                sell_shares = min(t.shares, buy.shares)
                pnl = sell_shares * (t.fill_price - buy.fill_price) - buy.fee_usd - t.fee_usd
                pairs.append(PairResult(
                    side=t.side,
                    buy_price=buy.fill_price,
                    sell_price=t.fill_price,
                    shares=sell_shares,
                    fee_usd=buy.fee_usd + t.fee_usd,
                    pnl_usd=pnl,
                    reason_buy=buy.reason,
                    reason_sell=t.reason,
                ))
    return pairs


def equity_curve_from_ticks(ticks: List[TickRow]) -> List[float]:
    return [t.equity_usd for t in ticks if t.equity_usd > 0]


def max_drawdown(curve: List[float]) -> tuple[float, float]:
    """Returns (max_drawdown_usd, max_drawdown_pct). 0 if no drawdown."""
    if len(curve) < 2:
        return 0.0, 0.0
    peak = curve[0]
    max_dd = 0.0
    max_dd_pct = 0.0
    for eq in curve:
        peak = max(peak, eq)
        dd = peak - eq
        if dd > max_dd:
            max_dd = dd
            max_dd_pct = dd / peak if peak > 0 else 0.0
    return max_dd, max_dd_pct


def sparkline(values: List[float], width: int = 60) -> str:
    """ASCII sparkline of a time series."""
    if not values:
        return "(no data)"
    # Downsample to `width` points.
    if len(values) > width:
        step = len(values) / width
        sampled = [values[int(i * step)] for i in range(width)]
    else:
        sampled = values
    mn, mx = min(sampled), max(sampled)
    chars = " ▁▂▃▄▅▆▇█"
    if mx == mn:
        return chars[4] * len(sampled)
    line = ""
    for v in sampled:
        idx = int((v - mn) / (mx - mn) * (len(chars) - 1))
        line += chars[idx]
    return line


def decision_counts(ticks: List[TickRow]) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for t in ticks:
        # Simplify: take the first token before '(' for grouping.
        key = t.decision.split("(")[0].rstrip("_")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: -x[1]))


def per_market_stats(trades: List[Trade]) -> Dict[str, Dict[str, float]]:
    """Per market-window: total buys, total sells, net cash flow."""
    stats: Dict[str, Dict[str, float]] = defaultdict(lambda: {"buys": 0, "sells": 0, "net_flow": 0.0})
    for t in trades:
        s = stats[t.market_slug]
        if t.action == "BUY":
            s["buys"] += 1
            s["net_flow"] -= t.notional_usd + t.fee_usd
        else:
            s["sells"] += 1
            s["net_flow"] += t.notional_usd - t.fee_usd
    return dict(stats)


def print_report(trades: List[Trade], ticks: List[TickRow]) -> None:
    print("=" * 70)
    print("  POLYMARKET-BOT PERFORMANCE REPORT")
    print("=" * 70)

    if not trades:
        print("\n  No trades recorded yet. Run the bot for at least one session first.")
        print("  (Looking for: logs/trades.csv)")
        return

    buys = [t for t in trades if t.action == "BUY"]
    sells = [t for t in trades if t.action == "SELL"]
    print(f"\n  Total trades : {len(trades)} ({len(buys)} buys, {len(sells)} sells)")
    print(f"  Time range   : {trades[0].timestamp} → {trades[-1].timestamp}")

    # --- Matched pairs P/L ---
    pairs = compute_pairs(trades)
    if pairs:
        wins = [p for p in pairs if p.pnl_usd > 0]
        losses = [p for p in pairs if p.pnl_usd < 0]
        flat = [p for p in pairs if p.pnl_usd == 0]
        total_pnl = sum(p.pnl_usd for p in pairs)
        avg_pnl = total_pnl / len(pairs)
        win_rate = len(wins) / len(pairs) * 100 if pairs else 0

        print(f"\n  --- Matched BUY→SELL pairs: {len(pairs)} ---")
        print(f"  Win rate     : {win_rate:.1f}% ({len(wins)}W / {len(losses)}L / {len(flat)}F)")
        print(f"  Total P/L    : ${total_pnl:+.4f}")
        print(f"  Avg P/L/trade: ${avg_pnl:+.4f}")
        if wins:
            avg_win = sum(p.pnl_usd for p in wins) / len(wins)
            print(f"  Avg win      : ${avg_win:+.4f}")
        if losses:
            avg_loss = sum(p.pnl_usd for p in losses) / len(losses)
            print(f"  Avg loss     : ${avg_loss:+.4f}")
        if wins and losses:
            profit_factor = sum(p.pnl_usd for p in wins) / abs(sum(p.pnl_usd for p in losses))
            print(f"  Profit factor: {profit_factor:.2f}x")

        # Best/worst
        best = max(pairs, key=lambda p: p.pnl_usd)
        worst = min(pairs, key=lambda p: p.pnl_usd)
        print(f"\n  Best trade   : ${best.pnl_usd:+.4f} ({best.side} buy@{best.buy_price:.4f}→sell@{best.sell_price:.4f})")
        print(f"                 reason: {best.reason_buy[:60]}")
        print(f"  Worst trade  : ${worst.pnl_usd:+.4f} ({worst.side} buy@{worst.buy_price:.4f}→sell@{worst.sell_price:.4f})")
        print(f"                 reason: {worst.reason_sell[:60]}")
    else:
        print("\n  No matched BUY→SELL pairs yet (all positions may still be open).")

    # --- Equity curve from ticks ---
    curve = equity_curve_from_ticks(ticks)
    if curve:
        dd_usd, dd_pct = max_drawdown(curve)
        print(f"\n  --- Equity curve ({len(curve)} ticks) ---")
        print(f"  Start        : ${curve[0]:.2f}")
        print(f"  End          : ${curve[-1]:.2f}")
        print(f"  Peak         : ${max(curve):.2f}")
        print(f"  Trough       : ${min(curve):.2f}")
        print(f"  Max drawdown : ${dd_usd:.2f} ({dd_pct*100:.1f}%)")
        print(f"  Net return   : ${curve[-1] - curve[0]:+.2f} ({(curve[-1]/curve[0]-1)*100:+.2f}%)")
        print(f"\n  {sparkline(curve)}")
        print(f"  {'$'+f'{min(curve):.0f}':<30}{'$'+f'{max(curve):.0f}':>30}")

    # --- Decision distribution ---
    if ticks:
        counts = decision_counts(ticks)
        print(f"\n  --- Decision distribution ({len(ticks)} ticks) ---")
        total_ticks = len(ticks)
        for dec, cnt in list(counts.items())[:15]:
            bar = "█" * max(1, int(cnt / total_ticks * 40))
            print(f"  {dec:<40} {cnt:>5} ({cnt/total_ticks*100:4.1f}%) {bar}")

    # --- Per-market stats ---
    mstats = per_market_stats(trades)
    if mstats:
        print(f"\n  --- Per-market window ({len(mstats)} windows) ---")
        for slug, st in sorted(mstats.items()):
            print(f"  {slug}: {int(st['buys'])}B/{int(st['sells'])}S net_flow=${st['net_flow']:+.2f}")

    # --- Fees ---
    total_fees = sum(t.fee_usd for t in trades)
    total_notional = sum(t.notional_usd for t in trades)
    print(f"\n  --- Costs ---")
    print(f"  Total fees   : ${total_fees:.4f}")
    print(f"  Total volume : ${total_notional:.2f}")
    if total_notional > 0:
        print(f"  Effective fee: {total_fees/total_notional*100:.3f}%")

    print("\n" + "=" * 70)
    print("  TIP: If win rate < 50% or total P/L is negative after a week of")
    print("  paper trading, do NOT go live. Change strategy parameters or accept")
    print("  that this market regime doesn't suit the strategy.")
    print("=" * 70)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze polymarket-bot trade performance")
    parser.add_argument("--trades", default="logs/trades.csv", help="Path to trades CSV")
    parser.add_argument("--ticks", default="logs/ticks.csv", help="Path to ticks CSV")
    args = parser.parse_args()

    trades = load_trades(Path(args.trades))
    ticks = load_ticks(Path(args.ticks))
    print_report(trades, ticks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
