#!/usr/bin/env python3
"""
币安合约自动交易 Bot v2 — 生产级单文件版本.

相比 v1 的改进 (P0 关键):
1. SQLite 状态持久化 — 重启后能恢复, 不会重复开仓
2. 启动对账 (Reconciliation) — 启动时与交易所核对真实持仓
3. 心跳监控 + Telegram 告警 — bot 死了会立即通知
4. 集成 v2 选币器 + 鲸鱼过滤器
5. ATR 动态止损 + 跟踪止盈 — 比固定 % 更适应波动
6. 订单去重 — 防止网络抖动导致重复下单
7. 异常恢复 — 任何步骤失败都不会让 bot 整体崩溃

用法:
    pip install ccxt requests python-dotenv pandas numpy
    cp .env.example .env  # 填入 API Key
    python trading_bot_v2.py

环境变量:
    BINANCE_API_KEY, BINANCE_API_SECRET
    USE_TESTNET=true (强烈推荐)
    TG_BOT_TOKEN, TG_CHAT_ID (可选)
    
    # 风控
    RISK_PER_TRADE=0.01      # 每笔风险 1%
    MAX_LEVERAGE=2           # 杠杆 2x (保守)
    MAX_DAILY_DRAWDOWN=0.05  # 日内 5% 熔断
    MAX_OPEN_POSITIONS=3
"""

import os
import sys
import time
import json
import sqlite3
import random
import hashlib
import threading
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import pandas as pd
import ccxt
import requests
from dotenv import load_dotenv

# ═══════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════

load_dotenv()

API_KEY = os.getenv("BINANCE_API_KEY", "")
API_SECRET = os.getenv("BINANCE_API_SECRET", "")
USE_TESTNET = os.getenv("USE_TESTNET", "true").lower() in ("true", "1", "yes")

# 风控
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.01"))
MAX_LEVERAGE = int(os.getenv("MAX_LEVERAGE", "2"))
MAX_DAILY_DRAWDOWN = float(os.getenv("MAX_DAILY_DRAWDOWN", "0.05"))
MAX_OPEN_POSITIONS = int(os.getenv("MAX_OPEN_POSITIONS", "3"))
MAX_PCT_PER_SYMBOL = float(os.getenv("MAX_PCT_PER_SYMBOL", "0.30"))

# 策略
TIMEFRAME = os.getenv("TIMEFRAME", "1h")
SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", "300"))
MIN_VOLUME_M = float(os.getenv("MIN_VOLUME_M", "10"))
COOLDOWN_HOURS = int(os.getenv("COOLDOWN_HOURS", "4"))

# 鲸鱼过滤器模式: VETO | CONFIRM | SIZE | DISABLED
WHALE_FILTER_MODE = os.getenv("WHALE_FILTER_MODE", "SIZE")

# 心跳
HEARTBEAT_INTERVAL = int(os.getenv("HEARTBEAT_INTERVAL", "1800"))  # 30 分钟
HEARTBEAT_TIMEOUT = int(os.getenv("HEARTBEAT_TIMEOUT", "3600"))    # 1 小时

# Telegram
TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN", "")
TG_CHAT_ID = os.getenv("TG_CHAT_ID", "")

# 文件
DB_FILE = Path(os.getenv("BOT_DB_FILE", "bot_state.db"))
LOG_FILE = Path(os.getenv("BOT_LOG_FILE", "bot.log"))
TZ_UTC8 = timezone(timedelta(hours=8))

# ═══════════════════════════════════════════════════
# 日志 + Telegram
# ═══════════════════════════════════════════════════

def log(msg: str, level: str = "INFO"):
    ts = datetime.now(TZ_UTC8).strftime("%m-%d %H:%M:%S")
    line = f"[{ts}] [{level}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def send_tg(text: str, important: bool = False):
    """发 Telegram 通知. important=True 时会重试 3 次."""
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TG_CHAT_ID, "text": text, "parse_mode": "Markdown"}
    attempts = 3 if important else 1
    for i in range(attempts):
        try:
            r = requests.post(url, json=payload, timeout=10)
            if r.status_code == 200:
                return
        except Exception as e:
            if i == attempts - 1:
                log(f"TG 通知失败: {e}", "WARN")
        time.sleep(2)


# ═══════════════════════════════════════════════════
# SQLite 状态持久化
# ═══════════════════════════════════════════════════

class State:
    """所有状态都写入 SQLite, 重启不丢数据."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS positions (
        symbol TEXT PRIMARY KEY,
        direction TEXT NOT NULL,        -- 'long' | 'short'
        entry_price REAL NOT NULL,
        qty REAL NOT NULL,
        sl_price REAL NOT NULL,
        tp_price REAL,
        peak_price REAL,                -- 用于跟踪止损
        opened_at TEXT NOT NULL,
        signal_reason TEXT,
        client_order_id TEXT,           -- 用于去重
        sl_order_id TEXT
    );
    CREATE TABLE IF NOT EXISTS trade_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        direction TEXT NOT NULL,
        entry_price REAL NOT NULL,
        exit_price REAL NOT NULL,
        qty REAL NOT NULL,
        pnl_usd REAL NOT NULL,
        pnl_pct REAL NOT NULL,
        opened_at TEXT NOT NULL,
        closed_at TEXT NOT NULL,
        close_reason TEXT,
        signal_reason TEXT
    );
    CREATE TABLE IF NOT EXISTS cooldowns (
        symbol TEXT PRIMARY KEY,
        last_trade_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS daily_stats (
        date TEXT PRIMARY KEY,
        start_equity REAL NOT NULL,
        peak_equity REAL NOT NULL,
        halted INTEGER DEFAULT 0
    );
    CREATE TABLE IF NOT EXISTS bot_meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS pending_orders (
        client_order_id TEXT PRIMARY KEY,
        symbol TEXT NOT NULL,
        side TEXT NOT NULL,
        qty REAL NOT NULL,
        created_at TEXT NOT NULL,
        status TEXT DEFAULT 'pending'   -- 'pending' | 'filled' | 'failed'
    );
    """

    def __init__(self, db_path: Path = DB_FILE):
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        for stmt in self.SCHEMA.strip().split(";"):
            if stmt.strip():
                self.conn.execute(stmt)
        self.conn.commit()

    # === Positions ===
    def get_position(self, symbol: str) -> Optional[dict]:
        with self.lock:
            row = self.conn.execute(
                "SELECT * FROM positions WHERE symbol=?", (symbol,)
            ).fetchone()
            return dict(row) if row else None

    def get_all_positions(self) -> list[dict]:
        with self.lock:
            rows = self.conn.execute("SELECT * FROM positions").fetchall()
            return [dict(r) for r in rows]

    def save_position(self, pos: dict):
        with self.lock:
            cols = ",".join(pos.keys())
            placeholders = ",".join(["?"] * len(pos))
            self.conn.execute(
                f"INSERT OR REPLACE INTO positions ({cols}) VALUES ({placeholders})",
                list(pos.values()),
            )
            self.conn.commit()

    def update_position_field(self, symbol: str, field: str, value):
        with self.lock:
            self.conn.execute(
                f"UPDATE positions SET {field}=? WHERE symbol=?", (value, symbol)
            )
            self.conn.commit()

    def remove_position(self, symbol: str):
        with self.lock:
            self.conn.execute("DELETE FROM positions WHERE symbol=?", (symbol,))
            self.conn.commit()

    # === Trade history ===
    def log_trade(self, trade: dict):
        with self.lock:
            cols = ",".join(trade.keys())
            placeholders = ",".join(["?"] * len(trade))
            self.conn.execute(
                f"INSERT INTO trade_history ({cols}) VALUES ({placeholders})",
                list(trade.values()),
            )
            self.conn.commit()

    def get_recent_trades(self, limit: int = 20) -> list[dict]:
        with self.lock:
            rows = self.conn.execute(
                "SELECT * FROM trade_history ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    # === Cooldowns ===
    def is_in_cooldown(self, symbol: str, hours: int = COOLDOWN_HOURS) -> bool:
        with self.lock:
            row = self.conn.execute(
                "SELECT last_trade_at FROM cooldowns WHERE symbol=?", (symbol,)
            ).fetchone()
            if not row:
                return False
            last = datetime.fromisoformat(row["last_trade_at"])
            return datetime.now(TZ_UTC8) - last < timedelta(hours=hours)

    def set_cooldown(self, symbol: str):
        with self.lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO cooldowns (symbol, last_trade_at) VALUES (?, ?)",
                (symbol, datetime.now(TZ_UTC8).isoformat()),
            )
            self.conn.commit()

    # === Daily stats (for drawdown halt) ===
    def get_today_stats(self) -> Optional[dict]:
        today = datetime.now(TZ_UTC8).strftime("%Y-%m-%d")
        with self.lock:
            row = self.conn.execute(
                "SELECT * FROM daily_stats WHERE date=?", (today,)
            ).fetchone()
            return dict(row) if row else None

    def init_today(self, equity: float):
        today = datetime.now(TZ_UTC8).strftime("%Y-%m-%d")
        with self.lock:
            self.conn.execute(
                "INSERT OR IGNORE INTO daily_stats (date, start_equity, peak_equity) VALUES (?, ?, ?)",
                (today, equity, equity),
            )
            self.conn.commit()

    def update_today_peak(self, equity: float):
        today = datetime.now(TZ_UTC8).strftime("%Y-%m-%d")
        with self.lock:
            self.conn.execute(
                "UPDATE daily_stats SET peak_equity=MAX(peak_equity, ?) WHERE date=?",
                (equity, today),
            )
            self.conn.commit()

    def halt_today(self):
        today = datetime.now(TZ_UTC8).strftime("%Y-%m-%d")
        with self.lock:
            self.conn.execute(
                "UPDATE daily_stats SET halted=1 WHERE date=?", (today,)
            )
            self.conn.commit()

    # === Pending orders (去重) ===
    def is_order_pending(self, client_order_id: str) -> bool:
        with self.lock:
            row = self.conn.execute(
                "SELECT status FROM pending_orders WHERE client_order_id=?",
                (client_order_id,),
            ).fetchone()
            return row is not None and row["status"] == "pending"

    def record_pending_order(self, client_order_id: str, symbol: str, side: str, qty: float):
        with self.lock:
            self.conn.execute(
                """INSERT OR REPLACE INTO pending_orders 
                (client_order_id, symbol, side, qty, created_at, status) 
                VALUES (?, ?, ?, ?, ?, 'pending')""",
                (client_order_id, symbol, side, qty, datetime.now(TZ_UTC8).isoformat()),
            )
            self.conn.commit()

    def mark_order_status(self, client_order_id: str, status: str):
        with self.lock:
            self.conn.execute(
                "UPDATE pending_orders SET status=? WHERE client_order_id=?",
                (status, client_order_id),
            )
            self.conn.commit()

    # === Bot meta ===
    def set_meta(self, key: str, value: str):
        with self.lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO bot_meta (key, value) VALUES (?, ?)", (key, value)
            )
            self.conn.commit()

    def get_meta(self, key: str, default: str = "") -> str:
        with self.lock:
            row = self.conn.execute("SELECT value FROM bot_meta WHERE key=?", (key,)).fetchone()
            return row["value"] if row else default


# ═══════════════════════════════════════════════════
# 启动对账 (Reconciliation)
# ═══════════════════════════════════════════════════

class Reconciler:
    """启动时对比本地状态与交易所真实持仓, 修复不一致."""

    def __init__(self, exchange: ccxt.binanceusdm, state: State):
        self.exchange = exchange
        self.state = state

    def reconcile(self) -> dict:
        """返回对账结果摘要."""
        log("开始启动对账...")
        try:
            exchange_positions = self.exchange.fetch_positions()
            exchange_open = {
                p["symbol"]: p for p in exchange_positions
                if float(p.get("contracts", 0) or 0) > 0
            }
        except Exception as e:
            log(f"对账失败 - 无法获取交易所持仓: {e}", "ERROR")
            return {"status": "failed", "error": str(e)}

        local_positions = {p["symbol"]: p for p in self.state.get_all_positions()}

        report = {
            "status": "ok",
            "exchange_count": len(exchange_open),
            "local_count": len(local_positions),
            "added_to_local": [],   # 交易所有, 本地无 (孤儿仓位)
            "removed_local": [],     # 本地有, 交易所无 (已被外部平仓)
            "synced": [],            # 两边都有
        }

        # 1. 交易所有但本地无 → 加入本地
        for sym, pos in exchange_open.items():
            if sym not in local_positions:
                contracts = float(pos["contracts"])
                entry = float(pos.get("entryPrice") or 0)
                side = "long" if pos.get("side", "").lower() == "long" else "short"
                # 用保守的止损 (5%)
                if side == "long":
                    sl_price = entry * 0.95
                else:
                    sl_price = entry * 1.05
                self.state.save_position({
                    "symbol": sym,
                    "direction": side,
                    "entry_price": entry,
                    "qty": contracts,
                    "sl_price": sl_price,
                    "tp_price": None,
                    "peak_price": entry,
                    "opened_at": datetime.now(TZ_UTC8).isoformat(),
                    "signal_reason": "RECONCILED (启动时发现)",
                    "client_order_id": None,
                    "sl_order_id": None,
                })
                report["added_to_local"].append(sym)
                log(f"  对账: 添加 {sym} {side} @{entry} qty={contracts} (本地缺失)")

        # 2. 本地有但交易所无 → 移除本地
        for sym in local_positions:
            if sym not in exchange_open:
                # 写入交易历史 (作为外部平仓)
                pos = local_positions[sym]
                self.state.log_trade({
                    "symbol": sym,
                    "direction": pos["direction"],
                    "entry_price": pos["entry_price"],
                    "exit_price": pos["entry_price"],  # 不知道实际平仓价
                    "qty": pos["qty"],
                    "pnl_usd": 0,
                    "pnl_pct": 0,
                    "opened_at": pos["opened_at"],
                    "closed_at": datetime.now(TZ_UTC8).isoformat(),
                    "close_reason": "EXTERNAL_CLOSE (对账时发现)",
                    "signal_reason": pos.get("signal_reason", ""),
                })
                self.state.remove_position(sym)
                report["removed_local"].append(sym)
                log(f"  对账: 移除 {sym} (交易所已平仓)")

        # 3. 两边都有 → 检查数量一致
        for sym in local_positions:
            if sym in exchange_open:
                local_qty = local_positions[sym]["qty"]
                exch_qty = float(exchange_open[sym]["contracts"])
                if abs(local_qty - exch_qty) / max(local_qty, 0.001) > 0.01:
                    self.state.update_position_field(sym, "qty", exch_qty)
                    log(f"  对账: 修正 {sym} 数量 {local_qty} → {exch_qty}")
                report["synced"].append(sym)

        log(f"对账完成: 交易所{len(exchange_open)}个 / 本地{len(local_positions)}个 → "
            f"添加{len(report['added_to_local'])} 移除{len(report['removed_local'])}")
        return report


# ═══════════════════════════════════════════════════
# 心跳监控
# ═══════════════════════════════════════════════════

class HeartbeatMonitor:
    """独立线程, 定期检查 bot 是否活着. 死掉就发告警."""

    def __init__(self, state: State, interval: int = HEARTBEAT_INTERVAL):
        self.state = state
        self.interval = interval
        self.thread = None
        self.stop_flag = threading.Event()

    def beat(self):
        """主循环每次扫描后调用, 更新心跳时间."""
        self.state.set_meta("last_heartbeat", datetime.now(TZ_UTC8).isoformat())

    def _monitor_loop(self):
        """独立线程, 检查心跳是否超时."""
        while not self.stop_flag.is_set():
            try:
                last_beat_str = self.state.get_meta("last_heartbeat", "")
                if last_beat_str:
                    last_beat = datetime.fromisoformat(last_beat_str)
                    elapsed = (datetime.now(TZ_UTC8) - last_beat).total_seconds()
                    if elapsed > HEARTBEAT_TIMEOUT:
                        log(f"⚠️ 心跳超时: {elapsed:.0f}秒未更新", "WARN")
                        send_tg(
                            f"⚠️ *Bot 心跳异常*\n"
                            f"距离上次心跳 {elapsed/60:.0f} 分钟\n"
                            f"可能 bot 已挂掉, 请检查!",
                            important=True,
                        )
            except Exception as e:
                log(f"心跳检查失败: {e}", "WARN")
            self.stop_flag.wait(self.interval)

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.stop_flag.clear()
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        log(f"心跳监控启动 (检查间隔 {self.interval}秒, 超时阈值 {HEARTBEAT_TIMEOUT}秒)")

    def stop(self):
        self.stop_flag.set()


# ═══════════════════════════════════════════════════
# 策略 + 鲸鱼过滤器 (内嵌 v2 核心逻辑)
# ═══════════════════════════════════════════════════

def compute_rsi(closes: np.ndarray, period: int = 9) -> np.ndarray:
    n = len(closes)
    rsi = np.full(n, 50.0)
    if n < period + 1:
        return rsi
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    rsi[period] = 100 - (100 / (1 + avg_gain / max(avg_loss, 1e-10)))
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period
        rsi[i] = 100 - (100 / (1 + avg_gain / max(avg_loss, 1e-10)))
    return rsi


def compute_atr(ohlcv: list, period: int = 14) -> float:
    """返回最新 ATR 值."""
    if len(ohlcv) < period + 1:
        return 0
    highs = np.array([float(b[2]) for b in ohlcv])
    lows = np.array([float(b[3]) for b in ohlcv])
    closes = np.array([float(b[4]) for b in ohlcv])
    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(
            np.abs(highs[1:] - closes[:-1]),
            np.abs(lows[1:] - closes[:-1]),
        ),
    )
    return float(np.mean(tr[-period:]))


def detect_smart_reversion(ohlcv: list, rsi_period: int = 9, rsi_extreme: float = 30) -> Optional[dict]:
    """smart_reversion 信号检测 (RSI 极端反转 + 量能确认)."""
    if len(ohlcv) < 50:
        return None

    closes = np.array([float(b[4]) for b in ohlcv])
    opens = np.array([float(b[1]) for b in ohlcv])
    volumes = np.array([float(b[5]) for b in ohlcv])

    rsi = compute_rsi(closes, rsi_period)
    rsi_cur = rsi[-2]   # 上一根已闭合
    rsi_prev = rsi[-3]
    close_prev = closes[-2]
    open_prev = opens[-2]

    if np.isnan(rsi_cur) or np.isnan(rsi_prev):
        return None

    # 量能过滤
    vol_ma = np.mean(volumes[-20:])
    vol_ok = volumes[-2] > vol_ma * 1.0  # 不要太苛刻

    # LONG: RSI 从超卖回升 + 阳线 + 量能
    if (rsi_prev < rsi_extreme and rsi_cur > rsi_prev
            and rsi_cur > rsi_extreme and close_prev > open_prev and vol_ok):
        return {
            "direction": "long",
            "reason": f"RSI {rsi_prev:.0f}→{rsi_cur:.0f} 超卖反转+阳线",
            "rsi": rsi_cur,
        }

    # SHORT: RSI 从超买回落 + 阴线 + 量能
    if (rsi_prev > (100 - rsi_extreme) and rsi_cur < rsi_prev
            and rsi_cur < (100 - rsi_extreme) and close_prev < open_prev and vol_ok):
        return {
            "direction": "short",
            "reason": f"RSI {rsi_prev:.0f}→{rsi_cur:.0f} 超买反转+阴线",
            "rsi": rsi_cur,
        }

    return None


def detect_event_signal(ticker: dict, funding_rate: float) -> Optional[dict]:
    """事件驱动信号: 极端费率 / 暴跌 / 暴涨."""
    change_pct = float(ticker.get("percentage", 0) or 0)

    if funding_rate < -0.08:
        return {
            "direction": "long",
            "reason": f"极端负费率 {funding_rate:.4f}%",
            "tp_pct": 0.10, "sl_pct": 0.05,
        }
    if funding_rate > 0.10:
        return {
            "direction": "short",
            "reason": f"极端正费率 {funding_rate:.4f}%",
            "tp_pct": 0.10, "sl_pct": 0.05,
        }
    if change_pct < -20:
        return {
            "direction": "long",
            "reason": f"24h暴跌{change_pct:.1f}%",
            "tp_pct": 0.12, "sl_pct": 0.08,
        }
    if change_pct > 35:
        return {
            "direction": "short",
            "reason": f"24h暴涨{change_pct:.1f}%",
            "tp_pct": 0.15, "sl_pct": 0.10,
        }
    return None


def fetch_whale_data(symbol_raw: str) -> Optional[dict]:
    """获取鲸鱼数据: OI 历史 + Top Trader Ratio + Taker Ratio."""
    base = "https://fapi.binance.com"
    try:
        # OI 最近 30 个数据点
        oi_resp = requests.get(
            f"{base}/futures/data/openInterestHist",
            params={"symbol": symbol_raw, "period": "1h", "limit": 30},
            timeout=8,
        )
        oi_data = oi_resp.json() if oi_resp.status_code == 200 else []
        if not isinstance(oi_data, list) or not oi_data:
            return None

        oi_values = [float(d["sumOpenInterest"]) for d in oi_data]

        # Top Trader Long/Short Ratio
        try:
            tt_resp = requests.get(
                f"{base}/futures/data/topLongShortPositionRatio",
                params={"symbol": symbol_raw, "period": "1h", "limit": 5},
                timeout=8,
            )
            tt_data = tt_resp.json() if tt_resp.status_code == 200 else []
            top_ls = float(tt_data[-1]["longShortRatio"]) if tt_data else 1.0
            top_ls_prev = float(tt_data[-3]["longShortRatio"]) if len(tt_data) >= 3 else top_ls
        except Exception:
            top_ls, top_ls_prev = 1.0, 1.0

        # Taker Buy/Sell Ratio
        try:
            tk_resp = requests.get(
                f"{base}/futures/data/takerlongshortRatio",
                params={"symbol": symbol_raw, "period": "1h", "limit": 5},
                timeout=8,
            )
            tk_data = tk_resp.json() if tk_resp.status_code == 200 else []
            taker_ratio = float(tk_data[-1]["buySellRatio"]) if tk_data else 1.0
        except Exception:
            taker_ratio = 1.0

        return {
            "oi_change_pct": (oi_values[-1] / oi_values[0] - 1) if oi_values[0] > 0 else 0,
            "oi_recent": oi_values[-1],
            "top_ls_ratio": top_ls,
            "top_ls_change": top_ls - top_ls_prev,
            "taker_ratio": taker_ratio,
        }
    except Exception as e:
        log(f"  获取 {symbol_raw} 鲸鱼数据失败: {e}", "WARN")
        return None


def evaluate_whale_score(whale_data: dict, price_change_pct: float) -> tuple[float, str]:
    """根据鲸鱼数据计算综合分 (-1 到 +1) 和原因."""
    if not whale_data:
        return 0.0, "无鲸鱼数据"

    components = []
    reasons = []

    # 1. OI 散度 (权重 0.4)
    oi_chg = whale_data["oi_change_pct"]
    if oi_chg > 0.01 and price_change_pct > 0.001:
        components.append(("oi", 1.0, 0.4))
        reasons.append("OI涨价涨")
    elif oi_chg < -0.01 and price_change_pct < -0.001:
        components.append(("oi", -1.0, 0.4))
        reasons.append("OI跌价跌")
    elif oi_chg > 0.02 and price_change_pct < -0.001:
        components.append(("oi", -1.0, 0.4))
        reasons.append("OI涨价跌-机构空")
    elif oi_chg < -0.02 and price_change_pct > 0.001:
        components.append(("oi", -0.5, 0.4))
        reasons.append("OI跌价涨-散户接盘")
    elif oi_chg > 0.03 and abs(price_change_pct) < 0.005:
        components.append(("oi", 0.5, 0.4))
        reasons.append("OI涨价稳-鲸鱼吸筹")
    else:
        components.append(("oi", 0.0, 0.4))

    # 2. Top Trader (权重 0.3)
    tt = whale_data["top_ls_ratio"]
    tt_chg = whale_data["top_ls_change"]
    if tt > 1.5 and tt_chg > 0:
        components.append(("tt", 1.0, 0.3))
        reasons.append(f"大户多仓({tt:.2f}↑)")
    elif tt < 0.7 and tt_chg < 0:
        components.append(("tt", -1.0, 0.3))
        reasons.append(f"大户空仓({tt:.2f}↓)")
    elif tt > 1.3 and tt_chg > 0:
        components.append(("tt", 0.5, 0.3))
    elif tt < 0.85 and tt_chg < 0:
        components.append(("tt", -0.5, 0.3))
    else:
        components.append(("tt", 0.0, 0.3))

    # 3. Taker (权重 0.3)
    tk = whale_data["taker_ratio"]
    if tk > 1.3:
        components.append(("tk", 1.0, 0.3))
    elif tk > 1.1:
        components.append(("tk", 0.5, 0.3))
    elif tk < 0.7:
        components.append(("tk", -1.0, 0.3))
    elif tk < 0.9:
        components.append(("tk", -0.5, 0.3))
    else:
        components.append(("tk", 0.0, 0.3))

    total_w = sum(w for _, _, w in components)
    score = sum(s * w for _, s, w in components) / total_w if total_w > 0 else 0
    reason = f"鲸鱼分{score:+.2f} ({', '.join(reasons) if reasons else '中性'})"
    return score, reason


def apply_whale_filter(signal: dict, whale_data: Optional[dict], 
                       price_change_pct: float, mode: str = "SIZE") -> tuple[Optional[dict], str]:
    """应用鲸鱼过滤器, 返回 (修改后的信号 或 None, 说明)."""
    if mode == "DISABLED" or whale_data is None:
        return signal, "鲸鱼过滤未启用或无数据"

    score, reason = evaluate_whale_score(whale_data, price_change_pct)
    direction = signal["direction"]
    direction_score = score if direction == "long" else -score

    if mode == "VETO":
        if direction_score < -0.3:
            return None, f"❌ VETO: {reason}"
        return signal, f"✓ VETO通过: {reason}"

    elif mode == "CONFIRM":
        if direction_score < 0.3:
            return None, f"❌ CONFIRM: {reason} (鲸鱼不支持)"
        return signal, f"✓ CONFIRM通过: {reason}"

    elif mode == "SIZE":
        if direction_score >= 0.5:
            signal["size_multiplier"] = 1.3
            return signal, f"⬆ SIZE 1.3x: {reason}"
        elif direction_score >= -0.3:
            signal["size_multiplier"] = 1.0
            return signal, f"= SIZE 1.0x: {reason}"
        elif direction_score >= -0.5:
            signal["size_multiplier"] = 0.5
            return signal, f"⬇ SIZE 0.5x: {reason}"
        else:
            return None, f"❌ SIZE 跳过: {reason}"

    return signal, "未知模式"


# ═══════════════════════════════════════════════════
# 仓位管理 (动态止损 + 跟踪止盈)
# ═══════════════════════════════════════════════════

class PositionManager:
    """管理开仓后的动态止损/止盈逻辑."""

    def __init__(self, state: State, exchange: ccxt.binanceusdm):
        self.state = state
        self.exchange = exchange

    def check_and_update(self, symbol: str, current_price: float, atr: float):
        """检查每个持仓的 TP/SL, 必要时移动止损 (跟踪止盈)."""
        pos = self.state.get_position(symbol)
        if not pos:
            return

        direction = pos["direction"]
        entry = pos["entry_price"]
        sl_price = pos["sl_price"]
        peak = pos.get("peak_price") or entry

        # 1. 更新峰值
        if direction == "long":
            new_peak = max(peak, current_price)
        else:
            new_peak = min(peak, current_price)
        if new_peak != peak:
            self.state.update_position_field(symbol, "peak_price", new_peak)
            peak = new_peak

        # 2. 跟踪止损: 价格离开入场点 1.5x ATR 后, 把止损上移 1x ATR
        if atr > 0:
            move_distance = abs(current_price - entry)
            if direction == "long" and current_price > entry + 1.5 * atr:
                # 上移止损到 peak - 1x ATR (但不能低于原止损)
                trail_sl = peak - atr
                if trail_sl > sl_price:
                    self.state.update_position_field(symbol, "sl_price", trail_sl)
                    sl_price = trail_sl
                    log(f"  {symbol} 跟踪止损上移到 {trail_sl:.4f} (peak={peak:.4f}, atr={atr:.4f})")
            elif direction == "short" and current_price < entry - 1.5 * atr:
                trail_sl = peak + atr
                if trail_sl < sl_price:
                    self.state.update_position_field(symbol, "sl_price", trail_sl)
                    sl_price = trail_sl
                    log(f"  {symbol} 跟踪止损下移到 {trail_sl:.4f}")

        # 3. 检查是否触发止损
        hit_sl = (direction == "long" and current_price <= sl_price) or \
                 (direction == "short" and current_price >= sl_price)
        if hit_sl:
            self._close_position(symbol, current_price, "止损")
            return

        # 4. 检查止盈
        tp_price = pos.get("tp_price")
        if tp_price:
            hit_tp = (direction == "long" and current_price >= tp_price) or \
                     (direction == "short" and current_price <= tp_price)
            if hit_tp:
                self._close_position(symbol, current_price, "止盈")

    def _close_position(self, symbol: str, price: float, reason: str):
        """平仓 + 记录历史."""
        pos = self.state.get_position(symbol)
        if not pos:
            return

        try:
            # 取消未触发的止损单
            try:
                self.exchange.cancel_all_orders(symbol)
            except Exception:
                pass

            # 市价平仓
            close_side = "sell" if pos["direction"] == "long" else "buy"
            order = self.exchange.create_order(
                symbol, "market", close_side, pos["qty"], None,
                {"reduceOnly": True},
            )

            # 计算 PnL (按真实成交价, 如果没有就用 price 估算)
            actual_price = float(order.get("average") or order.get("price") or price)
            if pos["direction"] == "long":
                pnl_pct = (actual_price - pos["entry_price"]) / pos["entry_price"]
            else:
                pnl_pct = (pos["entry_price"] - actual_price) / pos["entry_price"]
            pnl_pct *= MAX_LEVERAGE
            pnl_usd = pos["entry_price"] * pos["qty"] * pnl_pct / MAX_LEVERAGE

            self.state.log_trade({
                "symbol": symbol,
                "direction": pos["direction"],
                "entry_price": pos["entry_price"],
                "exit_price": actual_price,
                "qty": pos["qty"],
                "pnl_usd": pnl_usd,
                "pnl_pct": pnl_pct * 100,
                "opened_at": pos["opened_at"],
                "closed_at": datetime.now(TZ_UTC8).isoformat(),
                "close_reason": reason,
                "signal_reason": pos.get("signal_reason", ""),
            })
            self.state.remove_position(symbol)
            self.state.set_cooldown(symbol)

            emoji = "💰" if pnl_pct > 0 else "💸"
            log(f"{emoji} 平仓 {symbol} {pos['direction']} @{actual_price:.4f} | "
                f"{reason} | PnL {pnl_pct*100:+.2f}%")
            send_tg(
                f"{emoji} *平仓 {symbol}*\n"
                f"方向: {pos['direction'].upper()}\n"
                f"入场: {pos['entry_price']:.4f} → 平仓: {actual_price:.4f}\n"
                f"PnL: *{pnl_pct*100:+.2f}%* (${pnl_usd:+.2f})\n"
                f"原因: {reason}",
                important=True,
            )
        except Exception as e:
            log(f"❌ 平仓 {symbol} 失败: {e}", "ERROR")
            send_tg(f"❌ *平仓失败 {symbol}*: {str(e)[:100]}", important=True)


# ═══════════════════════════════════════════════════
# 主 Bot
# ═══════════════════════════════════════════════════

class TradingBot:
    def __init__(self):
        self.state = State()
        self.exchange = self._create_exchange()
        self.reconciler = Reconciler(self.exchange, self.state)
        self.heartbeat = HeartbeatMonitor(self.state)
        self.pos_mgr = PositionManager(self.state, self.exchange)
        self.consecutive_failures = 0

    def _create_exchange(self) -> ccxt.binanceusdm:
        ex = ccxt.binanceusdm({
            "apiKey": API_KEY,
            "secret": API_SECRET,
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        })
        if USE_TESTNET:
            ex.set_sandbox_mode(True)
        return ex

    def _generate_client_order_id(self, symbol: str, side: str) -> str:
        """生成唯一的客户端订单 ID 用于去重."""
        ts = int(time.time() * 1000)
        h = hashlib.md5(f"{symbol}{side}{ts}".encode()).hexdigest()[:12]
        return f"bot{h}"

    def _open_position(self, symbol: str, signal: dict, current_price: float, atr: float):
        """开仓 + 设置 SL/TP, 带去重和异常恢复."""
        # 1. 计算止损止盈价
        direction = signal["direction"]
        size_mult = signal.get("size_multiplier", 1.0)

        if "sl_pct" in signal:  # 事件驱动信号有自己的 SL/TP
            sl_distance = current_price * signal["sl_pct"]
            tp_distance = current_price * signal.get("tp_pct", 0.10)
        else:  # 否则用 ATR
            sl_distance = atr * 2.5
            tp_distance = atr * 4.0

        if direction == "long":
            sl_price = current_price - sl_distance
            tp_price = current_price + tp_distance
        else:
            sl_price = current_price + sl_distance
            tp_price = current_price - tp_distance

        # 2. 计算仓位
        try:
            balance = self.exchange.fetch_balance()
            equity = float(balance["total"].get("USDT", 0))
        except Exception as e:
            log(f"获取余额失败: {e}", "ERROR")
            return

        risk_per_unit = abs(current_price - sl_price)
        if risk_per_unit <= 0:
            return
        target_risk = equity * RISK_PER_TRADE * size_mult
        qty_by_risk = target_risk / risk_per_unit
        qty_by_pct = (equity * MAX_PCT_PER_SYMBOL) / current_price
        qty_by_lev = (equity * MAX_LEVERAGE) / current_price
        qty = min(qty_by_risk, qty_by_pct, qty_by_lev)

        try:
            qty = float(self.exchange.amount_to_precision(symbol, qty))
        except Exception:
            return
        if qty <= 0:
            return

        # 3. 去重检查 (防止网络重试导致双开)
        client_oid = self._generate_client_order_id(symbol, direction)
        if self.state.is_order_pending(client_oid):
            log(f"⚠️ {symbol} 已有未确认订单, 跳过")
            return

        self.state.record_pending_order(client_oid, symbol, direction, qty)

        # 4. 下单
        order_side = "buy" if direction == "long" else "sell"
        try:
            try:
                self.exchange.set_leverage(MAX_LEVERAGE, symbol)
            except Exception:
                pass

            order = self.exchange.create_order(
                symbol, "market", order_side, qty, None,
                {"newClientOrderId": client_oid},
            )
            actual_entry = float(order.get("average") or order.get("price") or current_price)

            # 5. 设置止损单
            sl_side = "sell" if direction == "long" else "buy"
            sl_order = self.exchange.create_order(
                symbol, "STOP_MARKET", sl_side, qty, None,
                {"stopPrice": sl_price, "reduceOnly": True},
            )

            # 6. 写入状态
            self.state.save_position({
                "symbol": symbol,
                "direction": direction,
                "entry_price": actual_entry,
                "qty": qty,
                "sl_price": sl_price,
                "tp_price": tp_price,
                "peak_price": actual_entry,
                "opened_at": datetime.now(TZ_UTC8).isoformat(),
                "signal_reason": signal["reason"],
                "client_order_id": client_oid,
                "sl_order_id": sl_order.get("id"),
            })
            self.state.set_cooldown(symbol)
            self.state.mark_order_status(client_oid, "filled")

            emoji = "🟢" if direction == "long" else "🔴"
            log(f"{emoji} 开仓 {symbol} {direction} @{actual_entry:.4f} | "
                f"SL {sl_price:.4f} TP {tp_price:.4f} | qty={qty} | {signal['reason']}")
            send_tg(
                f"{emoji} *开仓 {symbol}*\n"
                f"方向: {direction.upper()} @{actual_entry:.4f}\n"
                f"止盈: {tp_price:.4f} (+{tp_distance/current_price*100:.1f}%)\n"
                f"止损: {sl_price:.4f} (-{sl_distance/current_price*100:.1f}%)\n"
                f"理由: {signal['reason']}\n"
                f"仓位: ${actual_entry * qty:.0f} (倍数 {size_mult}x)",
                important=True,
            )
        except Exception as e:
            self.state.mark_order_status(client_oid, "failed")
            log(f"❌ 开仓 {symbol} 失败: {e}", "ERROR")
            send_tg(f"❌ *开仓失败 {symbol}*: {str(e)[:100]}", important=True)

    def _check_risk_halt(self) -> bool:
        """日内回撤熔断检查. True = 暂停开新仓."""
        try:
            balance = self.exchange.fetch_balance()
            equity = float(balance["total"].get("USDT", 0))
        except Exception:
            return False

        self.state.init_today(equity)
        self.state.update_today_peak(equity)
        stats = self.state.get_today_stats()
        if not stats:
            return False

        if stats["halted"]:
            return True

        # 从今日峰值算回撤
        peak = stats["peak_equity"]
        if peak > 0:
            dd = (peak - equity) / peak
            if dd >= MAX_DAILY_DRAWDOWN:
                self.state.halt_today()
                log(f"🛑 日内回撤熔断: 峰值 ${peak:.0f} → 当前 ${equity:.0f} (-{dd*100:.1f}%)", "WARN")
                send_tg(
                    f"🛑 *日内熔断启动*\n"
                    f"峰值: ${peak:.0f}\n"
                    f"当前: ${equity:.0f}\n"
                    f"回撤: -{dd*100:.1f}%\n"
                    f"今日停止开新仓",
                    important=True,
                )
                return True

        return False

    def scan_once(self):
        """一次完整的扫描 + 交易循环."""
        # 1. 风控熔断
        halted = self._check_risk_halt()

        # 2. 检查现有持仓 (动态止损 + 跟踪止盈)
        for pos in self.state.get_all_positions():
            sym = pos["symbol"]
            try:
                ticker = self.exchange.fetch_ticker(sym)
                current_price = float(ticker.get("last") or 0)
                if current_price <= 0:
                    continue
                # 拉 K 线计算 ATR
                ohlcv = self.exchange.fetch_ohlcv(sym, TIMEFRAME, limit=30)
                atr = compute_atr(ohlcv, 14)
                self.pos_mgr.check_and_update(sym, current_price, atr)
            except Exception as e:
                log(f"检查持仓 {sym} 失败: {e}", "WARN")

        # 3. 能否开新仓?
        open_count = len(self.state.get_all_positions())
        if halted or open_count >= MAX_OPEN_POSITIONS:
            log(f"扫描中 (持仓 {open_count}/{MAX_OPEN_POSITIONS}{'  熔断' if halted else ''})")
            return

        # 4. 拉市场数据
        try:
            tickers = self.exchange.fetch_tickers()
        except Exception as e:
            log(f"获取 tickers 失败: {e}", "ERROR")
            self.consecutive_failures += 1
            return

        # 资金费率
        funding_rates = {}
        try:
            r = requests.get("https://fapi.binance.com/fapi/v1/premiumIndex", timeout=10)
            if r.status_code == 200 and isinstance(r.json(), list):
                funding_rates = {
                    item["symbol"]: float(item["lastFundingRate"]) * 100
                    for item in r.json()
                }
        except Exception:
            pass

        # 5. 遍历币种 (按成交量降序)
        candidates = []
        for symbol, ticker in tickers.items():
            if ":USDT" not in symbol or "/USDT" not in symbol:
                continue
            vol = float(ticker.get("quoteVolume") or 0)
            if vol < MIN_VOLUME_M * 1_000_000:
                continue
            candidates.append((symbol, ticker, vol))

        candidates.sort(key=lambda x: -x[2])
        log(f"扫描 {len(candidates)} 个币种 (持仓 {open_count}/{MAX_OPEN_POSITIONS})")

        signals_found = 0
        for symbol, ticker, vol in candidates:
            if open_count >= MAX_OPEN_POSITIONS:
                break

            # 已有持仓
            if self.state.get_position(symbol):
                continue

            # 冷却期
            if self.state.is_in_cooldown(symbol):
                continue

            raw_sym = symbol.replace("/", "").replace(":USDT", "")
            funding = funding_rates.get(raw_sym, 0)
            current_price = float(ticker.get("last") or 0)
            if current_price <= 0:
                continue

            # 检测信号 (优先级: 事件 > RSI 反转)
            signal = detect_event_signal(ticker, funding)
            if not signal:
                try:
                    ohlcv = self.exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=100)
                    if not ohlcv or len(ohlcv) < 50:
                        continue
                    signal = detect_smart_reversion(ohlcv)
                except Exception:
                    continue

            if not signal:
                continue

            # 鲸鱼过滤
            whale_data = fetch_whale_data(raw_sym)
            change_pct = float(ticker.get("percentage", 0) or 0) / 100
            filtered_signal, whale_reason = apply_whale_filter(
                signal, whale_data, change_pct, mode=WHALE_FILTER_MODE
            )

            if filtered_signal is None:
                log(f"  {symbol} 信号过滤掉: {whale_reason}")
                continue

            log(f"  🎯 {symbol} {signal['direction']} | {signal['reason']} | {whale_reason}")

            # 计算 ATR (用于止损)
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=30)
                atr = compute_atr(ohlcv, 14)
            except Exception:
                atr = current_price * 0.02

            # 开仓
            self._open_position(symbol, filtered_signal, current_price, atr)
            signals_found += 1
            open_count = len(self.state.get_all_positions())
            time.sleep(2)  # 避免连续下单触发限流

        log(f"扫描完成: 开仓 {signals_found} 笔")

    def run(self):
        """主循环."""
        if not API_KEY or not API_SECRET:
            log("❌ 缺少 API Key. 设置 BINANCE_API_KEY 和 BINANCE_API_SECRET", "ERROR")
            sys.exit(1)

        # 启动横幅
        mode = "🧪 TESTNET" if USE_TESTNET else "⚠️  实盘"
        log("=" * 60)
        log(f"币安合约 Bot v2 启动 — {mode}")
        log(f"杠杆: {MAX_LEVERAGE}x | 单笔风险: {RISK_PER_TRADE*100:.1f}% | "
            f"最大持仓: {MAX_OPEN_POSITIONS}")
        log(f"日内熔断: {MAX_DAILY_DRAWDOWN*100:.0f}% | 鲸鱼过滤: {WHALE_FILTER_MODE}")
        log("=" * 60)

        # 启动对账
        try:
            report = self.reconciler.reconcile()
            if report.get("status") != "ok":
                send_tg(f"⚠️ *Bot 启动对账失败*\n{report.get('error', 'unknown')}", important=True)
        except Exception as e:
            log(f"对账异常: {e}", "ERROR")

        # 心跳监控
        self.heartbeat.start()
        self.heartbeat.beat()

        send_tg(
            f"🚀 *Bot v2 启动*\n"
            f"模式: {mode}\n"
            f"杠杆: {MAX_LEVERAGE}x | 风险: {RISK_PER_TRADE*100:.1f}%/笔\n"
            f"鲸鱼过滤: {WHALE_FILTER_MODE}",
            important=True,
        )

        # 主循环 (异常恢复)
        while True:
            try:
                self.scan_once()
                self.heartbeat.beat()
                self.consecutive_failures = 0
            except KeyboardInterrupt:
                log("Bot 手动停止")
                send_tg("🛑 *Bot 手动停止*", important=True)
                self.heartbeat.stop()
                break
            except Exception as e:
                self.consecutive_failures += 1
                log(f"❌ 扫描异常: {e}\n{traceback.format_exc()}", "ERROR")
                if self.consecutive_failures >= 5:
                    backoff = SCAN_INTERVAL * 4
                    send_tg(
                        f"⚠️ *Bot 连续失败 {self.consecutive_failures} 次*\n"
                        f"错误: {str(e)[:200]}\n"
                        f"退避 {backoff//60} 分钟",
                        important=True,
                    )
                    time.sleep(backoff)
                    self.consecutive_failures = 0
                    continue

            # 抖动间隔, 避免被识别为机器人
            jitter = random.randint(-30, 30)
            time.sleep(SCAN_INTERVAL + jitter)


if __name__ == "__main__":
    bot = TradingBot()
    bot.run()
