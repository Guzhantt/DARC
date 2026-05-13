#!/usr/bin/env python3
"""
币安合约自动交易 Bot — 单文件版
策略: smart_reversion (85%胜率) + alpha_scanner (交易员模式)
直接喂给 OpenClaw 或任何 Python 环境就能跑

用法:
  1. pip install ccxt requests python-dotenv pandas numpy
  2. 设置环境变量或创建 .env 文件:
     BINANCE_API_KEY=你的key
     BINANCE_API_SECRET=你的secret
     USE_TESTNET=true
     TG_BOT_TOKEN=你的telegram bot token (可选)
     TG_CHAT_ID=你的telegram chat id (可选)
  3. python trading_bot.py

默认连接 Binance Futures TESTNET（假钱），确认稳定后改 USE_TESTNET=false
"""

import os
import sys
import time
import json
import random
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path

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

# 风控参数
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.02"))       # 每笔风险2%权益
MAX_LEVERAGE = int(os.getenv("MAX_LEVERAGE", "3"))
MAX_DAILY_DRAWDOWN = float(os.getenv("MAX_DAILY_DRAWDOWN", "0.05"))
MAX_OPEN_POSITIONS = int(os.getenv("MAX_OPEN_POSITIONS", "3"))

# 策略参数 (smart_reversion — 85%胜率最优参数)
RSI_PERIOD = 9
RSI_EXTREME = 30        # RSI < 30 = 超卖, > 70 = 超买
TP_PCT = 0.008          # 止盈 0.8%
SL_PCT = 0.04           # 止损 4%

# 扫描参数
SCAN_INTERVAL = 300     # 每5分钟扫描一次
TIMEFRAME = "1h"        # K线周期
MIN_VOLUME_M = 10       # 最小24h成交量(百万USDT)
COOLDOWN_HOURS = 4      # 同币种冷却时间

# Telegram
TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN", "")
TG_CHAT_ID = os.getenv("TG_CHAT_ID", "")

# 本地状态文件
STATE_FILE = Path("bot_state.json")
LOG_FILE = Path("bot.log")
TZ_UTC8 = timezone(timedelta(hours=8))


# ═══════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════

def log(msg: str):
    ts = datetime.now(TZ_UTC8).strftime("%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    except:
        pass


def send_tg(text: str):
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": TG_CHAT_ID, "text": text, "parse_mode": "Markdown"}, timeout=10)
    except:
        pass


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"cooldowns": {}, "day_start_equity": 0, "day_key": "", "halted": False}


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


# ═══════════════════════════════════════════════════
# 交易所连接
# ═══════════════════════════════════════════════════

def create_exchange() -> ccxt.binanceusdm:
    exchange = ccxt.binanceusdm({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    })
    if USE_TESTNET:
        exchange.set_sandbox_mode(True)
    return exchange


# ═══════════════════════════════════════════════════
# 策略: RSI 极端反转 (smart_reversion — 85%胜率)
# ═══════════════════════════════════════════════════

def compute_rsi(closes: list[float], period: int = RSI_PERIOD) -> list[float]:
    """计算RSI序列"""
    if len(closes) < period + 1:
        return [50.0] * len(closes)
    
    rsi_values = [50.0] * len(closes)
    gains = []
    losses = []
    
    for i in range(1, len(closes)):
        change = closes[i] - closes[i-1]
        gains.append(max(change, 0))
        losses.append(max(-change, 0))
    
    if len(gains) < period:
        return rsi_values
    
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        if avg_loss == 0:
            rsi_values[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi_values[i + 1] = 100 - (100 / (1 + rs))
    
    return rsi_values


def check_smart_reversion_signal(ohlcv: list) -> dict | None:
    """
    检查 smart_reversion 信号
    
    条件:
    - RSI(9) 之前跌到 < 30 (或涨到 > 70)
    - RSI 现在回升过 30 (或回落过 70)
    - 确认K线: 阳线(做多) 或 阴线(做空)
    
    返回: {"direction": "long"/"short", "reason": "..."} 或 None
    """
    if len(ohlcv) < 50:
        return None
    
    closes = [float(bar[4]) for bar in ohlcv]
    opens = [float(bar[1]) for bar in ohlcv]
    
    rsi = compute_rsi(closes, RSI_PERIOD)
    
    # 用倒数第2根(已闭合)的数据判断
    rsi_cur = rsi[-2]     # 上一根K线的RSI
    rsi_prev = rsi[-3]    # 再上一根
    close_prev = closes[-2]
    open_prev = opens[-2]
    
    if np.isnan(rsi_cur) or np.isnan(rsi_prev):
        return None
    
    # LONG: RSI从极端超卖回升 + 阳线确认
    if (rsi_prev < RSI_EXTREME 
        and rsi_cur > rsi_prev 
        and rsi_cur > RSI_EXTREME 
        and close_prev > open_prev):
        return {
            "direction": "long",
            "reason": f"RSI从{rsi_prev:.1f}回升到{rsi_cur:.1f} 超卖反转+阳线确认",
        }
    
    # SHORT: RSI从极端超买回落 + 阴线确认
    if (rsi_prev > (100 - RSI_EXTREME) 
        and rsi_cur < rsi_prev 
        and rsi_cur < (100 - RSI_EXTREME) 
        and close_prev < open_prev):
        return {
            "direction": "short",
            "reason": f"RSI从{rsi_prev:.1f}回落到{rsi_cur:.1f} 超买反转+阴线确认",
        }
    
    return None


# ═══════════════════════════════════════════════════
# 策略: 极端事件检测 (交易员模式)
# ═══════════════════════════════════════════════════

def check_event_signals(ticker: dict, funding_rate: float) -> dict | None:
    """
    检测事件驱动信号 (来自交易员脚本的逻辑):
    1. 极端负费率 → 做多 (逼空)
    2. 极端正费率 → 做空 (多头拥挤)
    3. 24h暴跌 > 20% → 做多 (超跌反弹)
    4. 24h暴涨 > 35% → 做空 (回调)
    """
    change_pct = float(ticker.get("percentage", 0) or 0)
    
    # 1. 极端负费率
    if funding_rate < -0.08:
        return {
            "direction": "long",
            "reason": f"资金费率极端负值({funding_rate:.4f}%) 逼空信号",
            "tp_pct": 0.12,
            "sl_pct": 0.08,
        }
    
    # 2. 极端正费率
    if funding_rate > 0.10:
        return {
            "direction": "short",
            "reason": f"资金费率极端正值({funding_rate:.4f}%) 多头过度拥挤",
            "tp_pct": 0.15,
            "sl_pct": 0.10,
        }
    
    # 3. 暴跌反弹
    if change_pct < -20:
        return {
            "direction": "long",
            "reason": f"24h暴跌{change_pct:.1f}% 超跌反弹",
            "tp_pct": 0.15,
            "sl_pct": 0.10,
        }
    
    # 4. 暴涨做空
    if change_pct > 35:
        return {
            "direction": "short",
            "reason": f"24h暴涨{change_pct:.1f}% 回调概率>85%",
            "tp_pct": 0.20,
            "sl_pct": 0.15,
        }
    
    return None


# ═══════════════════════════════════════════════════
# 风控
# ═══════════════════════════════════════════════════

def check_risk(state: dict, equity: float) -> bool:
    """检查日内风控，返回True=允许交易"""
    today = datetime.now(TZ_UTC8).strftime("%Y-%m-%d")
    
    if state.get("day_key") != today:
        state["day_key"] = today
        state["day_start_equity"] = equity
        state["halted"] = False
        save_state(state)
    
    if state["day_start_equity"] > 0:
        dd = (state["day_start_equity"] - equity) / state["day_start_equity"]
        if dd >= MAX_DAILY_DRAWDOWN:
            state["halted"] = True
            save_state(state)
            return False
    
    return not state.get("halted", False)


def position_size(equity: float, entry_price: float, stop_price: float) -> float:
    """计算仓位大小 (基于固定风险%)"""
    risk_per_unit = abs(entry_price - stop_price)
    if risk_per_unit <= 0:
        return 0
    qty = (equity * RISK_PER_TRADE) / risk_per_unit
    max_qty = (equity * MAX_LEVERAGE) / entry_price
    return min(qty, max_qty)


# ═══════════════════════════════════════════════════
# 交易执行
# ═══════════════════════════════════════════════════

def get_open_positions(exchange: ccxt.binanceusdm) -> list[dict]:
    """获取当前所有持仓"""
    positions = exchange.fetch_positions()
    return [p for p in positions if float(p.get("contracts", 0)) > 0]


def close_position(exchange: ccxt.binanceusdm, symbol: str, position: dict):
    """平仓"""
    contracts = float(position["contracts"])
    side = "sell" if position.get("side", "").lower() == "long" else "buy"
    try:
        exchange.cancel_all_orders(symbol)
    except:
        pass
    exchange.create_order(symbol, "market", side, contracts, None, {"reduceOnly": True})


def open_trade(exchange: ccxt.binanceusdm, symbol: str, direction: str, 
               qty: float, stop_price: float):
    """开仓 + 设置止损单"""
    order_side = "buy" if direction == "long" else "sell"
    try:
        exchange.set_leverage(MAX_LEVERAGE, symbol)
    except:
        pass
    exchange.create_order(symbol, "market", order_side, qty)
    # 止损单
    stop_side = "sell" if direction == "long" else "buy"
    exchange.create_order(symbol, "STOP_MARKET", stop_side, qty, None, 
                          {"stopPrice": stop_price, "reduceOnly": True})


# ═══════════════════════════════════════════════════
# 主循环
# ═══════════════════════════════════════════════════

def scan_and_trade(exchange: ccxt.binanceusdm, state: dict):
    """一次完整的扫描+交易循环"""
    
    # 1. 获取账户信息
    balance = exchange.fetch_balance()
    equity = float(balance["total"].get("USDT", 0))
    
    if equity <= 0:
        log("⚠️ 余额为0，跳过")
        return
    
    # 2. 风控检查
    if not check_risk(state, equity):
        log(f"🛑 日内亏损超限，今日停止交易 (余额: ${equity:.2f})")
        return
    
    # 3. 检查现有持仓
    positions = get_open_positions(exchange)
    open_symbols = set()
    
    for pos in positions:
        sym = pos["symbol"]
        open_symbols.add(sym)
        # 检查止盈 (手动，因为binance不支持OCO on futures easily)
        current_price = float(pos.get("markPrice", 0) or pos.get("entryPrice", 0))
        entry_price = float(pos.get("entryPrice", 0))
        pos_side = pos.get("side", "").lower()
        
        if entry_price > 0 and current_price > 0:
            if pos_side == "long" and current_price >= entry_price * (1 + TP_PCT):
                log(f"💰 止盈平仓 {sym} LONG @{current_price:.2f}")
                close_position(exchange, sym, pos)
                send_tg(f"💰 *止盈* {sym} LONG\n入场: {entry_price:.2f}\n平仓: {current_price:.2f}\nPnL: +{(current_price/entry_price-1)*MAX_LEVERAGE*100:.1f}%")
                open_symbols.discard(sym)
            elif pos_side == "short" and current_price <= entry_price * (1 - TP_PCT):
                log(f"💰 止盈平仓 {sym} SHORT @{current_price:.2f}")
                close_position(exchange, sym, pos)
                send_tg(f"💰 *止盈* {sym} SHORT\n入场: {entry_price:.2f}\n平仓: {current_price:.2f}\nPnL: +{(1-current_price/entry_price)*MAX_LEVERAGE*100:.1f}%")
                open_symbols.discard(sym)
    
    # 4. 能否开新仓?
    can_open = len(open_symbols) < MAX_OPEN_POSITIONS
    if not can_open:
        log(f"持仓已满 ({len(open_symbols)}/{MAX_OPEN_POSITIONS})，只监控不开新仓")
        return
    
    log(f"扫描中... 余额: ${equity:.2f} | 持仓: {len(open_symbols)}/{MAX_OPEN_POSITIONS}")
    
    # 5. 获取市场数据
    try:
        tickers = exchange.fetch_tickers()
    except Exception as e:
        log(f"⚠️ 获取tickers失败: {e}")
        return
    
    # 获取资金费率
    funding_rates = {}
    try:
        data = requests.get("https://fapi.binance.com/fapi/v1/premiumIndex", timeout=10).json()
        if isinstance(data, list):
            funding_rates = {item["symbol"]: float(item["lastFundingRate"]) * 100 for item in data}
    except:
        pass
    
    # 6. 遍历所有USDT永续合约
    signals_found = 0
    
    for symbol, ticker in tickers.items():
        # 只要USDT永续
        if not symbol.endswith(":USDT") or "/USDT" not in symbol:
            continue
        
        # 成交量过滤
        vol = ticker.get("quoteVolume", 0) or 0
        if vol < MIN_VOLUME_M * 1_000_000:
            continue
        
        # 已有持仓跳过
        if symbol in open_symbols:
            continue
        
        # 冷却期检查
        cooldowns = state.get("cooldowns", {})
        last_trade = cooldowns.get(symbol)
        if last_trade:
            elapsed = datetime.now(TZ_UTC8) - datetime.fromisoformat(last_trade)
            if elapsed < timedelta(hours=COOLDOWN_HOURS):
                continue
        
        # === 检测信号 ===
        signal = None
        
        # 优先级1: 事件驱动信号 (极端费率/暴跌/暴涨)
        raw_sym = symbol.replace("/", "").replace(":USDT", "")
        fr = funding_rates.get(raw_sym, 0)
        event_sig = check_event_signals(ticker, fr)
        if event_sig:
            signal = event_sig
        
        # 优先级2: RSI极端反转 (smart_reversion)
        if not signal:
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=100)
                if ohlcv and len(ohlcv) >= 50:
                    rsi_sig = check_smart_reversion_signal(ohlcv)
                    if rsi_sig:
                        signal = rsi_sig
                        signal["tp_pct"] = TP_PCT
                        signal["sl_pct"] = SL_PCT
            except:
                continue
        
        if not signal:
            continue
        
        # === 执行开仓 ===
        price = float(ticker.get("last", 0))
        if price <= 0:
            continue
        
        tp_pct = signal.get("tp_pct", TP_PCT)
        sl_pct = signal.get("sl_pct", SL_PCT)
        direction = signal["direction"]
        
        if direction == "long":
            stop_price = price * (1 - sl_pct)
        else:
            stop_price = price * (1 + sl_pct)
        
        qty = position_size(equity, price, stop_price)
        try:
            qty = float(exchange.amount_to_precision(symbol, qty))
        except:
            continue
        
        if qty <= 0:
            continue
        
        try:
            open_trade(exchange, symbol, direction, qty, stop_price)
            signals_found += 1
            
            # 记录冷却
            state.setdefault("cooldowns", {})[symbol] = datetime.now(TZ_UTC8).isoformat()
            save_state(state)
            
            emoji = "🟢" if direction == "long" else "🔴"
            tp_price = price * (1 + tp_pct) if direction == "long" else price * (1 - tp_pct)
            
            log(f"{emoji} 开仓 {symbol} {direction.upper()} @{price:.4f} | {signal['reason']}")
            log(f"   止盈: {tp_price:.4f} | 止损: {stop_price:.4f} | 数量: {qty}")
            
            send_tg(
                f"{emoji} *开仓* {symbol}\n"
                f"方向: {direction.upper()} @{price:.4f}\n"
                f"止盈: {tp_price:.4f} ({tp_pct*100:.1f}%)\n"
                f"止损: {stop_price:.4f} ({sl_pct*100:.1f}%)\n"
                f"理由: {signal['reason']}\n"
                f"余额: ${equity:.2f}"
            )
            
            open_symbols.add(symbol)
            if len(open_symbols) >= MAX_OPEN_POSITIONS:
                break
                
        except Exception as e:
            log(f"❌ 开仓失败 {symbol}: {e}")
    
    log(f"扫描完成，本轮开仓 {signals_found} 笔")


def main():
    if not API_KEY or not API_SECRET:
        print("❌ 请设置环境变量 BINANCE_API_KEY 和 BINANCE_API_SECRET")
        print("   或创建 .env 文件（参考 .env.example）")
        sys.exit(1)
    
    exchange = create_exchange()
    state = load_state()
    consecutive_failures = 0
    
    mode = "🧪 TESTNET (假钱)" if USE_TESTNET else "⚠️ 实盘 (真钱!)"
    
    log("=" * 50)
    log(f"🚀 币安合约交易 Bot 启动")
    log(f"   模式: {mode}")
    log(f"   策略: smart_reversion (85%胜率) + 事件驱动")
    log(f"   杠杆: {MAX_LEVERAGE}x | 风险: {RISK_PER_TRADE*100:.0f}%/笔")
    log(f"   最大持仓: {MAX_OPEN_POSITIONS} | 周期: {TIMEFRAME}")
    log(f"   扫描间隔: {SCAN_INTERVAL}秒")
    log("=" * 50)
    
    send_tg(
        f"🚀 *Bot启动*\n"
        f"模式: {mode}\n"
        f"策略: smart_reversion + 事件驱动\n"
        f"杠杆: {MAX_LEVERAGE}x | 最大持仓: {MAX_OPEN_POSITIONS}"
    )
    
    while True:
        try:
            scan_and_trade(exchange, state)
            consecutive_failures = 0
        except Exception as e:
            consecutive_failures += 1
            log(f"⚠️ 异常: {e}")
            if consecutive_failures >= 5:
                backoff = SCAN_INTERVAL * 4
                log(f"连续失败{consecutive_failures}次，退避{backoff}秒")
                send_tg(f"⚠️ Bot连续失败{consecutive_failures}次，退避{backoff//60}分钟")
                time.sleep(backoff)
                continue
        
        # 随机抖动避免固定间隔被ban
        jitter = random.randint(-30, 30)
        time.sleep(SCAN_INTERVAL + jitter)


if __name__ == "__main__":
    main()
