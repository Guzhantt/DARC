"""Demo: fetch real Binance data and run backtest without any API key."""
import sys
sys.path.insert(0, '/projects/sandbox/DARC')

import requests
import pandas as pd
import numpy as np

# 1. 从币安公开API拉取 BTC 1h K线 (不需要API Key)
print('从币安拉取 BTCUSDT 1小时K线数据 (30天)...')
url = 'https://fapi.binance.com/fapi/v1/klines'
params = {'symbol': 'BTCUSDT', 'interval': '1h', 'limit': 720}
resp = requests.get(url, params=params, timeout=15)
raw = resp.json()
print(f'获取到 {len(raw)} 根K线')

df = pd.DataFrame(raw, columns=['ts','open','high','low','close','volume',
    'close_time','quote_vol','trades','taker_buy_base','taker_buy_quote','ignore'])
df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
for c in ['open','high','low','close','volume']:
    df[c] = df[c].astype(float)
df = df[['ts','open','high','low','close','volume']].copy()
print(f'时间范围: {df["ts"].iloc[0]} → {df["ts"].iloc[-1]}')
print(f'价格范围: {df["close"].min():.0f} - {df["close"].max():.0f} USDT')
print()

# 2. 拉取 OI + 多空比 + Taker + 资金费率
print('拉取衍生品市场数据...')
from src.data.oi_data import (
    fetch_oi_history, fetch_global_long_short_ratio,
    fetch_taker_long_short_ratio, fetch_funding_rate_history
)
from src.data.onchain_data import (
    compute_whale_score, compute_funding_signal,
    compute_exchange_flow_proxy, compute_oi_momentum
)

oi = fetch_oi_history('BTCUSDT', '1h', 500)
print(f'  OI 数据: {len(oi)} 条')

ls = fetch_global_long_short_ratio('BTCUSDT', '1h', 500)
print(f'  多空比数据: {len(ls)} 条')

taker = fetch_taker_long_short_ratio('BTCUSDT', '1h', 500)
print(f'  Taker数据: {len(taker)} 条')

funding = fetch_funding_rate_history('BTCUSDT', 500)
print(f'  资金费率: {len(funding)} 条')
print()

# 3. 合并所有数据
if df['ts'].dt.tz is None:
    df['ts'] = df['ts'].dt.tz_localize('UTC')

if not oi.empty:
    if oi['ts'].dt.tz is None:
        oi['ts'] = oi['ts'].dt.tz_localize('UTC')
    df = pd.merge_asof(df.sort_values('ts'), oi.sort_values('ts'), on='ts', direction='backward')

if not ls.empty:
    if ls['ts'].dt.tz is None:
        ls['ts'] = ls['ts'].dt.tz_localize('UTC')
    df = pd.merge_asof(df.sort_values('ts'), ls.sort_values('ts'), on='ts', direction='backward')

if not taker.empty:
    if taker['ts'].dt.tz is None:
        taker['ts'] = taker['ts'].dt.tz_localize('UTC')
    df = pd.merge_asof(df.sort_values('ts'), taker.sort_values('ts'), on='ts', direction='backward')

if not funding.empty:
    if funding['ts'].dt.tz is None:
        funding['ts'] = funding['ts'].dt.tz_localize('UTC')
    funding['funding_signal'] = compute_funding_signal(funding).values
    df = pd.merge_asof(df.sort_values('ts'), funding[['ts','funding_rate','funding_signal']].sort_values('ts'), on='ts', direction='backward')

# Compute derived features
df['whale_score'] = compute_whale_score(df).values
df['flow_proxy'] = compute_exchange_flow_proxy(df).values
df['oi_momentum'] = compute_oi_momentum(df).values

for col in ['oi','ls_ratio','taker_ratio','funding_rate','funding_signal','whale_score','flow_proxy','oi_momentum']:
    if col not in df.columns:
        df[col] = np.nan

df = df.sort_values('ts').reset_index(drop=True)
print(f'合并后数据: {len(df)} 行, {len(df.columns)} 列')
print()

# 4. 回测
print('='*60)
print(' BTC/USDT 1H 回测结果 (实际币安数据, 近30天)')
print('='*60)
from src.strategies import get_strategy, REGISTRY
from src.backtest.engine import run_backtest, BacktestConfig

cfg = BacktestConfig(initial_equity=10000, risk_per_trade=0.01, max_leverage=3)

print()
print(f'{"策略":<15} {"交易数":>6} {"胜率":>7} {"盈亏比":>7} {"收益":>9} {"最大回撤":>9} {"Sharpe":>7}')
print('-'*65)
for name in REGISTRY:
    s = get_strategy(name)
    r = run_backtest(df, s, cfg)
    m = r.metrics
    print(f'{name:<15} {m.trades:>6} {m.win_rate:>6.1%} {m.profit_factor:>7.2f} {m.total_return:>+8.2%} {m.max_drawdown:>8.2%} {m.sharpe:>7.2f}')

print()
print('='*60)
print(' oi_composite 详细结果')
print('='*60)
strat = get_strategy('oi_composite')
res = run_backtest(df, strat, cfg)
m = res.metrics
print(f'交易次数:   {m.trades}')
print(f'胜率:       {m.win_rate:.1%}')
print(f'盈亏比(PF): {m.profit_factor:.2f}')
print(f'总收益:     {m.total_return:+.2%}')
print(f'最大回撤:   {m.max_drawdown:.2%}')
print(f'Sharpe:     {m.sharpe:.2f}')
print(f'最终权益:   {m.final_equity:.2f} USDT (初始 10000)')
print(f'平均盈利:   {m.avg_win:.2f} USDT')
print(f'平均亏损:   {m.avg_loss:.2f} USDT')

if not res.trades.empty:
    print()
    print('=== 最近交易记录 ===')
    t = res.trades.tail(8).copy()
    t['entry'] = t['entry'].apply(lambda x: f'${x:,.1f}')
    t['exit'] = t['exit'].apply(lambda x: f'${x:,.1f}')
    t['pnl'] = t['pnl'].apply(lambda x: f'{x:+.2f}')
    print(t[['entry_ts','side','entry','exit','pnl','reason']].to_string(index=False))

print()
print('DONE.')
