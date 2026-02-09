import pandas as pd
import glob
import os
from tqdm import tqdm
import numpy as np

def load_data():
    files = sorted(glob.glob('data_cache/USDJPY_2025_*.parquet'))
    print(f"Loading {len(files)} shards...")
    dfs = []
    for f in tqdm(files):
        dfs.append(pd.read_parquet(f)[['ask', 'bid']])
    full = pd.concat(dfs).sort_index()
    if full.index.tz is not None: full.index = full.index.tz_localize(None)
    
    print("Resampling...")
    # 1m for execution
    ask_ohlc = full['ask'].resample('1min').ohlc().dropna()
    bid_ohlc = full['bid'].resample('1min').ohlc().dropna()
    
    # 1H ATR for SL normalization
    h1_ohlc = full['ask'].resample('1h').ohlc().dropna()
    h1_ohlc['atr'] = (h1_ohlc['high'] - h1_ohlc['low']).rolling(14).mean()
    
    # SMA120 Slope
    ask_ohlc['sma120'] = ask_ohlc['close'].rolling(120).mean()
    ask_ohlc['sma120_slope'] = ask_ohlc['sma120'].diff(10) # 10-min slope
    
    # Merge back to 1M
    ask_ohlc['h1_atr'] = h1_ohlc['atr'].reindex(ask_ohlc.index, method='ffill')
    
    return ask_ohlc, bid_ohlc

def run_regime_audit():
    ask_ohlc, bid_ohlc = load_data()
    
    sigs = pd.read_csv('oos_detailed_signals_sagemaker.csv')
    sigs['timestamp'] = pd.to_datetime(sigs['timestamp'])
    if sigs['timestamp'].dt.tz is not None: sigs['timestamp'] = sigs['timestamp'].dt.tz_localize(None)
    sigs_buy = sigs[(sigs['timestamp'] >= '2025-07-01') & (sigs['timestamp'] < '2026-01-01') & (sigs['side'] == 'BUY')]
    
    times = ask_ohlc.index
    time_to_idx = {t: i for i, t in enumerate(times)}
    
    a_open = ask_ohlc['open'].values
    a_close = ask_ohlc['close'].values
    a_sma120 = ask_ohlc['sma120'].values
    a_slope = ask_ohlc['sma120_slope'].values
    h1_atr = ask_ohlc['h1_atr'].values
    
    b_low = bid_ohlc['low'].values
    b_close = bid_ohlc['close'].values
    
    COST = 1.0
    HOLD_MIN = 240
    
    trades = []
    
    for _, row in tqdm(sigs_buy.iterrows(), total=len(sigs_buy), desc="Simulating Trades"):
        sig_ts = row['timestamp']
        if sig_ts not in time_to_idx:
            match_ts = times.asof(sig_ts)
            if pd.isna(match_ts) or abs((match_ts - sig_ts).total_seconds()) > 60: continue
            sig_idx = time_to_idx[match_ts]
        else:
            sig_idx = time_to_idx[sig_ts]
            
        # Filter: SMA120
        if np.isnan(a_sma120[sig_idx]) or a_close[sig_idx] <= a_sma120[sig_idx]:
            continue
            
        # SL: ATR x 3
        atr_val = h1_atr[sig_idx]
        sl_dist = (atr_val / 0.01) * 3.0 if not np.isnan(atr_val) else 50.0
        
        entry_idx = sig_idx + 1
        if entry_idx >= len(times): continue
        
        entry_price = a_open[entry_idx]
        exit_idx = min(entry_idx + HOLD_MIN, len(times) - 1)
        
        outcome_pips = None
        for j in range(entry_idx, exit_idx + 1):
            if (entry_price - b_low[j]) / 0.01 >= sl_dist:
                outcome_pips = -sl_dist
                break
        
        if outcome_pips is None:
            exit_price = b_close[exit_idx]
            outcome_pips = (exit_price - entry_price) / 0.01
            
        trades.append({
            'timestamp': sig_ts,
            'pips': outcome_pips - COST,
            'slope': a_slope[sig_idx],
            'hour': sig_ts.hour
        })
        
    df = pd.DataFrame(trades)
    
    # Analyze by Hour
    hour_stats = df.groupby('hour')['pips'].agg(['count', 'sum', 'mean']).reset_index()
    print("\n--- PERFORMANCE BY HOUR OF DAY (UTC) ---")
    print(hour_stats.to_string(index=False))
    
    # Analyze by Slope Regime
    # Define Regime: Up (slope > 0.001), Down (slope < -0.001), Flat (else)
    def get_regime(s):
        if s > 0.002: return 'Strong UP'
        if s > 0: return 'Weak UP'
        if s < -0.002: return 'Strong DOWN'
        return 'Weak DOWN/Flat'
    
    df['regime'] = df['slope'].apply(get_regime)
    regime_stats = df.groupby('regime')['pips'].agg(['count', 'sum', 'mean']).reset_index()
    print("\n--- PERFORMANCE BY SMA120 SLOPE REGIME ---")
    print(regime_stats.to_string(index=False))

if __name__ == "__main__":
    run_regime_audit()
