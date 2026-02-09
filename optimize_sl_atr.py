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
    
    print("Resampling and calculating ATR...")
    # Resample to 1min for execution
    ask_ohlc = full['ask'].resample('1min').ohlc().dropna()
    bid_ohlc = full['bid'].resample('1min').ohlc().dropna()
    
    # 1H ATR for SL normalization
    h1_ohlc = full['ask'].resample('1h').ohlc().dropna()
    h1_ohlc['atr'] = (h1_ohlc['high'] - h1_ohlc['low']).rolling(14).mean()
    
    # Merge H1 ATR back to 1M
    ask_ohlc['h1_atr'] = h1_ohlc['atr'].reindex(ask_ohlc.index, method='ffill')
    ask_ohlc['sma120'] = ask_ohlc['close'].rolling(120).mean()
    
    return ask_ohlc, bid_ohlc

def run_sl_optimization():
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
    h1_atr = ask_ohlc['h1_atr'].values
    
    b_low = bid_ohlc['low'].values
    b_close = bid_ohlc['close'].values
    
    COST = 1.0 # Round-trip
    HOLD_MIN = 240
    
    # Scenarios: Fixed 50 vs ATR multipliers
    scenarios = [
        {'name': 'Fixed 50', 'type': 'fixed', 'value': 50.0},
        {'name': 'ATR x 2', 'type': 'atr', 'value': 2.0},
        {'name': 'ATR x 3', 'type': 'atr', 'value': 3.0},
    ]
    
    results = []
    
    for scen in scenarios:
        net_pips_list = []
        sl_hits = 0
        
        for _, row in tqdm(sigs_buy.iterrows(), total=len(sigs_buy), desc=f"Testing {scen['name']}"):
            sig_ts = row['timestamp']
            
            if sig_ts not in time_to_idx:
                match_ts = times.asof(sig_ts)
                if pd.isna(match_ts) or abs((match_ts - sig_ts).total_seconds()) > 60: continue
                sig_idx = time_to_idx[match_ts]
            else:
                sig_idx = time_to_idx[sig_ts]
            
            # Filter
            if np.isnan(a_sma120[sig_idx]) or a_close[sig_idx] <= a_sma120[sig_idx]:
                continue
            
            # Dynamic SL Calculation
            if scen['type'] == 'fixed':
                sl_dist = scen['value']
            else:
                # Use H1 ATR if available, else fallback to 50
                atr_val = h1_atr[sig_idx]
                sl_dist = (atr_val / 0.01) * scen['value'] if not np.isnan(atr_val) else 50.0
            
            entry_idx = sig_idx + 1
            if entry_idx >= len(times): continue
            
            entry_price = a_open[entry_idx]
            exit_idx = min(entry_idx + HOLD_MIN, len(times) - 1)
            
            outcome_pips = None
            for j in range(entry_idx, exit_idx + 1):
                if (entry_price - b_low[j]) / 0.01 >= sl_dist:
                    outcome_pips = -sl_dist
                    sl_hits += 1
                    break
            
            if outcome_pips is None:
                exit_price = b_close[exit_idx]
                outcome_pips = (exit_price - entry_price) / 0.01
            
            net_pips_list.append(outcome_pips - COST)
            
        if net_pips_list:
            ps = pd.Series(net_pips_list)
            results.append({
                'Scenario': scen['name'],
                'Trades': len(ps),
                'TotalNet': ps.sum(),
                'AvgNet': ps.mean(),
                'WinRate': (ps > 0).mean(),
                'SL_Hits': sl_hits,
                'Max_Loss': ps.min()
            })
            
    res_df = pd.DataFrame(results)
    print("\n--- SL Optimization: Fixed vs ATR-Based (BUY, SMA120, 4h Exit) ---")
    print(res_df.to_string(index=False))

if __name__ == "__main__":
    run_sl_optimization()
