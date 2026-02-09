import pandas as pd
import numpy as np
import os
import ta
from tqdm import tqdm

# --- SETTINGS ---
SIGNAL_FILE = "rf_signals_final.csv"
DATA_FILE = "full_data_cache.parquet"
COST_PIPS = 1.0
ATR_LEN = 14
HOLD_MINUTES = 240
SL_ATR_MULT = 3.0
MAX_SL_PIPS = 80.0
MIN_SL_PIPS = 10.0
SMA_TESTS = [120, 240]

def calculate_max_consecutive_losses(net_pips_list):
    max_consecutive = 0
    current_consecutive = 0
    for pnl in net_pips_list:
        if pnl < 0:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 0
    return max_consecutive

def calculate_drawdown(net_pips_list):
    cumulative = np.cumsum(net_pips_list)
    peak = np.maximum.accumulate(cumulative)
    drawdown = peak - cumulative
    return np.max(drawdown)

def main():
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found.")
        return
    
    print(f"Loading data from {DATA_FILE}...")
    df = pd.read_parquet(DATA_FILE)
    if 'time' in df.columns: df = df.set_index('time')
    if df.index.tz is not None: df.index = df.index.tz_localize(None)
    df = df.sort_index()
    
    print("Resampling to 1-minute OHLC...")
    ohlc = df['ask'].resample('1min').ohlc().ffill()
    bid_ohlc = df['bid'].resample('1min').ohlc().ffill()
    ohlc['bid_close'] = bid_ohlc['close']
    ohlc['bid_low'] = bid_ohlc['low']
    ohlc['ask_high'] = ohlc['high']
    
    # ATR 1H (aligned to 1M)
    ohlc_1h = ohlc['close'].resample('1h').ohlc().dropna()
    atr_1h = ta.volatility.average_true_range(ohlc_1h['high'], ohlc_1h['low'], ohlc_1h['close'], window=ATR_LEN)
    atr_1h_long = pd.Series(atr_1h, index=ohlc_1h.index).reindex(ohlc.index, method='ffill')
    ohlc['ATR_SL_Dist'] = atr_1h_long * SL_ATR_MULT
    
    # Load Signals
    sig_df = pd.read_csv(SIGNAL_FILE)
    sig_df['timestamp'] = pd.to_datetime(sig_df['timestamp'])
    if sig_df['timestamp'].dt.tz is not None:
        sig_df['timestamp'] = sig_df['timestamp'].dt.tz_localize(None)
    
    # Filter only BUY for this audit
    sig_df = sig_df[sig_df['side'] == 'BUY']

    for sma_period in SMA_TESTS:
        print(f"\n>>> AUDIT: SMA {sma_period} (BUY ONLY) <<<")
        ohlc[f'SMA{sma_period}'] = ta.trend.sma_indicator(ohlc['close'], window=sma_period)
        
        results = []
        for _, row in tqdm(sig_df.iterrows(), total=len(sig_df), desc=f"SMA {sma_period}"):
            t = row['timestamp']
            if t not in ohlc.index: continue
            
            entry_data = ohlc.loc[t]
            if pd.isna(entry_data[f'SMA{sma_period}']): continue
            
            # Trend Filter (Price > SMA only)
            if entry_data['close'] <= entry_data[f'SMA{sma_period}']: continue
            
            entry_p = entry_data['close']
            sl_dist = entry_data['ATR_SL_Dist'] # This is directly in price units (e.g., 0.45)
            
            # SL floor/ceiling in price units
            if sl_dist < (MIN_SL_PIPS * 0.01): sl_dist = MIN_SL_PIPS * 0.01
            if sl_dist > (MAX_SL_PIPS * 0.01): sl_dist = MAX_SL_PIPS * 0.01
            
            sl_p = entry_p - sl_dist
            
            exit_window = ohlc.loc[t : t + pd.Timedelta(minutes=HOLD_MINUTES)]
            if len(exit_window) < 2: continue
            
            hit_sl = False
            exit_p = 0
            for ft, frow in exit_window.iterrows():
                if frow['bid_low'] <= sl_p:
                    exit_p = sl_p
                    hit_sl = True
                    break
            
            if not hit_sl:
                exit_p = exit_window.iloc[-1]['bid_close']
                
            pnl = (exit_p - entry_p) * 100
            results.append({'timestamp': t, 'net_pips': pnl - COST_PIPS})
            
        res_df = pd.DataFrame(results)
        if res_df.empty: 
            print("No trades passed filters.")
            continue
            
        net_pips = res_df['net_pips'].tolist()
        max_dd = calculate_drawdown(net_pips)
        max_losses = calculate_max_consecutive_losses(net_pips)
        
        print(f"\n--- Results for SMA {sma_period} ---")
        print(f"Total Trades: {len(res_df)}")
        print(f"Total Net Pips: {res_df['net_pips'].sum():.1f}")
        print(f"Average Pips: {res_df['net_pips'].mean():.2f}")
        print(f"Max Drawdown (Pips): {max_dd:.1f}")
        print(f"Max Consecutive Losses: {max_losses}")
        print(f"Win Rate (with SL): {(res_df['net_pips'] > 0).mean() * 100:.1f}%")
        
        res_df['month'] = res_df['timestamp'].dt.to_period('M')
        print(f"\nMonthly Net Pips (SMA {sma_period}):")
        print(res_df.groupby('month')['net_pips'].sum())

if __name__ == "__main__":
    main()
