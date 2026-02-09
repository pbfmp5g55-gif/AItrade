import pandas as pd
import numpy as np
import os
import ta
from tqdm import tqdm

# --- SETTINGS ---
SIGNAL_FILE = "rf_signals_final.csv"
POSSIBLE_DATA_FILES = ["full_data_consolidated_2023_2026.parquet", "full_data_cache.parquet"]
COST_PIPS = 1.0
SMA_FILTER_LEN = 120
ATR_LEN = 14
EXIT_SWEEP_MINUTES = [60, 120, 240, 480]

def main():
    data_file = None
    for f in POSSIBLE_DATA_FILES:
        if os.path.exists(f):
            data_file = f
            break
            
    if not data_file:
        print("Error: No data parquet file found locally.")
        return
    
    if not os.path.exists(SIGNAL_FILE):
        print(f"Error: {SIGNAL_FILE} not found.")
        return
    
    print(f"Loading data from {data_file}...")
    df = pd.read_parquet(data_file)
    if 'timestamp' in df.columns: 
        df = df.set_index('timestamp')
    elif 'time' in df.columns: 
        df = df.set_index('time')
    
    # Ensure Naive Local timestamps
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df.sort_index()
    
    print("Resampling to 1-minute OHLC for indicator calculation...")
    # Generate 1m bars from high-frequency data
    # We use 'ask' for core OHLC as in the training
    ohlc = df['ask'].resample('1min').ohlc().ffill()
    
    # Resample bid separately for exact exit calculations
    bid_ohlc = df['bid'].resample('1min').ohlc().ffill()
    
    print("Calculating Technical Indicators (SMA120, ATR_SL)...")
    # SMA120
    ohlc['SMA120'] = ta.trend.sma_indicator(ohlc['close'], window=SMA_FILTER_LEN)
    
    # ATR 1H (aligned to 1M)
    ohlc_1h = ohlc['close'].resample('1h').ohlc().dropna()
    if len(ohlc_1h) > 14:
        atr_1h = ta.volatility.average_true_range(ohlc_1h['high'], ohlc_1h['low'], ohlc_1h['close'], window=ATR_LEN)
        atr_1h_long = pd.Series(atr_1h, index=ohlc_1h.index).reindex(ohlc.index, method='ffill')
    else:
        atr_1h_long = pd.Series(0, index=ohlc.index)
        
    ohlc['ATR_SL_Dist'] = atr_1h_long * 3.0
    ohlc['bid_close'] = bid_ohlc['close']
    ohlc['bid_low'] = bid_ohlc['low']
    ohlc['ask_high'] = ohlc['high']
    
    # Load Signals
    sig_df = pd.read_csv(SIGNAL_FILE)
    sig_df['timestamp'] = pd.to_datetime(sig_df['timestamp'])
    if sig_df['timestamp'].dt.tz is not None:
        sig_df['timestamp'] = sig_df['timestamp'].dt.tz_localize(None)
        
    print(f"Total signals in CSV: {len(sig_df)}")
    
    for hold_m in EXIT_SWEEP_MINUTES:
        print(f"\n>>> Analyzing {hold_m}m Holding Period <<<")
        results = []
        for _, row in tqdm(sig_df.iterrows(), total=len(sig_df), desc=f"Hold={hold_m}"):
            t = row['timestamp']
            side = row['side']
            
            if t not in ohlc.index: continue
            
            # Entry point data
            entry_data = ohlc.loc[t]
            if pd.isna(entry_data['SMA120']): continue
            
            # EA logic for Price/SMA comparison
            # We use 'close' (ask) for Price in Buy, 'bid_close' for Price in Sell
            price_for_filter = entry_data['close'] if side=='BUY' else entry_data['bid_close']
            sma_val = entry_data['SMA120']
            sl_dist = entry_data['ATR_SL_Dist']
            
            # Trend Filter
            if side == 'BUY':
                if price_for_filter <= sma_val: continue
            else:
                if price_for_filter >= sma_val: continue
                
            # Simulated Entry Price
            entry_price = price_for_filter 
            sl_price = entry_price - sl_dist if side=='BUY' else entry_price + sl_dist
            
            # Simulation Window
            exit_window = ohlc.loc[t : t + pd.Timedelta(minutes=hold_m)]
            if len(exit_window) < 2: continue
            
            hit_sl = False
            exit_p = 0
            
            # Efficient SL check
            for ft, frow in exit_window.iterrows():
                if side=='BUY':
                    # Hit SL if Bid Low <= SL Price
                    if frow['bid_low'] <= sl_price:
                        exit_p = sl_price
                        hit_sl = True
                        break
                else:
                    # Hit SL if Ask High >= SL Price
                    if frow['ask_high'] >= sl_price:
                        exit_p = sl_price
                        hit_sl = True
                        break
            
            if not hit_sl:
                # Exit at fixed time
                final_bar = exit_window.iloc[-1]
                exit_p = final_bar['bid_close'] if side=='BUY' else final_bar['close']
                
            pips = (exit_p - entry_price) * 100 if side=='BUY' else (entry_price - exit_p) * 100
            net_pips = pips - COST_PIPS
            
            results.append({'side': side, 'net_pips': net_pips, 'hit_sl': hit_sl})
            
        res_df = pd.DataFrame(results)
        if res_df.empty:
            print("No trades passed Trend Filter.")
            continue
            
        summary = res_df.groupby('side')['net_pips'].agg(['sum', 'mean', 'count'])
        print(summary)
        
    print("\nLocal Audit Finalized.")

if __name__ == "__main__":
    main()
