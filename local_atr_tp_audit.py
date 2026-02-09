import pandas as pd
import numpy as np
import os
import ta
from tqdm import tqdm

# --- SETTINGS ---
SIGNAL_FILE = "rf_signals_final.csv"
DATA_FILE = "full_data_cache.parquet" # Using the large local tick data file
COST_PIPS = 1.0
SMA_FILTER_LEN = 120
ATR_LEN = 14
HOLD_MINUTES = 240
TP_ATR_MULTIPLIER = 1.5 # We'll test 1.5x ATR for TP

def main():
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found locally.")
        return
    
    if not os.path.exists(SIGNAL_FILE):
        print(f"Error: {SIGNAL_FILE} not found.")
        return
    
    print(f"Loading data from {DATA_FILE}...")
    df = pd.read_parquet(DATA_FILE)
    if 'timestamp' in df.columns: df = df.set_index('timestamp')
    elif 'time' in df.columns: df = df.set_index('time')
    
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df.sort_index()
    
    print("Resampling to 1-minute OHLC...")
    ohlc = df['ask'].resample('1min').ohlc().ffill()
    bid_ohlc = df['bid'].resample('1min').ohlc().ffill()
    
    print("Calculating Technical Indicators...")
    ohlc['SMA120'] = ta.trend.sma_indicator(ohlc['close'], window=SMA_FILTER_LEN)
    
    ohlc_1h = ohlc['close'].resample('1h').ohlc().dropna()
    atr_1h = ta.volatility.average_true_range(ohlc_1h['high'], ohlc_1h['low'], ohlc_1h['close'], window=ATR_LEN)
    atr_1h_merged = pd.Series(atr_1h, index=ohlc_1h.index).reindex(ohlc.index, method='ffill')
    
    ohlc['ATR_1H'] = atr_1h_merged
    ohlc['bid_close'] = bid_ohlc['close']
    ohlc['bid_low'] = bid_ohlc['low']
    ohlc['bid_high'] = bid_ohlc['high']
    ohlc['ask_high'] = ohlc['high']
    ohlc['ask_low'] = ohlc['low']
    
    # Load Signals
    sig_df = pd.read_csv(SIGNAL_FILE)
    sig_df['timestamp'] = pd.to_datetime(sig_df['timestamp'])
    if sig_df['timestamp'].dt.tz is not None:
        sig_df['timestamp'] = sig_df['timestamp'].dt.tz_localize(None)

    strategies = ['Time_Exit_4h', 'ATR_TP_1.5x']
    
    for strat in strategies:
        print(f"\n>>> Analyzing Strategy: {strat} <<<")
        results = []
        for _, row in tqdm(sig_df.iterrows(), total=len(sig_df)):
            t = row['timestamp']
            side = row['side']
            
            if t not in ohlc.index: continue
            
            entry_data = ohlc.loc[t]
            if pd.isna(entry_data['SMA120']): continue
            
            price = entry_data['close'] if side=='BUY' else entry_data['bid_close']
            if side=='BUY' and price <= entry_data['SMA120']: continue
            if side=='SELL' and price >= entry_data['SMA120']: continue
            
            # SL = 3.0x ATR
            # TP = 1.5x ATR (if strat is ATR_TP)
            atr_val = entry_data['ATR_1H']
            sl_price = price - (atr_val * 3.0) if side=='BUY' else price + (atr_val * 3.0)
            tp_price = price + (atr_val * TP_ATR_MULTIPLIER) if side=='BUY' else price - (atr_val * TP_ATR_MULTIPLIER)
            
            exit_window = ohlc.loc[t : t + pd.Timedelta(minutes=HOLD_MINUTES)]
            if len(exit_window) < 2: continue
            
            exit_p = 0
            hit_type = 'Time'
            
            for ft, frow in exit_window.iterrows():
                if side=='BUY':
                    # Check SL
                    if frow['bid_low'] <= sl_price:
                        exit_p = sl_price
                        hit_type = 'SL'
                        break
                    # Check TP (only if strategy is ATR_TP)
                    if strat == 'ATR_TP_1.5x' and frow['ask_high'] >= tp_price:
                        exit_p = tp_price
                        hit_type = 'TP'
                        break
                else:
                    # Check SL
                    if frow['ask_high'] >= sl_price:
                        exit_p = sl_price
                        hit_type = 'SL'
                        break
                    # Check TP
                    if strat == 'ATR_TP_1.5x' and frow['bid_low'] <= tp_price:
                        exit_p = tp_price
                        hit_type = 'TP'
                        break
            
            if exit_p == 0:
                final_bar = exit_window.iloc[-1]
                exit_p = final_bar['bid_close'] if side=='BUY' else final_bar['close']
            
            pips = (exit_p - price) * 100 if side=='BUY' else (price - exit_p) * 100
            results.append({
                'timestamp': t,
                'side': side,
                'net_pips': pips - COST_PIPS,
                'hit_type': hit_type
            })
            
        res_df = pd.DataFrame(results)
        if res_df.empty: continue
        
        res_df['month'] = res_df['timestamp'].dt.to_period('M')
        
        print(f"\nSummary for {strat}:")
        print(res_df.groupby('side')['net_pips'].agg(['sum', 'mean', 'count']))
        
        print(f"\nMonthly Profit (Sum of Net Pips) for {strat}:")
        monthly = res_df.groupby(['month', 'side'])['net_pips'].sum().unstack().fillna(0)
        print(monthly)

if __name__ == "__main__":
    main()
