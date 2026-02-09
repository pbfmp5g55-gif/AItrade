import pandas as pd
import numpy as np
import os

# --- SETTINGS ---
SIGNAL_FILE = "rf_signals_final.csv"
DATA_FILE = "full_data_consolidated_2023_2026.parquet"
COST_PIPS = 1.0
HOLD_MINUTES = 240 # 4 hours based on previous optimization

def main():
    if not os.path.exists(SIGNAL_FILE):
        print(f"Error: {SIGNAL_FILE} not found. Run training first.")
        return
    
    print("Loading data for validation...")
    df = pd.read_parquet(DATA_FILE)
    if 'timestamp' in df.columns: df = df.set_index('timestamp')
    elif 'time' in df.columns: df = df.set_index('time')
    df.index = pd.to_datetime(df.index).tz_localize(None)
    
    # Load Signals
    sig_df = pd.read_csv(SIGNAL_FILE)
    sig_df['timestamp'] = pd.to_datetime(sig_df['timestamp']).dt.tz_localize(None)
    
    print(f"Analyzing {len(sig_df)} signals with {HOLD_MINUTES}m exit and {COST_PIPS} pip cost...")
    
    results = []
    for _, row in sig_df.iterrows():
        entry_time = row['timestamp']
        side = row['side']
        
        # Simple exit at fixed time
        exit_time = entry_time + pd.Timedelta(minutes=HOLD_MINUTES)
        
        try:
            # Find closest available price
            entry_price = df.loc[entry_time:].iloc[0]['ask' if side=='BUY' else 'bid']
            exit_price_data = df.loc[exit_time:]
            if exit_price_data.empty: continue
            exit_price = exit_price_data.iloc[0]['bid' if side=='BUY' else 'ask']
            
            pnl = (exit_price - entry_price) * 100 if side=='BUY' else (entry_price - exit_price) * 100
            net_pnl = pnl - COST_PIPS
            
            results.append({
                'timestamp': entry_time,
                'side': side,
                'pnl': pnl,
                'net_pnl': net_pnl
            })
        except Exception:
            continue
            
    res_df = pd.DataFrame(results)
    if res_df.empty:
        print("No valid trades found for backtest.")
        return
        
    res_df['month'] = res_df['timestamp'].dt.to_period('M')
    
    print("\n--- PERFORMANCE SUMMARY (By Side) ---")
    summary = res_df.groupby(['side']).agg({
        'net_pnl': ['sum', 'mean', 'count'],
        'pnl': 'mean'
    })
    print(summary)
    
    # Save Split Detailed Results
    res_df[res_df['side']=='BUY'].to_csv("pnl_audit_buy.csv", index=False)
    res_df[res_df['side']=='SELL'].to_csv("pnl_audit_sell.csv", index=False)
    res_df.to_csv("pnl_audit_combined.csv", index=False)
    
    print("\n--- MONTHLY NET PIPS (BUY + SELL) ---")
    monthly = res_df.groupby(['month', 'side'])['net_pnl'].sum().unstack().fillna(0)
    if 'BUY' not in monthly.columns: monthly['BUY'] = 0
    if 'SELL' not in monthly.columns: monthly['SELL'] = 0
    monthly['TOTAL'] = monthly['BUY'] + monthly['SELL']
    print(monthly)
    
    print(f"\nDetailed logs saved to pnl_audit_buy.csv, pnl_audit_sell.csv, pnl_audit_combined.csv")

if __name__ == "__main__":
    main()
