import os
import glob
import pandas as pd
import numpy as np
import talib
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

# --- CONFIG ---
DATA_FILE = "full_data_cache.parquet" 
DATA_DIR = "data_cache"
OUTPUT_FILE = "rf_signals_final.csv"
TRAIN_CUTOFF = "2025-07-01"
ATR_H1_LIMIT = 12.0
WINDOW_SIZE = 15000
PERCENTILE = 99.0
Target_Pips = 5.0
PIP = 0.01

def load_and_prep():
    print(f"Loading data...")
    if os.path.exists(DATA_FILE):
        print(f"Loading from {DATA_FILE}...")
        df = pd.read_parquet(DATA_FILE)
    else:
        print(f"{DATA_FILE} not found. Searching for shards in {DATA_DIR}...")
        files = sorted(glob.glob(os.path.join(DATA_DIR, "USDJPY_*.parquet")))
        if not files:
            print("No data found in data_cache either.")
            return None
        print(f"Merging {len(files)} shards...")
        dfs = []
        for f in tqdm(files):
            dfs.append(pd.read_parquet(f))
        df = pd.concat(dfs)

    if 'timestamp' in df.columns: df = df.set_index('timestamp')
    elif 'time' in df.columns: df = df.set_index('time')
    df = df.sort_index()
    
    # Check if duplicate indices exist and drop them
    if df.index.duplicated().any():
        print("Dropping duplicate indices...")
        df = df[~df.index.duplicated(keep='first')]
    
    # 1. Resample to 1H for Filter
    print("Computing 1H ATR...")
    if 'ask' in df.columns:
        ohlc_1h = df['ask'].resample('1h').ohlc().dropna()
    else:
        ohlc_1h = df.resample('1h').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna()
        
    atr_1h_vals = talib.ATR(ohlc_1h['high'], ohlc_1h['low'], ohlc_1h['close'], timeperiod=14)
    # Convert to pips? User says "ATR12 filter". Usually 12 pips. 
    # JPY: Price 150. ATR 0.12. 
    # If using pct_change? No, standard ATR.
    # We assume raw ATR needs to be converted to Pips for comparison vs 12.0
    # Or maybe 12.0 IS the pip value (e.g. 0.12)? 
    # Let's assume user means Pips (Integer-like).
    # 0.12 * 100 = 12.0. So we multiply by 100 for JPY.
    # Note: If EURUSD, ATR 0.0012 -> * 10000 = 12.0.
    # We assume JPY here based on filename.
    ohlc_1h['ATR_Filter'] = atr_1h_vals * 100 
    
    # 2. Resample to 1M for Features
    print("Resampling to 1M...")
    if 'ask' in df.columns:
        ohlc = df['ask'].resample('1min').ohlc().dropna()
    else:
        ohlc = df.resample('1min').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna()
        
    # Merge ATR Filter (Backward fill to apply 1H stat to 1M bars)
    ohlc = pd.merge_asof(ohlc, ohlc_1h[['ATR_Filter']], left_index=True, right_index=True, direction='backward')
    
    return ohlc

def calc_features(df):
    c = df['close'].values
    h = df['high'].values
    l = df['low'].values
    
    sma20 = talib.SMA(c, timeperiod=20)
    sma60 = talib.SMA(c, timeperiod=60)
    rsi   = talib.RSI(c, timeperiod=14)
    atr   = talib.ATR(h, l, c, timeperiod=14)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        dist_sma = (c - sma20) / sma20
        mom_smas = (sma20 - sma60) / sma60
        
    feat = pd.DataFrame(index=df.index)
    feat['RSI'] = rsi
    feat['dist_sma'] = dist_sma
    feat['mom_smas'] = mom_smas
    feat['ATR'] = atr
    
    # Keep ATR_Filter
    feat['ATR_Filter'] = df['ATR_Filter']
    
    return feat.dropna(), df.loc[feat.dropna().index]

def get_labels(df, side):
    c = df['close'].values
    h = df['high'].values
    l = df['low'].values
    tp = Target_Pips * PIP
    labels = np.zeros(len(df))
    future_n = 10
    
    for i in range(len(df) - future_n - 1):
        if side=='BUY':
            if np.max(h[i+1:i+1+future_n]) >= c[i] + tp: labels[i]=1
        else:
            if np.min(l[i+1:i+1+future_n]) <= c[i] - tp: labels[i]=1
    return labels

def train_predict(df, side):
    print(f"\n--- Processing {side} ---")
    
    # Split
    train_end = pd.Timestamp(TRAIN_CUTOFF)
    if df.index.tz is not None and train_end.tz is None:
        train_end = train_end.tz_localize('UTC')
    elif df.index.tz is None and train_end.tz is not None:
        train_end = train_end.replace(tzinfo=None)
         
    train_mask = df.index < train_end
    test_mask  = df.index >= train_end
    
    X = df[['RSI', 'dist_sma', 'mom_smas', 'ATR']]
    y = get_labels(df, side)
    
    # Trim for label lookahead
    X_train = X[train_mask].iloc[:-10]
    y_train = y[train_mask][:-10]
    
    print(f"Train Size: {len(X_train)}")
    
    clf = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    # Predict Full (To get rolling stats for OOS start)
    probs = clf.predict_proba(X)[:, 1]
    prob_series = pd.Series(probs, index=df.index)
    
    # Rolling Threshold
    # shift(1) to avoid leakage
    rolling_thr = prob_series.shift(1).rolling(WINDOW_SIZE).quantile(PERCENTILE/100.0)
    
    # Filter Logic
    # 1. Score >= Threshold
    # 2. ATR_Filter >= Limit
    signals = (prob_series >= rolling_thr) & (df['ATR_Filter'] >= ATR_H1_LIMIT)
    
    # Extract only OOS signals
    oos_signals = signals[test_mask]
    
    # Return index of signals
    return oos_signals[oos_signals].index

def main():
    raw = load_and_prep()
    if raw is None: return
    
    feat, aligned_df = calc_features(raw)
    
    buy_indices = train_predict(aligned_df, 'BUY')
    sell_indices = train_predict(aligned_df, 'SELL')
    
    # Combine results
    # We want a DataFrame with trades
    trades = []
    
    # Add Buys
    for idx in buy_indices:
        trades.append({'timestamp': idx, 'side': 'BUY'})
    
    # Add Sells
    for idx in sell_indices:
        trades.append({'timestamp': idx, 'side': 'SELL'})
        
    if not trades:
        print("No trades generated.")
        return
        
    res = pd.DataFrame(trades).sort_values('timestamp')
    res.to_csv(OUTPUT_FILE, index=False) # Important: index=False, timestamp is column
    
    print(f"\nSaved {len(res)} signals to {OUTPUT_FILE}")
    print(res['side'].value_counts())

if __name__ == "__main__":
    main()
