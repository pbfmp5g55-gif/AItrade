import argparse
import os
import glob
import pandas as pd
import numpy as np
import ta
import joblib
from sklearn.ensemble import RandomForestClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import warnings

warnings.filterwarnings('ignore')

# --- CONFIG ---
TARGET_PIPS = 5.0
PIP = 0.01
ATR_H1_LIMIT = 12.0
WINDOW_SIZE = 15000
PERCENTILE = 99.0
TRAIN_START = "2023-01-01"
TRAIN_CUTOFF = "2025-07-01" # 2.5 years of training
FINAL_DATA_END = "2026-01-31" 

def load_data(data_dir):
    print(f"Loading data from {data_dir}...")
    if os.path.isfile(data_dir):
        files = [data_dir]
    else:
        files = sorted(glob.glob(os.path.join(data_dir, "**/*.parquet"), recursive=True))
        if not files:
            files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
        
    if not files:
        print("No files found!")
        return None
        
    print(f"Found {len(files)} files. Merging...")
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    if not dfs: return None
    
    df = pd.concat(dfs)
    print(f"Total raw rows loaded: {len(df)}")
    
    # Robust Time Detection
    ts_col = None
    for col in ['timestamp', 'time', 'date', 'DateTime', 'datetime']:
        if col in df.columns:
            ts_col = col
            break
            
    if ts_col:
        print(f"Detected time column: {ts_col}")
        df[ts_col] = pd.to_datetime(df[ts_col])
        df = df.set_index(ts_col)
    else:
        try:
            df.index = pd.to_datetime(df.index)
            print("Successfully converted index to datetime.")
        except:
            print("Warning: Could not detect or convert time index.")
            
    df = df.sort_index()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
        
    if df.index.duplicated().any():
        df = df[~df.index.duplicated(keep='first')]
        
    print(f"Data range: {df.index.min()} to {df.index.max()}")
    return df

def calc_features(df):
    print("Calculating Features...")
    if df.empty:
        print("Error: Input dataframe is empty.")
        return None, None
        
    # 1. Resample to 1H OHLC
    if 'ask' in df.columns:
        ohlc_1h = df['ask'].resample('1h').ohlc().dropna()
    else:
        # Fallback for OHLC input
        ohlc_1h = df.resample('1h').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna()
    
    # ATR 1H Filter Calculation
    if len(ohlc_1h) > 14:
        atr_1h_vals = ta.volatility.average_true_range(high=ohlc_1h['high'], low=ohlc_1h['low'], close=ohlc_1h['close'], window=14)
    else:
        print(f"Warning: Not enough 1H data ({len(ohlc_1h)} rows).")
        atr_1h_vals = pd.Series(0, index=ohlc_1h.index)
        
    ohlc_1h['ATR_Filter'] = atr_1h_vals * 100 
    
    # 2. Resample to 1M OHLC
    if 'ask' in df.columns:
        ohlc = df['ask'].resample('1min').ohlc().dropna()
    else:
        ohlc = df.resample('1min').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna()
        
    # Merge 1H Filter back to 1M data
    ohlc = pd.merge_asof(ohlc, ohlc_1h[['ATR_Filter']], left_index=True, right_index=True, direction='backward')
    
    c = ohlc['close']
    h = ohlc['high']
    l = ohlc['low']
    
    # 3. Indicator Calculation (ta library)
    print("Computing indicators...")
    sma20 = ta.trend.sma_indicator(c, window=20)
    sma60 = ta.trend.sma_indicator(c, window=60)
    rsi   = ta.momentum.rsi(c, window=14)
    atr   = ta.volatility.average_true_range(h, l, c, window=14)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        dist_sma = (c - sma20) / sma20
        mom_smas = (sma20 - sma60) / sma60
        
    feat = pd.DataFrame(index=ohlc.index)
    feat['RSI'] = rsi
    feat['dist_sma'] = dist_sma
    feat['mom_smas'] = mom_smas
    feat['ATR'] = atr
    feat['ATR_Filter'] = ohlc['ATR_Filter']
    
    return feat.dropna(), ohlc.loc[feat.dropna().index]

def get_labels(df, side):
    c = df['close'].values
    h = df['high'].values
    l = df['low'].values
    tp = TARGET_PIPS * PIP
    labels = np.zeros(len(df))
    future_n = 10
    
    for i in range(len(df) - future_n - 1):
        if side=='BUY':
            if np.max(h[i+1:i+1+future_n]) >= c[i] + tp: labels[i]=1
        else:
            if np.min(l[i+1:i+1+future_n]) <= c[i] - tp: labels[i]=1
    return labels

def export_onnx(clf, path):
    print(f"Exporting ONNX to {path}...")
    initial_type = [('float_input', FloatTensorType([None, 4]))] 
    options = {id(clf): {'zipmap': False}}
    onx = convert_sklearn(clf, initial_types=initial_type, options=options)
    with open(path, "wb") as f:
        f.write(onx.SerializeToString())

def train_predict(df, feat, side, model_dir):
    print(f"\n--- Training {side} ---")
    
    # Split
    train_end = pd.Timestamp(TRAIN_CUTOFF)
    if df.index.tz is not None and train_end.tz is None:
        train_end = train_end.tz_localize('UTC')
    elif df.index.tz is None and train_end.tz is not None:
        train_end = train_end.replace(tzinfo=None)
         
    train_mask = df.index < train_end
    test_mask  = df.index >= train_end
    
    X = feat[['RSI', 'dist_sma', 'mom_smas', 'ATR']]
    y = get_labels(df, side)
    
    # Train
    X_train = X[train_mask].iloc[:-10]
    y_train = y[train_mask][:-10]
    
    print(f"Train samples: {len(X_train)}")
    
    clf = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    # Export Model
    export_onnx(clf, os.path.join(model_dir, f"model_rf_{side.lower()}.onnx"))
    
    # Predict (Full) for Rolling
    probs = clf.predict_proba(X)[:, 1]
    prob_s = pd.Series(probs, index=df.index)
    
    rolling_thr = prob_s.shift(1).rolling(WINDOW_SIZE).quantile(PERCENTILE/100.0)
    
    # Generate Signals
    signals = (prob_s >= rolling_thr) & (feat['ATR_Filter'] >= ATR_H1_LIMIT)
    
    oos_sigs = signals[test_mask]
    return oos_sigs[oos_sigs].index

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '.'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', 'data_cache'))
    
    args = parser.parse_args()
    
    print("Starting SageMaker RF Training...")
    raw = load_data(args.train)
    if raw is None:
        print("Failed to load data")
        exit(1)
        
    feat, aligned = calc_features(raw)
    
    buy_idx = train_predict(aligned, feat, 'BUY', args.model_dir)
    sell_idx = train_predict(aligned, feat, 'SELL', args.model_dir)
    
    print("Consolidating signals...")
    buy_trades = [{'timestamp': idx, 'side': 'BUY'} for idx in buy_idx]
    sell_trades = [{'timestamp': idx, 'side': 'SELL'} for idx in sell_idx]
    
    buy_df = pd.DataFrame(buy_trades).sort_values('timestamp')
    sell_df = pd.DataFrame(sell_trades).sort_values('timestamp')
    
    buy_path = os.path.join(args.model_dir, "rf_signals_buy.csv")
    sell_path = os.path.join(args.model_dir, "rf_signals_sell.csv")
    final_path = os.path.join(args.model_dir, "rf_signals_final.csv")
    
    buy_df.to_csv(buy_path, index=False)
    sell_df.to_csv(sell_path, index=False)
    
    combined_df = pd.concat([buy_df, sell_df]).sort_values('timestamp')
    combined_df.to_csv(final_path, index=False)
    
    print(f"Saved split signals: {buy_path}, {sell_path}")
    print(f"Saved combined signals: {final_path}")
