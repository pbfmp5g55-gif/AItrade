import pandas as pd
import numpy as np
import os
import ta
import onnxruntime as ort
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

# --- CONFIGURATION ---
DATA_FILE = "full_data_cache.parquet"
MODEL_BUY = "model_rf_buy_old.onnx"
MODEL_SELL = "model_rf_sell_old.onnx"
OUT_DIR = "audit_results_old_model"

# Validation Periods (User Requested Split)
TRAIN_START = "2025-01-01"
TRAIN_END = "2025-06-30"
VAL_END = "2025-11-30"
OOS_START = "2025-12-01"

# Parameters for Sensitivity
DEFAULT_HOLD = 240
HOLD_SENSITIVITY = [170, 240, 310]
COST_PIPS_SENSITIVITY = [1.0, 2.0]

class SoundnessAudit:
    def __init__(self, data_path, model_buy_path, model_sell_path):
        self.data_path = data_path
        self.model_buy_path = model_buy_path
        self.model_sell_path = model_sell_path
        self.ohlc = None
        self.feat = None
        self.signals = None
        
        if not os.path.exists(OUT_DIR):
            os.makedirs(OUT_DIR)

    def load_and_preprocess(self):
        print(f"Loading data from {self.data_path}...")
        df = pd.read_parquet(self.data_path)
        
        ts_list = ['timestamp', 'time', 'datetime', 'date']
        ts_col = next((c for c in ts_list if c in df.columns), None)
        if ts_col is None:
            raise KeyError(f"No timestamp column found in {df.columns}")
        
        df = df.set_index(ts_col)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df = df.sort_index()
        
        print("Resampling to 1-minute OHLCV (Ask based)...")
        # Training used 'ask' for OHLC
        self.ohlc = df['ask'].resample('1min').ohlc().ffill()
        self.ohlc['volume'] = df['ask'].resample('1min').count() # Proxy for volume
        
        # Exact Bid for Exit
        self.bid_ohlc = df['bid'].resample('1min').ohlc().ffill()
        
        print("Calculating Indicators...")
        c = self.ohlc['close']
        h = self.ohlc['high']
        l = self.ohlc['low']
        
        self.ohlc['SMA120'] = ta.trend.sma_indicator(c, window=120)
        self.ohlc['RSI'] = ta.momentum.rsi(c, window=14)
        
        sma20 = ta.trend.sma_indicator(c, window=20)
        sma60 = ta.trend.sma_indicator(c, window=60)
        atr_1m = ta.volatility.average_true_range(h, l, c, window=14)
        
        # ATR 1H for SL (as in local_ultimate_audit)
        ohlc_1h = c.resample('1h').ohlc().dropna()
        atr_1h = ta.volatility.average_true_range(ohlc_1h['high'], ohlc_1h['low'], ohlc_1h['close'], window=14)
        self.ohlc['ATR_1H'] = pd.Series(atr_1h, index=ohlc_1h.index).reindex(self.ohlc.index, method='ffill')
        
        # Features for ONNX
        self.feat = pd.DataFrame(index=self.ohlc.index)
        self.feat['RSI'] = self.ohlc['RSI']
        self.feat['dist_sma'] = (c - sma20) / sma20
        self.feat['mom_smas'] = (sma20 - sma60) / sma60
        self.feat['ATR'] = atr_1m
        
        # Regime Indicators
        self.ohlc['SMA120_Slope'] = self.ohlc['SMA120'].diff(5) # 5-min slope
        self.ohlc['Hour_UTC'] = self.ohlc.index.hour
        
        self.feat = self.feat.dropna()
        self.ohlc = self.ohlc.loc[self.feat.index]
        self.bid_ohlc = self.bid_ohlc.loc[self.feat.index]

    def run_inference(self, side='BUY', window=15000, percentile=99.0):
        model_path = self.model_buy_path if side == 'BUY' else self.model_sell_path
        print(f"Running ONNX inference for {side}...")
        
        session = ort.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name
        
        # Data must be float32
        X = self.feat[['RSI', 'dist_sma', 'mom_smas', 'ATR']].values.astype(np.float32)
        
        # ONNX typically returns [labels, probabilities]
        # probabilities shape is [N, 2]
        outputs = session.run(None, {input_name: X})
        probs = outputs[1][:, 1]
        prob_s = pd.Series(probs, index=self.feat.index)
        
        # Rolling threshold
        rolling_thr = prob_s.shift(1).rolling(window).quantile(percentile/100.0)
        
        # Base Filter: ATR1H >= 12.0 (from training)
        atr_filter = self.ohlc['ATR_1H'] * 100 >= 12.0
        
        signals = (prob_s >= rolling_thr) & atr_filter
        return signals[signals].index

    def backtest(self, entry_timestamps, side, hold_m=240, cost_pips=1.0, sl_atr_mult=3.0, early_exit=False):
        results = []
        
        for t in entry_timestamps:
            # 3.1 Next-bar entry
            entry_idx = self.ohlc.index.get_loc(t) + 1
            if entry_idx >= len(self.ohlc): continue
            
            entry_time = self.ohlc.index[entry_idx]
            
            # Entry Price: BUY at Ask (ohlc['open']), SELL at Bid (bid_ohlc['open'])
            entry_price = self.ohlc.iloc[entry_idx]['open'] if side == 'BUY' else self.bid_ohlc.iloc[entry_idx]['open']
            
            # SL calculation (ATR-based)
            atr_val = self.ohlc.iloc[entry_idx-1]['ATR_1H']
            sl_dist = atr_val * sl_atr_mult
            sl_price = entry_price - sl_dist if side == 'BUY' else entry_price + sl_dist
            
            # Simulation Window
            end_idx = min(entry_idx + hold_m, len(self.ohlc) - 1)
            window_ohlc = self.ohlc.iloc[entry_idx : end_idx + 1]
            window_bid = self.bid_ohlc.iloc[entry_idx : end_idx + 1]
            
            exit_price = None
            exit_reason = "HOLD"
            mae = 0
            
            for i in range(len(window_ohlc)):
                curr_ohlc = window_ohlc.iloc[i]
                curr_bid = window_bid.iloc[i]
                
                # Check SL (Conservative: SL hit prioritized within bar)
                if side == 'BUY':
                    low_p = curr_bid['low']
                    if low_p <= sl_price:
                        exit_price = sl_price
                        exit_reason = "SL"
                        break
                    mae = max(mae, entry_price - low_p)
                else:
                    high_p = curr_ohlc['high']
                    if high_p >= sl_price:
                        exit_price = sl_price
                        exit_reason = "SL"
                        break
                    mae = max(mae, high_p - entry_price)
                
                # Test G: Early Exit (30 min check)
                if early_exit and i == 30:
                    current_profit_pips = (curr_bid['close'] - entry_price) * 100 if side == 'BUY' else (entry_price - curr_ohlc['close']) * 100
                    if current_profit_pips < 2.0:
                        exit_price = curr_bid['close'] if side == 'BUY' else curr_ohlc['close']
                        exit_reason = "EARLY_EXIT"
                        break
            
            if exit_price is None:
                # Time-based exit
                final_ohlc = window_ohlc.iloc[-1]
                final_bid = window_bid.iloc[-1]
                exit_price = final_bid['close'] if side == 'BUY' else final_ohlc['close']
            
            pips = (exit_price - entry_price) * 100 if side == 'BUY' else (entry_price - exit_price) * 100
            net_pips = pips - cost_pips
            
            # SMA Trend for Regime Analysis
            sma_trend = "UP" if self.ohlc.loc[t, 'SMA120_Slope'] > 0 else "DOWN"
            vol_regime = "HIGH" if self.ohlc.loc[t, 'ATR_1H'] > self.ohlc['ATR_1H'].median() else "LOW"
            
            results.append({
                'timestamp': t,
                'side': side,
                'entry_p': entry_price,
                'exit_p': exit_price,
                'net_pips': net_pips,
                'mae': mae * 100,
                'reason': exit_reason,
                'sma_trend': sma_trend,
                'vol_regime': vol_regime,
                'hour_utc': t.hour
            })
            
        return pd.DataFrame(results)

    def run_all_tests(self):
        # 1. Prediction for both sides
        buy_signals = self.run_inference('BUY')
        sell_signals = self.run_inference('SELL')
        
        # Test A/D: Period-wise + Side-wise
        print("Test A & D: Period/Side Analysis...")
        all_trades_list = []
        for side, sigs in [('BUY', buy_signals), ('SELL', sell_signals)]:
            trades = self.backtest(sigs, side)
            all_trades_list.append(trades)
        
        all_trades = pd.concat(all_trades_list)
        all_trades.to_csv(f"{OUT_DIR}/trades_base.csv", index=False)
        
        # Period Split
        all_trades['period'] = 'OOS'
        all_trades.loc[all_trades['timestamp'] <= pd.Timestamp(TRAIN_END), 'period'] = 'TRAIN'
        all_trades.loc[(all_trades['timestamp'] > pd.Timestamp(TRAIN_END)) & (all_trades['timestamp'] <= pd.Timestamp(VAL_END)), 'period'] = 'VAL'
        
        period_summary = all_trades.groupby(['period', 'side'])['net_pips'].agg(['sum', 'mean', 'count']).round(2)
        print("\n--- Period Summary (BUY/SELL) ---")
        print(period_summary)
        period_summary.to_csv(f"{OUT_DIR}/summary_period.csv")
        
        # Test B: HOLD Sensitivity
        print("Test B: HOLD Sensitivity...")
        sensitivity_results = []
        for hold in HOLD_SENSITIVITY:
            for side, sigs in [('BUY', buy_signals), ('SELL', sell_signals)]:
                res = self.backtest(sigs, side, hold_m=hold)
                if not res.empty:
                    sensitivity_results.append({'hold': hold, 'side': side, 'sum_pips': res['net_pips'].sum(), 'mean_pips': res['net_pips'].mean()})
        pd.DataFrame(sensitivity_results).to_csv(f"{OUT_DIR}/summary_hold_sensitivity.csv", index=False)

        # Test C: Cost Sensitivity
        print("Test C: Cost Sensitivity...")
        cost_results = []
        for cost in COST_PIPS_SENSITIVITY:
            for side, sigs in [('BUY', buy_signals), ('SELL', sell_signals)]:
                res = self.backtest(sigs, side, cost_pips=cost)
                if not res.empty:
                    cost_results.append({'cost': cost, 'side': side, 'sum_pips': res['net_pips'].sum()})
        pd.DataFrame(cost_results).to_csv(f"{OUT_DIR}/summary_cost_sensitivity.csv", index=False)

        # Test E: Regime
        print("Test E: Regime Analysis...")
        regime_summary = all_trades.groupby(['sma_trend', 'vol_regime'])['net_pips'].mean().unstack().round(2)
        regime_summary.to_csv(f"{OUT_DIR}/summary_regime.csv")

        # Test F: MAE
        print("Test F: MAE Distribution...")
        mae_stats = all_trades.groupby('side')['mae'].describe(percentiles=[0.5, 0.9, 0.95, 0.99])
        mae_stats.to_csv(f"{OUT_DIR}/summary_mae.csv")

        # Test G: Early Exit
        print("Test G: Early Exit Test...")
        early_res = []
        for side, sigs in [('BUY', buy_signals)]:
            base = self.backtest(sigs, side, early_exit=False)
            early = self.backtest(sigs, side, early_exit=True)
            early_res.append({'side': side, 'base_pips': base['net_pips'].sum(), 'early_pips': early['net_pips'].sum()})
        pd.DataFrame(early_res).to_csv(f"{OUT_DIR}/summary_early_exit.csv", index=False)

        self.generate_report(all_trades, sensitivity_results, cost_results, mae_stats)

    def generate_report(self, all_trades, sens, costs, mae):
        print("Generating Report...")
        
        # Scoring Logic
        # (Simplified for demonstration, can be refined)
        score = 0
        oos_pips = all_trades[(all_trades['period'] == 'OOS') & (all_trades['side'] == 'BUY')]['net_pips'].sum()
        if oos_pips > 0: score += 2
        
        # Check Hold stability
        buy_sens = [s['sum_pips'] for s in sens if s['side'] == 'BUY']
        if all(p > 0 for p in buy_sens): score += 2
        
        report = f"""# AI Soundness Audit Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. Executive Summary
- **Overall Score**: {score} / 10
- **Main Verdict**: {"PROCEED WITH CAUTION" if score < 7 else "ROBUST"}
- **OOS Performance (BUY)**: {oos_pips:.2f} pips

## 2. Test Results

### Period-wise Performance
| Period | Side | Trade Count | Avg Net Pips | Total Pips |
|--------|------|-------------|--------------|------------|
"""
        sum_period = all_trades.groupby(['period', 'side'])['net_pips'].agg(['count', 'mean', 'sum']).reset_index()
        for _, row in sum_period.iterrows():
            report += f"| {row['period']} | {row['side']} | {row['count']} | {row['mean']:.2f} | {row['sum']:.2f} |\n"

        report += "\n### HOLD Sensitivity\n"
        for s in sens:
            report += f"- HOLD {s['hold']}m ({s['side']}): {s['sum_pips']:.2f} pips\n"

        report += "\n### MAE Distribution (BUY)\n"
        buy_mae = mae.loc['BUY']
        report += f"- 95th Percentile MAE: {buy_mae['95%']:.2f} pips\n"
        report += f"- 99th Percentile MAE: {buy_mae['99%']:.2f} pips\n"

        with open(f"{OUT_DIR}/report.md", "w") as f:
            f.write(report)
        
        # Equity Curve for OOS BUY
        oos_buy = all_trades[(all_trades['period'] == 'OOS') & (all_trades['side'] == 'BUY')].copy()
        if not oos_buy.empty:
            oos_buy['cum_pips'] = oos_buy['net_pips'].cumsum()
            plt.figure(figsize=(10,6))
            plt.plot(oos_buy['timestamp'], oos_buy['cum_pips'])
            plt.title("OOS Equity Curve (BUY)")
            plt.grid(True)
            plt.savefig(f"{OUT_DIR}/equity_oos_buy.png")

if __name__ == "__main__":
    audit = SoundnessAudit(DATA_FILE, MODEL_BUY, MODEL_SELL)
    audit.load_and_preprocess()
    audit.run_all_tests()
