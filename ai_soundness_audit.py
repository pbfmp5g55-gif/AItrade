import argparse
import pandas as pd
import numpy as np
import os
import ta
import onnxruntime as ort
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

# --- CONFIGURATION (Defaults) ---
DATA_FILE = "full_data_cache.parquet"
MODEL_BUY = "model_rf_buy_old.onnx"
MODEL_SELL = "model_rf_sell_old.onnx"

# Validation Periods (User Requested Split)
TRAIN_START = "2025-01-01"
TRAIN_END = "2025-06-30"
VAL_END = "2025-11-30"
OOS_START = "2025-12-01"

class SoundnessAudit:
    def __init__(self, data_path, model_buy_path, model_sell_path, out_dir="audit_results"):
        self.data_path = data_path
        self.model_buy_path = model_buy_path
        self.model_sell_path = model_sell_path
        self.out_dir = out_dir
        self.ohlc = None
        self.bid_ohlc = None
        self.feat = None
        
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

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
        o = self.ohlc['open']
        
        self.ohlc['SMA120'] = ta.trend.sma_indicator(c, window=120)
        self.ohlc['RSI'] = ta.momentum.rsi(c, window=14)
        
        sma20 = ta.trend.sma_indicator(c, window=20)
        sma60 = ta.trend.sma_indicator(c, window=60)
        self.ohlc['ATR_1m'] = ta.volatility.average_true_range(h, l, c, window=14)
        self.ohlc['Body_Size'] = (c - o).abs()
        
        # ATR 1H for SL (as in local_ultimate_audit)
        ohlc_1h = c.resample('1h').ohlc().dropna()
        atr_1h = ta.volatility.average_true_range(ohlc_1h['high'], ohlc_1h['low'], ohlc_1h['close'], window=14)
        self.ohlc['ATR_1H'] = pd.Series(atr_1h, index=ohlc_1h.index).reindex(self.ohlc.index, method='ffill')
        
        # Features for ONNX
        self.feat = pd.DataFrame(index=self.ohlc.index)
        self.feat['RSI'] = self.ohlc['RSI']
        self.feat['dist_sma'] = (c - sma20) / sma20
        self.feat['mom_smas'] = (sma20 - sma60) / sma60
        self.feat['ATR'] = self.ohlc['ATR_1m']
        
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

    def backtest(self, entry_timestamps, side, exit_mode='baseline_hold', 
                 hold_m=240, cost_pips=1.0, sl_atr_mult=3.0, 
                 mae_cut_pips=35, decay_window=30, decay_ratio=0.5):
        results = []
        
        for t in entry_timestamps:
            # 3.1 Next-bar entry
            entry_idx = self.ohlc.index.get_loc(t) + 1
            if entry_idx >= len(self.ohlc): continue
            
            entry_time = self.ohlc.index[entry_idx]
            
            # Entry Price: BUY at Ask (ohlc['open']), SELL at Bid (bid_ohlc['open'])
            entry_price = self.ohlc.iloc[entry_idx]['open'] if side == 'BUY' else self.bid_ohlc.iloc[entry_idx]['open']
            
            # SL calculation (ATR-based)
            atr_h1_val = self.ohlc.iloc[entry_idx-1]['ATR_1H']
            sl_dist = atr_h1_val * sl_atr_mult
            sl_price = entry_price - sl_dist if side == 'BUY' else entry_price + sl_dist
            
            # Energy Decay Initial Stats (C1/C2)
            entry_atr_1m = self.ohlc.iloc[entry_idx]['ATR_1m']
            entry_body_avg = self.ohlc.iloc[entry_idx-5:entry_idx+1]['Body_Size'].mean() # 5-min average at entry

            # Simulation Window
            end_idx = min(entry_idx + hold_m, len(self.ohlc) - 1)
            window_ohlc = self.ohlc.iloc[entry_idx : end_idx + 1]
            window_bid = self.bid_ohlc.iloc[entry_idx : end_idx + 1]
            
            exit_price = None
            exit_time = None
            exit_reason = "HOLD"
            mae = 0
            mfe = 0
            
            for i in range(len(window_ohlc)):
                curr_ohlc = window_ohlc.iloc[i]
                curr_bid = window_bid.iloc[i]
                curr_time = window_ohlc.index[i]
                
                # Update MAE/MFE
                if side == 'BUY':
                    low_p = curr_bid['low']
                    high_p = curr_ohlc['high']
                    mae = max(mae, entry_price - low_p)
                    mfe = max(mfe, high_p - entry_price)
                    
                    if low_p <= sl_price:
                        exit_price = sl_price
                        exit_time = curr_time
                        exit_reason = "SL"
                        break
                else:
                    high_p = curr_ohlc['high']
                    low_p = curr_bid['low']
                    mae = max(mae, high_p - entry_price)
                    mfe = max(mfe, entry_price - low_p)
                    
                    if high_p >= sl_price:
                        exit_price = sl_price
                        exit_time = curr_time
                        exit_reason = "SL"
                        break

                # --- Exit Mode Logic ---
                
                # Exit B: MAE Monitoring Cut (at 30m)
                if exit_mode == 'mae_cut' and i == 30:
                    current_mae_pips = mae * 100
                    if current_mae_pips >= mae_cut_pips:
                        exit_price = curr_bid['close'] if side == 'BUY' else curr_ohlc['close']
                        exit_time = curr_time
                        exit_reason = "MAE_CUT"
                        break

                # Exit C: Energy Decay (starting at 20m)
                if exit_mode == 'energy_decay' and i >= 20:
                    # Lookback Window
                    start_look = max(0, i - decay_window)
                    window_segment = window_ohlc.iloc[start_look : i + 1]
                    
                    # C1: Body Decay
                    current_body_avg = window_segment['Body_Size'].mean()
                    if current_body_avg < entry_body_avg * decay_ratio:
                        exit_price = curr_bid['close'] if side == 'BUY' else curr_ohlc['close']
                        exit_time = curr_time
                        exit_reason = "DECAY_BODY"
                        break
                        
                    # C2: Volatility Decay
                    current_atr_avg = window_segment['ATR_1m'].mean()
                    if current_atr_avg < entry_atr_1m * decay_ratio:
                        exit_price = curr_bid['close'] if side == 'BUY' else curr_ohlc['close']
                        exit_time = curr_time
                        exit_reason = "DECAY_VOL"
                        break
            
            if exit_price is None:
                # Time-based exit
                final_ohlc = window_ohlc.iloc[-1]
                final_bid = window_bid.iloc[-1]
                exit_price = final_bid['close'] if side == 'BUY' else final_ohlc['close']
                exit_time = window_ohlc.index[-1]
            
            pips = (exit_price - entry_price) * 100 if side == 'BUY' else (entry_price - exit_price) * 100
            net_pips = pips - cost_pips
            
            # SMA Trend for Regime Analysis
            sma_trend = "UP" if self.ohlc.loc[t, 'SMA120_Slope'] > 0 else "DOWN"
            vol_regime = "HIGH" if self.ohlc.loc[t, 'ATR_1H'] > self.ohlc['ATR_1H'].median() else "LOW"
            
            results.append({
                'signal_time': t,
                'entry_time': entry_time,
                'exit_time': exit_time,
                'side': side,
                'entry_p': entry_price,
                'exit_p': exit_price,
                'net_pips': net_pips,
                'mae': mae * 100,
                'mfe': mfe * 100,
                'reason': exit_reason,
                'sma_trend': sma_trend,
                'vol_regime': vol_regime,
                'hour_utc': t.hour,
                'hold_min_actual': i
            })
            
        return pd.DataFrame(results)

    def extract_dd_decomposition(self, trades_df):
        if trades_df.empty:
            return
            
        # 1. Build Realized Equity Curve
        equity_df = trades_df.sort_values('exit_time').copy()
        equity_df['cum_pips'] = equity_df['net_pips'].cumsum()
        equity_df['running_max'] = equity_df['cum_pips'].cummax()
        equity_df['drawdown'] = equity_df['cum_pips'] - equity_df['running_max']
        equity_df.to_csv(f"{self.out_dir}/equity_oos_trades.csv", index=False)
        
        # 2. Extract DD Events
        dd_events = []
        event_id = 0
        in_dd = False
        peak_time = equity_df.iloc[0]['exit_time']
        peak_val = equity_df.iloc[0]['cum_pips']
        
        for idx, row in equity_df.iterrows():
            if row['cum_pips'] >= peak_val:
                if in_dd:
                    # DD Event Recovered
                    event['recovered'] = True
                    event['recovery_time'] = row['exit_time']
                    dd_events.append(event)
                    in_dd = False
                peak_time = row['exit_time']
                peak_val = row['cum_pips']
            else:
                if not in_dd:
                    # New DD Event Starts
                    in_dd = True
                    event_id += 1
                    event = {
                        'event_id': event_id,
                        'peak_time': peak_time,
                        'peak_equity': round(peak_val, 2),
                        'trough_time': row['exit_time'],
                        'trough_equity': round(row['cum_pips'], 2),
                        'max_dd_pips': round(row['drawdown'], 2),
                        'recovered': False,
                        'recovery_time': None
                    }
                else:
                    # Update Trough
                    if row['cum_pips'] < event['trough_equity']:
                        event['trough_time'] = row['exit_time']
                        event['trough_equity'] = round(row['cum_pips'], 2)
                        event['max_dd_pips'] = round(row['drawdown'], 2)
        
        if in_dd:
            dd_events.append(event)
            
        df_events = pd.DataFrame(dd_events)
        if not df_events.empty:
            df_events['duration_to_trough_min'] = (df_events['trough_time'] - df_events['peak_time']).dt.total_seconds() / 60
            df_events.to_csv(f"{self.out_dir}/dd_events_oos.csv", index=False)
            
            # 3. Culprit Ranking (Top 10)
            culprits_list = []
            for _, ev in df_events.iterrows():
                # Trades that CLOSED during the [peak, trough] period
                culprits = equity_df[(equity_df['exit_time'] >= ev['peak_time']) & (equity_df['exit_time'] <= ev['trough_time'])].copy()
                if culprits.empty: continue
                
                culprits['dd_contribution_pips'] = culprits['net_pips'].clip(upper=0)
                culprits = culprits.sort_values('dd_contribution_pips').head(10)
                culprits['event_id'] = ev['event_id']
                culprits_list.append(culprits)
            
            if culprits_list:
                df_culprits = pd.concat(culprits_list)
                df_culprits.to_csv(f"{self.out_dir}/dd_culprits_oos_top10.csv", index=False)
                
                # 4. Culprit Summary
                summary_list = []
                for eid, group in df_culprits.groupby('event_id'):
                    summary_list.append({
                        'event_id': eid,
                        'culprit_count': len(group),
                        'avg_mae': group['mae'].mean().round(2),
                        'avg_hold': group['hold_min_actual'].mean().round(2),
                        'reasons': group['reason'].value_counts().to_dict(),
                        'hours': group['hour_utc'].value_counts().to_dict()
                    })
                pd.DataFrame(summary_list).to_csv(f"{self.out_dir}/dd_culprits_summary_oos.csv", index=False)

    def calculate_risk_metrics(self, trades_df):
        if trades_df.empty:
            return {}
            
        metrics = {}
        for period in ['TRAIN', 'VAL', 'OOS']:
            p_df = trades_df[trades_df['period'] == period]
            if p_df.empty: continue
            
            pos = p_df[p_df['net_pips'] > 0]['net_pips'].sum()
            neg = abs(p_df[p_df['net_pips'] < 0]['net_pips'].sum())
            pf = pos / neg if neg != 0 else np.inf
            win_rate = (p_df['net_pips'] > 0).mean() * 100
            
            # Max DD (Realized)
            # Need to sort by exit_time for correct DD calculation
            e_df = p_df.sort_values('exit_time')
            cum_pips = e_df['net_pips'].cumsum()
            running_max = cum_pips.cummax()
            dd = cum_pips - running_max
            max_dd = dd.min()
            
            metrics[period] = {
                'pf': round(pf, 2),
                'winrate': round(win_rate, 2),
                'max_dd_pips': round(max_dd, 2),
                'mae_p95': round(p_df['mae'].quantile(0.95), 2),
                'mae_p99': round(p_df['mae'].quantile(0.99), 2)
            }
        return metrics

    def run_all_tests(self, exit_mode='baseline_hold', hold_m=240, cost_pips=1.0, mae_cut_pips=35, decay_ratio=0.5):
        # 1. Prediction for both sides
        buy_signals = self.run_inference('BUY')
        
        print(f"--- Running Audit: Mode={exit_mode}, Cost={cost_pips} ---")
        buy_trades = self.backtest(buy_signals, 'BUY', exit_mode=exit_mode, hold_m=hold_m, cost_pips=cost_pips, mae_cut_pips=mae_cut_pips, decay_ratio=decay_ratio)
        
        if buy_trades.empty:
            print("No trades generated.")
            return []

        # Period Split
        buy_trades['period'] = 'OOS'
        buy_trades.loc[buy_trades['signal_time'] <= pd.Timestamp(TRAIN_END), 'period'] = 'TRAIN'
        buy_trades.loc[(buy_trades['signal_time'] > pd.Timestamp(TRAIN_END)) & (buy_trades['signal_time'] <= pd.Timestamp(VAL_END)), 'period'] = 'VAL'
        
        # DD Decomposition for OOS
        oos_trades = buy_trades[buy_trades['period'] == 'OOS'].copy()
        self.extract_dd_decomposition(oos_trades)

        # Save Details
        buy_trades.to_csv(f"{self.out_dir}/trades.csv", index=False)
        
        # summary_period.csv (Standard)
        summary = buy_trades.groupby(['period', 'side'])['net_pips'].agg(['sum', 'mean', 'count']).round(2)
        summary.to_csv(f"{self.out_dir}/summary_period.csv")
        
        # summary_period_risk.csv (Extended)
        risk_metrics = self.calculate_risk_metrics(buy_trades)
        risk_list = []
        for period, m in risk_metrics.items():
            if (period, 'BUY') not in summary.index: continue
            row = summary.loc[(period, 'BUY')].to_dict()
            row.update(m)
            row['period'] = period
            row['side'] = 'BUY'
            risk_list.append(row)
        pd.DataFrame(risk_list).to_csv(f"{self.out_dir}/summary_period_risk.csv", index=False)
        
        # Equity Curve for OOS BUY
        if not oos_trades.empty:
            oos_trades = oos_trades.sort_values('exit_time')
            oos_trades['cum_pips'] = oos_trades['net_pips'].cumsum()
            plt.figure(figsize=(10,6))
            plt.plot(oos_trades['exit_time'], oos_trades['cum_pips'])
            plt.title(f"OOS Equity Curve (BUY - {exit_mode})")
            plt.grid(True)
            plt.savefig(f"{self.out_dir}/equity_oos.png")
            plt.close()

        return risk_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exit_mode', type=str, default='baseline_hold', choices=['baseline_hold', 'mae_cut', 'energy_decay'])
    parser.add_argument('--hold_minutes', type=int, default=240)
    parser.add_argument('--cost_pips', type=float, default=1.0)
    parser.add_argument('--mae_cut_pips', type=float, default=35.0)
    parser.add_argument('--decay_ratio', type=float, default=0.5)
    parser.add_argument('--decay_window', type=int, default=30)
    parser.add_argument('--compare_all', action='store_true', help="Run all 3 modes and generate comparison report")

    args = parser.parse_args()

    audit = SoundnessAudit(DATA_FILE, MODEL_BUY, MODEL_SELL)
    audit.load_and_preprocess()

    if args.compare_all:
        print("\n=== RUNNING ALL EXIT MODES FOR COMPARISON ===")
        
        modes = [
            ('baseline_hold', 'audit_results_exit_compare/exitA_baseline'),
            ('mae_cut', 'audit_results_exit_compare/exitB_mae_cut'),
            ('energy_decay', 'audit_results_exit_compare/exitC_energy_decay')
        ]
        
        summary_compare = []
        
        for mode_name, out_path in modes:
            audit.out_dir = out_path
            if not os.path.exists(out_path): os.makedirs(out_path)
            
            res_list = audit.run_all_tests(exit_mode=mode_name, hold_m=args.hold_minutes, cost_pips=args.cost_pips, mae_cut_pips=args.mae_cut_pips, decay_ratio=args.decay_ratio)
            
            if res_list:
                # Find OOS result
                oos_res = next((r for r in res_list if r['period'] == 'OOS'), None)
                if oos_res:
                    oos_res['exit_mode'] = mode_name
                    summary_compare.append(oos_res)

        # Generate Master Comparison Report
        if summary_compare:
            print("\nGenerating Master Comparison Report...")
            comp_df = pd.DataFrame(summary_compare)
            comp_df = comp_df[['exit_mode', 'count', 'sum', 'mean', 'max_dd_pips', 'mae_p95', 'mae_p99', 'pf']]
            comp_df.to_csv("audit_results_exit_compare/comparison_oos.csv", index=False)
            
            report = "# Exit Layer Comparison Report (OOS Period)\n\n"
            report += comp_df.to_markdown(index=False)
            report += "\n\n### Evaluation Results\n"
            
            # Recommendation logic: MaxDD is best when closest to 0
            best_dd_mode = comp_df.loc[comp_df['max_dd_pips'].idxmax()] 
            report += f"\n- **Lowest Drawdown**: {best_dd_mode['exit_mode']} ({best_dd_mode['max_dd_pips']} pips)\n"
            
            best_mae_mode = comp_df.loc[comp_df['mae_p99'].idxmin()]
            report += f"- **Best MAE Tail Control**: {best_mae_mode['exit_mode']} (P99: {best_mae_mode['mae_p99']} pips)\n"
            
            with open("audit_results_exit_compare/report.md", "w") as f:
                f.write(report)
            print("Comparison finished. See 'audit_results_exit_compare/report.md'")

    else:
        audit.run_all_tests(exit_mode=args.exit_mode, hold_m=args.hold_minutes, cost_pips=args.cost_pips, mae_cut_pips=args.mae_cut_pips, decay_ratio=args.decay_ratio)
