# AI Soundness Audit Report
Generated: 2026-02-09 09:37:56

## 1. Executive Summary
- **Overall Score**: 4 / 10
- **Main Verdict**: PROCEED WITH CAUTION
- **OOS Performance (BUY)**: 8.30 pips

## 2. Test Results

### Period-wise Performance
| Period | Side | Trade Count | Avg Net Pips | Total Pips |
|--------|------|-------------|--------------|------------|
| OOS | BUY | 3 | 2.77 | 8.30 |
| TRAIN | BUY | 2748 | 2.09 | 5742.15 |
| TRAIN | SELL | 2922 | -6.44 | -18830.18 |
| VAL | BUY | 2991 | 7.45 | 22278.54 |
| VAL | SELL | 2918 | -11.15 | -32545.66 |

### HOLD Sensitivity
- HOLD 170m (BUY): 16581.29 pips
- HOLD 170m (SELL): -39382.21 pips
- HOLD 240m (BUY): 28028.99 pips
- HOLD 240m (SELL): -51375.84 pips
- HOLD 310m (BUY): 33333.20 pips
- HOLD 310m (SELL): -59668.35 pips

### MAE Distribution (BUY)
- 95th Percentile MAE: 84.99 pips
- 99th Percentile MAE: 100.76 pips
