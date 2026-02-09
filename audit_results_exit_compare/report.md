# Exit Layer Comparison Report (OOS Period)

| exit_mode      |   count |    sum |   mean |   max_dd_pips |   mae_p95 |   mae_p99 |   pf |
|:---------------|--------:|-------:|-------:|--------------:|----------:|----------:|-----:|
| baseline_hold  |     405 | 1001   |   2.47 |       -5257.6 |     69.22 |     98.26 | 1.15 |
| energy_decay   |     405 | -501.4 |  -1.24 |       -2934.5 |     61.04 |     82.59 | 0.89 |
| unified_3phase |     405 |  151.6 |   0.37 |       -2929.4 |     51.88 |     80.5  | 1.03 |

### Evaluation Results

- **Lowest Drawdown**: unified_3phase (-2929.4 pips)
- **Best MAE Tail Control**: unified_3phase (P99: 80.5 pips)
