//+------------------------------------------------------------------+
//|                           Antigravity_Ultimate_System_v1.mq5     |
//|                                  Copyright 2026, Antigravity AI  |
//|                                             https://google.com   |
//+------------------------------------------------------------------+
#property copyright "Copyright 2026, Antigravity AI"
#property link      "https://google.com"
#property version   "1.01" // ULTRA FIX
#property strict

// --- Resources (BUY Model Only as verified) ---
#resource "model_rf_buy.onnx" as uchar model_buy_data[]

// --- Input Parameters ---
input string          InpSymbol            = "USDJPY";      // Symbol
input ENUM_TIMEFRAMES InpTimeframe         = PERIOD_M1;      // Base Timeframe
input int             InpHoldMinutes       = 240;            // Holding time (4h)
input double          InpRiskPercent       = 0.5;            // Risk per trade (%)
input int             InpSmaPeriod         = 240;            // Trend Filter SMA Period
input double          InpAtrMultiplier     = 3.0;            // ATR SL Multiplier
input double          InpMinSLPips         = 10.0;           // Min SL Pips (Floor)
input double          InpMaxSLPips         = 80.0;           // Max SL Pips (Ceiling)
input double          InpAtrEntryLimit     = 12.0;           // ATR Filter (H1-ATR*100)
input int             InpMagicNumber       = 888888;         // Magic Number
input int             InpMaxPositions      = 1;              // Max Open Positions
input double          InpRollingPct        = 99.0;           // Rolling Quantile (Top 1%)
input int             InpWindowSize        = 15000;          // Window Size (2 weeks)
input double          InpThresholdBuy      = 0.50;           // Absolute Buy Threshold
input bool            InpDebugMode         = true;           // Enable Debug Mode

#include <Trade\Trade.mqh>
CTrade trade;

// --- Global Handles ---
long     handle_onnx = INVALID_HANDLE;
int      h_rsi, h_atr_m1, h_atr_h1, h_sma20, h_sma60, h_sma240;
const ulong input_shape[] = {1, 4}; 
const ulong output_label_shape[] = {1};
const ulong output_probs_shape[] = {1, 2};

// --- Rolling Quantile Monitor ---
class CQuantileMonitor {
private:
    int      m_window_size;
    int      m_bins[1001];      
    float    m_history[];       
    int      m_head;            
    int      m_count;           
public:
    void Init(int window) {
        m_window_size = window;
        ArrayResize(m_history, window);
        ArrayInitialize(m_bins, 0);
        m_head = 0; m_count = 0;
    }
    void Add(float val) {
        if(val < 0.0f) val = 0.0f; if(val > 1.0f) val = 1.0f;
        if(m_count >= m_window_size) {
            int old_bin = (int)(m_history[m_head] * 1000.0f);
            if(old_bin > 1000) old_bin = 1000;
            if(m_bins[old_bin] > 0) m_bins[old_bin]--;
        } else m_count++;
        m_history[m_head] = val;
        int new_bin = (int)(val * 1000.0f);
        if(new_bin > 1000) new_bin = 1000;
        m_bins[new_bin]++;
        m_head++; if(m_head >= m_window_size) m_head = 0;
    }
    float GetQuantile(double percentile) {
        if(m_count == 0) return 0.5f;
        int needed = (int)(m_count * (1.0 - (percentile / 100.0)));
        if(needed < 1) needed = 1;
        int sum = 0;
        for(int i = 1000; i >= 0; i--) {
            sum += m_bins[i];
            if(sum >= needed) return (float)i / 1000.0f;
        }
        return 0.0f;
    }
};

CQuantileMonitor mon_buy;

//+------------------------------------------------------------------+
//| Initialization                                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    Print("=== Antigravity ULTIMATE System v1.01 ULTRA FIX ===");

    handle_onnx = OnnxCreateFromBuffer(model_buy_data, ONNX_DEFAULT);
    if(handle_onnx == INVALID_HANDLE) { 
        Print("ERR: OnnxCreateFromBuffer failed ", GetLastError()); 
        return(INIT_FAILED); 
    }
    
    if(!OnnxSetInputShape(handle_onnx, 0, input_shape)) { 
        Print("ERR: OnnxSetInputShape failed ", GetLastError()); 
        return(INIT_FAILED); 
    }
    if(!OnnxSetOutputShape(handle_onnx, 0, output_label_shape)) { 
        Print("ERR: OnnxSetOutputShape 0 failed ", GetLastError()); 
        return(INIT_FAILED); 
    }
    if(!OnnxSetOutputShape(handle_onnx, 1, output_probs_shape)) { 
        Print("ERR: OnnxSetOutputShape 1 failed ", GetLastError()); 
        return(INIT_FAILED); 
    }

    h_rsi    = iRSI(_Symbol, PERIOD_M1, 14, PRICE_CLOSE);
    h_atr_m1 = iATR(_Symbol, PERIOD_M1, 14);
    h_atr_h1 = iATR(_Symbol, PERIOD_H1, 14);
    h_sma20  = iMA(_Symbol, PERIOD_M1, 20, 0, MODE_SMA, PRICE_CLOSE);
    h_sma60  = iMA(_Symbol, PERIOD_M1, 60, 0, MODE_SMA, PRICE_CLOSE);
    h_sma240 = iMA(_Symbol, PERIOD_M1, InpSmaPeriod, 0, MODE_SMA, PRICE_CLOSE);

    mon_buy.Init(InpWindowSize);
    trade.SetExpertMagicNumber(InpMagicNumber);
    trade.SetTypeFilling(ORDER_FILLING_IOC);

    // --- INSTANT WARM-UP ---
    Print("... Pre-loading history for Rolling Quantile ...");
    int limit = MathMin(Bars(_Symbol, PERIOD_M1) - 10, InpWindowSize);
    int warm_up_count = 0;
    long  out_l[1];
    float out_p[1][2];

    for(int i = limit; i >= 1; i--)
    {
        double r[1], a_m1[1], sma20[1], sma60[1], c[1];
        if(CopyBuffer(h_rsi, 0, i, 1, r) < 1) continue;
        if(CopyBuffer(h_atr_m1, 0, i, 1, a_m1) < 1) continue;
        if(CopyBuffer(h_sma20, 0, i, 1, sma20) < 1) continue;
        if(CopyBuffer(h_sma60, 0, i, 1, sma60) < 1) continue;
        if(CopyClose(_Symbol, PERIOD_M1, i, 1, c) < 1) continue;

        float feat[4];
        feat[0] = (float)r[0];
        feat[1] = (float)((c[0] - sma20[0]) / sma20[0]);
        feat[2] = (float)((sma20[0] - sma60[0]) / sma60[0]);
        feat[3] = (float)a_m1[0];

        if(OnnxRun(handle_onnx, ONNX_NO_CONVERSION, feat, out_l, out_p)) {
            mon_buy.Add(out_p[0][1]);
            warm_up_count++;
        }
    }
    PrintFormat("Warm-up complete. Pre-loaded %d bars.", warm_up_count);
    return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason)
{
    if(handle_onnx != INVALID_HANDLE) OnnxRelease(handle_onnx);
    IndicatorRelease(h_rsi);
    IndicatorRelease(h_atr_m1); IndicatorRelease(h_atr_h1);
    IndicatorRelease(h_sma20); IndicatorRelease(h_sma60); IndicatorRelease(h_sma240);
}

void OnTick()
{
    CheckTimeExits();

    static datetime last_bar_time = 0;
    datetime current_bar_time = iTime(_Symbol, PERIOD_M1, 0);
    if(current_bar_time == last_bar_time) return;
    last_bar_time = current_bar_time;

    MqlDateTime dt; TimeGMT(dt);
    if(dt.hour > 15) return;

    double r[1], a_m1[1], a_h1[1], sma20[1], sma60[1], sma240[1], c[1];
    if(CopyBuffer(h_rsi, 0, 1, 1, r) < 1) return;
    if(CopyBuffer(h_atr_m1, 0, 1, 1, a_m1) < 1) return;
    if(CopyBuffer(h_atr_h1, 0, 1, 1, a_h1) < 1) return;
    if(CopyBuffer(h_sma20, 0, 1, 1, sma20) < 1) return;
    if(CopyBuffer(h_sma60, 0, 1, 1, sma60) < 1) return;
    if(CopyBuffer(h_sma240, 0, 1, 1, sma240) < 1) return;
    if(CopyClose(_Symbol, PERIOD_M1, 1, 1, c) < 1) return;

    float feat[4];
    feat[0] = (float)r[0];
    feat[1] = (float)((c[0] - sma20[0]) / sma20[0]);
    feat[2] = (float)((sma20[0] - sma60[0]) / sma60[0]);
    feat[3] = (float)a_m1[0];

    long  out_l[1]; float out_p[1][2];
    if(!OnnxRun(handle_onnx, ONNX_NO_CONVERSION, feat, out_l, out_p)) return;

    float score_buy = out_p[0][1];
    mon_buy.Add(score_buy);

    float thr_rolling = mon_buy.GetQuantile(InpRollingPct);
    bool ai_signal    = (score_buy >= thr_rolling) && (score_buy >= InpThresholdBuy);
    bool trend_filter = (c[0] > sma240[0]);
    bool vol_filter   = (a_h1[0] * 100.0 >= InpAtrEntryLimit);

    if(ai_signal && trend_filter && vol_filter && (PositionsTotal() < InpMaxPositions))
    {
        double sl_dist_pips = a_h1[0] * InpAtrMultiplier;
        if(_Digits == 3 || _Digits == 5) sl_dist_pips /= 0.01;
        
        if(sl_dist_pips < InpMinSLPips) sl_dist_pips = InpMinSLPips;
        if(sl_dist_pips > InpMaxSLPips) sl_dist_pips = InpMaxSLPips;

        double risk_amt = AccountInfoDouble(ACCOUNT_BALANCE) * (InpRiskPercent / 100.0);
        double tick_val = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
        double tick_sz  = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
        double pips_sz  = (_Digits == 3 || _Digits == 5) ? 0.01 : 0.0001;
        
        double loss_per_lot = (sl_dist_pips * pips_sz / tick_sz) * tick_val;
        double lots = MathFloor(risk_amt / loss_per_lot * 100.0) / 100.0;
        lots = MathMax(lots, SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN));
        lots = MathMin(lots, SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX));

        double sl_price = SymbolInfoDouble(_Symbol, SYMBOL_ASK) - (sl_dist_pips * pips_sz);
        trade.Buy(lots, _Symbol, 0, sl_price, 0, "Ultimate BUY");
    }
}

void CheckTimeExits()
{
    int total = PositionsTotal();
    for(int i=total-1; i>=0; i--) {
        if(PositionGetSymbol(i) != _Symbol) continue;
        ulong ticket = PositionGetTicket(i);
        if(PositionSelectByTicket(ticket)) {
            if(PositionGetInteger(POSITION_MAGIC) == InpMagicNumber) {
                if(TimeCurrent() - PositionGetInteger(POSITION_TIME) >= InpHoldMinutes * 60) {
                    trade.PositionClose(ticket);
                }
            }
        }
    }
}