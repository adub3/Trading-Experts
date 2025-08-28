import numpy as np
import pandas as pd
from typing import List, Dict

def run_backtest(
    prices: pd.DataFrame,
    experts: List,
    meta,
    lookback: int = 60,
    cost_bps: float = 0.0,
    hold_on_tie: bool = False,
) -> Dict[str, pd.DataFrame]:
    if "Close" not in prices.columns:
        raise ValueError("prices must contain a 'Close' column")

    close = prices["Close"].astype(float).copy()
    ret = close.pct_change().fillna(0.0)

    rows, weight_rows = [], []
    pos_prev = 0

    # main loop
    for t in range(lookback, len(close) - 1):
        window = prices.iloc[:t+1]
        next_ret = (close.iloc[t+1] / close.iloc[t]) - 1.0
        outcome_sign = 1 if next_ret > 0 else (-1 if next_ret < 0 else 0)

        preds = {}
        for e in experts:
            s = float(e.predict(window))
            preds[e.name] = max(-1.0, min(1.0, s))

        # debug: print first few steps
        if t < lookback + 3:   # only show the first 3 steps after warmup
            print(f"[dbg] t={t} preds: {preds}")

        decision = meta.aggregate(preds)
        if decision == 0 and hold_on_tie:
            decision = pos_prev
        position = int(np.sign(decision))

        trade_cost = 0.0
        if position != pos_prev and cost_bps > 0:
            trade_cost = abs(position - pos_prev) * (cost_bps * 1e-4)

        pnl = position * next_ret - trade_cost

        meta.update(preds, outcome_sign=outcome_sign)

        rows.append({
            "date": close.index[t+1],
            "close": close.iloc[t+1],
            "ret": next_ret,
            "decision": decision,
            "position": position,
            "trade_cost": trade_cost,
            "pnl": pnl
        })
        weight_rows.append({"date": close.index[t+1], **meta.weights})
        pos_prev = position

    if not rows:
        raise ValueError(
            f"No backtest rows produced. len(prices)={len(prices)}, lookback={lookback}. "
            "Lower lookback or widen your date range."
        )

    trades = pd.DataFrame(rows).set_index("date")
    weights = pd.DataFrame(weight_rows).set_index("date")
    trades["equity"] = (1.0 + trades["pnl"]).cumprod()

    # strategy summary
    summary = _summarize(trades["pnl"])

    # buy-and-hold baseline (same period as trades)
    bh = _buy_and_hold(close.loc[trades.index.min(): trades.index.max()])
    summary["BH_CAGR"] = bh["CAGR"].iloc[0]
    summary["BH_Sharpe"] = bh["Sharpe"].iloc[0]
    summary["BH_MaxDrawdown"] = bh["MaxDrawdown"].iloc[0]

    return {"trades": trades, "weights": weights, "summary": summary}

def _summarize(pnl_series: pd.Series) -> pd.DataFrame:
    eq = (1.0 + pnl_series).cumprod()
    ann = 252
    cagr = eq.iloc[-1] ** (ann / len(eq)) - 1 if len(eq) > 0 else float("nan")
    vol = pnl_series.std(ddof=0) * (ann**0.5)
    sharpe = (pnl_series.mean() * ann) / vol if vol > 0 else float("nan")
    mdd = (eq / eq.cummax() - 1.0).min() if len(eq) > 0 else float("nan")
    hit = (pnl_series > 0).mean() if len(pnl_series) > 0 else float("nan")
    return pd.DataFrame({"CAGR":[cagr], "Sharpe":[sharpe], "MaxDrawdown":[mdd], "HitRate":[hit]})

def _buy_and_hold(close: pd.Series) -> pd.DataFrame:
    ret = close.pct_change().fillna(0.0)
    eq = (1.0 + ret).cumprod()
    ann = 252
    cagr = eq.iloc[-1] ** (ann / len(eq)) - 1 if len(eq) > 0 else float("nan")
    vol = ret.std(ddof=0) * (ann**0.5)
    sharpe = (ret.mean() * ann) / vol if vol > 0 else float("nan")
    mdd = (eq / eq.cummax() - 1.0).min() if len(eq) > 0 else float("nan")
    return pd.DataFrame({"CAGR":[cagr], "Sharpe":[sharpe], "MaxDrawdown":[mdd]})
