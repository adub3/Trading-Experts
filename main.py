# main.py
import os
import pandas as pd

from simulation.temploader import load_smp_csv
from simulation.backtester import run_backtest
from experts import AlwaysLong, AlwaysShort, MovingAverageCrossover, RSIExpert, MomentumExpert
from meta.hedge import HedgeMeta

# ---- config you can tweak ----
CSV_PATH = os.path.join("simulation", "archive", "sp500_stocks.csv")
SYMBOL   = "AXP"       # try "AXP" if JPM still gives NaNs in your file
START    = "2018-01-01"
LOOKBACK = 60          # will auto-adjust if history is short
COST_BPS = 1.0
# ------------------------------

def _print_prices_info(df: pd.DataFrame, label: str):
    print(f"[{label}] rows={len(df)} "
          f"| range={df.index.min()} → {df.index.max()} "
          f"| cols={list(df.columns)}")

def _ensure_enough_history(df: pd.DataFrame, lookback: int) -> int:
    min_needed = lookback + 2
    if len(df) < min_needed:
        new_lb = max(5, len(df) - 2)
        print(f"[warn] not enough rows for lookback={lookback}. "
              f"len={len(df)}, need>={min_needed}. Using lookback={new_lb}.")
        return new_lb
    return lookback

def main():
    # 1) load once WITHOUT date filter to confirm symbol parses correctly
    raw_prices = load_smp_csv(CSV_PATH, symbol=SYMBOL)
    _print_prices_info(raw_prices, "loaded(no-start)")
    if raw_prices.empty:
        raise SystemExit("[fatal] loader returned empty frame for symbol. Check CSV and symbol spelling.")

    # 2) now apply start filter (optional)
    prices = load_smp_csv(CSV_PATH, symbol=SYMBOL, start=START)
    _print_prices_info(prices, f"loaded(start>={START})")
    if prices.empty:
        print("[warn] No rows after start filter; falling back to full history.")
        prices = raw_prices

    # 3) sanity check required column
    if "Close" not in prices.columns:
        raise SystemExit("[fatal] 'Close' column missing after load.")

    # 4) auto-adjust lookback if needed
    lookback = _ensure_enough_history(prices, LOOKBACK)

    # 5) set up experts + meta
    experts = [
        AlwaysLong(),
        AlwaysShort(),
        MovingAverageCrossover(10, 50),
        RSIExpert(14, 30, 70),
        MomentumExpert(lookback=20),
    ]
    meta = HedgeMeta(
        experts,
        eta=0.05,
        alpha_fixed_share=0.05,
        decay_lambda=0.99,
        weight_floor=1e-6,
        allow_specialists=True,
        random_tie_break=True,
        loss="logistic",
        loss_scale=2.0,
    )

    # 6) run backtest
    print(f"[bt] running backtest: symbol={SYMBOL}, lookback={lookback}, cost_bps={COST_BPS}")
    results = run_backtest(
        prices,
        experts,
        meta,
        lookback=lookback,
        cost_bps=COST_BPS,
        hold_on_tie=True,
    )

    # 7) show results
    print("\n=== Summary ===")
    print(results["summary"])
    print("\n=== Trades (head) ===")
    print(results["trades"].head())
    print("\n=== Weights (head) ===")
    print(results["weights"].head())

    out_path = os.path.join("simulation", "results_summary.txt")
    with open(out_path, "w") as f:
        f.write("=== Strategy Performance Summary ===\n")
        f.write(results["summary"].to_string(index=False))
        f.write("\n\n=== Config ===\n")
        f.write(f"Symbol: {SYMBOL}\n")
        f.write(f"Start: {START}\n")
        f.write(f"Lookback: {lookback}\n")
        f.write(f"Cost (bps): {COST_BPS}\n")
        f.write(f"Experts: {[e.name for e in experts]}\n")
    print(f"[saved] summary written to {out_path}")


if __name__ == "__main__":
    # be sure to run from the project root (folder containing main.py)
    main()
