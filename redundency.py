import os
import pandas as pd

from simulation.temploader import load_smp_csv
from analysis import TradingAnalyzer  # adjust if your class lives elsewhere
from experts import *
from diagnostic import diagnose_expert_redundancy_selfcontained  # or `from diagnostics import ...`

def _predict_fn(experts, prices, t):
    """
    Adapter: returns {expert_name: signal} for time t.
    Assumes each expert has .name and .predict(prices, t) -> float in [-1,1] (or R).
    """
    out = {}
    for e in experts:
        try:
            out[e.name] = float(e.predict(prices, t))
        except Exception:
            out[e.name] = 0.0
    return out

from experts import *
import experts

def get_expert_pool():
    pool = []
    for name in experts.__all__:
        cls = getattr(experts, name)
        try:
            pool.append(cls())   # assume no-arg constructor
        except TypeError:
            # special cases that need args
            if name == "MovingAverageCrossover":
                pool.append(cls(10, 50))
            elif name == "MomentumExpert":
                pool.append(cls(lookback=20))
            # add other one-liners here if needed
    return pool


def run_redundancy_diagnostic(prices, lookback, save_dir):
    """
    Builds signals from all experts and runs the redundancy diagnostic.
    Saves summary CSV and shows a correlation heatmap.
    """
    experts = get_expert_pool()

    diag = diagnose_expert_redundancy_selfcontained(
        experts=experts,
        prices=prices,                 # pd.Series or DataFrame indexed by date
        predict_fn=_predict_fn,
        build_start=max(lookback, 5),  # skip warmup so experts have data
        cluster_threshold=0.80,        # tune: 0.70–0.85
        top_k_per_cluster=1,           # keep 1 representative per cluster
        plot=True
    )

    # Console summary
    print("\n=== Expert Redundancy Diagnostic ===")
    print("Kept   :", diag["kept"])
    print("Dropped:", diag["dropped"])
    print("\nSummary (avg_abs_corr lower = more unique):")
    print(diag["summary"])

    # Persist artifacts
    os.makedirs(save_dir, exist_ok=True)
    diag["summary"].to_csv(os.path.join(save_dir, "redundancy_summary.csv"))
    diag["corr"].to_csv(os.path.join(save_dir, "redundancy_corr_matrix.csv"))

    return diag

def main():
    # Config
    CSV_PATH = os.path.join("simulation", "archive", "sp500_stocks.csv")
    SYMBOL = "AXP"  # "USB"
    START = "2018-01-01"
    LOOKBACK = 60
    COST_BPS = 1.0
    ETA_VALUES = [0.40]  # tweak as you like

    print(f"Loading data for {SYMBOL}...")
    prices = load_smp_csv(CSV_PATH, symbol=SYMBOL, start=START)
    if prices.empty:
        print("No data after start filter. Loading full history…")
        prices = load_smp_csv(CSV_PATH, symbol=SYMBOL)
    if prices.empty:
        raise SystemExit(f"Could not load data for symbol {SYMBOL}")

    print(f"Loaded {len(prices)} rows: {prices.index.min()} → {prices.index.max()}")

    if len(prices) < LOOKBACK + 2:
        LOOKBACK = max(5, len(prices) - 2)
        print(f"[info] Adjusted lookback to {LOOKBACK}")

    analyzer = TradingAnalyzer()
    analyzer.run_eta_analysis(
        prices,
        eta_values=ETA_VALUES,
        lookback=LOOKBACK,
        cost_bps=COST_BPS
    )

    save_path = f"analysis_results_{SYMBOL}"
    results_df = analyzer.generate_comprehensive_analysis(SYMBOL, save_path=save_path)

    print("\n=== Performance Comparison ===")
    print(results_df)

    # ---- NEW: redundancy diagnostic ----
    diag = run_redundancy_diagnostic(prices, lookback=LOOKBACK, save_dir=save_path)

    return analyzer, results_df, diag


if __name__ == "__main__":
    analyzer, results, diag = main()
