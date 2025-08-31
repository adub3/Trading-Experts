# enhanced_analysis.py
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from simulation.temploader import load_smp_csv
from simulation.backtester import run_backtest
from experts import AlwaysLong, AlwaysShort, MovingAverageCrossover, MomentumExpert
from meta.hedge import HedgeMeta, HedgeMetaEWMAProb


# ---------------------------
# Expert pool assembly
# ---------------------------
from experts import *

def build_expert_pool():
    return [
        AlwaysLong(),
        AlwaysShort(),
        MovingAverageCrossover(10, 50),
        MomentumExpert(lookback=20),
        VolatilityRegimeExpert(),
        MeanReversionExpert(),
        TrendQualityExpert(),
        DrawdownAvoidanceExpert(),
        PriceAccelerationExpert(),
        RelativeStrengthExpert(),
        VolumePriceExpert(),
        BreakoutExpert(),
    ]

class TradingAnalyzer:
    def __init__(self):
        self.prices = None
        self.results: dict[str, dict] = {}
        self.expert_weights_history: dict[str, pd.DataFrame] = {}
        self.performance_metrics: dict[str, dict] = {}

    # ---------------------------
    # Core run (eta sweep)
    # ---------------------------
    def run_eta_analysis(self, prices, eta_values=(0.05, 0.15), lookback=60, cost_bps=1.0):
        """
        Run analysis comparing different eta values using a fresh expert pool per run.
        """
        self.prices = prices

        for eta in eta_values:
            print(f"\n=== Running analysis with eta = {eta} ===")
            experts = build_expert_pool()
            print("[experts]", [e.name for e in experts])
            
        meta = HedgeMetaEWMAProb(
            experts,
            eta=eta,
            alpha_fixed_share=0.10,
            decay_lambda=0.99,
            weight_floor=1e-6,
            allow_specialists=True,
            random_tie_break=True,
            # new-style args for the subclass:
            prob_loss="log",        # proper probability scoring (or "brier")
            conf_mode="power",
            conf_power=2.0,         # ↑ if you want sharper confidence effect
            conf_min=0.0,
            label_ewma_tau=0.2,     # tracks slow drifts
            drift_ewma_tau=0.05,
            drift_mix_gamma=0.15,
            weight_momentum=0.2,
        )

        results = run_backtest(
            prices,
            experts,
            meta,
            lookback=lookback,
            cost_bps=cost_bps,
            hold_on_tie=True,
        )

        key = f"eta_{eta}"
        self.results[key] = results
        self.expert_weights_history[key] = results.get("weights", pd.DataFrame())
        self.performance_metrics[key] = self._calculate_performance_metrics(results)


    # ---------------------------
    # Helpers
    # ---------------------------
    def _get_all_expert_names(self):
        """Union of expert names across all η runs (column union of weights frames)."""
        names = set()
        for wdf in self.expert_weights_history.values():
            if wdf is not None and not wdf.empty:
                names.update(wdf.columns.tolist())
        return sorted(names)

    # ---------------------------
    # Metrics
    # ---------------------------
    def _calculate_performance_metrics(self, results: dict) -> dict:
        trades: pd.DataFrame = results["trades"]
        if trades.empty:
            return {}

        # --- Strategy metrics ---
        portfolio_col = next((c for c in ["portfolio_value", "portfolio", "nav", "value"] if c in trades.columns), None)
        returns_col   = next((c for c in ["strategy_return", "returns", "daily_return", "pnl"] if c in trades.columns), None)
        cumret_col    = next((c for c in ["cumulative_return", "cum_return", "cumret", "total_return"] if c in trades.columns), None)

        if portfolio_col:
            portfolio_values = trades[portfolio_col].astype(float)
            strat_returns = portfolio_values.pct_change().dropna()
            total_return = portfolio_values.iloc[-1] / portfolio_values.iloc[0] - 1.0
        elif returns_col:
            strat_returns = trades[returns_col].astype(float).dropna()
            portfolio_values = (1.0 + strat_returns).cumprod()
            total_return = portfolio_values.iloc[-1] - 1.0
        elif cumret_col:
            cumret = trades[cumret_col].astype(float)
            total_return = float(cumret.iloc[-1])
            portfolio_values = 1.0 + cumret
            strat_returns = portfolio_values.pct_change().dropna()
        else:
            price_col = next((c for c in trades.columns if "close" in c.lower() or "price" in c.lower()), None)
            if not price_col:
                return {}
            portfolio_values = trades[price_col].astype(float)
            strat_returns = portfolio_values.pct_change().dropna()
            total_return = portfolio_values.iloc[-1] / portfolio_values.iloc[0] - 1.0

        if len(strat_returns) == 0 or strat_returns.std(ddof=0) == 0:
            vol = sharpe = 0.0
        else:
            vol = strat_returns.std(ddof=0) * np.sqrt(252)
            sharpe = (strat_returns.mean() * 252) / vol

        roll_max = portfolio_values.expanding().max()
        drawdown = (portfolio_values - roll_max) / roll_max
        max_dd = float(drawdown.min())

        # --- Baseline Buy & Hold ---
        bh_total = bh_sharpe = bh_dd = bh_vol = np.nan
        price_col = next((c for c in trades.columns if "close" in c.lower()), None)
        if price_col:
            bh_close = trades[price_col].astype(float)
            bh_ret = bh_close.pct_change().dropna()
            bh_eq = (1 + bh_ret).cumprod()
            if len(bh_ret) > 0 and bh_ret.std(ddof=0) > 0:
                bh_vol = bh_ret.std(ddof=0) * np.sqrt(252)
                bh_sharpe = (bh_ret.mean() * 252) / bh_vol
            bh_dd = float((bh_eq / bh_eq.cummax() - 1.0).min())
            bh_total = float(bh_eq.iloc[-1] - 1.0)

        return {
            # strategy
            "total_return": float(total_return),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_dd),
            "volatility": float(vol),
            "avg_daily_return": float(strat_returns.mean()) if len(strat_returns) else 0.0,
            "win_rate": float((strat_returns > 0).mean()) if len(strat_returns) else 0.0,
            "num_trades": float(len(trades)),
            # baseline
            "BH_total_return": bh_total,
            "BH_sharpe_ratio": bh_sharpe,
            "BH_max_drawdown": bh_dd,
            "BH_volatility": bh_vol,
        }

    # ---------------------------
    # Orchestration: export + plots
    # ---------------------------
    def generate_comprehensive_analysis(self, symbol, save_path="analysis_results"):
        """
        Save CSVs and plots for everything we computed.
        """
        os.makedirs(save_path, exist_ok=True)

        # 1) performance comparison table
        perf_df = pd.DataFrame(self.performance_metrics).T
        perf_df.to_csv(os.path.join(save_path, f"{symbol}_performance_comparison.csv"))

        # 2) expert weights analysis / export
        self._analyze_expert_weights(save_path, symbol)

        # 3) plots
        self._create_comprehensive_plots(save_path, symbol)

        # 4) detailed exports
        self._export_detailed_results(save_path, symbol)

        print("\n=== Analysis Complete ===")
        print(f"Results saved to: {save_path}/")
        print(f"Performance comparison: {symbol}_performance_comparison.csv")
        return perf_df

    def _analyze_expert_weights(self, save_path, symbol):
        """
        Export full weight histories and average weights per eta.
        """
        weights_comparison = {}
        for eta_key, wdf in self.expert_weights_history.items():
            if wdf is None or wdf.empty:
                continue
            # average weights across time
            weights_comparison[eta_key] = wdf.mean()
            # full history export
            wdf.to_csv(os.path.join(save_path, f"{symbol}_{eta_key}_weights_history.csv"))

        if weights_comparison:
            weights_comp_df = pd.DataFrame(weights_comparison)
            weights_comp_df.to_csv(os.path.join(save_path, f"{symbol}_weights_comparison.csv"))

    # ---------------------------
    # Plots
    # ---------------------------
    def _create_comprehensive_plots(self, save_path, symbol):
        plt.style.use('default')
        sns.set_palette("husl")

        fig = plt.figure(figsize=(24, 22))

        # 1. Price/strategy (rebased percent)
        ax1 = plt.subplot(4, 2, (1, 2))
        self._plot_price_and_strategy_performance(ax1)

        # 2. Stacked weight share (best η)
        ax2 = plt.subplot(4, 2, 3)
        self._plot_expert_weights_evolution(ax2)

        # 3. Performance metrics comparison (Strategy vs BH)
        ax3 = plt.subplot(4, 2, 4)
        self._plot_performance_comparison(ax3)

        # 4. Heatmap: average weights (experts × η)
        ax4 = plt.subplot(4, 2, 5)
        self._plot_weights_heatmap_all_etas(ax4, top_k=None)  # set top_k=20 if crowded

        # 5. Latest weights across all etas (grouped bars)
        ax5 = plt.subplot(4, 2, 6)
        self._plot_latest_weights_all_etas(ax5)

        # 6. Rolling metrics (best η)
        ax6 = plt.subplot(4, 2, 7)
        self._plot_rolling_metrics(ax6)

        # 7. Momentum deep dive (best η)
        ax7 = plt.subplot(4, 2, 8)
        self._plot_momentum_analysis(ax7)

        plt.tight_layout()
        out_png = os.path.join(save_path, f"{symbol}_comprehensive_analysis.png")
        plt.savefig(out_png, dpi=300, bbox_inches='tight')
        print(f"[saved] plot -> {out_png}")
        try:
            plt.show()
        except Exception:
            pass

    def _plot_price_and_strategy_performance(self, ax):
        """
        Plot % change of the underlying ticker (buy & hold) and
        % cumulative returns of each strategy (one line per η).
        """
        ax.set_title("Buy & Hold vs Strategy Cumulative Returns (rebased to 0%)", fontsize=14, fontweight='bold')
        ax.set_ylabel("Cumulative Return (%)")
        ax.grid(True, alpha=0.3)

        # ----- Buy & Hold from raw prices -----
        bh_line_plotted = False
        if self.prices is not None and len(self.prices) > 1:
            px = None
            for col in ["Close", "Adj Close", "close", "adj_close"]:
                if col in self.prices.columns:
                    px = self.prices[col].astype(float).copy()
                    break
            if px is not None and len(px) > 1:
                bh_px = px.copy()  # present in locals()

        # ----- Strategy curves -----
        for eta_key, res in self.results.items():
            trades = res.get("trades", None)
            if trades is None or trades.empty:
                continue

            # Strategy cumulative % (prefer equity; else portfolio_value; else cumulative_return)
            strat_curve = None
            if "equity" in trades.columns:
                eq = trades["equity"].astype(float)
                strat_curve = (eq / eq.iloc[0]) - 1.0
            else:
                col = next((c for c in ["portfolio_value", "nav", "cumulative_return", "cum_return"] if c in trades.columns), None)
                if col is not None:
                    s = trades[col].astype(float)
                    if col in ("portfolio_value", "nav"):
                        strat_curve = (s / s.iloc[0]) - 1.0
                    else:  # cumulative_return already cumulative
                        strat_curve = s
                        if strat_curve.iloc[0] != 0.0:
                            strat_curve = strat_curve - strat_curve.iloc[0]
                else:
                    # last resort: try 'pnl' cumprod
                    if "pnl" in trades.columns:
                        eq = (1.0 + trades["pnl"].astype(float)).cumprod()
                        strat_curve = (eq / eq.iloc[0]) - 1.0

            if strat_curve is None or strat_curve.empty:
                continue

            # Plot strategy curve (%)
            ax.plot(
                strat_curve.index,
                (strat_curve * 100.0).values,
                label=f"Strategy ({eta_key})",
                linewidth=1.8,
            )

            # Plot aligned buy & hold once (match the same index window so rebasing is fair)
            if (not bh_line_plotted) and ("bh_px" in locals()):
                idx = strat_curve.index.intersection(bh_px.index)
                if len(idx) > 1:
                    bh_win = bh_px.loc[idx]
                    bh_curve = (bh_win / bh_win.iloc[0]) - 1.0
                    ax.plot(
                        bh_curve.index,
                        (bh_curve * 100.0).values,
                        label="Buy & Hold (Price)",
                        linewidth=1.2,
                        linestyle="--",
                        alpha=0.8,
                    )
                    bh_line_plotted = True

        ax.legend()

    def _plot_expert_weights_evolution(self, ax):
        """
        Stacked area of ALL experts' weights over time for the best η (by Sharpe).
        Normalized each day so the stack sums to 1.
        """
        if not self.performance_metrics:
            ax.text(0.5, 0.5, 'No performance metrics yet', ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Expert Weights Evolution - No Data", fontsize=14)
            return

        best_key = max(
            self.performance_metrics.keys(),
            key=lambda k: (self.performance_metrics[k] or {}).get("sharpe_ratio", float("-inf"))
        )
        wdf = self.expert_weights_history.get(best_key)
        if wdf is None or wdf.empty:
            ax.text(0.5, 0.5, f'No weights for {best_key}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Expert Weights Evolution - No Data", fontsize=14)
            return

        w = wdf.copy().astype(float)
        w = w.clip(lower=0)  # guard against tiny negative floors
        row_sum = w.sum(axis=1).replace(0, np.nan)
        w = w.div(row_sum, axis=0).fillna(0.0)

        experts = w.columns.tolist()
        ax.stackplot(w.index, [w[c].values for c in experts], labels=experts, alpha=0.9, linewidth=0.0)

        ax.set_title(f"Expert Weight Share Over Time (stacked) — {best_key}", fontsize=14, fontweight='bold')
        ax.set_ylabel("Share (sum = 1)")
        ax.set_ylim(0, 1)
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8, ncol=1)
        ax.grid(True, alpha=0.25)

    def _plot_weights_heatmap_all_etas(self, ax, top_k=None):
        """
        Heatmap of average weights per expert per η (experts × η).
        Set top_k to show only top experts by mean weight across all η.
        """
        all_experts = self._get_all_expert_names()
        if not all_experts:
            ax.text(0.5, 0.5, 'No weights available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Average Weights Heatmap - No Data", fontsize=14)
            return

        etas = list(self.expert_weights_history.keys())
        data = []
        for exp in all_experts:
            row = []
            for eta_key in etas:
                wdf = self.expert_weights_history.get(eta_key)
                if wdf is not None and not wdf.empty and exp in wdf.columns:
                    row.append(float(wdf[exp].mean()))
                else:
                    row.append(np.nan)
            data.append(row)

        mat = pd.DataFrame(data, index=all_experts, columns=etas)

        if top_k is not None and len(all_experts) > top_k:
            means = mat.mean(axis=1).fillna(0.0)
            top_idx = means.sort_values(ascending=False).head(top_k).index
            mat = mat.loc[top_idx]

        mat = mat.fillna(0.0)

        im = ax.imshow(mat.values, aspect='auto', interpolation='nearest')
        ax.set_yticks(np.arange(len(mat.index)))
        ax.set_yticklabels(mat.index, fontsize=8)
        ax.set_xticks(np.arange(len(mat.columns)))
        ax.set_xticklabels(mat.columns, rotation=45, ha='right')
        ax.set_title("Average Expert Weights by η (Heatmap)", fontsize=14, fontweight='bold')
        ax.grid(False)

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Average Weight', rotation=90)

    def _plot_latest_weights_all_etas(self, ax):
        """
        Grouped bars: latest expert weights for each η, covering ALL experts found across runs.
        """
        all_experts = self._get_all_expert_names()
        if not all_experts:
            ax.text(0.5, 0.5, 'No weights data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Latest Expert Weights by η - No Data", fontsize=14)
            return

        etas = list(self.expert_weights_history.keys())
        latest = {}
        for eta_key in etas:
            wdf = self.expert_weights_history.get(eta_key)
            latest[eta_key] = (wdf.iloc[-1] if (wdf is not None and not wdf.empty) else pd.Series(dtype=float))

        idx = np.arange(len(all_experts))
        width = max(0.1, 0.8 / max(1, len(etas)))

        for i, eta_key in enumerate(etas):
            vals = np.array([float(latest[eta_key].get(exp, 0.0)) for exp in all_experts])
            ax.bar(idx + i * width, vals, width=width, label=eta_key, alpha=0.85)

        ax.set_xticks(idx + width * (len(etas) - 1) / 2)
        ax.set_xticklabels(all_experts, rotation=45, ha='right')
        ax.set_ylabel("Weight")
        ax.set_title("Latest Expert Weights by η (All Experts)", fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_performance_comparison(self, ax):
        """
        Side-by-side bars per eta: Strategy vs Buy&Hold for key metrics.
        Metrics: total_return, sharpe_ratio, max_drawdown, volatility.
        """
        metrics_df = pd.DataFrame(self.performance_metrics).T
        if metrics_df.empty:
            ax.text(0.5, 0.5, 'No metrics yet', ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Performance Metrics Comparison - No Data", fontsize=14)
            return

        # Ensure baseline columns exist (if missing for some runs, fill with NaN)
        for col in ["BH_total_return", "BH_sharpe_ratio", "BH_max_drawdown", "BH_volatility"]:
            if col not in metrics_df.columns:
                metrics_df[col] = np.nan

        display_rows = metrics_df.index.tolist()
        x = np.arange(len(display_rows))
        width = 0.35  # bar width for paired bars

        pairs = [
            ("total_return", "BH_total_return", "Total Return"),
            ("sharpe_ratio", "BH_sharpe_ratio", "Sharpe Ratio"),
            ("max_drawdown", "BH_max_drawdown", "Max Drawdown"),
            ("volatility", "BH_volatility", "Volatility"),
        ]

        offsets = [-1.5, -0.5, 0.5, 1.5]
        cluster = width / 2

        for (i, (m_strat, m_bh, title)) in enumerate(pairs):
            xs = x + offsets[i] * cluster / 2
            ax.bar(xs - width/2, metrics_df[m_strat].values, width=width*0.48, label=f"{title} (Strategy)" if i==0 else None, alpha=0.85)
            ax.bar(xs + width/2, metrics_df[m_bh].values,   width=width*0.48, label=f"{title} (Buy&Hold)" if i==0 else None, alpha=0.65)

        ax.set_xticks(x)
        ax.set_xticklabels(display_rows, rotation=0)
        ax.set_title("Performance: Strategy vs Buy & Hold (per η)", fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)

    def _plot_drawdown_analysis(self, ax):
        has_data = False
        for eta_key, res in self.results.items():
            trades = res["trades"]
            if trades.empty:
                continue

            perf = None
            if 'portfolio_value' in trades.columns:
                perf = trades['portfolio_value'].astype(float)
            elif 'cumulative_return' in trades.columns:
                perf = (1.0 + trades['cumulative_return'].astype(float))
            elif 'nav' in trades.columns:
                perf = trades['nav'].astype(float)
            else:
                price_col = next((c for c in trades.columns if any(k in c.lower() for k in ['price', 'close', 'value'])), None)
                if price_col:
                    perf = trades[price_col].astype(float)

            if perf is None or len(perf) < 2:
                continue

            if perf.min() <= 0:
                perf = perf - perf.min() + 1.0

            roll_max = perf.expanding().max()
            dd = (perf - roll_max) / roll_max
            if not dd.isna().all():
                ax.fill_between(perf.index, dd.values, 0, alpha=0.45, label=f"Drawdown ({eta_key})")
                has_data = True

        if has_data:
            ax.set_title("Drawdown Analysis", fontsize=14, fontweight='bold')
            ax.set_ylabel("Drawdown")
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No drawdown data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Drawdown Analysis - No Data", fontsize=14)

    def _plot_rolling_metrics(self, ax):
        """
        60-day rolling Sharpe and annualized return for the best eta (by Sharpe).
        """
        if not self.performance_metrics:
            ax.text(0.5, 0.5, 'No results yet', ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Rolling Performance Metrics - No Data", fontsize=14)
            return

        best_key = max(self.performance_metrics.keys(),
                       key=lambda k: (self.performance_metrics[k] or {}).get("sharpe_ratio", float("-inf")))
        trades = self.results.get(best_key, {}).get("trades", pd.DataFrame())
        if trades.empty:
            ax.text(0.5, 0.5, 'No trades available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Rolling Performance Metrics - No Data", fontsize=14)
            return

        returns = None
        if 'strategy_return' in trades.columns:
            returns = trades['strategy_return'].astype(float)
        elif 'returns' in trades.columns:
            returns = trades['returns'].astype(float)
        elif 'portfolio_value' in trades.columns:
            returns = trades['portfolio_value'].astype(float).pct_change().dropna()
        elif 'cumulative_return' in trades.columns:
            eq = 1.0 + trades['cumulative_return'].astype(float)
            returns = eq.pct_change().dropna()
        elif 'pnl' in trades.columns:
            returns = trades['pnl'].astype(float)

        if returns is None or len(returns) < 60:
            ax.text(0.5, 0.5, 'Insufficient data for rolling metrics', ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Rolling Performance Metrics - Insufficient Data", fontsize=14)
            return

        returns = returns.dropna()
        roll_mean = returns.rolling(60, min_periods=30).mean()
        roll_std = returns.rolling(60, min_periods=30).std()
        roll_sharpe = (roll_mean / roll_std) * np.sqrt(252)
        roll_sharpe = roll_sharpe.replace([np.inf, -np.inf], np.nan).dropna()

        if len(roll_sharpe) > 0:
            ax.plot(roll_sharpe.index, roll_sharpe.values, label="60-day Rolling Sharpe", linewidth=2)

            roll_ret_ann = (roll_mean * 252).dropna()
            if len(roll_ret_ann) > 0:
                ax2 = ax.twinx()
                ax2.plot(roll_ret_ann.index, roll_ret_ann.values, label="60-day Rolling Return (Ann.)",
                         linewidth=1, alpha=0.7, color='red', linestyle='--')
                ax2.set_ylabel("Annualized Return", fontsize=10, color='red')
                ax2.tick_params(axis='y', labelcolor='red')

            ax.set_title(f"Rolling Performance Metrics ({best_key})", fontsize=14, fontweight='bold')
            ax.set_ylabel("Rolling Sharpe")
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Insufficient data after filtering', ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Rolling Performance Metrics - Insufficient Data", fontsize=14)

    def _plot_momentum_analysis(self, ax):
        """
        Deep dive into momentum behavior using available close/price series in trades.
        """
        if not self.results:
            ax.text(0.5, 0.5, 'No results yet', ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Momentum Analysis - No Data", fontsize=14)
            return

        # pick best η by Sharpe
        best_key = max(self.performance_metrics.keys(),
                       key=lambda k: (self.performance_metrics[k] or {}).get("sharpe_ratio", float("-inf")))
        trades = self.results.get(best_key, {}).get("trades", pd.DataFrame())
        if trades.empty:
            ax.text(0.5, 0.5, 'No momentum data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Momentum Analysis - No Data", fontsize=14)
            return

        price_col = next((c for c in trades.columns if 'close' in c.lower() or 'price' in c.lower()), None)
        if not price_col:
            ax.text(0.5, 0.5, 'No price column found', ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Momentum Analysis - No Price Data", fontsize=14)
            return

        px = trades[price_col].astype(float)
        mom20 = px.pct_change(20)
        mom5 = px.pct_change(5)
        mom60 = px.pct_change(60)

        ax.plot(px.index, mom20.values, label='20-day Momentum', alpha=0.8)
        ax.plot(px.index, mom5.values, label='5-day Momentum', alpha=0.6)
        ax.plot(px.index, mom60.values, label='60-day Momentum', alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)

        ax.set_title('Momentum Analysis', fontsize=14, fontweight='bold')
        ax.set_ylabel('Momentum (% Change)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        stats_text = (
            f"20d mean: {mom20.mean():.6f}\n"
            f"20d std:  {mom20.std():.6f}\n"
            f"20d min:  {mom20.min():.6f}\n"
            f"20d max:  {mom20.max():.6f}"
        )
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=9)

    # ---------------------------
    # Exports
    # ---------------------------
    def _export_detailed_results(self, save_path, symbol):
        for eta_key, res in self.results.items():
            trades = res.get("trades", pd.DataFrame())
            if not trades.empty:
                trades.to_csv(os.path.join(save_path, f"{symbol}_{eta_key}_trades.csv"))

            summary = res.get("summary", None)
            if summary is not None:
                with open(os.path.join(save_path, f"{symbol}_{eta_key}_summary.txt"), "w") as f:
                    if hasattr(summary, "to_string"):
                        f.write(f"=== Strategy Performance Summary ({eta_key}) ===\n")
                        f.write(summary.to_string(index=False))
                    else:
                        f.write(str(summary))


# ---------------------------
# Script entry
# ---------------------------
def main():
    # Config
    CSV_PATH = os.path.join("simulation", "archive", "sp500_stocks.csv")
    SYMBOL = "AXP" #"USB"
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

    return analyzer, results_df


if __name__ == "__main__":
    analyzer, results = main()
