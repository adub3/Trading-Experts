# enhanced_analysis.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Your existing imports
from simulation.temploader import load_smp_csv
from simulation.backtester import run_backtest
from experts import AlwaysLong, AlwaysShort, MovingAverageCrossover, RSIExpert, MomentumExpert

# Import the new quant experts (assuming they're in the same experts module or imported separately)
# from quant_experts import VolatilityRegimeExpert, MeanReversionExpert, TrendQualityExpert, DrawdownAvoidanceExpert, PriceAccelerationExpert, RelativeStrengthExpert, VolumePriceExpert, BreakoutExpert

from meta.hedge import HedgeMeta

class TradingAnalyzer:
    def __init__(self):
        self.results = {}
        self.expert_weights_history = {}
        self.performance_metrics = {}
    
    def run_eta_analysis(self, prices, eta_values=[0.05, 0.15], lookback=60, cost_bps=1.0):
        """Run analysis comparing different eta values"""
        
        # Original experts
        base_experts = [
            AlwaysLong(),
            AlwaysShort(),
            MovingAverageCrossover(10, 50),
            MomentumExpert(lookback=20),
        ]
        
        # Enhanced quant experts (commented out the imports above, so creating simplified versions here)
        # For now, I'll create some basic additional experts to demonstrate the concept
        enhanced_experts = base_experts.copy()
        
        for eta in eta_values:
            print(f"\n=== Running analysis with eta = {eta} ===")
            
            meta = HedgeMeta(
                enhanced_experts,
                eta=eta,
                alpha_fixed_share=0.1,
                decay_lambda=0.99,
                weight_floor=1e-6,
                allow_specialists=True,
                random_tie_break=True,
                loss="logistic",
                loss_scale=2.0,
            )
            
            results = run_backtest(
                prices,
                enhanced_experts,
                meta,
                lookback=lookback,
                cost_bps=cost_bps,
                hold_on_tie=True,
            )
            
            self.results[f"eta_{eta}"] = results
            self.expert_weights_history[f"eta_{eta}"] = results["weights"]
            
            # Calculate additional performance metrics
            self.performance_metrics[f"eta_{eta}"] = self._calculate_performance_metrics(results)
    
    def _calculate_performance_metrics(self, results):
        """Calculate comprehensive performance metrics"""
        trades = results["trades"]
        if trades.empty:
            return {}
        
        print(f"[debug] Available columns in trades: {trades.columns.tolist()}")
        
        # Try to find the right column names
        cumret_col = None
        for col in ['cumulative_return', 'cum_return', 'cumret', 'total_return']:
            if col in trades.columns:
                cumret_col = col
                break
        
        portfolio_col = None
        for col in ['portfolio_value', 'portfolio', 'value', 'nav']:
            if col in trades.columns:
                portfolio_col = col
                break
        
        returns_col = None
        for col in ['strategy_return', 'returns', 'daily_return', 'pnl']:
            if col in trades.columns:
                returns_col = col
                break
        
        # Basic metrics
        if cumret_col:
            total_return = trades[cumret_col].iloc[-1]
        elif portfolio_col:
            total_return = (trades[portfolio_col].iloc[-1] / trades[portfolio_col].iloc[0]) - 1
        else:
            total_return = 0
        
        # Calculate returns series for other metrics
        if portfolio_col:
            portfolio_values = trades[portfolio_col]
            returns = portfolio_values.pct_change().dropna()
        elif returns_col:
            returns = trades[returns_col].dropna()
            portfolio_values = (1 + returns).cumprod()
        else:
            # Fallback: assume we have price data
            price_cols = [col for col in trades.columns if 'price' in col.lower() or 'close' in col.lower()]
            if price_cols:
                portfolio_values = trades[price_cols[0]]
                returns = portfolio_values.pct_change().dropna()
            else:
                return {"total_return": total_return, "sharpe_ratio": 0, "max_drawdown": 0, "volatility": 0}
        
        if len(returns) == 0:
            return {"total_return": total_return, "sharpe_ratio": 0, "max_drawdown": 0, "volatility": 0}
        
        # Performance metrics
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
        
        # Max drawdown
        rolling_max = portfolio_values.expanding().max()
        drawdown = (portfolio_values - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Volatility (annualized)
        volatility = returns.std() * np.sqrt(252)
        
        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "volatility": volatility,
            "avg_daily_return": returns.mean(),
            "win_rate": (returns > 0).mean(),
            "num_trades": len(trades)
        }
    
    def generate_comprehensive_analysis(self, symbol, save_path="analysis_results"):
        """Generate comprehensive analysis with visualizations and CSV exports"""
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        # 1. Create performance comparison DataFrame
        perf_df = pd.DataFrame(self.performance_metrics).T
        perf_df.to_csv(os.path.join(save_path, f"{symbol}_performance_comparison.csv"))
        
        # 2. Expert weights analysis over time
        self._analyze_expert_weights(save_path, symbol)
        
        # 3. Generate visualizations
        self._create_comprehensive_plots(save_path, symbol)
        
        # 4. Export detailed results
        self._export_detailed_results(save_path, symbol)
        
        print(f"\n=== Analysis Complete ===")
        print(f"Results saved to: {save_path}/")
        print(f"Performance comparison: {symbol}_performance_comparison.csv")
        
        return perf_df
    
    def _analyze_expert_weights(self, save_path, symbol):
        """Analyze how expert weights change over time"""
        
        weights_comparison = {}
        
        for eta_key, weights_df in self.expert_weights_history.items():
            if weights_df.empty:
                continue
                
            # Calculate average weights over time
            avg_weights = weights_df.mean()
            weights_comparison[eta_key] = avg_weights
            
            # Export full weight history
            weights_df.to_csv(os.path.join(save_path, f"{symbol}_{eta_key}_weights_history.csv"))
        
        # Create weights comparison DataFrame
        if weights_comparison:
            weights_comp_df = pd.DataFrame(weights_comparison)
            weights_comp_df.to_csv(os.path.join(save_path, f"{symbol}_weights_comparison.csv"))
    
    def _create_comprehensive_plots(self, save_path, symbol):
        """Create comprehensive visualization plots"""
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create a large figure with subplots
        fig = plt.figure(figsize=(24, 20))
        
        # 1. Price chart with strategy performance (top row, full width)
        ax1 = plt.subplot(4, 2, (1, 2))
        self._plot_price_and_strategy_performance(ax1)
        
        # 2. Expert weights evolution
        ax2 = plt.subplot(4, 2, 3)
        self._plot_expert_weights_evolution(ax2)
        
        # 3. Performance metrics comparison
        ax3 = plt.subplot(4, 2, 4)
        self._plot_performance_comparison(ax3)
        
        # 4. Drawdown analysis
        ax4 = plt.subplot(4, 2, 5)
        self._plot_drawdown_analysis(ax4)
        
        # 5. Rolling metrics
        ax5 = plt.subplot(4, 2, 6)
        self._plot_rolling_metrics(ax5)
        
        # 6. NEW: Expert predictions analysis
        ax6 = plt.subplot(4, 2, 7)
        self._plot_expert_predictions_analysis(ax6)
        
        # 7. NEW: Momentum deep dive
        ax7 = plt.subplot(4, 2, 8)
        self._plot_momentum_analysis(ax7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{symbol}_comprehensive_analysis.png"), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_price_and_strategy_performance(self, ax):
        """Plot price and strategy performance side by side"""
        
        for eta_key, results in self.results.items():
            trades = results["trades"]
            if trades.empty:
                continue
            
            print(f"[debug] Plotting {eta_key}, columns: {trades.columns.tolist()}")
            
            # Find strategy performance column
            strategy_col = None
            for col in ['cumulative_return', 'cum_return', 'portfolio_value', 'nav']:
                if col in trades.columns:
                    strategy_col = col
                    break
            
            if strategy_col:
                if 'portfolio_value' in strategy_col or 'nav' in strategy_col:
                    # Convert to returns if it's a value series
                    strategy_returns = (trades[strategy_col] / trades[strategy_col].iloc[0]) - 1
                else:
                    strategy_returns = trades[strategy_col]
                    
                ax.plot(trades.index, strategy_returns, 
                       label=f"Strategy ({eta_key})", linewidth=2)
            
            # Plot underlying asset performance
            underlying_col = None
            for col in ['underlying_return', 'benchmark_return', 'buy_hold_return']:
                if col in trades.columns:
                    underlying_col = col
                    break
            
            if underlying_col:
                ax.plot(trades.index, trades[underlying_col], 
                       label="Buy & Hold", linewidth=1, alpha=0.7, linestyle='--')
            else:
                # Try to construct buy & hold from price data
                price_cols = [col for col in trades.columns if 'price' in col.lower() or 'close' in col.lower()]
                if price_cols:
                    buy_hold = (trades[price_cols[0]] / trades[price_cols[0]].iloc[0]) - 1
                    ax.plot(trades.index, buy_hold, 
                           label="Buy & Hold", linewidth=1, alpha=0.7, linestyle='--')
        
        ax.set_title("Strategy Performance vs Buy & Hold", fontsize=14, fontweight='bold')
        ax.set_ylabel("Cumulative Return", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_expert_weights_evolution(self, ax):
        """Plot how expert weights evolve over time"""
        
        eta_15_weights = self.expert_weights_history.get("eta_0.15")
        if eta_15_weights is not None and not eta_15_weights.empty:
            # Plot each expert's weight over time
            for col in eta_15_weights.columns:
                ax.plot(eta_15_weights.index, eta_15_weights[col], 
                       label=col, alpha=0.8, linewidth=1.5)
        
        ax.set_title("Expert Weights Evolution (eta=0.15)", fontsize=14, fontweight='bold')
        ax.set_ylabel("Weight", fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    def _plot_performance_comparison(self, ax):
        """Plot performance metrics comparison"""
        
        metrics_df = pd.DataFrame(self.performance_metrics).T
        if metrics_df.empty:
            return
        
        # Select key metrics to plot
        key_metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'volatility']
        plot_metrics = [m for m in key_metrics if m in metrics_df.columns]
        
        x = np.arange(len(metrics_df.index))
        width = 0.15
        
        for i, metric in enumerate(plot_metrics):
            ax.bar(x + i * width, metrics_df[metric], width, label=metric, alpha=0.8)
        
        ax.set_title("Performance Metrics Comparison", fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * (len(plot_metrics) - 1) / 2)
        ax.set_xticklabels(metrics_df.index)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_drawdown_analysis(self, ax):
        """Plot drawdown analysis"""
        
        has_data = False
        
        for eta_key, results in self.results.items():
            trades = results["trades"]
            if trades.empty:
                continue
            
            print(f"[debug] Drawdown analysis for {eta_key}, columns: {list(trades.columns)}")
            
            # Find the right column for performance data
            performance_data = None
            
            # Try different column combinations
            if 'portfolio_value' in trades.columns:
                performance_data = trades['portfolio_value']
                print(f"[debug] Using portfolio_value: {performance_data.head()}")
            elif 'cumulative_return' in trades.columns:
                performance_data = 1 + trades['cumulative_return']  # Convert to value
                print(f"[debug] Using cumulative_return: {performance_data.head()}")
            elif 'nav' in trades.columns:
                performance_data = trades['nav']
                print(f"[debug] Using nav: {performance_data.head()}")
            else:
                # Try to construct from any price-like column
                price_cols = [col for col in trades.columns if any(keyword in col.lower() 
                             for keyword in ['price', 'close', 'value', 'return'])]
                if price_cols:
                    col = price_cols[0]
                    performance_data = trades[col]
                    print(f"[debug] Using fallback column '{col}': {performance_data.head()}")
            
            if performance_data is not None and len(performance_data) > 1:
                # Ensure we have positive values (convert if necessary)
                if performance_data.min() < 0:
                    performance_data = performance_data - performance_data.min() + 1
                
                # Calculate drawdown
                rolling_max = performance_data.expanding().max()
                drawdown = (performance_data - rolling_max) / rolling_max
                
                print(f"[debug] Drawdown range: {drawdown.min():.4f} to {drawdown.max():.4f}")
                
                if not drawdown.isna().all():
                    ax.fill_between(trades.index, drawdown, 0, alpha=0.5, label=f"Drawdown ({eta_key})")
                    has_data = True
        
        if has_data:
            ax.set_title("Drawdown Analysis", fontsize=14, fontweight='bold')
            ax.set_ylabel("Drawdown", fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No drawdown data available', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title("Drawdown Analysis - No Data", fontsize=14)
    
    def _plot_rolling_metrics(self, ax):
        """Plot rolling performance metrics"""
        
        eta_15_results = self.results.get("eta_0.15")
        if not eta_15_results or eta_15_results["trades"].empty:
            ax.text(0.5, 0.5, 'No eta=0.15 results available', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title("Rolling Performance Metrics - No Data", fontsize=14)
            return
        
        trades = eta_15_results["trades"]
        print(f"[debug] Rolling metrics - available columns: {list(trades.columns)}")
        
        # Try to find or calculate returns
        returns = None
        
        if 'strategy_return' in trades.columns:
            returns = trades['strategy_return']
            print(f"[debug] Using strategy_return: {returns.describe()}")
        elif 'returns' in trades.columns:
            returns = trades['returns']
            print(f"[debug] Using returns: {returns.describe()}")
        elif 'portfolio_value' in trades.columns:
            returns = trades['portfolio_value'].pct_change().dropna()
            print(f"[debug] Calculated returns from portfolio_value: {returns.describe()}")
        elif 'cumulative_return' in trades.columns:
            # Convert cumulative returns to period returns
            cum_rets = trades['cumulative_return']
            returns = cum_rets.diff().fillna(cum_rets.iloc[0])
            print(f"[debug] Calculated returns from cumulative_return: {returns.describe()}")
        else:
            # Try any numeric column that looks like returns
            numeric_cols = trades.select_dtypes(include=[np.number]).columns
            potential_cols = [col for col in numeric_cols if any(keyword in col.lower() 
                             for keyword in ['return', 'pnl', 'profit'])]
            if potential_cols:
                returns = trades[potential_cols[0]].diff().dropna()
                print(f"[debug] Using fallback column '{potential_cols[0]}': {returns.describe()}")
        
        if returns is not None and len(returns) > 60:
            returns = returns.dropna()
            
            if len(returns) > 0 and returns.std() > 0:
                # Calculate rolling Sharpe ratio
                rolling_mean = returns.rolling(window=60, min_periods=30).mean()
                rolling_std = returns.rolling(window=60, min_periods=30).std()
                rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252)
                
                # Remove infinite values
                rolling_sharpe = rolling_sharpe.replace([np.inf, -np.inf], np.nan).dropna()
                
                print(f"[debug] Rolling Sharpe range: {rolling_sharpe.min():.4f} to {rolling_sharpe.max():.4f}")
                
                if len(rolling_sharpe) > 0:
                    ax.plot(rolling_sharpe.index, rolling_sharpe, 
                           label="60-day Rolling Sharpe", linewidth=2, color='blue')
                    
                    # Also plot rolling returns
                    rolling_ret_annualized = rolling_mean * 252
                    rolling_ret_annualized = rolling_ret_annualized.dropna()
                    
                    if len(rolling_ret_annualized) > 0:
                        ax2 = ax.twinx()
                        ax2.plot(rolling_ret_annualized.index, rolling_ret_annualized, 
                               label="60-day Rolling Return (Ann.)", linewidth=1, 
                               alpha=0.7, color='red', linestyle='--')
                        ax2.set_ylabel("Annualized Return", fontsize=10, color='red')
                        ax2.tick_params(axis='y', labelcolor='red')
                    
                    ax.set_title("Rolling Performance Metrics (eta=0.15)", fontsize=14, fontweight='bold')
                    ax.set_ylabel("Rolling Sharpe Ratio", fontsize=12)
                    ax.legend(loc='upper left')
                    ax.grid(True, alpha=0.3)
                    return
        
        ax.text(0.5, 0.5, 'Insufficient data for rolling metrics', 
               horizontalalignment='center', verticalalignment='center',
               transform=ax.transAxes, fontsize=12)
        ax.set_title("Rolling Performance Metrics - Insufficient Data", fontsize=14)
    
    def _plot_expert_predictions_analysis(self, ax):
        """Analyze expert predictions over time"""
        
        eta_15_results = self.results.get("eta_0.15")
        if not eta_15_results or eta_15_results["trades"].empty:
            ax.text(0.5, 0.5, 'No prediction data available', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title("Expert Predictions - No Data", fontsize=14)
            return
        
        # Get the expert predictions from the debug output or weights
        # This is a simplified version - you'd need to modify your backtester to save predictions
        weights_df = eta_15_results.get("weights")
        if weights_df is not None and not weights_df.empty:
            # Show the most recent predictions distribution
            latest_weights = weights_df.iloc[-1]
            
            # Create a bar plot of latest expert weights
            experts = latest_weights.index
            weights = latest_weights.values
            
            bars = ax.bar(range(len(experts)), weights, alpha=0.7)
            ax.set_xticks(range(len(experts)))
            ax.set_xticklabels(experts, rotation=45, ha='right')
            ax.set_ylabel('Weight')
            ax.set_title('Latest Expert Weights (eta=0.15)', fontsize=14, fontweight='bold')
            
            # Color bars based on weight magnitude
            max_weight = max(abs(w) for w in weights)
            for bar, weight in zip(bars, weights):
                if weight > 0:
                    bar.set_color('green')
                    bar.set_alpha(0.3 + 0.7 * (weight / max_weight))
                else:
                    bar.set_color('red')
                    bar.set_alpha(0.3 + 0.7 * (abs(weight) / max_weight))
            
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No weights data available', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title("Expert Predictions - No Data", fontsize=14)
    
    def _plot_momentum_analysis(self, ax):
        """Deep dive into momentum expert behavior"""
        
        eta_15_results = self.results.get("eta_0.15")
        if not eta_15_results or eta_15_results["trades"].empty:
            ax.text(0.5, 0.5, 'No momentum data available', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title("Momentum Analysis - No Data", fontsize=14)
            return
        
        trades = eta_15_results["trades"]
        
        # Try to reconstruct momentum signals from the underlying price data
        # Look for price column
        price_col = None
        for col in trades.columns:
            if any(keyword in col.lower() for keyword in ['close', 'price']):
                price_col = col
                break
        
        if price_col:
            prices = trades[price_col]
            
            # Calculate what momentum should be (20-day momentum as in your debug output)
            momentum_20 = prices.pct_change(20)  # 20-day momentum
            momentum_5 = prices.pct_change(5)    # 5-day momentum
            momentum_60 = prices.pct_change(60)  # 60-day momentum
            
            ax.plot(trades.index, momentum_20, label='20-day Momentum', alpha=0.8)
            ax.plot(trades.index, momentum_5, label='5-day Momentum', alpha=0.6)
            ax.plot(trades.index, momentum_60, label='60-day Momentum', alpha=0.7)
            
            # Add horizontal line at zero
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            
            ax.set_title('Momentum Analysis - Why Signals Are Small', fontsize=14, fontweight='bold')
            ax.set_ylabel('Momentum (% Change)', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            stats_text = f"""20-day Momentum Stats:
Mean: {momentum_20.mean():.6f}
Std: {momentum_20.std():.6f}
Range: [{momentum_20.min():.6f}, {momentum_20.max():.6f}]"""
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   fontsize=9)
        else:
            ax.text(0.5, 0.5, f'No price data found in columns: {list(trades.columns)}', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=10)
            ax.set_title("Momentum Analysis - No Price Data", fontsize=14)
    
    def _export_detailed_results(self, save_path, symbol):
        """Export detailed results for each eta value"""
        
        for eta_key, results in self.results.items():
            # Export trades
            if not results["trades"].empty:
                results["trades"].to_csv(
                    os.path.join(save_path, f"{symbol}_{eta_key}_trades.csv"))
            
            # Export summary
            if "summary" in results:
                with open(os.path.join(save_path, f"{symbol}_{eta_key}_summary.txt"), "w") as f:
                    f.write(f"=== Strategy Performance Summary ({eta_key}) ===\n")
                    f.write(results["summary"].to_string(index=False) if hasattr(results["summary"], 'to_string') else str(results["summary"]))

def main():
    # Configuration
    CSV_PATH = os.path.join("simulation", "archive", "sp500_stocks.csv")
    SYMBOL = "AXP"
    START = "2018-01-01"
    LOOKBACK = 60
    COST_BPS = 1.0
    
    print(f"Loading data for {SYMBOL}...")
    
    # Load data
    prices = load_smp_csv(CSV_PATH, symbol=SYMBOL, start=START)
    
    if prices.empty:
        print("No data loaded, falling back to full history...")
        prices = load_smp_csv(CSV_PATH, symbol=SYMBOL)
    
    if prices.empty:
        raise SystemExit(f"Could not load data for symbol {SYMBOL}")
    
    print(f"Loaded {len(prices)} rows of data from {prices.index.min()} to {prices.index.max()}")
    
    # Ensure enough history
    if len(prices) < LOOKBACK + 2:
        LOOKBACK = max(5, len(prices) - 2)
        print(f"Adjusted lookback to {LOOKBACK}")
    
    # Run analysis
    analyzer = TradingAnalyzer()
    analyzer.run_eta_analysis(
        prices, 
        eta_values=[0.05, 0.15],  # Original eta and the requested eta=15 (0.15)
        lookback=LOOKBACK, 
        cost_bps=COST_BPS
    )
    
    # Generate comprehensive analysis
    results_df = analyzer.generate_comprehensive_analysis(
        SYMBOL, 
        save_path=f"analysis_results_{SYMBOL}"
    )
    
    print("\n=== Performance Comparison ===")
    print(results_df)
    
    return analyzer, results_df

if __name__ == "__main__":
    analyzer, results = main()