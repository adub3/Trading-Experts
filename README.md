# Trading Experts Bot 🧠📈

An adaptive trading framework that combines multiple simple **experts** (strategies) under a meta-learning algorithm (**HedgeMeta**) to dynamically allocate trust across them. Inspired by Hagelbarger’s *SEER* (1956) and early “mind-reading” machines, this bot explores how expert aggregation can adapt to changing market regimes.

---

## 🔑 Key Concepts

- **Expert Pool**
  - Always Long / Always Short
  - Moving Average Crossover
  - RSI Threshold
  - Momentum
  - (easily extendable with new experts and increased complexity)

- **Meta Algorithm (HedgeMeta)**
  - Aggregates expert predictions into one decision
  - Updates weights online using multiplicative weights
  - Supports multiple loss functions: sign-accuracy, logistic, log-wealth
  - Allows **dominance by regime**: the right expert takes over in the right environment

- **Backtester**
  - OHLCV data (CSV / Yahoo Finance / custom loaders)
  - Transaction costs (basis points)
  - Position sizing and equity curve tracking
  - Baseline comparison vs. buy-and-hold

---

## 📊 Metrics

- **CAGR** – Compound Annual Growth Rate  
- **Sharpe Ratio** – Risk-adjusted return  
- **Max Drawdown** – Worst peak-to-trough equity loss  
- **Hit Rate** – % of days/trades profitable  
- Plus: total return, volatility, daily returns, win rate, number of trades

---

## 🚀 Quickstart

### 1. Clone & Install
```bash
git clone https://github.com/<your-username>/trading-experts.git
cd trading-experts
pip install -r requirements.txt
