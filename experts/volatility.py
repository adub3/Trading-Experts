from .base import Expert
import pandas as pd
import numpy as np

class VolatilityRegimeExpert(Expert):
    """Trades based on volatility regimes - stronger signal in extreme regimes"""
    def __init__(self, vol_window=20):
        super().__init__(f"VolRegime_w{vol_window}")
        self.vol_window = vol_window
    
    def predict(self, data):
        if len(data) < self.vol_window + 50:
            return 0.0
        
        returns = data['Close'].pct_change().dropna()
        rolling_vol = returns.rolling(window=self.vol_window).std()
        vol_percentile = rolling_vol.rolling(window=min(252, len(rolling_vol))).rank(pct=True).iloc[-1]
        
        # Continuous signal: -1 at 100th percentile, +1 at 0th percentile
        # High vol regimes get negative signal, low vol gets positive
        signal = 2 * (1 - vol_percentile) - 1
        return np.clip(signal, -1.0, 1.0)
