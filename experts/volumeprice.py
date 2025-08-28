from .base import Expert
import pandas as pd
import numpy as np

class VolumePriceExpert(Expert):
    """Volume-weighted price momentum"""
    def __init__(self, window=20):
        super().__init__(f"VolumePrice_w{window}")
        self.window = window
    
    def predict(self, data):
        if len(data) < self.window or 'Volume' not in data.columns:
            # Fallback to simple price momentum if no volume
            if len(data) < self.window:
                return 0.0
            returns = data['Close'].pct_change().rolling(window=self.window).mean().iloc[-1]
            return np.clip(np.tanh(returns * 50), -1.0, 1.0)
        
        prices = data['Close']
        volumes = data['Volume']
        
        # Volume-weighted returns
        returns = prices.pct_change()
        vol_weighted_returns = (returns * volumes).rolling(window=self.window).sum() / volumes.rolling(window=self.window).sum()
        
        signal_value = vol_weighted_returns.iloc[-1]
        if pd.isna(signal_value):
            return 0.0
        
        # Scale to signal range
        signal = np.tanh(signal_value * 50)
        return np.clip(signal, -1.0, 1.0)
