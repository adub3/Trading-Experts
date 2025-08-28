from .base import Expert
import pandas as pd
import numpy as np

class PriceAccelerationExpert(Expert):
    """Trades on price acceleration (second derivative)"""
    def __init__(self, window=10, smoothing=5):
        super().__init__(f"PriceAccel_w{window}_s{smoothing}")
        self.window = window
        self.smoothing = smoothing
    
    def predict(self, data):
        if len(data) < self.window + self.smoothing + 5:
            return 0.0
        
        prices = data['Close']
        returns = prices.pct_change()
        
        # Calculate acceleration (change in momentum)
        momentum = returns.rolling(window=self.smoothing).mean()
        acceleration = momentum.diff().rolling(window=self.window).mean().iloc[-1]
        
        if pd.isna(acceleration):
            return 0.0
        
        # Scale acceleration to reasonable signal range
        signal = np.tanh(acceleration * 1000)  # Scale factor to make signal meaningful
        return np.clip(signal, -1.0, 1.0)