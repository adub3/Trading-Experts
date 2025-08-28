from .base import Expert
import pandas as pd
import numpy as np

class MeanReversionExpert(Expert):
    """Continuous mean reversion signal based on z-score"""
    def __init__(self, window=20, max_zscore=3.0):
        super().__init__(f"MeanRevert_w{window}_z{max_zscore}")
        self.window = window
        self.max_zscore = max_zscore
    
    def predict(self, data):
        if len(data) < self.window:
            return 0.0
        
        prices = data['Close']
        ma = prices.rolling(window=self.window).mean().iloc[-1]
        current_price = prices.iloc[-1]
        std = prices.rolling(window=self.window).std().iloc[-1]
        
        if std <= 0:
            return 0.0
        
        z_score = (current_price - ma) / std
        
        # Mean reversion: stronger signal for larger deviations
        # Negative z_score (below mean) -> positive signal (buy)
        # Positive z_score (above mean) -> negative signal (sell)
        signal = -z_score / self.max_zscore
        return np.clip(signal, -1.0, 1.0)