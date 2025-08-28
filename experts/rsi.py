import pandas as pd
from .base import Expert
import numpy as np

class RelativeStrengthExpert(Expert):
    """Continuous RSI-based signal"""
    def __init__(self, window=14):
        super().__init__(f"RelativeStrength_w{window}")
        self.window = window
    
    def predict(self, data):
        if len(data) < self.window + 1:
            return 0.0
        
        prices = data['Close']
        delta = prices.diff()
        
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        avg_gains = gains.rolling(window=self.window).mean().iloc[-1]
        avg_losses = losses.rolling(window=self.window).mean().iloc[-1]
        
        if avg_losses == 0:
            return 1.0
        
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        # Convert RSI to continuous signal (-1 to 1)
        # RSI 50 = neutral (0), RSI 0 = oversold (+1), RSI 100 = overbought (-1)
        signal = (50 - rsi) / 50
        return np.clip(signal, -1.0, 1.0)

    