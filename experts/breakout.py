from .base import Expert
import pandas as pd
import numpy as np

class BreakoutExpert(Expert):
    """Detects price breakouts from recent ranges"""
    def __init__(self, lookback=20, breakout_threshold=0.02):
        super().__init__(f"Breakout_lb{lookback}_th{breakout_threshold}")
        self.lookback = lookback
        self.breakout_threshold = breakout_threshold
    
    def predict(self, data):
        if len(data) < self.lookback:
            return 0.0
        
        prices = data['Close']
        recent_prices = prices.tail(self.lookback)
        
        price_high = recent_prices.max()
        price_low = recent_prices.min()
        current_price = prices.iloc[-1]
        
        price_range = price_high - price_low
        if price_range == 0:
            return 0.0
        
        # Distance from range boundaries
        dist_from_high = (current_price - price_high) / price_range
        dist_from_low = (price_low - current_price) / price_range
        
        # Breakout signals
        if dist_from_high > self.breakout_threshold:
            # Upward breakout
            signal = min(dist_from_high * 5, 1.0)
        elif dist_from_low > self.breakout_threshold:
            # Downward breakout
            signal = -min(dist_from_low * 5, 1.0)
        else:
            # No breakout
            signal = 0.0
        
        return np.clip(signal, -1.0, 1.0)