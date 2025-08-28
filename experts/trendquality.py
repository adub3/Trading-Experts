from .base import Expert
import pandas as pd
import numpy as np

class TrendQualityExpert(Expert):
    """Continuous trend signal weighted by trend quality"""
    def __init__(self, short_window=10, long_window=50, quality_window=20):
        super().__init__(f"TrendQual_s{short_window}_l{long_window}_q{quality_window}")
        self.short_window = short_window
        self.long_window = long_window
        self.quality_window = quality_window
    
    def predict(self, data):
        if len(data) < self.long_window + self.quality_window:
            return 0.0
        
        prices = data['Close']
        ma_short = prices.rolling(window=self.short_window).mean()
        ma_long = prices.rolling(window=self.long_window).mean()
        
        # Trend direction strength
        trend_strength = (ma_short.iloc[-1] - ma_long.iloc[-1]) / ma_long.iloc[-1]
        
        # Trend consistency (quality measure)
        trend_up = (ma_short > ma_long)
        trend_consistency = trend_up.rolling(window=self.quality_window).mean().iloc[-1]
        
        # Convert consistency to quality weight (0.5 = no trend, 1.0 = perfect trend)
        quality_weight = 2 * abs(trend_consistency - 0.5)
        
        # Direction signal (-1 to 1) weighted by quality
        direction_signal = np.tanh(trend_strength * 10)  # tanh for smooth continuous signal
        return direction_signal * quality_weight
    