from .base import Expert
import pandas as pd
import numpy as np

class DrawdownAvoidanceExpert(Expert):
    """Reduces signal strength during drawdowns"""
    def __init__(self, lookback=60):
        super().__init__(f"DrawdownAvoid_lb{lookback}")
        self.lookback = lookback
    
    def predict(self, data):
        if len(data) < self.lookback:
            return 0.0
        
        prices = data['Close']
        recent_prices = prices.tail(self.lookback)
        
        # Calculate current drawdown from recent high
        rolling_max = recent_prices.expanding().max()
        current_drawdown = (recent_prices.iloc[-1] - rolling_max.iloc[-1]) / rolling_max.iloc[-1]
        
        # Calculate drawdown duration (days since new high)
        days_since_high = 0
        for i in range(len(recent_prices)-1, -1, -1):
            if recent_prices.iloc[i] >= rolling_max.iloc[-1] * 0.999:  # Within 0.1% of high
                break
            days_since_high += 1
        
        # Signal strength decreases with drawdown severity and duration
        drawdown_penalty = abs(current_drawdown) * 2  # 0 to ~0.4 typically
        duration_penalty = min(days_since_high / 30, 1.0)  # 0 to 1 over 30 days
        
        # Base signal is slightly positive (small equity bias)
        base_signal = 0.2
        total_penalty = drawdown_penalty + duration_penalty * 0.5
        
        signal = base_signal * (1 - total_penalty)
        return np.clip(signal, -1.0, 1.0)
