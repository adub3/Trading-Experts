#regression based momentum expert

import numpy as np
from .base import Expert

class MomentumExpert(Expert):
    def __init__(self, lookback=20):
        super().__init__(f"Momentum {lookback}d")
        self.lookback = lookback

    def predict(self, data):
        if len(data) < self.lookback:
            return 0
        prices = data["Close"].iloc[-self.lookback:]
        log_prices = np.log(prices.values)
        x = np.arange(len(log_prices))
        slope = np.polyfit(x, log_prices, 1)[0]  # regression slope
        return slope
