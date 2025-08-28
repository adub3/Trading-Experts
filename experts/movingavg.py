from .base import Expert

class MovingAverageCrossover(Expert):
    def __init__(self, short_window=10, long_window=50):
        super().__init__(f"MA Crossover {short_window}/{long_window}")
        self.short = short_window
        self.long = long_window

    def predict(self, data):
        if len(data) < self.long:
            return 0
        short_ma = data["Close"].tail(self.short).mean()
        long_ma = data["Close"].tail(self.long).mean()
        return max(short_ma, long_ma)

