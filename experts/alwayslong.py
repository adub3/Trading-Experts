from .base import Expert

class AlwaysLong(Expert):
    def __init__(self):
        super().__init__("Always Long")

    def predict(self, data):
        return 1
