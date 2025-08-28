from .base import Expert

class AlwaysShort(Expert):
    def __init__(self):
        super().__init__("Always Short")

    def predict(self, data):
        return -1

