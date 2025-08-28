from typing import Dict, Any

class MetaAlgorithm:
    def __init__(self, experts):
        self.experts = experts  # list of Expert instances
        self.weights = {e.name: 1.0 / len(experts) for e in experts}

    def aggregate(self, predictions: Dict[str, float]) -> float:
        raise NotImplementedError

    def update(self, predictions: Dict[str, float], outcome: Any, **kwargs):
        raise NotImplementedError
