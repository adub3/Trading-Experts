import math
from typing import Dict

def normalize_weights(w: Dict[str, float], floor: float = 0.0) -> Dict[str, float]:
    if floor > 0:
        for k in w:
            w[k] = max(w[k], floor)
    s = sum(w.values())
    if s == 0:
        n = len(w)
        return {k: 1.0 / n for k in w}
    return {k: v / s for k, v in w.items()}

def sign(x: float) -> int:
    return 1 if x > 0 else (-1 if x < 0 else 0)

# losses expect outcome in {-1,+1}; predictions in [-1,1] or {-1,0,1}
def loss_sign_accuracy(pred: float, y: int) -> float:
    p = sign(pred)
    if p == 0: return 0.5
    return 0.0 if p == y else 1.0

def loss_squared(pred: float, y: int) -> float:
    # map y∈{-1,+1} to [-1,1] and square error
    return (y - pred) ** 2

def loss_logistic(pred: float, y: int, scale: float = 1.0) -> float:
    # logistic loss on margin y*pred
    return math.log1p(math.exp(-scale * y * pred))

def loss_neg_log_wealth(position: float, realized_return: float, eps: float = 1e-6) -> float:
    # penalizes by -log(1 + position * r)
    x = 1.0 + position * realized_return
    return -math.log(max(x, eps))
