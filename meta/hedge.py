from typing import Dict, Optional, Callable
import random
import math
from .base import MetaAlgorithm
from .utils import normalize_weights, sign, loss_sign_accuracy, loss_squared, loss_logistic, loss_neg_log_wealth

class HedgeMeta(MetaAlgorithm):
    def __init__(
        self,
        experts,
        eta: float = 0.3,                 # learning rate
        alpha_fixed_share: float = 0.02,   # 0 = off
        decay_lambda: float = 0.98,        # recency (0..1], lower = faster forgetting
        weight_floor: float = 0.0,         # e.g., 1e-6 to avoid collapse
        allow_specialists: bool = True,    # inactive experts not updated
        random_tie_break: bool = True,     # break 0 with random sign
        loss: str = "logistic",       # 'sign_accuracy' | 'squared' | 'logistic' | 'log_wealth'
        loss_scale: float = 1.0            # for logistic margin scaling
    ):
        super().__init__(experts)
        self.eta = eta
        self.alpha = alpha_fixed_share
        self.decay = decay_lambda
        self.weight_floor = weight_floor
        self.allow_specialists = allow_specialists
        self.random_tie_break = random_tie_break
        self.loss_name = loss
        self.loss_scale = loss_scale
        self._cum_loss = {e.name: 0.0 for e in experts}

        if loss not in {"sign_accuracy", "squared", "logistic", "log_wealth"}:
            raise ValueError("Unsupported loss.")

    def _loss_fn(self, pred: float, y: int, realized_return: Optional[float] = None) -> float:
        if self.loss_name == "sign_accuracy":
            return loss_sign_accuracy(pred, y)
        if self.loss_name == "squared":
            return loss_squared(pred, y)
        if self.loss_name == "logistic":
            return loss_logistic(pred, y, scale=self.loss_scale)
        # log_wealth expects realized_return
        if realized_return is None:
            raise ValueError("realized_return required for log_wealth loss")
        return loss_neg_log_wealth(pred, realized_return)

    def aggregate(self, predictions: Dict[str, float]) -> int:
        # weighted signal in [-1,1]; map to sign decision
        s = sum(self.weights[name] * predictions.get(name, 0.0) for name in self.weights)
        d = sign(s)
        if d == 0 and self.random_tie_break:
            d = random.choice([-1, 1])
        return d

    def update(
        self,
        predictions: Dict[str, float],
        outcome_sign: Optional[int] = None,      # {-1,+1} for direction-based losses
        realized_return: Optional[float] = None, # for log_wealth
        active: Optional[Dict[str, bool]] = None # optional specialist mask
    ):
        # decay cum loss
        for k in self._cum_loss:
            self._cum_loss[k] *= self.decay

        # per-expert instantaneous loss
        for name in self._cum_loss:
            if self.allow_specialists and active is not None and not active.get(name, True):
                continue
            pred = predictions.get(name, 0.0)
            if self.loss_name == "log_wealth":
                ell = self._loss_fn(pred, 0, realized_return=realized_return)
            else:
                if outcome_sign is None:
                    raise ValueError("outcome_sign required for non-wealth losses")
                ell = self._loss_fn(pred, outcome_sign)
            self._cum_loss[name] += ell

        # exponential weights
        max_neg = -min(self._cum_loss.values())  # for numerical stability
        new_w = {k: math.exp(-self.eta * (self._cum_loss[k])) for k in self._cum_loss}

        # fixed-share (mix a bit of uniform mass)
        new_w = normalize_weights(new_w)
        if self.alpha > 0.0:
            n = len(new_w)
            new_w = {k: (1 - self.alpha) * v + self.alpha / n for k, v in new_w.items()}

        # floor + renorm
        self.weights = normalize_weights(new_w, self.weight_floor)
