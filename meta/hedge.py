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


from collections import deque
from typing import Dict, Optional
import math, random

from collections import deque
from typing import Dict, Optional
import math, random

def clip(x, lo=1e-6, hi=1-1e-6):
    return max(lo, min(hi, x))

class HedgeMetaEWMAProb(HedgeMeta):
    """
    - Proper probability scoring (log loss or Brier)
    - EWMA target (soft label) to track slow drifts
    - Confidence scaling: overconfident mistakes hurt more
    - Drift prior: mix expert probs toward base-rate trend
    - Per-step updates (uses EWMA instead of chunky windows)
    """
    def __init__(
        self,
        experts,
        eta: float = 0.25,
        alpha_fixed_share: float = 0.04,
        decay_lambda: float = 0.95,
        weight_floor: float = 1e-6,
        allow_specialists: bool = True,
        random_tie_break: bool = True,
        # scoring
        prob_loss: str = "log",         # 'log' or 'brier'
        conf_mode: str = "power",       # 'power'|'linear'|'none'
        conf_power: float = 2.0,
        conf_min: float = 0.0,
        # EWMA of labels and drift prior
        label_ewma_tau: float = 0.2,    # higher => faster to follow drift (0<tau<=1)
        drift_ewma_tau: float = 0.05,   # slow global base-rate tracker
        drift_mix_gamma: float = 0.15,  # mix strength toward drift prior (0..1)
        # weight momentum (inertia)
        weight_momentum: float = 0.2,   # 0=no inertia, 0.2–0.4 is common
        
    ):
        super().__init__(
            experts,
            eta=eta,
            alpha_fixed_share=alpha_fixed_share,
            decay_lambda=decay_lambda,
            weight_floor=weight_floor,
            allow_specialists=allow_specialists,
            random_tie_break=random_tie_break,
            loss="logistic",  # placeholder; we override scoring below
            loss_scale=1.0
        )
        self.prob_loss = prob_loss
        self.conf_mode = conf_mode
        self.conf_power = conf_power
        self.conf_min = conf_min

        # EWMA soft label y∈[0,1] and a slower drift prior
        self._y_ewma: Optional[float] = None
        self._drift_prob: Optional[float] = None
        self.label_ewma_tau = label_ewma_tau
        self.drift_ewma_tau = drift_ewma_tau
        self.drift_mix_gamma = drift_mix_gamma

        self.weight_momentum = weight_momentum
        self._prev_weights = dict(self.weights)

    def _conf_scale(self, pred: float) -> float:
        a = abs(pred)
        if self.conf_mode == "none":
            return 1.0
        if self.conf_mode == "linear":
            return max(self.conf_min, a)
        return max(self.conf_min, a ** self.conf_power)

    @staticmethod
    def _to_prob(pred: float) -> float:
        # map [-1,1] -> [0,1]
        return 0.5 * (pred + 1.0)

    @staticmethod
    def _log_loss(p: float, y: float) -> float:
        p = clip(p)
        return -(y * math.log(p) + (1 - y) * math.log(1 - p))

    @staticmethod
    def _brier(p: float, y: float) -> float:
        return (p - y) ** 2

    def aggregate(self, predictions: Dict[str, float]) -> int:
        # Same aggregation (weighted sign), but you can also do a probability vote:
        # p_meta = sum(w_i * p_i); d = 1 if p_meta>=0.5 else -1
        s = sum(self.weights[name] * predictions.get(name, 0.0) for name in self.weights)
        d = sign(s)
        if d == 0 and self.random_tie_break:
            d = random.choice([-1, 1])
        return d

    def update(
        self,
        predictions: Dict[str, float],
        outcome_sign: Optional[int] = None,      # {-1,+1}
        realized_return: Optional[float] = None, # ignored in prob mode; you can extend similarly
        active: Optional[Dict[str, bool]] = None
    ):
        if outcome_sign is None:
            raise ValueError("outcome_sign required")
        # y in {0,1}
        y = 1.0 if outcome_sign > 0 else 0.0 if outcome_sign < 0 else 0.5

        # 1) Update soft label EWMA and drift prior
        if self._y_ewma is None:
            self._y_ewma = y
        else:
            self._y_ewma = (1 - self.label_ewma_tau) * self._y_ewma + self.label_ewma_tau * y

        if self._drift_prob is None:
            self._drift_prob = y
        else:
            self._drift_prob = (1 - self.drift_ewma_tau) * self._drift_prob + self.drift_ewma_tau * y

        y_soft = self._y_ewma                       # target prob in [0,1]
        p_drift = self._drift_prob                  # slow trend prior

        # 2) Forgetting on cumulative losses (EMA over time)
        for k in self._cum_loss:
            self._cum_loss[k] *= self.decay

        # 3) Per-expert probability scoring with confidence scaling and drift mix
        for name in self._cum_loss:
            if self.allow_specialists and active is not None and not active.get(name, True):
                continue

            pred = predictions.get(name, 0.0)
            conf = self._conf_scale(pred)

            p_i = self._to_prob(pred)              # expert prob in [0,1]
            # mix toward drift prior to encourage alignment with gentle rises
            p_i = (1 - self.drift_mix_gamma) * p_i + self.drift_mix_gamma * p_drift
            p_i = clip(p_i)

            if self.prob_loss == "log":
                ell = self._log_loss(p_i, y_soft)
            else:
                ell = self._brier(p_i, y_soft)

            self._cum_loss[name] += conf * ell

        # 4) Multiplicative-weights update (stable)
        c = min(self._cum_loss.values())
        new_w = {k: math.exp(-self.eta * (self._cum_loss[k] - c)) for k in self._cum_loss}
        new_w = normalize_weights(new_w)

        # fixed-share (helps shifting experts)
        if self.alpha > 0.0:
            n = len(new_w)
            new_w = {k: (1 - self.alpha) * v + self.alpha / n for k, v in new_w.items()}

        # floor + renorm
        new_w = normalize_weights(new_w, self.weight_floor)

        # 5) Weight momentum (helps slow trends “stick”)
        if self.weight_momentum > 0.0:
            smoothed = {}
            for k in new_w:
                prev = self._prev_weights.get(k, new_w[k])
                smoothed[k] = (1 - self.weight_momentum) * new_w[k] + self.weight_momentum * prev
            new_w = normalize_weights(smoothed)

        self._prev_weights = dict(new_w)
        self.weights = new_w
