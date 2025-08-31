from .base import Expert
import numpy as np

# Optional pandas import (safe if you already use pandas for data)
try:
    import pandas as pd
except Exception:
    pd = None


# --------- small helpers (robust to column naming) ---------
def _get_close(data):
    if pd is not None and isinstance(data, (pd.Series, pd.DataFrame)):
        if isinstance(data, pd.Series):
            return data
        for k in ["Close", "close", "Adj Close", "adj_close", "Adj_Close", "price"]:
            if k in data.columns:
                return data[k]
    # fallback: try dict-like
    for k in ["Close", "close", "Adj Close", "adj_close", "Adj_Close", "price"]:
        if isinstance(data, dict) and k in data:
            return data[k]
    raise ValueError("Could not locate a 'Close' price series in data.")


def _rolling_std(x, w):
    if pd is not None and isinstance(x, (pd.Series, pd.DataFrame)):
        return x.pct_change().rolling(w).std()
    # numpy fallback (expects 1D array)
    xr = np.diff(np.asarray(x), axis=0) / np.asarray(x[:-1])
    out = np.empty_like(x, dtype=float)
    out[:] = np.nan
    for i in range(w, len(xr)+1):
        out[i] = xr[i-w:i].std()
    return out


def _ema(x, span):
    if pd is not None and isinstance(x, (pd.Series, pd.DataFrame)):
        return x.ewm(span=span, adjust=False).mean()
    # simple numpy EMA fallback
    x = np.asarray(x, dtype=float)
    alpha = 2.0 / (span + 1.0)
    ema = np.empty_like(x)
    ema[0] = x[0]
    for i in range(1, len(x)):
        ema[i] = alpha * x[i] + (1 - alpha) * ema[i-1]
    return ema


def _pct_slope_norm(series, lag, vol):
    """Percent slope over 'lag' days, normalized by recent vol (std of returns)."""
    if pd is not None and isinstance(series, (pd.Series, pd.DataFrame)):
        s = (series / series.shift(lag) - 1.0)
        rv = vol
        z = s / (rv.replace(0, np.nan))
        return z
    # numpy fallback
    arr = np.asarray(series, dtype=float)
    s = np.empty_like(arr)
    s[:] = np.nan
    s[lag:] = arr[lag:] / arr[:-lag] - 1.0
    rv = np.asarray(vol, dtype=float)
    z = s / np.where(rv == 0, np.nan, rv)
    return z


def _clip(x, lo=-1.0, hi=1.0):
    return float(np.nan_to_num(np.clip(x, lo, hi), nan=0.0))


# ===========================================================
# 1) Long-Horizon Slope (200d EMA slope normalized by vol)
#     — captures gentle, persistent trends (slow drift)
# ===========================================================
class LongHorizonSlope200(Expert):
    def __init__(self, ema_span=200, lag=5, vol_win=20):
        super().__init__(f"LongHorizonSlope_{ema_span}")
        self.ema_span = ema_span
        self.lag = lag
        self.vol_win = vol_win

    def predict(self, data):
        close = _get_close(data)
        ema = _ema(close, self.ema_span)
        if pd is not None and isinstance(close, (pd.Series, pd.DataFrame)):
            rv20 = close.pct_change().rolling(self.vol_win).std()
            z = _pct_slope_norm(pd.Series(ema, index=close.index), self.lag, rv20)
            val = z.iloc[-1]
        else:
            rv20 = _rolling_std(close, self.vol_win)
            z = _pct_slope_norm(ema, self.lag, rv20)
            val = z[-1]
        return _clip(val)


# ===========================================================
# 2) Low-Vol Grind (trend-following only when vol is quiet)
#     — participates in calm, upward markets without overtrading
# ===========================================================
class LowVolGrind(Expert):
    def __init__(self, fast=50, slow=200, vol_win=20, vol_ref=252):
        super().__init__("LowVolGrind")
        self.fast = fast
        self.slow = slow
        self.vol_win = vol_win
        self.vol_ref = vol_ref

    def predict(self, data):
        close = _get_close(data)
        ema_fast = _ema(close, self.fast)
        ema_slow = _ema(close, self.slow)

        if pd is not None and isinstance(close, (pd.Series, pd.DataFrame)):
            ret = close.pct_change()
            rv = ret.rolling(self.vol_win).std()
            rv_ref = ret.rolling(self.vol_ref).std()
            # low-vol score in [0,1]
            lv = ((rv_ref - rv) / (rv_ref + 1e-12)).clip(lower=0.0, upper=1.0)
            slope = _pct_slope_norm(pd.Series(ema_fast, index=close.index), 5, rv).clip(lower=0.0)
            trend = (pd.Series(ema_fast, index=close.index) > pd.Series(ema_slow, index=close.index)).astype(float)
            score = (0.5 * lv + 0.5 * slope) * trend
            val = score.iloc[-1]
        else:
            rv = _rolling_std(close, self.vol_win)
            rv_ref = _rolling_std(close, self.vol_ref)
            lv = np.clip((rv_ref - rv) / (rv_ref + 1e-12), 0.0, 1.0)
            slope = _pct_slope_norm(ema_fast, 5, rv)
            slope = np.clip(slope, 0.0, None)
            trend = (np.asarray(ema_fast) > np.asarray(ema_slow)).astype(float)
            score = (0.5 * lv + 0.5 * slope) * trend
            val = score[-1]
        return _clip(val, 0.0, 1.0)  # long-only in quiet uptrends


# ===========================================================
# 3) Pullback-in-Uptrend (buy shallow dips within an uptrend)
#     — gentle mean reversion toward trend during healthy climbs
# ===========================================================
class PullbackInUptrend(Expert):
    def __init__(self, mid=100, trend=200, bb_win=20, bb_k=2.0):
        super().__init__("PullbackInUptrend")
        self.mid = mid
        self.trend = trend
        self.bb_win = bb_win
        self.bb_k = bb_k

    def predict(self, data):
        close = _get_close(data)
        ema_mid = _ema(close, self.mid)
        ema_trend = _ema(close, self.trend)

        if pd is not None and isinstance(close, (pd.Series, pd.DataFrame)):
            m = close.rolling(self.bb_win).mean()
            s = close.rolling(self.bb_win).std()
            upper = m + self.bb_k * s
            lower = m - self.bb_k * s
            width = (upper - lower).replace(0, np.nan)
            bb = ((close - lower) / width).clip(0, 1)
            uptrend = ((pd.Series(ema_trend, index=close.index).diff() > 0) &
                       (close > pd.Series(ema_mid, index=close.index)))
            # long signal grows as %b dips below 0.5 (pullback)
            sig = (0.5 - bb).clip(lower=0.0) * 2.0
            val = (sig * uptrend.astype(float)).iloc[-1]
        else:
            # numpy fallback (coarse)
            c = np.asarray(close, dtype=float)
            # simple rolling mean/std
            if len(c) < self.bb_win + 2:
                return 0.0
            m = pd.Series(c).rolling(self.bb_win).mean().values
            s = pd.Series(c).rolling(self.bb_win).std().values
            upper = m + self.bb_k * s
            lower = m - self.bb_k * s
            width = np.where((upper - lower) == 0, np.nan, (upper - lower))
            bb = np.clip((c - lower) / width, 0, 1)
            ema_mid = np.asarray(ema_mid)
            ema_tr = np.asarray(ema_trend)
            uptrend = ((np.diff(ema_tr, prepend=ema_tr[0]) > 0) & (c > ema_mid)).astype(float)
            sig = np.clip(0.5 - bb, 0, None) * 2.0
            val = (sig * uptrend)[-1]
        return _clip(val, 0.0, 1.0)  # long-only pullback buyer


# ===========================================================
# 4) Range Revert (Keltner/mean-reversion when trend is flat)
#     — handles sideways chop; reverts toward a midline
# ===========================================================
class RangeRevert(Expert):
    def __init__(self, mid=50, vol_win=20, flat_thresh=0.10, k=1.5):
        super().__init__("RangeRevert")
        self.mid = mid
        self.vol_win = vol_win
        self.flat_thresh = flat_thresh
        self.k = k

    def predict(self, data):
        close = _get_close(data)
        ema_mid = _ema(close, self.mid)

        if pd is not None and isinstance(close, (pd.Series, pd.DataFrame)):
            ret = close.pct_change()
            rv = ret.rolling(self.vol_win).std()
            slope = _pct_slope_norm(pd.Series(ema_mid, index=close.index), 5, rv)
            flat = (slope.abs() <= self.flat_thresh).astype(float)
            z = ((close / pd.Series(ema_mid, index=close.index)) - 1.0) / (rv + 1e-12)
            sig = (-z).clip(-1, 1) * flat  # revert toward mid when flat
            val = sig.iloc[-1]
        else:
            rv = _rolling_std(close, self.vol_win)
            ema_mid = np.asarray(ema_mid)
            slope = _pct_slope_norm(ema_mid, 5, rv)
            flat = (np.abs(slope) <= self.flat_thresh).astype(float)
            z = ((np.asarray(close) / ema_mid) - 1.0) / (rv + 1e-12)
            sig = np.clip(-z, -1, 1) * flat
            val = sig[-1]
        return _clip(val)


# ===========================================================
# 5) Streak Persistence (Beta-Bernoulli up-probability)
#     — captures modest “drift” via conditional up-prob
# ===========================================================
class StreakPersistence(Expert):
    def __init__(self, window=5, alpha=1.0, beta=1.0):
        super().__init__(f"StreakPersistence_w{window}")
        self.window = window
        self.alpha = alpha
        self.beta = beta

    def predict(self, data):
        close = _get_close(data)
        if pd is not None and isinstance(close, (pd.Series, pd.DataFrame)):
            ret = close.pct_change()
            if len(ret) < self.window + 1:
                return 0.0
            last = ret.iloc[-self.window:]
            ups = (last > 0).sum()
            p = (ups + self.alpha) / (self.window + self.alpha + self.beta)
            # confidence: mean |ret| vs recent vol
            rv = ret.rolling(self.window).std().iloc[-1]
            conf = float(min(1.0, (last.abs().mean() / (rv + 1e-12)) if rv == rv else 0.0))
        else:
            c = np.asarray(close, dtype=float)
            if len(c) < self.window + 1:
                return 0.0
            r = np.diff(c) / c[:-1]
            last = r[-self.window:]
            ups = int((last > 0).sum())
            p = (ups + self.alpha) / (self.window + self.alpha + self.beta)
            rv = last.std() if last.size > 1 else 0.0
            conf = float(min(1.0, (np.mean(np.abs(last)) / (rv + 1e-12)) if rv > 0 else 0.0))
        # map to [-1,1] with confidence scaling
        sig = (2.0 * p - 1.0) * conf
        return _clip(sig)
