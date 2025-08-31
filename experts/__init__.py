# experts/__init__.py
from .base import Expert
from .alwayslong import AlwaysLong
from .alwaysshort import AlwaysShort
from .movingavg import MovingAverageCrossover
from .rsi import RelativeStrengthExpert
from .momentum import MomentumExpert

# Quant finance experts
from .breakout import BreakoutExpert
from .drawdown import DrawdownAvoidanceExpert
from .meanreversion import MeanReversionExpert
from .priceaccel import PriceAccelerationExpert
from .trendquality import TrendQualityExpert
from .volatility import VolatilityRegimeExpert
from .volumeprice import VolumePriceExpert

# New regime experts (all inside 831experts.py)
from .newexperts import (
    LongHorizonSlope200,
    LowVolGrind,
    PullbackInUptrend,
    RangeRevert,
    StreakPersistence,
)

__all__ = [
    "Expert",
    # Baseline experts
    "AlwaysLong",
    "AlwaysShort",
    "MovingAverageCrossover",
    "RelativeStrengthExpert",
    "MomentumExpert",
    # Quant finance experts
    "BreakoutExpert",
    "DrawdownAvoidanceExpert",
    "MeanReversionExpert",
    "PriceAccelerationExpert",
    "TrendQualityExpert",
    "VolatilityRegimeExpert",
    "VolumePriceExpert",
    # Regime-specific experts
    "LongHorizonSlope200",
    "LowVolGrind",
    "PullbackInUptrend",
    "RangeRevert",
    "StreakPersistence",
]
