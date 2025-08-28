# experts/__init__.py
from .base import Expert
from .alwayslong import AlwaysLong
from .alwaysshort import AlwaysShort
from .movingavg import MovingAverageCrossover
from .rsi import RelativeStrengthExpert as RSIExpert
from .momentum import MomentumExpert

# New quant finance experts
from .breakout import BreakoutExpert
from .drawdown import DrawdownAvoidanceExpert
from .meanreversion import MeanReversionExpert
from .priceaccel import PriceAccelerationExpert
from .trendquality import TrendQualityExpert
from .volatility import VolatilityRegimeExpert
from .volumeprice import VolumePriceExpert

# If you have a vol.py file (seems to be cut off in the screenshot)

__all__ = [
    "Expert",
    # Original experts
    "AlwaysLong",
    "AlwaysShort", 
    "MovingAverageCrossover",
    "RelativeStrengthExpert",
    "MomentumExpert",
    # New quant finance experts
    "BreakoutExpert",
    "DrawdownAvoidanceExpert",
    "MeanReversionExpert",
    "PriceAccelerationExpert",
    "TrendQualityExpert",
    "VolatilityRegimeExpert",
    "VolumePriceExpert",
]