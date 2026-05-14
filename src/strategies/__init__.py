from .base import Signal, Strategy
from .ma_cross import MACrossStrategy
from .rsi import RSIStrategy
from .donchian import DonchianStrategy
from .oi_composite import OICompositeStrategy
from .smart_reversion import SmartReversionStrategy
from .alpha_scanner import AlphaScannerStrategy
from .trend_filter import TrendFilterStrategy

REGISTRY = {
    "ma_cross": MACrossStrategy,
    "rsi": RSIStrategy,
    "donchian": DonchianStrategy,
    "oi_composite": OICompositeStrategy,
    "smart_reversion": SmartReversionStrategy,
    "alpha_scanner": AlphaScannerStrategy,
    "trend_filter": TrendFilterStrategy,
}


def get_strategy(name: str, **kwargs) -> Strategy:
    if name not in REGISTRY:
        raise ValueError(f"Unknown strategy '{name}'. Available: {list(REGISTRY)}")
    return REGISTRY[name](**kwargs)
