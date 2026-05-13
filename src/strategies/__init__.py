from .base import Signal, Strategy
from .ma_cross import MACrossStrategy
from .rsi import RSIStrategy
from .donchian import DonchianStrategy
from .oi_composite import OICompositeStrategy
from .smart_reversion import SmartReversionStrategy

REGISTRY = {
    "ma_cross": MACrossStrategy,
    "rsi": RSIStrategy,
    "donchian": DonchianStrategy,
    "oi_composite": OICompositeStrategy,
    "smart_reversion": SmartReversionStrategy,
}


def get_strategy(name: str, **kwargs) -> Strategy:
    if name not in REGISTRY:
        raise ValueError(f"Unknown strategy '{name}'. Available: {list(REGISTRY)}")
    return REGISTRY[name](**kwargs)
