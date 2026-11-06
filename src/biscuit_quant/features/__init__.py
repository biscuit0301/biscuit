"""Causal features computed through the observation date only."""

from biscuit_quant.data import Bar


def build_features(bars: list[Bar], lookback: int) -> list[tuple[float, float]]:
    if type(lookback) is not int or lookback < 1:
        raise ValueError("lookback must be a positive integer")
    result = []
    for i, bar in enumerate(bars):
        daily_return = bar.close / bars[i - 1].close - 1 if i else 0.0
        momentum = bar.close / bars[i - lookback].close - 1 if i >= lookback else 0.0
        result.append((daily_return, momentum))
    return result
