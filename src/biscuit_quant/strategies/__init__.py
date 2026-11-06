"""Strategies consume observations only, never future bars."""

from typing import Protocol


class Policy(Protocol):
    def predict(self, observation: tuple[float, ...]) -> float: ...


class BuyAndHold:
    def predict(self, observation: tuple[float, ...]) -> float:
        return 1.0


class Momentum:
    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold

    def predict(self, observation: tuple[float, ...]) -> float:
        return float(observation[1] > self.threshold)
