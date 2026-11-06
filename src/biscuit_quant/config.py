"""Validated, immutable experiment configuration."""

import math
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExecutionConfig:
    initial_cash: float = 100_000.0
    lot_size: int = 100
    commission_rate: float = 0.0003
    minimum_commission: float = 5.0
    sell_tax_rate: float = 0.0005
    slippage_bps: float = 5.0

    def __post_init__(self):
        for key, value in asdict(self).items():
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid execution parameter: {key}")
        if self.initial_cash <= 0 or type(self.lot_size) is not int or self.lot_size < 1:
            raise ValueError("initial_cash must be positive; lot_size must be a positive integer")
        if self.slippage_bps >= 10_000:
            raise ValueError("slippage_bps must be below 10000")


@dataclass(frozen=True)
class ResearchConfig:
    execution: ExecutionConfig = ExecutionConfig()
    lookback: int = 5
    strategy: str = "momentum"
    threshold: float = 0.0

    def __post_init__(self):
        if type(self.lookback) is not int or self.lookback < 1:
            raise ValueError("lookback must be a positive integer")
        if self.strategy not in {"momentum", "buy_hold"}:
            raise ValueError("strategy must be momentum or buy_hold")
        if not math.isfinite(self.threshold):
            raise ValueError("threshold must be finite")


def load_config(path: str | Path | None = None) -> ResearchConfig:
    if path is None:
        return ResearchConfig()
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)
    unknown = set(raw) - {"execution", "features", "strategy"}
    if unknown:
        raise ValueError(f"Unknown config sections: {sorted(unknown)}")
    for section, allowed in [("features", {"lookback"}), ("strategy", {"name", "threshold"})]:
        if set(raw.get(section, {})) - allowed:
            raise ValueError(f"Unknown keys in {section}")
    return ResearchConfig(
        execution=ExecutionConfig(**raw.get("execution", {})),
        lookback=raw.get("features", {}).get("lookback", 5),
        strategy=raw.get("strategy", {}).get("name", "momentum"),
        threshold=raw.get("strategy", {}).get("threshold", 0.0),
    )
