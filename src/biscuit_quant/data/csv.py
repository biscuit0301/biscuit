import csv
import math
import random
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

FIELDS = ["date", "open", "high", "low", "close", "volume", "can_buy", "can_sell"]


@dataclass(frozen=True)
class Bar:
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    can_buy: bool = True
    can_sell: bool = True


def validate_bars(bars: list[Bar]) -> None:
    if not bars:
        raise ValueError("Empty market data")
    previous = None
    for bar in bars:
        day = date.fromisoformat(bar.date)
        if bar.date != day.isoformat() or (previous is not None and day <= previous):
            raise ValueError("Dates must be ISO dates, unique and strictly increasing")
        previous = day
        prices = [bar.open, bar.high, bar.low, bar.close]
        if any(not math.isfinite(x) or x <= 0 for x in prices):
            raise ValueError("Prices must be finite and positive")
        if not bar.low <= min(bar.open, bar.close) <= max(bar.open, bar.close) <= bar.high:
            raise ValueError("Inconsistent OHLC range")
        if not math.isfinite(bar.volume) or bar.volume < 0:
            raise ValueError("Volume must be finite and nonnegative")
        if type(bar.can_buy) is not bool or type(bar.can_sell) is not bool:
            raise ValueError("Trading flags must be boolean")


def load_bars(path: str | Path) -> list[Bar]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != FIELDS:
            raise ValueError(f"CSV columns must be exactly: {','.join(FIELDS)}")
        bars = []
        for line, row in enumerate(reader, 2):
            try:
                if row["can_buy"] not in {"0", "1"} or row["can_sell"] not in {"0", "1"}:
                    raise ValueError("Trading flags must be 0 or 1")
                bars.append(
                    Bar(
                        row["date"],
                        *(float(row[k]) for k in FIELDS[1:6]),
                        row["can_buy"] == "1",
                        row["can_sell"] == "1",
                    )
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid CSV row {line}: {exc}") from exc
    validate_bars(bars)
    return bars


def write_demo(path: str | Path, *, seed: int = 42, days: int = 120) -> Path:
    """Synthetic weekday bars only; no real security or exchange calendar."""
    if days < 2:
        raise ValueError("days must be at least 2")
    rng = random.Random(seed)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    day, close = date(2024, 1, 2), 10.0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDS)
        for _ in range(days):
            while day.weekday() >= 5:
                day += timedelta(days=1)
            opening = close * (1 + rng.uniform(-0.015, 0.015))
            close = opening * (1 + rng.uniform(-0.025, 0.025))
            writer.writerow(
                [
                    day.isoformat(),
                    f"{opening:.4f}",
                    f"{max(opening, close) * 1.01:.4f}",
                    f"{min(opening, close) * 0.99:.4f}",
                    f"{close:.4f}",
                    rng.randint(100_000, 1_000_000),
                    1,
                    1,
                ]
            )
            day += timedelta(days=1)
    return path
