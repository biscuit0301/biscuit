"""Single asset, daily, long-only, cash-account research environment."""

import math

from biscuit_quant.config import ExecutionConfig
from biscuit_quant.data import Bar
from biscuit_quant.data.csv import validate_bars
from biscuit_quant.features import build_features


class TradingEnv:
    """Observe close[t], submit target exposure, fill open[t+1], mark close[t+1].

    One rebalance per session makes today's purchases sellable from the next
    session only. No leverage, shorting, intraday trading or live broker.
    """

    def __init__(self, bars: list[Bar], config: ExecutionConfig, lookback: int = 5):
        validate_bars(bars)
        self.features = build_features(bars, lookback)
        if len(bars) <= lookback + 1:
            raise ValueError("Need at least lookback + 2 bars")
        self.bars, self.config, self.lookback = list(bars), config, lookback
        self.reset()

    def reset(self) -> tuple[float, ...]:
        self.index = self.lookback
        self.cash, self.shares = self.config.initial_cash, 0
        self.equity = self.cash
        self.done = False
        self.trades: list[dict] = []
        self.history = [
            {
                "date": self.bars[self.index].date,
                "equity": self.equity,
                "cash": self.cash,
                "shares": self.shares,
            }
        ]
        return self.observation()

    def observation(self) -> tuple[float, ...]:
        position = self.shares * self.bars[self.index].close / self.equity
        return (*self.features[self.index], position, self.cash / self.equity)

    def _fee(self, notional: float, sell: bool) -> float:
        cfg = self.config
        return max(cfg.minimum_commission, notional * cfg.commission_rate) + (
            notional * cfg.sell_tax_rate if sell else 0.0
        )

    def step(self, target: float) -> tuple[tuple[float, ...], float, bool, dict]:
        if self.done:
            raise RuntimeError("Episode complete; call reset")
        if isinstance(target, bool) or not math.isfinite(target) or not 0 <= target <= 1:
            raise ValueError("Target exposure must be finite and in [0, 1]")
        bar, cfg = self.bars[self.index + 1], self.config
        previous_equity = self.equity
        opening_equity = self.cash + self.shares * bar.open
        desired = int(target * opening_equity / bar.open / cfg.lot_size) * cfg.lot_size
        delta = desired - self.shares
        reason = "hold"
        if delta and (
            bar.volume == 0 or (delta > 0 and not bar.can_buy) or (delta < 0 and not bar.can_sell)
        ):
            reason = "blocked"
        elif delta:
            sell = delta < 0
            price = bar.open * (1 + (-1 if sell else 1) * cfg.slippage_bps / 10_000)
            quantity = abs(delta)
            if not sell:
                # Bound by available cash, then account for the minimum commission.
                quantity = min(quantity, int(self.cash / price / cfg.lot_size) * cfg.lot_size)
                while (
                    quantity and quantity * price + self._fee(quantity * price, False) > self.cash
                ):
                    quantity -= cfg.lot_size
            if quantity:
                notional = quantity * price
                fee = self._fee(notional, sell)
                # Extreme fee assumptions must never turn a cash account negative.
                if sell and self.cash + notional < fee:
                    reason = "unaffordable_fee"
                else:
                    self.cash += notional - fee if sell else -notional - fee
                    self.shares += -quantity if sell else quantity
                    reason = "filled"
                    self.trades.append(
                        {
                            "date": bar.date,
                            "signal_date": self.bars[self.index].date,
                            "side": "sell" if sell else "buy",
                            "quantity": quantity,
                            "price": price,
                            "fee": fee,
                        }
                    )
            else:
                reason = "insufficient_cash"
        self.index += 1
        self.equity = self.cash + self.shares * bar.close
        self.done = self.index == len(self.bars) - 1
        reward = self.equity / previous_equity - 1
        self.history.append(
            {"date": bar.date, "equity": self.equity, "cash": self.cash, "shares": self.shares}
        )
        return (
            self.observation(),
            reward,
            self.done,
            {"date": bar.date, "status": reason, "equity": self.equity},
        )
