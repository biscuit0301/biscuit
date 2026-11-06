"""Offline paper replay with an auditable event ledger; no broker connectivity."""

import json
from pathlib import Path

from biscuit_quant.envs import TradingEnv
from biscuit_quant.strategies import Policy


def replay(env: TradingEnv, policy: Policy, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    observation = env.reset()
    with path.open("w", encoding="utf-8") as handle:
        while not env.done:
            signal_date = env.bars[env.index].date
            target = policy.predict(observation)
            count = len(env.trades)
            observation, reward, _, info = env.step(target)
            event = {
                "mode": "offline_paper",
                "signal_date": signal_date,
                "target": target,
                "reward": reward,
                **info,
                "cash": env.cash,
                "shares": env.shares,
                "fills": env.trades[count:],
            }
            handle.write(json.dumps(event, allow_nan=False) + "\n")
