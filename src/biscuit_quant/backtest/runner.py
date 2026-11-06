from biscuit_quant.envs import TradingEnv
from biscuit_quant.strategies import Policy


def run_backtest(env: TradingEnv, policy: Policy) -> dict:
    observation = env.reset()
    while not env.done:
        observation, _, _, _ = env.step(policy.predict(observation))
    peak, max_drawdown = env.config.initial_cash, 0.0
    for row in env.history:
        peak = max(peak, row["equity"])
        max_drawdown = max(max_drawdown, 1 - row["equity"] / peak)
    return {
        "initial_equity": env.config.initial_cash,
        "final_equity": env.equity,
        "total_return": env.equity / env.config.initial_cash - 1,
        "max_drawdown": max_drawdown,
        "trade_count": len(env.trades),
        "fees_paid": sum(trade["fee"] for trade in env.trades),
        "steps": len(env.history) - 1,
        "terminal_position": env.shares,
        "valuation": "mark_to_market_no_forced_liquidation",
    }
