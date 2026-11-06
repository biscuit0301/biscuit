"""Optional PPO training with chronological holdout, never a performance claim."""

import argparse
import hashlib
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from biscuit_quant.backtest import run_backtest
from biscuit_quant.cli import write_json
from biscuit_quant.config import load_config
from biscuit_quant.data import load_bars
from biscuit_quant.envs import TradingEnv
from biscuit_quant.envs.gymnasium import GymTradingEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/ppo"))
    parser.add_argument("--timesteps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.timesteps < 1:
        parser.error("timesteps must be positive")
    config = load_config(args.config)
    bars = load_bars(args.data)
    split = int(len(bars) * 0.8)
    if split < config.lookback + 2 or len(bars) - split < 2:
        parser.error("Insufficient data for 80/20 chronological split and feature warmup")
    train = GymTradingEnv(TradingEnv(bars[:split], config.execution, config.lookback))
    # Warmup ends at split - 1; first holdout fill occurs at split.
    test = TradingEnv(bars[split - config.lookback - 1 :], config.execution, config.lookback)
    check_env(train)
    args.output.mkdir(parents=True, exist_ok=False)
    model = PPO(
        "MlpPolicy", train, seed=args.seed, device="cpu", verbose=1, n_steps=128, batch_size=64
    )
    model.learn(total_timesteps=args.timesteps)
    model.save(args.output / "model")

    class Policy:
        def predict(self, observation):
            action, _ = model.predict(observation, deterministic=True)
            return float(action)

    write_json(args.output / "holdout_metrics.json", run_backtest(test, Policy()))
    write_json(
        args.output / "manifest.json",
        {
            "seed": args.seed,
            "requested_timesteps": args.timesteps,
            "actual_timesteps": model.num_timesteps,
            "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
            "train_last_date": bars[split - 1].date,
            "test_first_fill_date": bars[split].date,
            "versions": {
                p: version(p) for p in ["stable-baselines3", "gymnasium", "numpy", "torch"]
            },
            "config": asdict(config),
            "note": "Single seed holdout; no tuning on holdout or profitability claim.",
        },
    )


if __name__ == "__main__":
    main()
