"""Offline command line workflows and reproducibility records."""

import argparse
import csv
import hashlib
import json
import platform
from dataclasses import asdict
from pathlib import Path

from biscuit_quant import __version__
from biscuit_quant.backtest import run_backtest
from biscuit_quant.config import load_config
from biscuit_quant.data import load_bars, write_demo
from biscuit_quant.envs import TradingEnv
from biscuit_quant.paper import replay
from biscuit_quant.strategies import BuyAndHold, Momentum


def write_json(path: Path, data: dict) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Offline single-asset quant research")
    parser.add_argument("command", choices=["demo", "backtest", "paper"])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data", type=Path, help="CSV required for backtest and paper")
    parser.add_argument("--output", type=Path, default=Path("artifacts/demo"))
    parser.add_argument("--seed", type=int, default=42, help="Synthetic demo generator seed")
    args = parser.parse_args(argv)
    if args.command != "demo" and args.data is None:
        parser.error("--data is required for backtest and paper")
    if args.command == "demo" and args.data is not None:
        parser.error("demo generates synthetic data; use backtest for an input CSV")
    config = load_config(args.config)
    # Each run gets a new directory, so prior evidence cannot be silently overwritten.
    args.output.mkdir(parents=True, exist_ok=False)
    data_path = (
        write_demo(args.output / "synthetic.csv", seed=args.seed)
        if (args.command == "demo")
        else args.data
    )
    bars = load_bars(data_path)
    env = TradingEnv(bars, config.execution, config.lookback)
    policy = BuyAndHold() if config.strategy == "buy_hold" else Momentum(config.threshold)
    manifest = {
        "package_version": __version__,
        "python": platform.python_version(),
        "command": args.command,
        "data_path": str(data_path.resolve()),
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "config": asdict(config),
        "synthetic": args.command == "demo",
        "seed": args.seed if args.command == "demo" else None,
        "start_date": bars[config.lookback].date,
        "end_date": bars[-1].date,
    }
    write_json(args.output / "manifest.json", manifest)
    if args.command == "paper":
        replay(env, policy, args.output / "ledger.jsonl")
        print(f"Offline paper ledger: {args.output / 'ledger.jsonl'}")
        return
    metrics = run_backtest(env, policy)
    write_json(args.output / "metrics.json", metrics)
    for name, rows, fields in [
        ("equity.csv", env.history, ["date", "equity", "cash", "shares"]),
        ("trades.csv", env.trades, ["date", "signal_date", "side", "quantity", "price", "fee"]),
    ]:
        with (args.output / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(metrics, indent=2, allow_nan=False))
