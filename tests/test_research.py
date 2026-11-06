import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from biscuit_quant.backtest import run_backtest
from biscuit_quant.cli import main
from biscuit_quant.config import ExecutionConfig, load_config
from biscuit_quant.data import Bar, load_bars, write_demo
from biscuit_quant.data.csv import validate_bars
from biscuit_quant.envs import TradingEnv
from biscuit_quant.features import build_features
from biscuit_quant.paper import replay
from biscuit_quant.strategies import BuyAndHold


def sample():
    return [Bar(f"2024-01-{i:02d}", 10, 11, 9, 10, 10000) for i in range(1, 9)]


class ResearchTests(unittest.TestCase):
    def test_features_do_not_use_future(self):
        bars = sample()
        before = build_features(bars, 2)
        bars[-1] = replace(bars[-1], close=11)
        self.assertEqual(before[:-1], build_features(bars, 2)[:-1])

    def test_next_open_fill_and_t_plus_one(self):
        bars = sample()
        bars[3] = replace(bars[3], open=11)
        env = TradingEnv(bars, ExecutionConfig(slippage_bps=0), 2)
        env.step(1)
        self.assertEqual(env.trades[0]["price"], 11)
        self.assertEqual(env.trades[0]["signal_date"], bars[2].date)
        env.step(0)
        self.assertEqual(env.trades[1]["date"], bars[4].date)
        self.assertEqual(env.shares, 0)

    def test_hand_calculated_fees_and_equity(self):
        cfg = ExecutionConfig(
            initial_cash=10005,
            commission_rate=0,
            minimum_commission=5,
            sell_tax_rate=0.001,
            slippage_bps=0,
        )
        env = TradingEnv(sample(), cfg, 2)
        env.step(1)
        self.assertEqual((env.cash, env.shares, env.equity), (0, 1000, 10000))
        env.step(0)
        self.assertEqual(env.cash, 9985)
        self.assertEqual(env.trades[-1]["fee"], 15)

    def test_slippage_and_lots(self):
        env = TradingEnv(sample(), ExecutionConfig(slippage_bps=100), 2)
        env.step(1)
        self.assertAlmostEqual(env.trades[0]["price"], 10.1)
        self.assertEqual(env.shares % 100, 0)
        self.assertGreaterEqual(env.cash, 0)

    def test_account_identity_and_reward_compounding(self):
        with tempfile.TemporaryDirectory() as tmp:
            bars = load_bars(write_demo(Path(tmp) / "data.csv"))
            env = TradingEnv(bars, ExecutionConfig(), 5)
            growth = 1.0
            while not env.done:
                _, reward, _, _ = env.step(float(env.index % 2))
                growth *= 1 + reward
                self.assertGreaterEqual(env.cash, 0)
                self.assertGreaterEqual(env.shares, 0)
                self.assertEqual(env.shares % env.config.lot_size, 0)
                self.assertAlmostEqual(env.equity, env.cash + env.shares * bars[env.index].close)
            self.assertAlmostEqual(growth, env.equity / env.config.initial_cash)

    def test_flat_cash_and_drawdown_metrics(self):
        class Flat:
            def predict(self, observation):
                return 0.0

        env = TradingEnv(sample(), ExecutionConfig(), 2)
        metrics = run_backtest(env, Flat())
        self.assertEqual(metrics["total_return"], 0)
        self.assertEqual(metrics["max_drawdown"], 0)
        self.assertEqual(metrics["fees_paid"], 0)

    def test_insufficient_cash_does_not_create_position(self):
        env = TradingEnv(sample(), ExecutionConfig(initial_cash=100), 2)
        self.assertEqual(env.step(1)[3]["status"], "hold")
        self.assertEqual(env.shares, 0)
        self.assertEqual(env.cash, 100)

    def test_blocked_and_suspended_buys(self):
        for change in [{"volume": 0}, {"can_buy": False}]:
            bars = sample()
            bars[3] = replace(bars[3], **change)
            env = TradingEnv(bars, ExecutionConfig(), 2)
            self.assertEqual(env.step(1)[3]["status"], "blocked")
            self.assertEqual(env.shares, 0)

    def test_blocked_sell_keeps_position(self):
        bars = sample()
        bars[4] = replace(bars[4], can_sell=False)
        env = TradingEnv(bars, ExecutionConfig(), 2)
        env.step(1)
        shares = env.shares
        self.assertEqual(env.step(0)[3]["status"], "blocked")
        self.assertEqual(env.shares, shares)

    def test_invalid_actions(self):
        env = TradingEnv(sample(), ExecutionConfig(), 2)
        for action in [-1, 2, float("nan"), float("inf"), True]:
            with self.assertRaises(ValueError):
                env.step(action)

    def test_terminal_and_reset(self):
        env = TradingEnv(sample(), ExecutionConfig(), 2)
        run_backtest(env, BuyAndHold())
        with self.assertRaises(RuntimeError):
            env.step(1)
        env.reset()
        self.assertEqual((env.shares, len(env.trades)), (0, 0))
        self.assertEqual(env.cash, env.config.initial_cash)

    def test_validation(self):
        bars = sample()
        for bad in [
            replace(bars[0], close=float("nan")),
            replace(bars[0], volume=-1),
            replace(bars[0], low=12),
        ]:
            with self.assertRaises(ValueError):
                validate_bars([bad])
        with self.assertRaises(ValueError):
            validate_bars([bars[0], bars[0]])
        with self.assertRaises(ValueError):
            TradingEnv(bars[:3], ExecutionConfig(), 2)
        with self.assertRaises(ValueError):
            ExecutionConfig(lot_size=1.5)

    def test_paper_matches_backtest(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = TradingEnv(sample(), ExecutionConfig(), 2)
            metrics = run_backtest(env, BuyAndHold())
            path = Path(tmp) / "ledger.jsonl"
            replay(env, BuyAndHold(), path)
            events = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(events[-1]["equity"], metrics["final_equity"])
            self.assertEqual(len(events), metrics["steps"])

    def test_demo_deterministic_and_csv_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a.csv", Path(tmp) / "b.csv"
            write_demo(a)
            write_demo(b)
            self.assertEqual(a.read_bytes(), b.read_bytes())
            self.assertEqual(len(load_bars(a)), 120)
            a.write_text("date,close\n2024-01-01,10\n")
            with self.assertRaises(ValueError):
                load_bars(a)

    def test_cli_artifacts_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            main(["demo", "--output", str(output)])
            self.assertTrue(json.loads((output / "manifest.json").read_text())["synthetic"])
            self.assertTrue((output / "trades.csv").exists())
            with self.assertRaises(FileExistsError):
                main(["demo", "--output", str(output)])

    def test_unknown_config_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.toml"
            path.write_text("[features]\nlookbak = 5\n")
            with self.assertRaises(ValueError):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
