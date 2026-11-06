import importlib.util
import unittest

from test_research import sample


@unittest.skipUnless(importlib.util.find_spec("stable_baselines3"), "Install [rl] extra")
class GymTests(unittest.TestCase):
    def test_gym_contract_and_ppo_smoke(self):
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_checker import check_env

        from biscuit_quant.config import ExecutionConfig
        from biscuit_quant.envs import TradingEnv
        from biscuit_quant.envs.gymnasium import GymTradingEnv

        env = GymTradingEnv(TradingEnv(sample(), ExecutionConfig(), 2))
        check_env(env)
        model = PPO("MlpPolicy", env, n_steps=8, batch_size=8, seed=42, device="cpu")
        model.learn(total_timesteps=16)
        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        self.assertIn(int(action), [0, 1])
