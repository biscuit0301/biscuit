"""Optional Gymnasium adapter. Install biscuit-quant[rl] to use it."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from biscuit_quant.envs.trading import TradingEnv


class GymTradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, core: TradingEnv):
        super().__init__()
        self.core = core
        # PPO chooses flat or fully invested; the core also supports fractional targets.
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.asarray(self.core.reset(), dtype=np.float32), {}

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError("Action must be 0 or 1")
        observation, reward, done, info = self.core.step(float(action))
        return np.asarray(observation, dtype=np.float32), reward, done, False, info
