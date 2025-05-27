import argparse
import sys

sys.path.append("../")

import pandas as pd
import numpy as np
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from envs.feature_engineering import FeatureEngineer 
from gym_trading_env.environments import TradingEnv

import warnings
from envs.env import make_env
warnings.filterwarnings("ignore", category=ResourceWarning)


if __name__ == "__main__":
  # 参数解析
  parser = argparse.ArgumentParser()
  parser.add_argument('--symbol', type=str, default='300604',
                      help='股票代码')
  
  args = parser.parse_args()
  symbol = args.symbol  
  env = make_env(symbol,window_size=6, eval=False)
  for _ in range(10):
    terminated, truncated = False, False
    observation, info = env.reset()
    while not terminated  or not truncated:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        # print(observation,info)
        # print(env._features_columns)
  # Save for render
  # env.save_for_render()  