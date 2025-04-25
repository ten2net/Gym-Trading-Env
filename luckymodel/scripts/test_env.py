import sys

sys.path.append("./src")

import pandas as pd
import numpy as np
import time
from gym_trading_env.environments import TradingEnv
import gymnasium as gym
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from stable_baselines3.common.utils import get_schedule_fn
from basic_model.utils.feature_engineering import FeatureEngineer  

import warnings
warnings.filterwarnings("ignore", category=ResourceWarning)

# Create your own reward function with the history object
def reward_function(history):
    
    return 100 * np.log(history["portfolio_valuation", -1] / history["portfolio_valuation", -2]) #log (p_t / p_t-1 )

def dynamic_feature_last_position_taken(history):
    return history['position', -1]

def dynamic_feature_real_position(history):
    return history['real_position', -1]

def make_env(symbol:str,eval: bool = False):
    df = pd.read_csv(f"../../../raw_data/csv/m5/{symbol}.csv", parse_dates=["date"], index_col= "date")
    df.sort_index(inplace= True)
    df.dropna(inplace= True)
    df.drop_duplicates(inplace=True)

    # Generating features
    # WARNING : the column names need to contain keyword 'feature' !
    df["feature_close"] = df["close"].pct_change()
    df["feature_open"] = df["open"]/df["close"]
    df["feature_high"] = df["high"]/df["close"]
    df["feature_low"] = df["low"]/df["close"]
    df["feature_volume"] = df["volume"] / df["volume"].rolling(7*24).max()
    
    fe =FeatureEngineer(window_size=20)
    df = fe.compute_features(df)
    # print(df.iloc[-1])
    numeric_cols = df.columns
    for col in numeric_cols:
        if col.startswith("feature"):
            df[col] = df[col].round(4)
    df.to_csv(f"{symbol}.csv", index=True, encoding='utf_8_sig')
    df.dropna(inplace= True)  
    env =  gym.make(
          "TradingEnv",
          name= symbol,
          df = df,
          windows= None,
          # positions = [ -1, -0.5, 0, 0.5, 1, 1.5, 2], # From -1 (=SHORT), to +1 (=LONG)
          positions = [0,  1], # From -1 (=SHORT), to +1 (=LONG)
          initial_position = 0, #'random', #Initial position
          trading_fees = 0.01/100, # 0.01% per stock buy / sell
          borrow_interest_rate= 0.0003/100, #per timestep (= 1h here)
          reward_function = reward_function,
          portfolio_initial_value = 10000, # in FIAT (here, USD)
          max_episode_duration = 1000,
          disable_env_checker= True
      )

    # env.add_metric('Position Changes', lambda history : np.sum(np.diff(history['position']) != 0) )
    # env.add_metric('Episode Lenght', lambda history : len(history['position']) )
    env.add_metric('Reward sum', lambda history : f"{np.sum(history["reward"]):.4f}")
    env.add_metric('Reward svg', lambda history : f"{np.sum(history["reward"]) / len(history['position']):.6f}") 
    
    eval_env = Monitor(env, './eval_logs')  
    return env if not eval else eval_env


# 测试模型
env = make_env("300059",eval=False)
model_file = "./ppo_trading_model.zip" 
model = PPO.load(model_file, env=env, device="cpu") 
obs, info = env.reset()
for _ in range(10000):
    action, _states = model.predict(obs)
    obs, reward, done, truncated, info = env.step(action)
    print(reward,info)
    env.render()  # 显示当前状态
    if done:
        break

