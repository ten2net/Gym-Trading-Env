import argparse
import sys
sys.path.append("./src")

import pandas as pd
import numpy as np
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from basic_model.utils.feature_engineering import FeatureEngineer 
from gym_trading_env.environments import TradingEnv

import warnings
warnings.filterwarnings("ignore", category=ResourceWarning)

# Create your own reward function with the history object
def reward_function(history): 
    log_return = np.log(history["portfolio_valuation", -1] / history["portfolio_valuation", -2])  #log (p_t / p_t-1 )
    return np.clip(log_return, -0.002,0.005) 
    # return max(0,(history["portfolio_valuation", -1] -history["portfolio_valuation", -2] )/ history["portfolio_valuation", -2])

def dynamic_feature_last_position_taken(history):
    return history['position', -1]

def dynamic_feature_real_position(history):
    return history['real_position', -1]

def make_env(symbol:str,window_size: int | None =24, eval: bool = False):
    df = pd.read_csv(f"../raw_data/csv/m5/{symbol}.csv", parse_dates=["date"], index_col= "date")
    df.sort_index(inplace= True)
    df.dropna(inplace= True)
    df.drop_duplicates(inplace=True)

    # Generating features
    # WARNING : the column names need to contain keyword 'feature' !
    df["feature_close"] = 100 * df["close"].pct_change()
    df["feature_open"] = df["open"]- df["close"] / (df["open"] + df["close"])
    df["feature_high"] = df["high"]- df["close"] / (df["high"] + df["close"])
    df["feature_low"] = df["low"]- df["close"] / (df["low"] + df["close"])
    df['dt'] = df.index.date
    # 2. 获取每日开盘价
    daily_open = df.groupby('dt')['open'].transform('first')
    # 3. 将每日开盘价合并回原始数据框
    df = df.merge(daily_open.rename('daily_open'), 
                  left_on='date', 
                  right_index=True)   
    df['feature_close_open_yoy'] = df['close'] - df['daily_open'] / (df['close'] + df['daily_open'])
    # df["feature_volume"] = df["volume"] / df["volume"].rolling(12).max()
    points_per_day = 48  # 24小时*60分钟/5分钟=288，但实际交易时间可能更少
    df['close_prev'] = df['close'].shift(points_per_day)
    df['volume_prev'] = df['volume'].shift(points_per_day)
    df['cum_volume'] = df.groupby('dt')['volume'].cumsum()
    df['cum_volume_prev'] = df["cum_volume"].shift(points_per_day)
    
    df['feature_close_yoy'] = (df['close'] - df['close_prev']) / (df['close'] + df['close_prev'])
    df['feature_volume_sum'] = (df['cum_volume'] - df['cum_volume_prev']) / (df['cum_volume'] + df['cum_volume_prev'])
    df['feature_volume'] = (df['volume'] - df['volume_prev']) / (df['volume'] + df['volume_prev'])
    df = df.drop(columns=['dt','daily_open', 'volume_prev', 'cum_volume', 'cum_volume_prev'])
    # print(df[-50:])
    fe =FeatureEngineer(window_size=3)
    df = fe.compute_features(df)
    numeric_cols = df.columns
    for col in numeric_cols:
        if col.startswith("feature"):
            df[col] = df[col].round(3)
    df.dropna(inplace= True)  
    df.to_csv(f"{symbol}.csv", index=True, encoding='utf_8_sig')
    env =  gym.make(
          "TradingEnv",
          name= symbol,
          df = df,
          windows= window_size,
        #   positions = [ -1, -0.5, 0, 0.5, 1, 1.5, 2], # From -1 (=SHORT), to +1 (=LONG)
          positions = [0, 0.5, 1], # From -1 (=SHORT), to +1 (=LONG)
          initial_position = 'random', #Initial position
          trading_fees = 0.01/100, # 0.01% per stock buy / sell
          reward_function = reward_function,
          # dynamic_feature_functions = [dynamic_feature_last_position_taken, dynamic_feature_real_position],
          portfolio_initial_value = 1000000, # in FIAT (here, USD)
          max_episode_duration = 1 , # "max" ,# 500,
          disable_env_checker= True
      )

    # env.add_metric('Position Changes', lambda history : np.sum(np.diff(history['position']) != 0) )
    # env.add_metric('Episode Lenght', lambda history : len(history['position']) )
    # env.add_metric('Reward sum', lambda history : f"{np.sum(history["reward"]):.3f}")
    # env.add_metric('Reward svg', lambda history : f"{np.sum(history["reward"]) / len(history['position']):.4f}") 
    env.add_metric('valuation', lambda history : f"{history['portfolio_valuation',-1]:.1f}") 
    # env.add_metric('pos_index', lambda history : f"{history['position_index',-1]}") 
    # env.add_metric('pos', lambda history : f"{history['position',-1]}") 
    env.add_metric('asset', lambda history : f"{history['portfolio_distribution_asset',-1]:.1f}") 
    env.add_metric('fiat', lambda history : f"{history['portfolio_distribution_fiat',-1]:.1f}") 

    
    eval_env = Monitor(env, './eval_logs')  
    return env if not eval else eval_env

if __name__ == "__main__":
  # 参数解析
  parser = argparse.ArgumentParser()
  parser.add_argument('--symbol', type=str, default='300604',
                      help='股票代码')
  
  args = parser.parse_args()
  symbol = args.symbol  
  for _ in range(1):
    done, truncated = False, False
    env = make_env(symbol,eval=False)
    observation, info = env.reset()
    while not done or not truncated:
    # while not truncated :
        action = env.action_space.sample()
        observation, reward, done, truncated, info = env.step(action)
        # print(observation,info)
        # print(env._features_columns)
  # Save for render
  # env.save_for_render()  