import argparse
import sys
sys.path.append("./src")

import pandas as pd
import numpy as np
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from basic_model.utils.feature_engineering import FeatureEngineer 
from gym_trading_env.environments_V1 import TradingEnv

import warnings
warnings.filterwarnings("ignore", category=ResourceWarning)

# Create your own reward function with the history object
def reward_function(history): 
    log_return = 100 * np.log(history["portfolio_valuation", -1] / history["portfolio_valuation", -2])  #log (p_t / p_t-1 )
    return np.clip(log_return, -0.5,0.5) 
    # return max(0,(history["portfolio_valuation", -1] -history["portfolio_valuation", -2] )/ history["portfolio_valuation", -2])
 
def one_step_reward(history, 
                    alpha=2.0, 
                    beta=3.0, 
                    gamma=0.5,
                    max_clip=5.0):
    """
    非对称指数奖励函数
    :param portfolio_return: 投资组合收益率（相对于初始值）
    :param base_return: 基准收益率阈值（例如1.0表示保本）
    :param alpha: 正向收益放大系数
    :param beta: 负向风险惩罚系数
    :param gamma: 收益饱和系数（控制高收益区间的奖励增长速度）
    :param max_clip: 奖励最大值截断（防止数值溢出）
    """
   
    excess_return = 100 * np.log(history["portfolio_valuation", -1] / history["portfolio_valuation", -2])
    
    if excess_return >= 0:
        # 正向收益：使用修正指数函数控制增长
        reward = np.sign(excess_return) * (np.abs(excess_return)**gamma) * alpha
    else:
        # 负向亏损：使用指数惩罚
        reward = -np.exp(beta * np.abs(excess_return)) + 1
    
    # 数值截断
    return np.clip(reward, -max_clip, max_clip)   
  

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
          initial_position = 0, # 'random', #Initial position
          trading_fees = 0.01/100, # 0.01% per stock buy / sell
          reward_function = one_step_reward, #reward_function,
          # dynamic_feature_functions = [dynamic_feature_last_position_taken, dynamic_feature_real_position],
          portfolio_initial_value = 1000000, # in FIAT (here, USD)
          max_episode_duration = 48 * 5 , # "max" ,# 500,  
          target_return=0.15,
          max_drawdown=-0.05,
          daily_loss_limit=-0.03,        
          disable_env_checker= True,
          render_mode="logs",
          verbose=1          
      )

    env.add_metric('Position Changes', lambda history : np.sum(np.diff(history['real_position']) != 0) )
    env.add_metric('Episode Lenght', lambda history : len(history['real_position']) )
    env.add_metric('Reward sum', lambda history : f"{np.sum(history["reward"]):.5f}")
    env.add_metric('Reward svg', lambda history : f"{np.sum(history["reward"]) / len(history['real_position']):.5f}") 
    # env.add_metric('valuation', lambda history : f"{history['portfolio_valuation',-1]:.1f}") 
    # env.add_metric('pos_index', lambda history : f"{history['position_index',-1]}") 
    # env.add_metric('pos', lambda history : f"{history['position',-1]}") 
    # env.add_metric('asset', lambda history : f"{history['portfolio_distribution_asset',-1]:.1f}") 
    # env.add_metric('fiat', lambda history : f"{history['portfolio_distribution_fiat',-1]:.1f}") 

    
    eval_env = Monitor(env, './eval_logs')  
    return env if not eval else eval_env

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