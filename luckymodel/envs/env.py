import sys
sys.path.append("../")
from envs.feature_engineering import FeatureEngineer
import warnings
from gym_trading_env.environments import TradingEnv
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Optional, List, Literal
import argparse



warnings.filterwarnings("ignore", category=ResourceWarning)

def calculate_reward(
    current_value: float,
    prev_value: float,
    step: int,
    max_steps: int = 480,
    target_profit: float = 0.15,
    stop_loss: float = -0.1,
    consecutive_ups: int = 0,
    consecutive_downs: int = 0
) -> tuple[float, bool, bool]:
    """
    强化学习交易策略的奖励计算函数
    该奖励函数用于5分钟的OHLCV数据训练PPO模型，我发现训练到1m步后，平均奖励似乎不再震荡爬升，且震荡剧烈，如何抑制震荡，且回合平均步数在440到470区间震荡，最大步数是480。如何在不对奖励结构做大调整的情况下，抑制震荡，让平均奖励趋势上升直到收敛，让回合平均步数趋势下降也收敛到较低水平    
    
    参数：
    - current_value: 当前资产净值（标准化后的值，初始为1.0）
    - prev_value: 前一步的资产净值
    - step: 当前步数（0-based）
    - max_steps: 最大允许步数
    
    返回：
    - reward: 当前步的奖励值
    - done: 是否终止当前回合
    """
    # ========== 参数配置 ==========
    TARGET_PROFIT = target_profit   # 目标收益率15%
    STOP_LOSS = stop_loss      # 最大亏损10%
    
    # 计算当前收益状态
    current_return = current_value - 1.0   # 标准化收益率
    prev_return = prev_value - 1.0  if prev_value is not None else 0.0
    momentum = current_return - prev_return
    # 更新连续上涨/下跌次数
    new_ups, new_downs = consecutive_ups, consecutive_downs    
    
    # 奖励系数配置    
    PROFIT_STEP_COEFF = 1.0
    LOSS_STEP_COEFF = 0.8 * PROFIT_STEP_COEFF    
    TARGET_BONUS_BASE = 25 * PROFIT_STEP_COEFF
    STOP_LOSS_PENALTY_BASE = -8 * LOSS_STEP_COEFF

    done, truncated = False, False
    reward = 0.0
    rewards = []
    
    # 动态进度感知衰减
    progress_ratio = step / max_steps
    DYNAMIC_COEFF = 1.8 / (1 + np.exp(6*(progress_ratio - 0.7)))  # 关键点0.7步时开始衰减
    # ----------------------------
    # 基础奖励
    # ----------------------------
    if current_return >= 0:
        # 计算收益进度变化
        curr_progress = current_return / TARGET_PROFIT
        prev_progress = prev_return / TARGET_PROFIT if prev_return >= 0 else 0.0
        # 引入加速奖励：后期进度变化奖励加倍
        reward += (curr_progress - prev_progress) *  PROFIT_STEP_COEFF * DYNAMIC_COEFF
    else:
        # 计算亏损进度变化（相对止损）
        curr_loss = current_return/STOP_LOSS 
        prev_loss =prev_return/STOP_LOSS if prev_return<0 else 0.0
        # 添加损失变化的平滑处理
        loss_change = min(curr_loss - prev_loss, 0.15)
        reward -= loss_change * LOSS_STEP_COEFF
    
    # 情况1：达到目标收益
    if current_return >= TARGET_PROFIT :
        # 增加时间奖励递减因子 
        time_bonus = TARGET_BONUS_BASE * (1 - 0.5*progress_ratio) 
        reward += time_bonus
        done = True
    # 情况2：触发止损
    elif current_return <= STOP_LOSS :
        # 动态止损惩罚：后期惩罚加倍
        reward += STOP_LOSS_PENALTY_BASE * (1 + progress_ratio*0.8)
        done = True
    # 情况3：达到最大步数
    elif step >= max_steps - 1 :  # 考虑0-based索引
        # 终点惩罚梯度重构
        if current_return > 0:
            # 进度奖励指数增长 
            reward += TARGET_BONUS_BASE * (current_return/TARGET_PROFIT)**2 * 2
        else:
            # 亏损惩罚立方增长
            reward += 3 * STOP_LOSS_PENALTY_BASE * (abs(current_return)/STOP_LOSS)**3         
        truncated = True
    else:         
        # ----------------------------
        # 趋势奖励（抑制震荡）趋势延续奖励可使后期训练更稳定 # 在上升趋势中​​保持持仓​​   # 在下跌趋势中​​加速止损​​    # 在震荡行情中​​减少无效交易
        # ----------------------------

        if momentum > 0:
            new_ups += 1
            new_downs = 0
        else: # 不增长视为下跌
            new_downs += 1
            new_ups = 0

        if new_ups > 0:
            reward +=  (1 - progress_ratio) ** new_ups
        if new_downs > 0:
            reward -=  (1 - progress_ratio) ** new_downs 

    return round(reward, 6), done, truncated, new_ups, new_downs


def dynamic_feature_last_position_taken(history):
    return history['position', -1]


def dynamic_feature_real_position(history):
    return history['real_position', -1]


def make_env(
    symbol: str,
    eval: bool = False,
    window_size: int | None = 24,
    positions: List[float] = [0, 0.5, 1],
    trading_fees: float = 0.01/100,  # 0.01% per stock buy / sell
    portfolio_initial_value: float = 1000000.0,  # in FIAT (here, YMB)
    max_episode_duration: int | Literal["max"] = 48 * 22,  # "max" ,# 500,
    target_return: float = 0.15,  # 策略目标收益率，超过视为成功完成，给予高额奖励
    stop_loss: float = -0.10,  # 止损，低于视为失败，给予惩罚
    render_mode: Literal["logs", "human"] = "logs",
    verbose: Literal[0, 1, 2] = 1
):
    df = pd.read_csv(
        f"../raw_data/csv/m5/{symbol}.csv", parse_dates=["date"], index_col="date")
    df.sort_index(inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    # Generating features
    # WARNING : the column names need to contain keyword 'feature' !
    df["feature_close"] = 100 * df["close"].pct_change()
    df["feature_open"] = df["open"] - df["close"] / (df["open"] + df["close"])
    df["feature_high"] = df["high"] - df["close"] / (df["high"] + df["close"])
    df["feature_low"] = df["low"] - df["close"] / (df["low"] + df["close"])
    df['dt'] = df.index.date
    # 2. 获取每日开盘价
    daily_open = df.groupby('dt')['open'].transform('first')
    # 3. 将每日开盘价合并回原始数据框
    df = df.merge(daily_open.rename('daily_open'),
                  left_on='date',
                  right_index=True)
    df['feature_close_open_yoy'] = df['close'] - \
        df['daily_open'] / (df['close'] + df['daily_open'])
    # df["feature_volume"] = df["volume"] / df["volume"].rolling(12).max()
    points_per_day = 48  # 24小时*60分钟/5分钟=288，但实际交易时间可能更少
    df['close_prev'] = df['close'].shift(points_per_day)
    df['volume_prev'] = df['volume'].shift(points_per_day)
    df['cum_volume'] = df.groupby('dt')['volume'].cumsum()
    df['cum_volume_prev'] = df["cum_volume"].shift(points_per_day)

    df['feature_close_yoy'] = (
        df['close'] - df['close_prev']) / (df['close'] + df['close_prev'])
    df['feature_volume_sum'] = (
        df['cum_volume'] - df['cum_volume_prev']) / (df['cum_volume'] + df['cum_volume_prev'])
    df['feature_volume'] = (df['volume'] - df['volume_prev']) / \
        (df['volume'] + df['volume_prev'])
    df = df.drop(columns=['dt', 'daily_open',
                 'volume_prev', 'cum_volume', 'cum_volume_prev'])
    # print(df[-50:])
    fe = FeatureEngineer(window_size=3)
    df = fe.compute_features(df)
    numeric_cols = df.columns
    for col in numeric_cols:
        if col.startswith("feature"):
            df[col] = df[col].round(3)
    df.dropna(inplace=True)
    df.to_csv(f"{symbol}.csv", index=True, encoding='utf_8_sig')
    env = gym.make(
        "TradingEnv",
        name=symbol,
        df=df,
        initial_position=0,  # 'random', #Initial position
        reward_function=calculate_reward,
        # dynamic_feature_functions = [dynamic_feature_last_position_taken, dynamic_feature_real_position],
        windows=window_size,
        positions=positions,
        trading_fees=trading_fees,
        portfolio_initial_value=portfolio_initial_value,
        max_episode_duration=max_episode_duration,
        target_return=target_return,
        stop_loss=stop_loss,
        disable_env_checker=True,
        render_mode=render_mode,
        verbose=verbose
    )

    # env.add_metric('调参次数', lambda history: np.sum(
    #     np.diff(history['real_position']) != 0))
    # env.add_metric(
    #     '奖励 sum', lambda history: f"{np.sum(history["reward"]):.4f}")
    # env.add_metric(
    #     '奖励 max', lambda history: f"{np.max(history["reward"]):.4f}")
    # env.add_metric(
    #     '奖励 min', lambda history: f"{np.min(history["reward"]):.4f}")
    # env.add_metric(
    #     '奖励 avg', lambda history: f"{np.mean(history["reward"]):.4f}")
    # env.add_metric(
    #     '奖励 median', lambda history: f"{np.median(history["reward"]):.5f}")
    # env.add_metric('最大回撤', lambda history: f"{cal_max_drawdown(history):.2f}")
    # env.add_metric('IDX', lambda history: f"{history['idx',0]}")
    # env.add_metric('IDXLAST', lambda history: f"{history['idx',-1]}")

    eval_env = Monitor(env, './eval_logs')
    return env if not eval else eval_env


if __name__ == "__main__":
    # 参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', type=str, default='300604',
                        help='股票代码')

    args = parser.parse_args()
    symbol = args.symbol
    env = make_env(symbol, window_size=5, eval=False)
    for _ in range(2):
        terminated, truncated = False, False
        observation, info = env.reset()
        while not terminated or not truncated:
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()
            # print(observation,info)
            # print(env._features_columns)
    # Save for render
    # env.save_for_render()
