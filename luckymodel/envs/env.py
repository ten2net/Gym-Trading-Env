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

# Create your own reward function with the history object



def calculate_reward(
    current_value: float,
    prev_value: float,
    step: int,
    max_steps: int = 480,
    target_profit: float = 0.15,
    consecutive_ups: int = 0,
    consecutive_downs: int = 0
) -> tuple[float, bool, bool]:
    """
    强化学习交易策略的奖励计算函数
    
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
    STOP_LOSS = -0.10      # 最大亏损10%
    
    # 计算当前收益状态
    current_return = current_value - 1.0   # 标准化收益率
    prev_return = prev_value - 1.0  if prev_value is not None else 0.0
    
    # 奖励系数配置    
    PROFIT_STEP_COEFF = 1.0
    LOSS_STEP_COEFF = 3 * PROFIT_STEP_COEFF
    
    TARGET_BONUS = 60 * PROFIT_STEP_COEFF * (1 - (step/max_steps)**0.6)
    STOP_LOSS_PENALTY = -10 * LOSS_STEP_COEFF  * (step/max_steps)
    distance_to_target = (current_return - TARGET_PROFIT) / TARGET_PROFIT
    TIMEOUT_REWARD_COEFF = 0
    if distance_to_target > 0.6 :
       TIMEOUT_REWARD_COEFF = -1 * STOP_LOSS_PENALTY * 1/5 
    elif distance_to_target< -0.2 :
        TIMEOUT_REWARD_COEFF = STOP_LOSS_PENALTY
        
    done, truncated = False, False
    reward = 0.0

    # ----------------------------
    # 基础奖励计算（每步动态奖励）
    # ----------------------------
    if current_return >= 0:
        # 盈利区域：计算相对于目标的进度变化,添加进度限制防止过冲
        curr_progress = min(current_return / TARGET_PROFIT, 1.2)  # 允许20%超调
        prev_progress = min(prev_return / TARGET_PROFIT, 1.2) if prev_return >= 0 else 0.0
        reward += (curr_progress - prev_progress) * PROFIT_STEP_COEFF
    else:
        # 亏损区域：计算相对于止损线的进度变化
        curr_loss = min(abs(current_return)/abs(STOP_LOSS), 1.2)  # 允许120%亏损缓冲
        prev_loss = min(abs(prev_return)/abs(STOP_LOSS), 1.2) if prev_return < 0 else 0.0
        reward += abs((prev_loss - curr_loss)) * LOSS_STEP_COEFF  # 亏损扩大则惩罚

    # # 增强非线性特征，后期惩罚提升3倍
    # step_penalty = 0.025 * ((step/max_steps)**2.5)  # 指数从1.5→2.5
    # reward -= step_penalty * (2 if current_return < TARGET_PROFIT else 1)  # 未达标时双倍惩罚
    # ----------------------------
    # 终止条件判断
    # ----------------------------
    # 情况1：达到目标收益
    if current_return >= TARGET_PROFIT :
        # print(f"达到目标 {step} {current_return:.4f} {prev_return: .4f} {TARGET_PROFIT: .2f}  {STOP_LOSS: .2f}")
        time_decay = 0.5 + 0.5*(max_steps - step)/max_steps  # 越早完成效率越高
        reward += TARGET_BONUS * time_decay 
        done = True
    
    # 情况2：触发止损
    elif current_return <= STOP_LOSS  and step > max_steps * 0.05:
        # print(f"触发止损 {step} {current_return:.4f} {prev_return: .4f} {TARGET_PROFIT: .2f}  {STOP_LOSS: .2f}")
        time_decay = 0.3 + 0.7*(step/max_steps) # 越早触发惩罚越重
        reward += STOP_LOSS_PENALTY * time_decay
        done = True
    
    # 情况3：达到最大步数
    elif step >= max_steps :  # 考虑0-based索引
        # 根据最终状态计算处罚,加强超时惩罚（双系数机制）
        # print(f"达到最大步数 {step} {current_return:.4f} {prev_return: .4f} {TARGET_PROFIT: .2f}  {STOP_LOSS: .2f}")
        if current_return >= 0:
            reward += TIMEOUT_REWARD_COEFF * (current_return / TARGET_PROFIT)
        else:
            reward += TIMEOUT_REWARD_COEFF * (current_return / STOP_LOSS)
        truncated = True
        
    # 添加完成速度奖励
    if done and not truncated:
        speed_bonus = 200 * (1 - step/(max_steps*0.8))  # 前80%步数完成有额外奖励
        reward += max(speed_bonus, 0)        
    # ----------------------------
    # 添加趋势奖励（抑制震荡）趋势延续奖励可使后期训练更稳定
    # 该机制通过识别收益曲线的二阶导数（动量），能有效引导模型：
    # 在上升趋势中​​保持持仓​​
    # 在下跌趋势中​​加速止损​​
    # 在震荡行情中​​减少无效交易
    # ----------------------------
    momentum = current_return - prev_return
    momentum_coeff = 0.4 if abs(momentum)<0.005 else 0.8  # 小幅波动时降低敏感性
    if momentum > 0 :
        reward += momentum_coeff * momentum  # 上升趋势奖励
        if current_return >=0:
            consecutive_ups += 1
            reward += 0.15 * (consecutive_ups**0.5)  # 连续上涨奖励
            if consecutive_ups >= 3:
                reward += 0.5  # 降低持续奖励
    else:
        consecutive_ups = 0  
              
    if momentum <0:
        reward += momentum_coeff * momentum  # 下跌趋势惩罚  
        if current_return <0:
            consecutive_downs += 1
            reward -= 0.15 * (consecutive_downs**0.5)  # 连续下跌惩罚
            if consecutive_downs>=3:
                reward -= 0.5   
    else:
        consecutive_downs = 0   

    return round(reward, 6), done, truncated, consecutive_ups, consecutive_downs


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
    target_return: float = 0.3,  # 策略目标收益率，超过视为成功完成，给予高额奖励
    min_target_return: float = 0.05,  # 最小目标收益率，低于视为失败，给予惩罚
    max_drawdown: float = -0.3,
    daily_loss_limit: float = -0.1,
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
        min_target_return=min_target_return,
        max_drawdown=max_drawdown,
        daily_loss_limit=daily_loss_limit,
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
