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
    history,
    current_price: float,
    current_value: float,
    step: int = 0,
    max_steps: int = 480,
    target_profit: float = 0.15,
    stop_loss: float = 0.1,
    consecutive_ups: int = 0,
    consecutive_downs: int = 0
) -> tuple[float, bool, bool, int, int]:
    """
    强化学习交易策略的奖励计算函数
    修改后的版本更注重趋势延续和稳定性
    
    参数：
    - history: 历史数据
    - current_price: 当前价格
    - current_value: 当前资产市值
    - step: 当前步数
    - max_steps: 最大允许步数
    - target_profit: 目标收益率
    - stop_loss: 止损线
    - consecutive_ups: 连续上涨次数
    - consecutive_downs: 连续下跌次数
    
    返回：
    - reward: 当前步的奖励值
    - done: 是否终止当前回合
    - truncated: 是否因步数限制终止
    - new_ups: 更新后的连续上涨次数
    - new_downs: 更新后的连续下跌次数
    """
    TARGET_PROFIT = np.log(1 + target_profit)
    STOP_LOSS = np.log(1 - stop_loss)
    
    prev_price = float(history["price", -1])
    prev_value = float(history["portfolio_valuation", -1])
    
    current_return_base = np.log(current_price / prev_price)
    current_return = np.log(current_value / prev_value)
    
    new_ups, new_downs = consecutive_ups, consecutive_downs    
    
    # 重新平衡的奖励系数
    BASE_PROFIT_COEFF = 1.0
    BASE_LOSS_COEFF = 1.5  # 惩罚略大于奖励
    EARLY_BONUS = 50       # 提前完成的额外奖励
    
    progress_ratio = step / max_steps
    TIME_PENALTY = -0.05 # * (1 + 2*progress_ratio**2)  # 每步的时间成本
    
    done, truncated = False, False
    reward = TIME_PENALTY  # 基础时间惩罚
    
    
    # 情况1：达到目标收益
    if current_return >= TARGET_PROFIT:
        # 提前完成奖励 = 基础奖励 + 提前完成奖励(指数衰减)
        early_bonus = EARLY_BONUS * np.exp(-5 * progress_ratio)
        reward += BASE_PROFIT_COEFF * (TARGET_PROFIT + early_bonus)
        done = True
    
    # 情况2：触发止损
    elif current_return <= STOP_LOSS:
        # 止损惩罚 = 基础惩罚 × 亏损严重程度
        loss_severity = min(abs(current_return/STOP_LOSS), 2.0)
        reward += BASE_LOSS_COEFF * STOP_LOSS * loss_severity
        done = True
    
    # 情况3：达到最大步数
    elif step >= max_steps:
        # 终点补偿 = 当前收益 × 进度补偿系数
        reward += current_return * (EARLY_BONUS *  (current_return / TARGET_PROFIT) if current_return > 0 else EARLY_BONUS * 2 * np.sqrt(abs(current_return / STOP_LOSS)))
        truncated = True
    
    else:
        # ===== 动态趋势奖励 =====
        momentum = current_return - current_return_base
        TREND_THRESHOLD = 0.001 * (1 - 0.5*progress_ratio)
        # 更新趋势连续性 (增加最小变化阈值)
        if momentum > TREND_THRESHOLD:  # 0.1%以上变化才视为趋势
            new_ups += 1
            new_downs = 0
        elif momentum < -TREND_THRESHOLD:
            new_downs += 1
            new_ups = 0
        
        # 趋势奖励机制改进
        if new_ups >= 1:  
            trend_strength = min(new_ups/5, 1.0)  # 标准化强度
            reward += BASE_PROFIT_COEFF * (
                current_return  # 价格变动基础奖励
                + trend_strength   # 趋势延续奖励
            )
        
        if new_downs >= 1:  # 只要下跌就惩罚
            trend_weakness = min(new_downs/3, 1.0)
            reward += BASE_LOSS_COEFF * (
                current_return  # 价格变动基础惩罚
                - trend_weakness * 0.8  # 趋势延续惩罚
            )

    return reward, done, truncated, new_ups, new_downs


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
    stop_loss: float = 0.10,  # 止损，低于视为失败，给予惩罚
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
