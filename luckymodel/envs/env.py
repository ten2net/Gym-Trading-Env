from basic_model.utils.feature_engineering import FeatureEngineer
import warnings
from gym_trading_env.environments import TradingEnv
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Optional, List, Literal
import argparse
import sys
sys.path.append("./src")


warnings.filterwarnings("ignore", category=ResourceWarning)

# Create your own reward function with the history object


def reward_function(history):
    log_return = 100 * np.log(history["portfolio_valuation", -1] /
                              # log (p_t / p_t-1 )
                              history["portfolio_valuation", -2])
    return np.clip(log_return, -0.5, 0.5)

def excess_return_reward(history, step: int = 0, alpha: float = 1.5, scale_factor: float = 10.0):
    """
    基于累计超额收益的分层奖励函数
    :param history: 历史数据容器
    :param alpha: 超额收益非线性系数 (α>1时奖励随超额收益加速增长)
    :param scale_factor: 奖励缩放系数
    """
    # 获取初始及当前投资组合价值
    initial_portfolio = history["portfolio_valuation", 0]
    current_portfolio = history["portfolio_valuation", -1]
    
    # 计算策略累计收益率
    strategy_return = (current_portfolio - initial_portfolio) / initial_portfolio
    
    # 获取市场基准收益率（假设data_close包含标的资产价格）
    initial_benchmark = history["data_close", 0]
    current_benchmark = history["data_close", -1]
    benchmark_return = (current_benchmark - initial_benchmark) / initial_benchmark
    
    # 计算超额收益
    excess_return = strategy_return - benchmark_return
    
    if excess_return <= 0:
        # 未跑赢基准时给予微小负奖励（避免随机扰动）
        return -0.01 * np.abs(excess_return)
    
    # 非线性奖励计算（可调整alpha控制激励强度）
    reward = scale_factor * (np.exp(alpha * excess_return) - 1)
    
    # 添加波动率惩罚项（可选）
    if step > 10:  # 避免初期波动计算不稳定
        portfolio_volatility = np.std(np.diff(np.log(history["portfolio_valuation", -10:])) )
        reward *= np.exp(-2.0 * portfolio_volatility)
    
    return float(reward)
def cal_max_drawdown(history):
    # 添加最大回撤
    peak = np.maximum.accumulate(history['portfolio_valuation'])
    drawdown = (peak - history['portfolio_valuation']) / peak
    return np.max(drawdown) * 100


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

    excess_return = 100 * \
        np.log(history["portfolio_valuation", -1] /
               history["portfolio_valuation", -2])

    if excess_return >= 0:
        # 正向收益：使用修正指数函数控制增长
        reward = np.sign(excess_return) * \
            (np.abs(excess_return)**gamma) * alpha
    else:
        # 负向亏损：使用指数惩罚
        reward = -np.exp(beta * np.abs(excess_return)) + 1

    # 数值截断
    return np.clip(reward, -max_clip, max_clip)


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
        reward_function=excess_return_reward, ##one_step_reward,  # reward_function,
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

    env.add_metric('调参次数', lambda history: np.sum(
        np.diff(history['real_position']) != 0))
    env.add_metric(
        '奖励 sum', lambda history: f"{np.sum(history["reward"]):.4f}")
    env.add_metric(
        '奖励 max', lambda history: f"{np.max(history["reward"]):.4f}")
    env.add_metric(
        '奖励 min', lambda history: f"{np.min(history["reward"]):.4f}")
    env.add_metric(
        '奖励 avg', lambda history: f"{np.mean(history["reward"]):.4f}")
    env.add_metric(
        '奖励 median', lambda history: f"{np.median(history["reward"]):.5f}")
    env.add_metric('最大回撤', lambda history: f"{cal_max_drawdown(history):.2f}")

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
