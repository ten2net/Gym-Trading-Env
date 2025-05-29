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
from pathlib import Path



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
    consecutive_downs: int = 0,
    training_steps: int = 0
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
    - training_steps : 训练步数，用于调整时间惩罚
    
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
    # 系数从-0.05→-0.04,降低基础惩罚，避免初期探索不足 
    # 指数从2→1.8 ,减缓后期惩罚增长速率，防止模型过早止损。
    # 乘数从2→1.2​​：缩小动态范围，平衡探索与利用。
    # progress_ratio_coeff = 1 + 0.5 * (1 - progress_ratio **1.8)
    # progress_ratio_coeff = 1 + 0.5 * (1 - np.sqrt(progress_ratio))  # 平方根衰减
    progress_ratio_coeff = 1 + 0.5 * np.exp(-2 * progress_ratio) 
    # TIME_PENALTY =-0.1 * (1 + 1.2 * progress_ratio_coeff) if training_steps > 1e6 else -0.1 * (1 + 1.8 * progress_ratio_coeff)
    # 计算线性变化的 coeff (1.8 → 1.2)
    max_training_steps = 4e6  # 根据实际情况调整
    # 优化方案：Sigmoid 平滑过渡
    # coeff = 1.8 - (training_steps / max_training_steps) * (1.8 - 1.2)
    # coeff = max(1.2, min(1.8, coeff))  # 限制在 [1.2, 1.8] 之间
    def smooth_clip(x, low=1.2, high=1.8, k=10):
        """Sigmoid 平滑截断，避免突变"""
        scale = high - low
        return low + scale / (1 + np.exp(-k * (x - 0.5)))
    coeff = smooth_clip(1.8 - (training_steps / max_training_steps) * 0.6)    
    TIME_PENALTY = -0.05 * (1 + coeff * progress_ratio_coeff)
    
    done, truncated = False, False
    reward = TIME_PENALTY  # 基础时间惩罚
    # 中间奖励（持仓波动率惩罚）
    # volatility_penalty = -0.1 * np.std(history["portfolio_valuation",-10:])  
    volatility_window=10
    market_volatility = max(0.00001,np.std(history["price", -20:]))  # 市场基准波动率
    low_vol_threshold = 0.5 * market_volatility    # 低波动阈值（如0.002~0.008）
    penalty_coeff = 0.1 / low_vol_threshold  # 系数与阈值反向相关
    volatility = np.std(history["portfolio_valuation",-volatility_window:]) /np.mean(history["portfolio_valuation",-volatility_window:]) # 相对波动率  
    if volatility < low_vol_threshold:
        volatility_penalty = penalty_coeff * (low_vol_threshold - volatility)  # 波动率越低，惩罚越重
        reward -= volatility_penalty
        # print(f"{market_volatility} volatility: {volatility}",volatility_penalty)
    # 情况1：达到目标收益
    if current_return >= TARGET_PROFIT:
        # 提前完成奖励 = 基础奖励 + 提前完成奖励(指数衰减)
        # TARGET_PROFIT=np.log(1 + target_profit) 是预设目标收益率（如15%对应≈0.1398），代表策略的基本任务要求。
        # EARLY_BONUS * np.exp(-5 * progress_ratio) 是动态附加奖励，鼓励模型尽快达成目标。
        # 基础奖励（TARGET_PROFIT）​​
        #     ​​确保策略有效性​​：保证模型至少获得与目标收益率匹配的基础奖励，建立收益与风险的基准对应关系。
        #     ​​数学必要性​​：对数收益率 TARGET_PROFIT 本身已包含复利特性，直接作为奖励基准符合金融逻辑。
        # 提前奖励（early_bonus）​​
        #     通过 np.exp(-5 * progress_ratio) 实现指数衰减，体现"越早达成奖励越高"的原则：
        #     当 progress_ratio=0（第一步达成）：奖励最大（early_bonus = EARLY_BONUS）
        #     当 progress_ratio→1：奖励趋近于0（early_bonus ≈ EARLY_BONUS*0.0067） 
        # 机会成本补偿​​
        #     在金融中，早期实现的收益具有更高时间价值。例如：
        #     第1步赚15% vs 第480步赚15%，前者可立即复投获取额外收益。
        #     early_bonus 实质是对这种机会成本的量化补偿。
        # 衰减系数（当前为5）​​
        #     若需更陡峭的时间衰减（强化"尽早达成"）：
        #     np.exp(-7 * progress_ratio)  # 中期奖励衰减更快
        #     np.exp(-3 * progress_ratio)  # 中期保留更多奖励
        # EARLY_BONUS 与 TARGET_PROFIT 的比例​​
        #     当前 50/0.1398≈358x 的倍数关系较为激进，适合高波动市场。
        #     对于稳定市场可调整为： EARLY_BONUS = 20  # 降为≈143x    
        # 改进公式（平滑极端值）​​
        #     early_bonus = EARLY_BONUS * np.exp(-5 * (progress_ratio**0.8))
        #     这种次线性变换（progress_ratio**0.8）可使中期奖励衰减更平缓，同时保持后期快速收敛。
        # 数学本质
        #     该设计实质是 ​​时间折扣奖励​​ 的变体：
        #     总奖励 = 即时收益奖励 + 时间效率奖励
        # 其中：
        #     TARGET_PROFIT 对应贝尔曼方程中的 R(s,a)
        #     early_bonus 对应折扣因子 γ^t 的逆应用（越早达成，γ^t越小，但奖励越大）
        # 这种结构在期权定价早期执行溢价中有类似应用，符合金融数学原理。                          
        # early_bonus = EARLY_BONUS * np.exp(-5 * progress_ratio)
        # 分级奖励机制
        return_ratio = current_return / TARGET_PROFIT
        if return_ratio < 1.005:  # 基础达标区
            early_bonus = EARLY_BONUS * np.exp(-5 * (progress_ratio**0.7))
        else:  # 超额达标区
            early_bonus = EARLY_BONUS * 5 * np.exp(-3 * (progress_ratio**0.5))        
        reward += BASE_PROFIT_COEFF * (TARGET_PROFIT + early_bonus + np.log1p(return_ratio - 1))
        done = True
    
    # 情况2：触发止损
    elif current_return <= STOP_LOSS:
        # 止损惩罚 = 基础惩罚 × 亏损严重程度
        # 当前方法对超跌的惩罚增长更平缓（通过 STOP_LOSS * loss_severity 实现非线性约束）。
        # 直接使用 current_return 在极端情况下会产生过大的梯度，导致训练不稳定（与您展示的剧烈震荡曲线相关）。
        # 鼓励模型学会遵守止损纪律，而非预测市场极端波动（后者几乎不可能稳定实现）。
        # 通过 loss_severity 的梯度惩罚（1.0→2.0），模型会更早主动止损，从而缩短平均回合长度（从475→460区间）。        
        # 惩罚 = 系数 × 基准 × (实际损失/基准)^k
        # 其中 k=1 时为线性惩罚，k>1 时为超线性惩罚（对超跌更敏感）。这种结构在强化学习中被称为 ​​Huberized Penalty​​，
        # 这样处理后，平衡了对异常值的鲁棒性和梯度有效性。
        # loss_severity = min(abs(current_return/STOP_LOSS), 2.0)
        loss_severity = min(abs(current_return/STOP_LOSS)**1.5, 3.0)  # 非线性增长
        reward += BASE_LOSS_COEFF * STOP_LOSS * loss_severity
        done = True
    
    # 情况3：达到最大步数
    elif step >= max_steps:
        # 终点补偿 = 当前收益 × 目标达成度补偿系数
        # 盈利情况的线性处理​​
        #     (current_return / TARGET_PROFIT) 实现：
        #     收益达标时（current_return ≈ TARGET_PROFIT）：奖励 ≈ EARLY_BONUS
        #     超额收益时（如 current_return = 2*TARGET_PROFIT）：奖励线性增长到 2*EARLY_BONUS
        # ​​亏损情况的非线性处理（关键创新）​​
        #     总惩罚 = 基础系数 × 压缩后亏损程度
        #     np.sqrt(abs(current_return / STOP_LOSS)) 专门解决：
        # ​    ​梯度爆炸抑制​​：您原始曲线中出现的-40极端值，通过平方根压缩大幅降低异常惩罚。
        # 金融合理性​​
        #     符合 ​​"边际风险递增"​​ 原则：
        #         小亏损（如-5%）需要敏感反应
        #         大亏损（如-20%）应避免过度反应（因可能含市场异常波动）
        # 终点补偿逻辑​中，若轻微盈利或轻微亏损，不奖励或惩罚
        # 引入​​死区机制​​，在保持对显著盈亏合理反应的同时，过滤了微小波动带来的噪声信号，
        # 更符合实际交易中"忽略小额摩擦成本"的原则。建议初始设置 DEAD_ZONE_RATIO=0.01，后续根据训练曲线调整。
        # 定义死区阈值（例如±1%收益率）
        DEAD_ZONE_RATIO = 0.01  
        dead_zone_lower = np.log(1 - DEAD_ZONE_RATIO)  # ≈-0.01005
        dead_zone_upper = np.log(1 + DEAD_ZONE_RATIO)  # ≈0.00995 
                
        if current_return > dead_zone_upper:
            # 显著盈利：线性奖励
            reward += EARLY_BONUS * (current_return / TARGET_PROFIT)
        elif current_return < dead_zone_lower:
            # 显著亏损：平方根压缩惩罚
            # 亏损超出死区部分才惩罚
            excess_loss = abs(current_return) - abs(dead_zone_lower)            
            loss_ratio = min(abs(current_return / STOP_LOSS), 4.0)
            reward -= excess_loss  * 10 * np.sqrt(loss_ratio)
        else:
            # 轻微波动：零奖惩（仅保留基础时间惩罚）
            pass                         
        truncated = True    
    else:
        # ===== 动态趋势奖励 =====
        momentum = current_return - current_return_base
        # print(f'{momentum:.6f}')
        # TREND_THRESHOLD = 0.0001 * (1 - 0.5*progress_ratio)
        # TREND_THRESHOLD = 0.0002 if training_steps > 1e6 else 0.0003

        TREND_THRESHOLD = 0.0001 + (training_steps / max_training_steps) * (0.001 - 0.0001)
        TREND_THRESHOLD = min(0.001 ,TREND_THRESHOLD)
        # 更新趋势连续性 (增加最小变化阈值)
        if momentum > TREND_THRESHOLD:  # 0.1%以上变化才视为趋势
            new_ups += 1
            new_downs = 0
        elif momentum < -TREND_THRESHOLD:
            new_downs += 1
            new_ups = 0
        
        # 趋势奖励机制改进
        if new_ups >= 1:  
            # tanh 的输出范围始终在 (-1, 1)，天然避免奖励爆炸。
            # 双曲正切控制增长速率（越小增长越快，3是一个平衡值）
            # tanh 在早期（new_ups=1→3）提供更积极的激励，有助于缩短回合长度（对应您下图中的高步数问题）。
            # 在后期（new_ups>5）自动抑制过拟合，稳定训练曲线。
            trend_strength = np.tanh(new_ups/3)  # 改用双曲正切平滑增长
            log_return = np.log1p(abs(current_return))  # 保证输入>0
            reward += BASE_PROFIT_COEFF * (
                log_return  # 价格变动基础奖励
                + trend_strength   # 趋势延续奖励
            )
        
        if new_downs >= 1:  # 只要下跌就惩罚
            # 对数收益率log(1+x)在x接近0时≈x，在x较大时增长变缓，能自动抑制极端惩罚
            # 训练曲线剧烈波动正是需要这种非线性抑制
            # ​​风险控制​​：防止单次大亏损导致奖励值"悬崖式"下降
            log_return = np.log1p(abs(min(current_return, -0.0005)))  # 保证输入>0
            reward -= BASE_LOSS_COEFF * (log_return + np.log1p(new_downs) /2)  # 双重对数平滑
            # trend_weakness = min(new_downs/3, 1.0)
            # reward += BASE_LOSS_COEFF * (
            #     current_return  # 价格变动基础惩罚
            #     - trend_weakness * 0.8  # 趋势延续惩罚
            # )

    return reward, done, truncated, new_ups, new_downs


def dynamic_feature_last_position_taken(history):
    return history['position', -1]


def dynamic_feature_real_position(history):
    return history['real_position', -1]

def preprocess(df : pd.DataFrame) ->  pd.DataFrame:
    # if multi_dataset:
    # df["date"] = pd.to_datetime(df["timestamp"], unit= "ms")
    df.sort_index(inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    # Generating features
    # WARNING : the column names need to contain keyword 'feature' !
    df["feature_close"] = df["close"].pct_change()
    df["feature_high"] = df["high"].pct_change()
    df["feature_low"] = df["low"].pct_change()
    df["feature_volume"] = df["volume"].pct_change()
    df['dt'] = df.index.date
    # 2. 获取每日开盘价
    daily_open = df.groupby('dt')['open'].transform('first')
    daily_volume = df.groupby('dt')['volume'].transform('first')
    # 3. 将每日开盘价合并回原始数据框
    df = df.merge(daily_open.rename('daily_open'),
                  left_on='date',
                  right_index=True)
    df = df.merge(daily_volume.rename('daily_volume'),
                  left_on='date',
                  right_index=True)
    # df['feature_close_open_yoy'] = df['close'] / df['daily_open']
    # df['feature_high_open_yoy'] = df['high'] / df['daily_open']
    # df['feature_low_open_yoy'] = df['low'] / df['daily_open']
    # df['feature_volume_open_yoy'] = df['volume'] / df['daily_volume']
    # df["feature_volume"] = df["volume"] / df["volume"].rolling(12).max()
    points_per_day = 48  # 24小时*60分钟/5分钟=288，但实际交易时间可能更少
    # 滞后特征
    for lag in [1, points_per_day, points_per_day*5]:  # 1个点前，1天前，5天前
        df[f'feature_close_lag_{lag}'] = df['close'] / df['close'].shift(lag)
        df[f'feature_volume_lag_{lag}'] =  df['volume'] / df['volume'].shift(lag)    
    # df['close_prev'] = df['close'].shift(points_per_day)
    # df['volume_prev'] = df['volume'].shift(points_per_day)
    # df['cum_volume'] = df.groupby('dt')['volume'].cumsum()
    # df['cum_volume_prev'] = df["cum_volume"].shift(points_per_day)

    # df['feature_close_yoy'] = df['close'] / df['close_prev']
    # df['feature_volume_sum'] = df['cum_volume'] / df['cum_volume_prev']
    # df = df.drop(columns=['dt', 'daily_open',
    #              'volume_prev', 'cum_volume', 'cum_volume_prev'])
    numeric_cols = df.columns
    for col in numeric_cols:
        if col.startswith("feature"):
            df[col] = df[col].round(6)
    df.dropna(inplace=True)
    # df.to_csv(f"{symbol}.csv", index=True, encoding='utf_8_sig')
    return df
def make_multi_dataset_env(
    raw_data_dir: str ="../raw_data/pkl/m5",
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
    env = gym.make(
        "MultiDatasetTradingEnv",
        dataset_dir= f'{raw_data_dir}/',
        preprocess= preprocess,
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
    eval_env = Monitor(env, './eval_logs')
    return env if not eval else eval_env

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
    csv_path = Path("../raw_data/csv/m5") / f"{symbol}.csv"
    df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
    df = (
         df.sort_index()
        .dropna()
        .drop_duplicates()
    )

    # 3. 特征工程（收益率特征）
    df = df.assign(
        feature_close=df["close"].pct_change(),
        feature_high=df["high"].pct_change(),
        feature_low=df["low"].pct_change(),
        feature_volume=df["volume"].pct_change(),
        feature_volatility = df["high"] / df["low"] - 1, # 波动率特征
        dt=df.index.date,  # 日期列（不含时间）
    ).dropna()  # 删除pct_change引入的NaN
    # 4. 添加星期和时间列
    df = df.assign(
        weekday=df.index.day_name(),  # 星期几（英文，如 "Monday"）
        feature_weekday=df.index.dayofweek / 5,  # 星期几的数字（0=周一，6=周日）
        feature_hour=df.index.hour / 24,  # 小时（0-23）
    )    
    # 2. 获取每日开盘价
    daily_open = df.groupby('dt')['open'].transform('first')
    daily_volume = df.groupby('dt')['volume'].transform('first')
    # 3. 将每日开盘价合并回原始数据框
    df = df.merge(daily_open.rename('daily_open'),
                  left_on='date',
                  right_index=True)
    df = df.merge(daily_volume.rename('daily_volume'),
                  left_on='date',
                  right_index=True)
    df['feature_close_open_yoy'] = df['close'] / df['daily_open']
    df['feature_high_open_yoy'] = df['high'] / df['daily_open']
    df['feature_low_open_yoy'] = df['low'] / df['daily_open']
    df['feature_volume_open_yoy'] = df['volume'] / df['daily_volume']
    # df["feature_volume"] = df["volume"] / df["volume"].rolling(12).max()
    points_per_day = 48  # 24小时*60分钟/5分钟=288，但实际交易时间可能更少
    # 滞后特征
    for lag in [1, points_per_day, points_per_day*5]:  # 1个点前，1天前，5天前
        df[f'feature_close_lag_{lag}'] = df['close'] / df['close'].shift(lag)
        df[f'feature_volume_lag_{lag}'] =  df['volume'] / df['volume'].shift(lag)    
    df['close_prev'] = df['close'].shift(points_per_day)
    df['volume_prev'] = df['volume'].shift(points_per_day)
    df['cum_volume'] = df.groupby('dt')['volume'].cumsum()
    df['cum_volume_prev'] = df["cum_volume"].shift(points_per_day)

    df['feature_close_yoy'] = df['close'] / df['close_prev']
    df['feature_volume_sum'] = df['cum_volume'] / df['cum_volume_prev']
    df = df.drop(columns=['dt', 'daily_open',
                 'volume_prev', 'cum_volume', 'cum_volume_prev'])
    # print(df[-50:])
    # fe = FeatureEngineer(window_size=window_size)
    # df = fe.compute_features(df)
    numeric_cols = df.columns
    for col in numeric_cols:
        if col.startswith("feature"):
            df[col] = df[col].round(6)
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
