from typing import Literal
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np
import datetime
import glob
from pathlib import Path
from collections import Counter
from .utils.history import History
from .utils.portfolio import Portfolio, TargetPortfolio


def basic_reward_function(history: History, step: int = 0, lambda_param: float = 0.1, eta: float = 0.5):
    R_t = np.log(history["portfolio_valuation", -1] /
                 history["portfolio_valuation", -2])


def dynamic_feature_last_position_taken(history):
    return history['position', -1]


def dynamic_feature_real_position(history):
    return history['real_position', -1]


class TradingEnv(gym.Env):
    metadata = {'render_modes': ['logs']}

    def __init__(self,
                 df: pd.DataFrame,
                 positions: list = [0, 1],
                 dynamic_feature_functions=[
                     dynamic_feature_real_position],
                 reward_function=basic_reward_function,
                 windows=None,
                 trading_fees=0,
                 portfolio_initial_value=1000000,
                 initial_position='random',
                 max_episode_duration='max',
                 target_return=0.15,  # 目标收益率15%
                 daily_loss_limit=-0.03,  # 单日亏损限制
                 max_drawdown=-0.05,  # 最大回撤限制
                 name="Stock",
                 render_mode="logs",
                 verbose=1
                 ):
        # 参数校验（限制仓位范围0-1）
        assert all(
            0 <= pos <= 1 for pos in positions), "All positions must be between 0 and 1"
        assert initial_position in positions or initial_position == 'random', "Invalid initial position"

        self.max_episode_duration = max_episode_duration
        self.name = name
        self.verbose = verbose
        self.positions = positions
        self.dynamic_feature_functions = dynamic_feature_functions
        self.reward_function = reward_function
        self.windows = windows
        self.trading_fees = trading_fees
        self.portfolio_initial_value = float(portfolio_initial_value)
        self.initial_position = initial_position
        self.render_mode = render_mode

        self.target_return = target_return
        self.daily_loss_limit = daily_loss_limit
        self.max_drawdown = max_drawdown

        self.terminated = False
        self.truncated = False
        # 数据设置
        self._set_df(df)

        # 动作和观测空间
        self.action_space = spaces.Discrete(len(positions))
        self.observation_space = spaces.Box(
            -np.inf,
            np.inf,
            shape=[self._nb_features]
        )
        if self.windows is not None:
            self.observation_space = spaces.Box(
                -np.inf,
                np.inf,
                shape=[self.windows, self._nb_features]
            )
        self.epid = 0
        self.log_metrics = []

    def _set_df(self, df):
        df = df.copy()
        self._features_columns = [
            col for col in df.columns if "feature" in col]
        self._info_columns = list(
            set(df.columns) - set(self._features_columns))
        self._nb_features = len(self._features_columns)
        self._nb_static_features = self._nb_features

        # 添加动态特征
        for i in range(len(self.dynamic_feature_functions)):
            df[f"dynamic_feature__{i}"] = 0
            self._features_columns.append(f"dynamic_feature__{i}")
            self._nb_features += 1

        self.df = df
        self._obs_array = np.array(
            self.df[self._features_columns], dtype=np.float32)
        self._info_array = np.array(self.df[self._info_columns])
        self._price_array = np.array(self.df["close"])


    def _get_price(self, delta=0):
            return self._price_array[self._idx + delta]
    def _get_ohlcv(self, item: Literal["open", "high", "low", "close", "volume"], delta: int = 0) -> float:
        """
        获取指定OHLCV字段的值（带索引偏移）
        
        参数:
            item: 必须是 "open", "high", "low", "close" 或 "volume"
            delta: 索引偏移量（0=当前，-1=前一根K线等）
        
        返回:
            float: 对应的价格或成交量
        
        异常:
            ValueError: 当item不是OHLCV字段时
            IndexError: 当索引超出数据范围时
        """
        # 静态类型检查（通过Literal已保证，此处为运行时双保险）
        valid_items = {"open", "high", "low", "close", "volume"}
        if item not in valid_items:
            raise ValueError(
                f"Invalid item '{item}'. Must be one of: {', '.join(valid_items)}"
            )

        # 索引安全检查
        target_idx = self._idx + delta
        if not 0 <= target_idx < len(self.df):
            raise IndexError(
                f"Index {target_idx} out of bounds for DataFrame with length {len(self.df)}"
            )
        current_date  = self.df.index[target_idx].date()
        daily_mask = (self.df.index.date == current_date)
        daily_data = self.df[daily_mask]
        if item == 'open':
            return float(daily_data['open'].iloc[0])
        elif item == 'high':
            return float(daily_data['high'].max())
        elif item == 'low':
            return float(daily_data['low'].min())
        elif item == 'close':
            return float(daily_data['close'].iloc[-1])
        elif item == 'volume':
            return float(daily_data['volume'].sum())

    def _get_obs(self):
        # 更新动态特征
        for i, func in enumerate(self.dynamic_feature_functions):
            self._obs_array[self._idx, self._nb_static_features +
                            i] = func(self.historical_info)

        return self._obs_array[self._idx - self.windows + 1: self._idx + 1] if self.windows else self._obs_array[self._idx]

    def layered_reward_function(self):
        """分层奖励计算"""
        # 单步奖励（已实现盈亏 - 手续费）
        step_reward = self._calculate_step_reward()

        # 当日奖励（若为新交易日）
        daily_reward = 0
        if self._is_new_day():
            daily_reward = self._calculate_daily_reward()

        # 回合奖励（仅在终止时计算）
        episode_reward = 0
        if self.terminated or self.truncated:
            episode_reward = self._calculate_episode_reward()

        # 加权综合
        return 0.4 * step_reward + 0.5 * daily_reward + 0.1 * episode_reward

    def _get_daily_log_volatility(self):
        """基于日内各时间步对数收益计算波动率"""
        current_date = self.df.index[self._idx].date()
        # 筛选当日所有净值记录
        day_values = []
        for t in range(self.historical_info.size):
            entry = self.historical_info[t]
            if pd.Timestamp(entry['date']).date() == current_date:
                day_values.append(entry['portfolio_valuation'])
        # 至少需要2个数据点计算波动率
        if len(day_values) < 2:
            return 0.0                
        log_returns = np.diff(np.log(day_values))  # 日内对数收益序列
        return np.std(log_returns)
    def _calculate_step_reward(self):
        """单步：log收益 - 交易惩罚"""
        current_value = self._portfolio.valorisation(self._get_price())
        prev_value = self.historical_info['portfolio_valuation', -2]
        log_return = np.log(current_value / prev_value) if prev_value > 0 else 0
        return log_return 

    def _calculate_daily_reward(self):
        """当日：收益率 - 波动惩罚"""
        daily_log_return = np.log(self.day_close_value / self.day_open_value)
        daily_volatility = self._get_daily_log_volatility()  
        risk_penalty = np.sqrt(daily_volatility) * 0.2
        risk_penalty = daily_volatility * 0.5  
        return daily_log_return - risk_penalty

    def _calculate_episode_reward(self):
        """回合：目标达成奖励 + 夏普比率"""
        current_value = self._portfolio.valorisation(self._get_price())
        total_return = (current_value /
                        self.portfolio_initial_value) - 1
        if total_return >= self.target_return:
            target_bonus = 50  # 目标达成奖励
        else:
            target_bonus = -30 * (self.target_return - total_return)

        sharpe_ratio = self._calculate_sharpe_ratio(annualized=True, risk_free_rate=0.0)  # 需实现夏普计算
        return target_bonus + sharpe_ratio * 10
    
    def _calculate_sharpe_ratio(self, annualized=True, risk_free_rate=0.0):
        """
        计算夏普比率
        :param annualized: 是否年化（适用于日内数据）
        :param risk_free_rate: 年化无风险利率（默认0）
        :return: 夏普比率
        """
        # 获取净值序列
        portfolio_values = self.historical_info["portfolio_valuation"]
        if len(portfolio_values) < 2:
            return 0.0  # 数据不足无法计算

        # 计算对数收益率
        returns = np.diff(np.log(portfolio_values))
        
        # 去除无效值
        returns = returns[~np.isnan(returns)]
        if len(returns) == 0:
            return 0.0

        # 计算平均收益率和标准差
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        if std_return < 1e-8:  # 防止除零
            return 0.0

        # 计算基础夏普比率
        sharpe = (mean_return - risk_free_rate/252) / std_return  # 假设无风险利率为年化
        
        # 年化处理（假设每个step为5分钟，1年=252天*48步=12096步）
        if annualized:
            sharpe *= np.sqrt(252 * 48)  # 按波动率平方根法则年化

        return float(sharpe)    

    def _handle_new_day(self):
        """新交易日初始化"""
        self.current_day += 1
        self.day_open_value =self._portfolio.valorisation(self._get_ohlcv('open')) 

    def _update_day_close_value(self):
        """更新当日净值极值"""
        current_value = self._portfolio.valorisation(self._get_ohlcv('close'))
        self.day_close_value = current_value

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 初始化状态
        self.terminated = False
        self.truncated = False
        # 动态交易日追踪
        self.current_day = 0
        self.day_open_value = self.portfolio_initial_value  # 每日初始净值
        
        self.epid += 1
        self._step = 0
        self._position = (np.random.choice(self.positions)
                          if self.initial_position == 'random'
                          else self.initial_position)

        # 设置初始索引
        self._idx = self.windows - 1 if self.windows else 0
        if isinstance(self.max_episode_duration, int):
            self._idx = np.random.randint(
                self._idx,
                len(self.df) - self.max_episode_duration
            )
        # 初始化投资组合
        self._portfolio = TargetPortfolio(
            position=self._position,
            value=self.portfolio_initial_value,
            price=self._get_price()
        )

        initial_distribution = {
            "asset": self.portfolio_initial_value / self._get_price(),
            "fiat": 0.0,
            "next_day_available_asset": 0.0
        }

        self.current_reward = 0

        # 初始化历史记录
        self.historical_info = History(max_size=len(self.df))
        self.historical_info.set(
            idx=self._idx,
            step=self._step,
            date=self.df.index.values[self._idx],
            position_index=self.positions.index(self._position),
            position=self._position,
            real_position=self._position,
            data=dict(zip(self._info_columns, self._info_array[self._idx])),
            portfolio_valuation=self.portfolio_initial_value,
            portfolio_distribution=initial_distribution,  # 使用预定义结构
            reward=self.current_reward,
        )
        self._update_history()

        return self._get_obs(), {}

    def step(self, action):
        if self._is_new_day():
            self._handle_new_day()
        prev_dt = self.df.index.date[self._idx - 1]
        curr_dt = self.df.index.date[self._idx]
        is_new_day = self._is_new_day()
        if is_new_day:
            self._portfolio.update_day()
        # 执行交易
        self._position = self.positions[action]
        self._portfolio.trade_to_position(
            self._position,
            self._get_price(),
            self.trading_fees
        )
        # 更新历史记录以反映最后一次交易
        self._update_history()

        # 计算基本探索奖励
        reward = self.reward_function(self.historical_info)

        # 更新索引和日期
        self._idx += 1
        self._step += 1
        terminated, truncated = False, False
        # 检查是否超出数据范围
        if self._idx >= len(self.df) - 1:
            self._idx -= 1
            truncated = True
            reward += self.episode_reward_function(lambda_param=0.1, eta=0.5)
            self.current_reward = reward
            if self.verbose:
                print(
                    f"回合​​截断（到达数据尾部) step {self._step}.")
            # 记录最终指标
            self._log_metrics()
            return self._get_obs(), self.current_reward, terminated, truncated, self.historical_info[-1]

        # 检查终止条件
        portfolio_value = self._portfolio.valorisation(self._get_price())
        portfolio_return = portfolio_value / self.portfolio_initial_value
        if abs(portfolio_return - 1) >= 0.15:
            terminated = True
            reward += self.episode_reward_function(lambda_param=0.1, eta=0.5)

        if terminated:
            reward += self.episode_reward_function(lambda_param=0.1, eta=0.5)
            if self.verbose:
                print(
                    f"回合完成。 step {self._step}，奖励：{reward:.4f}")
        truncated = self._check_truncation()  # 当前时间步长超过最大时间步长 或 到达数据末尾
        if truncated:
            terminated = True
            if self.verbose:
                print(
                    f"回合​​截断（达到最大时间步数或到达数据尾部) step {self._step}，奖励：{reward:.4f}")

        # 记录最终指标
        if terminated or truncated:
            self._log_metrics()
        self.current_reward = reward

        self._update_day_close_value()
        # print(f"{self.epid} {self._step} {is_new_day} {prev_dt} {curr_dt} {self._position:.1f}  {self._portfolio.position(self._get_price()):.1f} {self._get_price()}   ,{self._portfolio.next_day_available_asset:.0f},            {self._portfolio.asset:.0f},  {self._portfolio.fiat:.2f}      {self._portfolio.valorisation(self._get_price()):.2f}")
        return self._get_obs(), self.current_reward, terminated, truncated, self.historical_info[-1]

    def _log_metrics(self):
        self.calculate_metrics()
        if self.verbose:
            print(" | ".join(f"{k}: {v}" for k,
                  v in self.results_metrics.items()))

    def _is_new_day(self):
        try:
            return self.df.index[self._idx].date() != self.df.index[self._idx-1].date()
        except IndexError:
            return False

    def _check_bankruptcy(self):
        portfolio_value = self._portfolio.valorisation(self._get_price())
        return (portfolio_value / self.portfolio_initial_value) <= 0.4

    def _check_truncation(self):
        return (self._idx >= len(self.df) - 1 or
                (isinstance(self.max_episode_duration, int) and
                self._step >= self.max_episode_duration))

    def _update_history(self):
        price = self._get_price()
        # 调整资产分布字段
        portfolio_dist = {
            "asset": self._portfolio.asset,
            "fiat": self._portfolio.fiat,
            "next_day_available_asset": self._portfolio.next_day_available_asset
        }
        self.historical_info.add(
            idx=self._idx,
            step=self._step,
            date=self.df.index.values[self._idx],
            position_index=self.positions.index(self._position),
            position=self._position,
            real_position=self._portfolio.position(price),
            data=dict(zip(self._info_columns, self._info_array[self._idx])),
            portfolio_valuation=self._portfolio.valorisation(price),
            portfolio_distribution=portfolio_dist,
            reward=self.current_reward
        )

    # 保留指标计算方法
    def add_metric(self, name, function):
        self.log_metrics.append({
            'name': name,
            'function': function
        })

    def calculate_metrics(self):
        market_return = (
            self.historical_info['data_close', -1] / self.historical_info['data_close', 0] - 1) * 100
        portfolio_return = (
            self.historical_info['portfolio_valuation', -1] / self.portfolio_initial_value - 1) * 100

        self.results_metrics = {
            "Market Return": f"{market_return:.2f}%",
            "Portfolio Return": f"{portfolio_return:.2f}%"
        }
        for metric in self.log_metrics:
            self.results_metrics[metric['name']
                                 ] = metric['function'](self.historical_info)

    def log(self):
        if self.verbose:
            print(" | ".join(f"{k}: {v}" for k,
                  v in self.results_metrics.items()))


class MultiDatasetTradingEnv(TradingEnv):
    def __init__(self,
                 dataset_dir,
                 *args,
                 preprocess=lambda df: df,
                 episodes_between_dataset_switch=1,
                 **kwargs):

        # 过滤掉可能存在的无效参数
        kwargs.pop('borrow_interest_rate', None)

        self.dataset_dir = dataset_dir
        self.preprocess = preprocess
        self.episodes_between_dataset_switch = episodes_between_dataset_switch
        self.dataset_pathes = glob.glob(self.dataset_dir)

        if not self.dataset_pathes:
            raise FileNotFoundError(
                f"No datasets found matching: {self.dataset_dir}")

        super().__init__(self._load_next_dataset(), *args, **kwargs)

    def _load_next_dataset(self):
        self.current_dataset = np.random.choice(self.dataset_pathes)
        return self.preprocess(pd.read_pickle(self.current_dataset))

    def reset(self, seed=None, options=None):
        if hasattr(self, 'episode_count'):
            self.episode_count += 1
            if self.episode_count % self.episodes_between_dataset_switch == 0:
                self._set_df(self._load_next_dataset())
        else:
            self.episode_count = 0

        return super().reset(seed=seed, options=options)
