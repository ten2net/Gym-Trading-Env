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

def basic_reward_function(history : History):
    return np.log(history["portfolio_valuation", -1] / history["portfolio_valuation", -2])

def dynamic_feature_last_position_taken(history):
    return history['position', -1]

def dynamic_feature_real_position(history):
    return history['real_position', -1]

class TradingEnv(gym.Env):
    metadata = {'render_modes': ['logs']}
    
    def __init__(self,
                df : pd.DataFrame,
                positions : list = [0, 1],
                dynamic_feature_functions = [dynamic_feature_last_position_taken, dynamic_feature_real_position],
                reward_function = basic_reward_function,
                windows = None,
                trading_fees = 0,
                portfolio_initial_value = 1000,
                initial_position ='random',
                max_episode_duration = 'max',
                verbose = 1,
                name = "Stock",
                render_mode= "logs"
                ):
        # 参数校验（限制仓位范围0-1）
        assert all(0 <= pos <= 1 for pos in positions), "All positions must be between 0 and 1"
        assert initial_position in positions or initial_position == 'random', "Invalid initial position"

        # 初始化参数（移除了借贷利率）
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
        
        # 数据设置
        self._set_df(df)
        
        # 动作和观测空间
        self.action_space = spaces.Discrete(len(positions))
        self.observation_space = spaces.Box(
            -np.inf,
            np.inf,
            shape = [self._nb_features]
        )
        if self.windows is not None:
            self.observation_space = spaces.Box(
                -np.inf,
                np.inf,
                shape = [self.windows, self._nb_features]
            )
        
        self.log_metrics = []

    def _set_df(self, df):
        df = df.copy()
        self._features_columns = [col for col in df.columns if "feature" in col]
        self._info_columns = list(set(df.columns) - set(self._features_columns))
        self._nb_features = len(self._features_columns)
        self._nb_static_features = self._nb_features

        # 添加动态特征
        for i  in range(len(self.dynamic_feature_functions)):
            df[f"dynamic_feature__{i}"] = 0
            self._features_columns.append(f"dynamic_feature__{i}")
            self._nb_features += 1

        self.df = df
        self._obs_array = np.array(self.df[self._features_columns], dtype= np.float32)
        self._info_array = np.array(self.df[self._info_columns])
        self._price_array = np.array(self.df["close"])

    def _get_price(self, delta = 0):
        return self._price_array[self._idx + delta]
    
    def _get_obs(self):
        # 更新动态特征
        for i, func in enumerate(self.dynamic_feature_functions):
            self._obs_array[self._idx, self._nb_static_features + i] = func(self.historical_info)

        return self._obs_array[self._idx - self.windows + 1: self._idx + 1] if self.windows else self._obs_array[self._idx]

    def reset(self, seed = None, options=None):
        super().reset(seed = seed)
        
        # 初始化状态
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
        
        # 初始化投资组合（使用不支持借贷的版本）
        self._portfolio = TargetPortfolio(
            position = self._position,
            value = self.portfolio_initial_value,
            price = self._get_price()
        )
        
        initial_distribution = {
            "asset": self.portfolio_initial_value / self._get_price(),
            "fiat": 0.0,
            "next_day_available_asset": 0.0
        }
                
        # 初始化历史记录
        self.historical_info = History(max_size=len(self.df))
        self.historical_info.set(
            idx = self._idx,
            step = self._step,
            date = self.df.index.values[self._idx],
            position_index = self.positions.index(self._position),
            position = self._position,
            real_position = self._position,
            data = dict(zip(self._info_columns, self._info_array[self._idx])),
            portfolio_valuation = self.portfolio_initial_value,
            portfolio_distribution = initial_distribution,  # 使用预定义结构
            reward = 0,
        )        
        self._update_history()
        
        return self._get_obs(), {}

    def step(self, action):
        # 执行交易
        self._position = self.positions[action]
        self._portfolio.trade_to_position(
            self._position, 
            self._get_price(), 
            self.trading_fees
        )
        
        # 更新索引和日期
        self._idx += 1
        self._step += 1
        
        # 检查是否超出数据范围
        if self._idx >= len(self.df):
            # 触发截断并恢复索引
            done = True
            truncated = True
            self._idx -= 1  # 恢复到有效索引
            self._step -= 1
            # 更新历史记录以反映最后一次交易
            self._update_history()
            reward = 0
            if self.verbose:
                print(f"Episode truncated due to exceeding data bounds at step {self._step}.")
            # 记录最终指标
            self._log_metrics()
            return self._get_obs(), reward, done, truncated, self.historical_info[-1]
        
        # 处理自然日更新
        if self._is_new_day():
            self._portfolio.update_day()
        
        # 检查终止条件
        done = self._check_bankruptcy()
        truncated = self._check_truncation()
        
        # 更新历史记录
        self._update_history()
        
        # 计算奖励
        reward = 0 if done else self.reward_function(self.historical_info)
        
        # 记录最终指标
        if done or truncated:
            self._log_metrics()
            
        return self._get_obs(), reward, done, truncated, self.historical_info[-1]

    def _log_metrics(self):
        self.calculate_metrics()
        if self.verbose:
            print(" | ".join(f"{k}: {v}" for k, v in self.results_metrics.items()))
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
            idx = self._idx,
            step = self._step,
            date = self.df.index.values[self._idx],
            position_index = self.positions.index(self._position),
            position = self._position,
            real_position = self._portfolio.position(price),
            data = dict(zip(self._info_columns, self._info_array[self._idx])),
            portfolio_valuation = self._portfolio.valorisation(price),
            portfolio_distribution = portfolio_dist,
            reward = 0
        )

    # 保留指标计算方法
    def add_metric(self, name, function):
        self.log_metrics.append({
            'name': name,
            'function': function
        })
        
    def calculate_metrics(self):
        market_return = (self.historical_info['data_close', -1] / self.historical_info['data_close', 0] -1) * 100
        portfolio_return = (self.historical_info['portfolio_valuation', -1] / self.portfolio_initial_value -1) * 100
        
        self.results_metrics = {
            "Market Return": f"{market_return:.2f}%",
            "Portfolio Return": f"{portfolio_return:.2f}%"
        }
        for metric in self.log_metrics:
            self.results_metrics[metric['name']] = metric['function'](self.historical_info)

    def log(self):
        if self.verbose:
            print(" | ".join(f"{k}: {v}" for k, v in self.results_metrics.items()))

class MultiDatasetTradingEnv(TradingEnv):
    def __init__(self,
                dataset_dir, 
                *args,
                preprocess = lambda df: df,
                episodes_between_dataset_switch = 1,
                **kwargs):
        
        # 过滤掉可能存在的无效参数
        kwargs.pop('borrow_interest_rate', None)
        
        self.dataset_dir = dataset_dir
        self.preprocess = preprocess
        self.episodes_between_dataset_switch = episodes_between_dataset_switch
        self.dataset_pathes = glob.glob(self.dataset_dir)
        
        if not self.dataset_pathes:
            raise FileNotFoundError(f"No datasets found matching: {self.dataset_dir}")
            
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