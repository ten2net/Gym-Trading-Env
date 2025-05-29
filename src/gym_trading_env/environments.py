from typing import Literal, Optional, Dict, Any
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np
import glob
from pathlib import Path
import logging
from .utils.history import History
from .utils.portfolio import Portfolio, TargetPortfolio


def basic_reward_function(history: History, step: int = 0, lambda_param: float = 0.1, eta: float = 0.5):
    R_t = np.log(history["portfolio_valuation", -1] /
                 history["portfolio_valuation", -2])
    return R_t


def dynamic_feature_last_position_taken(history):
    return history['position', -1]


def dynamic_feature_real_position(history):
    return history['real_position', -1]
def dynamic_feature_step_progress(history):
    return history['step_progress', -1]
def dynamic_feature_portfolio_return(history):
    return history['portfolio_return', -1]


class TradingEnv(gym.Env):
    metadata = {'render_modes': ['human', 'logs'], 'render_fps': 10}

    def __init__(
        self,
        df: pd.DataFrame,
        positions: list = [0, 1],
        dynamic_feature_functions=[dynamic_feature_real_position, dynamic_feature_step_progress, dynamic_feature_portfolio_return],
        reward_function=basic_reward_function,
        windows: Optional[int] = None,
        trading_fees: float = 0.0005,
        portfolio_initial_value: float = 1e6,
        initial_position: str = 'random',
        max_episode_duration: Any = 'max',
        target_return: float = 0.15,
        stop_loss=0.10,
        name: str = "Stock",
        render_mode: Optional[str] = None,
        verbose: int = 1
    ):
        # 参数校验
        self._validate_parameters(positions, initial_position)

        # 初始化参数
        self._init_parameters(locals())

        # 数据处理
        self._process_data(df)

        # 初始化空间
        self._init_spaces()

        # 初始化日志
        self._init_logging()

        # 运行时状态
        self.reset()

    def _validate_parameters(self, positions, initial_position):
        if not all(0 <= pos <= 1 for pos in positions):
            raise ValueError("All positions must be between 0 and 1")
        if initial_position not in positions and initial_position != 'random':
            raise ValueError("Invalid initial position")

    def _init_parameters(self, params):
        # 省略参数初始化细节
        self.max_episode_duration = params['max_episode_duration']
        self.name = params['name']
        self.verbose = params['verbose']
        self.positions = params['positions']
        self.dynamic_feature_functions = params['dynamic_feature_functions']
        self.reward_function = params['reward_function']
        self.windows = params['windows']
        self.trading_fees = params['trading_fees']
        self.portfolio_initial_value = float(params['portfolio_initial_value'])
        self.initial_position = params['initial_position']
        self.render_mode = params['render_mode']
        self.target_return = params['target_return']
        self.stop_loss = params['stop_loss']
        self.training_steps = 0

        self.log_metrics = []

    def _process_data(self, df):
        df = df.copy()

        # 分离特征列和信息列
        self._features_columns = [
            col for col in df.columns if "feature" in col]
        self._info_columns = list(
            set(df.columns) - set(self._features_columns))
        self._nb_features = len(self._features_columns)
        self._nb_static_features = self._nb_features

        # 添加动态特征占位
        for i in range(len(self.dynamic_feature_functions)):
            df[f"dynamic_feature__{i}"] = 0.0
            self._features_columns.append(f"dynamic_feature__{i}")
            self._nb_features += 1

        # 转换为numpy数组
        self.df = df
        self._obs_array = np.array(
            df[self._features_columns], dtype=np.float32)
        self._info_array = np.array(df[self._info_columns])
        self._price_array = np.array(df["close"])
        self._dates = df.index

    def _init_spaces(self):
        self.action_space = spaces.Discrete(len(self.positions))

        obs_shape = [self._nb_features]
        if self.windows is not None:
            obs_shape = [self.windows, self._nb_features]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs_shape,
            dtype=np.float32
        )

    def _init_logging(self):
        self.logger = logging.getLogger(f"TradingEnv-{self.name}")
        self.logger.setLevel(logging.INFO if self.verbose else logging.WARNING)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 重置运行时状态
        self._idx = self.windows - 1 if self.windows else 0
        if isinstance(self.max_episode_duration, int):
            self._idx = np.random.randint(
                self._idx,
                len(self.df) - self.max_episode_duration
            )        
        self._step = 0
        self._day_open_value = self.portfolio_initial_value
        self._day_close_value = self.portfolio_initial_value
        self._max_portfolio_value = self.portfolio_initial_value
        self._terminated = False
        self._truncated = False
        self._successful_termination = False
        self._termination_reason = ""
        self.consecutive_ups = 0 # 趋势持续时间计数器，用于添加趋势持续时间奖励
        self.consecutive_downs = 0 # 趋势持续时间计数器，用于添加趋势持续时间奖励

        # 初始化投资组合
        self._init_portfolio()

        # 初始化历史记录
        initial_distribution = {
            "asset": self.portfolio_initial_value / self._get_price(),
            "fiat": 0.0,
            "next_day_available_asset": 0.0
        }
        self.history = History(max_size=len(self.df))
        self.history.set(
            idx=self._idx,
            step=self._step,
            step_progress=(1 - self._step / self.max_episode_duration) if isinstance(self.max_episode_duration, int) else (1-self._step / len(self.df)),
            date=self.current_date,
            position_index=self.positions.index(self._position),
            position=self._position,
            real_position=self._position,
            data=dict(zip(self._info_columns, self._info_array[self._idx])),
            portfolio_valuation=self.portfolio_initial_value,
            portfolio_return=0,
            portfolio_distribution=initial_distribution,  # 使用预定义结构
            price=self._get_price(),
            reward=0,
            target_achieved=self._successful_termination,
            termination=self._terminated,
            termination_reason=self._termination_reason,
            truncated=self._truncated
        )
        self._update_history()

        return self._get_obs(), {}

    def _init_portfolio(self):
        self._position = (np.random.choice(self.positions)
                          if self.initial_position == 'random'
                          else self.initial_position)
        self.portfolio = TargetPortfolio(
            position=self._position,
            value=self.portfolio_initial_value,
            price=self._get_price()
        )

    def add_metric(self, name, function):
        self.log_metrics.append({
            'name': name,
            'function': function
        })

    def calculate_metrics(self):
        market_return = (
            self.history['data_close', -1] / self.history['data_close', 0] - 1) * 100
        portfolio_return = (
            self.history['portfolio_valuation', -1] / self.portfolio_initial_value - 1) * 100
        self.results_metrics = {
            "回合结束原因": self._termination_reason,
            "总步数": self._step,
            "基准收益率": f"{market_return:.2f}%",  # 市场基准收益率
            "策略收益率": f"{portfolio_return:.2f}%",  # 策略组合收益率

        }
        for metric in self.log_metrics:
            self.results_metrics[metric['name']
                                 ] = metric['function'](self.history)

    def _log_metrics(self):
        self.calculate_metrics()
        if self.verbose:
            print(" | ".join(f"{k}: {v}" for k,
                  v in self.results_metrics.items()))
            self.logger.info(" | ".join(f"{k}: {v}" for k,
                                        v in self.results_metrics.items()))

    def step(self, action):

        # 执行交易
        self._execute_trade(action)

        # 更新索引
        self._idx += 1
        self._step += 1
        self.training_steps += 1

        # 在更新历史记录前检查终止状态
        # self._check_termination()

        # 在更新历史记录前计算奖励
        reward = self._calculate_reward()

        # 更新历史记录
        self._update_history(reward)

        # 处理新交易日
        if self._is_new_day():
            self._handle_new_day()

        return self._get_obs(), reward, self._terminated, self._truncated, self.history[-1]

    def _execute_trade(self, action):
        target_position = self.positions[action]
        current_price = self._get_price()

        self._position = self.positions[action]

        self.portfolio.trade_to_position(
            position=target_position,
            price=current_price,
            trading_fees=self.trading_fees
        )

    def _calculate_reward(self):
        current_price =  self._get_price()      
        current_value: float =  float(self.portfolio.valorisation(current_price))
        step: int = self._step
        max_steps: int = self.max_episode_duration if isinstance(self.max_episode_duration, int) else len(self.df)        
        
        reward, self._terminated, self._truncated, self.consecutive_ups, self.consecutive_downs = self.reward_function(
            history=self.history,
            current_price = current_price,
            current_value = current_value, 
            step = step, 
            max_steps = max_steps,
            target_profit=self.target_return,
            stop_loss=self.stop_loss,
            consecutive_ups=self.consecutive_ups,
            consecutive_downs=self.consecutive_downs,
            training_steps=self.training_steps
            )
        return reward

    def _handle_new_day(self):
        self._day_open_value = self.portfolio.valorisation(
            self._get_ohlcv('open'))
        self.portfolio.update_day()
        self.logger.info(f"New trading day: {self.current_date}")

    def _update_history(self, reward=0):

        price = self._get_price()
        # 调整资产分布字段
        portfolio_dist = {
            "asset": self.portfolio.asset,
            "fiat": self.portfolio.fiat,
            "next_day_available_asset": self.portfolio.next_day_available_asset
        }
        self.history.add(
            idx=self._idx,
            step=self._step,
            step_progress=self._step / self.max_episode_duration if isinstance(self.max_episode_duration, int) else self._step / len(self.df),
            date=self.current_date,
            position_index=self.positions.index(self._position),
            position=self._position,
            real_position=self.portfolio.position(price),
            data=dict(zip(self._info_columns, self._info_array[self._idx])),
            portfolio_valuation=self.portfolio.valorisation(price),
            portfolio_return=self.portfolio.valorisation(price)  / self.portfolio_initial_value - 1,
            portfolio_distribution=portfolio_dist,
            price=self._get_price(),
            reward=reward,
            target_achieved=self._successful_termination,
            termination=self._terminated,
            termination_reason=self._termination_reason,
            truncated=self._truncated
        )

    def _get_obs(self):
        # 更新动态特征
        for i, func in enumerate(self.dynamic_feature_functions):
            self._obs_array[self._idx,
                            self._nb_static_features + i] = func(self.history)

        if self.windows:
            return self._obs_array[self._idx - self.windows + 1: self._idx + 1]
        return self._obs_array[self._idx]

    def _get_price(self, delta=0):
        return self._price_array[self._idx + delta]

    def _get_ohlcv(self, item: Literal["open", "high", "low", "close", "volume"], delta: int = 0) -> float:
        target_idx = self._idx + delta
        if not 0 <= target_idx < len(self.df):
            raise IndexError(f"Index {target_idx} out of bounds")

        current_date = self._dates[target_idx].date()
        mask = (self._dates.date == current_date) & (
            self._dates <= self._dates[target_idx])
        daily_data = self.df[mask]

        if item == 'open':
            return daily_data['open'].iloc[0]
        elif item == 'high':
            return daily_data['high'].max()
        elif item == 'low':
            return daily_data['low'].min()
        elif item == 'close':
            return daily_data['close'].iloc[-1]
        elif item == 'volume':
            return daily_data['volume'].sum()
        else:
            raise ValueError(f"Invalid OHLCV item: {item}")

    def _is_new_day(self):
        if self._idx == 0:
            return True
        return self._dates[self._idx].date() != self._dates[self._idx-1].date()

    @property
    def current_date(self):
        return self._dates[self._idx]

    def render(self):
        if self.render_mode == 'human':
            self._render_human()
        elif self.render_mode == 'logs':
            self._render_logs()

    def _render_human(self):
        print(f"\nStep: {self._step}")
        print(f"Date: {self.current_date}")
        print(
            f"Portfolio Value: {self.portfolio.valorisation(self._get_price()):,.2f}")
        print(
            f"Current Position: {self.portfolio.position(self._get_price()):.1%}")

    def _render_logs(self):
        # 增强日志显示终止原因
        if self._terminated:
            status_msg = f"回合完成: {self._termination_reason}"
        elif self._truncated:
            status_msg = "回合截断 (达到最大步数)"
        else:
            status_msg = "正在运行"
        self.logger.info(f"[Step {self._step}] {status_msg} | "
                         f"Value: {self.portfolio.valorisation(self._get_price()):,.2f} "
                         f"Position: {self.portfolio.position(self._get_price()):.1%}")


class MultiDatasetTradingEnv(TradingEnv):
    def __init__(
        self,
        dataset_dir: str,
        *args,
        preprocess=lambda df: df,
        episodes_between_switch=2000,
        **kwargs
    ):
        self.dataset_paths = sorted(
            glob.glob(str(Path(dataset_dir) / "*.pkl")))
        if not self.dataset_paths:
            raise FileNotFoundError(f"No datasets found in {dataset_dir}")
        self.current_dataset = None
        self.episodes_between_switch = episodes_between_switch
        self.episode_count = 0
        self.preprocess = preprocess
        init_dataset = np.random.choice(self.dataset_paths)
        raw_df = pd.read_pickle(init_dataset)
        processed_df = self.preprocess(raw_df)
        super().__init__(processed_df, *args, **kwargs)
        self._load_next_dataset()
  

    def _load_next_dataset(self):
        self.current_dataset = np.random.choice(self.dataset_paths)
        raw_df = pd.read_pickle(self.current_dataset)
        processed_df = self.preprocess(raw_df)
        self._process_data(processed_df)

    def reset(self, **kwargs):
        # print( self.episode_count)
        if self.episode_count % self.episodes_between_switch == 0:
            self._load_next_dataset()
            print(self.current_dataset)

        self.episode_count += 1
        return super().reset(**kwargs)
