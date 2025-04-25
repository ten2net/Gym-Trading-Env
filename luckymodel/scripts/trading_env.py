from datetime import datetime
import math
import random
import sys
from pathlib import Path
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
from stable_baselines3.common.utils import get_schedule_fn


# 路径处理（必须放在其他导入之前）
sys.path.append(str(Path(__file__).resolve().parent.parent))
import os
import argparse
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import List, Optional, Dict, Any
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback,EvalCallback

from configs_loader import load_config
from data.datasets.train_dataset import StockDataset
from data.processor.normalizer import MinMaxNormalizer
from evaluators.callbackes.lstm_callback import EnhancedLSTMCallback
from evaluators.callbackes.stats_callback import EpisodeStatsCallback
from envs.trade_history import TradeHistory

class StockTradingEnv(gym.Env):
    """适用于LSTM-PPO的股票交易强化学习环境"""
    
    metadata = {'render.modes': ['human']}

    def __init__(self, 
                 data: Dict[str, np.ndarray],
                 feature_names: List[str],
                 normalizers: Dict[str, MinMaxNormalizer],
                 mode: str = 'train',
                 initial_balance: float = 1e6,
                 commission: float = 0.001,
                 window_size: int = 10,
                 seed:int = 42,
                 debug=False):
        """
        :param data: 包含'train'和'val'键的数据字典
        :param feature_names: 特征名称列表
        :param normalizers: 归一化器字典
        :param mode: 环境模式 ('train'/'val')
        :param initial_balance: 初始资金
        :param commission: 交易佣金率
        :param window_size: 观测窗口大小
        """
        super(StockTradingEnv, self).__init__()
        
        # 环境参数
        self.mode = mode
        self.window_size = window_size
        self.commission = commission
        self.initial_balance = initial_balance
        self.feature_names = feature_names
        self.n_features = len(feature_names)
        self.normalizers = normalizers
        self.debug=debug
        
        self.dates = None  # 存储实际日期序列
        self.current_date = None  # 当前步骤对应日期
        
        # 初始化交易历史记录
        self.trade_history = TradeHistory()  
        
        # 添加净值历史记录容器
        self.net_worth_history = []
        self.max_net_worth = initial_balance  # 追踪历史最高净值  
        
        self.position_history = []  # 每日持仓
        self.action_history = []  # 动作记录 
        self.reward_history =[]       
        
        # 加载数据集
        self._load_dataset(data)
        
        # 定义动作和观测空间
        self.action_space = spaces.Box(
          low=-1, 
          high=1, 
          shape=(1,), 
          dtype=np.float32)  # [-1,1]表示卖出到买入
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(window_size, self.n_features),  # LSTM需要序列输入
            dtype=np.float32
        )
        
        # 重置环境
        self.reset(seed=seed)

    def _load_dataset(self, data: Dict[str, np.ndarray]):
        """加载预处理后的数据集"""
        if self.mode == 'train':
            self.dataset = data['train']
            self.prices = self._denormalize_prices(data['train'])
            self.dates = data['train_dates']  # 新增日期存储
            self.raw_data = pd.DataFrame(
                {
                    "Date":self.dates,
                    "Close" : self.prices
                }
            )            
        else:
            self.dataset = data['val']
            self.prices = self._denormalize_prices(data['val'])
            self.dates = data['val_dates']  # 验证集日期
            self.raw_data = pd.DataFrame(
                {
                    "Date":self.dates,
                    "Close" : self.prices
                }
            )             
            
        # 添加日期长度校验
        assert len(self.dates) == len(self.dataset), \
            f"日期与数据长度不匹配: 数据{len(self.dataset)}条，日期{len(self.dates)}条"            
        self.n_steps = len(self.dataset) - self.window_size
        
    def _denormalize_prices(self, data: np.ndarray) -> np.ndarray:
        """反归一化价格数据"""
        # 获取close特征索引
        close_idx = self.feature_names.index('close')
        
        # 正确重塑数据形状为 (n_samples, 1)
        close_scaled = data[:, -1, close_idx].reshape(-1, 1)  # 三维转二维
        
        # 转换为DataFrame并命名列
        df_close = pd.DataFrame(close_scaled, columns=['close'])
        
        # 获取当前股票的归一化器
        stock_code = next(iter(self.normalizers.keys()))
        normalizer = self.normalizers[stock_code]
        
        # 执行逆变换
        denorm_close = normalizer.inverse_transform(df_close)
        return denorm_close['close'].values.flatten()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)      
        """重置环境到初始状态"""
        super().reset(seed=seed)
        starting_point = random.choice(range(int(len(self.dates) * 0.5))) 
        # print("...starting_point=",starting_point)
        self.current_step = self.window_size #+ starting_point # 跳过初始窗口
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.done = False
        
        # 初始化当前日期
        self.current_date = self.dates[self.current_step] if self.dates is not None else None
                
        # 重置时初始化历史记录
        self.trade_history.reset()   
        self.position_history.clear()
        self.action_history.clear()               
        self.net_worth_history = [self.initial_balance] * (self.window_size + 1)
        self.max_net_worth = self.initial_balance 
        self.reward_history.clear()        
        
        obs = self._next_observation()
        
        return obs,{} 

    def _next_observation(self) -> np.ndarray:
        """获取当前观测"""
        # obs = self.dataset[self.current_step - self.window_size : self.current_step]
        obs = self.dataset[self.current_step]
        
        # 确保形状为 (window_size, n_features)
        assert obs.shape == (self.window_size, len(self.feature_names)), \
            f"观测形状错误: 期望({self.window_size}, {len(self.feature_names)}), 实际{obs.shape}" 
                   
        return obs.astype(np.float32)

    def _take_action(self, action: float):
        """执行交易动作"""
        current_price = self.prices[self.current_step]
        # print(self.current_step,self.current_date, current_price)
        action_type = action[0]  # 从[-1,1]转换为交易比例
        
        # 计算目标持仓比例
        target_ratio = (action_type + 1) / 2  # 转换为[0,1]
        target_value = self.net_worth * target_ratio
        delta = target_value - self.shares_held * current_price
        
        # 滑点模拟（0.1%随机波动）
        slippage = current_price * np.random.uniform(-0.001, 0.001)
        executed_price = current_price + slippage        
        
        transaction_cost = 0.0
        shares_bought = 0
        shares_sold = 0               
            
        # 执行交易
        if delta > 0:  # 买入
            max_buyable = self.balance / executed_price
            shares_bought = min(delta / executed_price, max_buyable)
            shares_bought = (shares_bought // 100) * 100
            shares_bought = max(shares_bought, 0)
            if shares_bought > 0:
                transaction_cost = self._calc_commission(shares_bought, executed_price, 'buy')
                self.shares_held += shares_bought
                self.balance -= shares_bought * executed_price            
        else:  # 卖出
            shares_to_sell = -delta / executed_price
            max_sellable = self.shares_held
            
            # 计算最少卖出数量（持仓的2/3，向上取整到100的整数倍）
            min_shares = math.ceil((max_sellable * 1 / 4) / 100) * 100
            min_shares = min(min_shares, max_sellable)  # 不能超过持仓量
            
            # 确定最终要卖出的数量
            shares_sold_unrounded = max(shares_to_sell, min_shares)
            shares_sold_unrounded = min(shares_sold_unrounded, max_sellable)
            
            # 转换为100的整数倍（向下取整）
            shares_sold = (shares_sold_unrounded // 100) * 100
            
            # 确保最终卖出量不小于min_shares的向下取整
            min_shares_floor = (min_shares // 100) * 100
            shares_sold = max(shares_sold, min_shares_floor)
            shares_sold = min(shares_sold, max_sellable)
            shares_sold = max(shares_sold, 0)  # 防止负数

            if shares_sold > 0:
                transaction_cost = self._calc_commission(shares_sold, executed_price, 'sell')
                self.shares_held -= shares_sold
                self.balance += shares_sold * executed_price        

        # 更新净资产
        self.net_worth = self.balance + self.shares_held * current_price
        
        # 记录交易历史
        if shares_bought > 0 or shares_sold > 0:
            transaction_cost = abs(delta) * self.commission
            self.trade_history.add_trade(
                step=self.current_step,
                action=delta,
                price=executed_price,
                shares=shares_bought if delta > 0 else -shares_sold,
                cost=transaction_cost,
                current_date=self.dates[self.current_step]
            )
            # 更新持仓量
            self.trade_history.position = self.shares_held        

    def step(self, action: np.ndarray) -> tuple:
        """执行一步环境更新"""
        self.current_date = self.dates[self.current_step] if self.dates is not None else None
        
        self._take_action(action) # 执行交易
        
        # 获取下一观测
        self.current_step += 1
        obs = self._next_observation()
        
        # 计算奖励
        reward, reward_detail = self.calculate_reward()
    
        self.last_action = action  # 记录action，用于render函数显示
        self.last_reward = reward # 记录奖励，用于render函数显示
        self.last_reward_detail = reward_detail # 记录奖励构成，用于render函数显示
        # 终止条件
        self.done = (self.current_step >= len(self.dataset) - 1) or (self.net_worth < (self.initial_balance * 0.5))
        
        truncated=False #truncated标志
        
        # 更新净值历史（保持固定长度）
        if len(self.net_worth_history) >= 10000:  # 限制历史长度防止内存溢出
            self.net_worth_history.pop(0)
        self.net_worth_history.append(self.net_worth)
        
        self.position_history.append({
            'timestamp': pd.to_datetime(self.dates[self.current_step]),
            'position': self.shares_held,
            'price': self.prices[self.current_step],
            'cash': self.balance
        })         
        
        # 更新历史最高净值
        self.max_net_worth = max(self.max_net_worth, self.net_worth)        
        
        # 调用render（每100步记录）
        self.render(mode='human', log_freq=512)  
              
        return (obs, 
                reward, 
                self.done,
                truncated,
                {})
        
    # 新奖励函数设计目标
    GOALS = {
        'positive_mean': True,    # 保持平均奖励为正
        'std_range': (10, 50),    # 奖励标准差合理区间
        'scale_factor': 1.0       # 输出范围[-1,1]
    }
    
    def calculate_returns(self, history: list, window: int = 50) -> np.ndarray:
        """安全计算收益率
        
        Args:
            history: 净值历史列表
            window: 计算窗口长度
            
        Returns:
            收益率数组，形状(window-1,)
        """
        # 转换为NumPy数组
        arr = np.array(history[-window:], dtype=np.float32)
        
        # 安全检查
        if len(arr) < 2:
            return np.zeros(0)
        
        # 计算差分
        diffs = np.diff(arr)
        
        # 分母处理
        denominators = arr[:-1]
        denominators = np.where(np.abs(denominators) > 1e-6, denominators, 1e-6)
        
        # 收益率计算
        returns = diffs / denominators
        
        return returns 
      
    def _calc_commission(self, shares: float, price: float, side: str) -> float:
        """计算交易手续费
        :param side: 'buy'/'sell'
        """
        # 费率结构示例：买入0.03%，卖出0.1%，最低5元
        rate = 0.00015 if side == 'buy' else 0.00015
        amount = abs(shares) * price
        commission = amount * rate
        
        # 最低手续费限制
        min_fee = 5.0
        return max(commission, min_fee)       
    
    def calculate_alpha(self):
        """对数收益率版阿尔法效应计算"""
        # 策略对数收益率
        strategy_return = np.log1p((self.net_worth - self.initial_balance) / self.initial_balance)
        
        # 市场对数收益率（30日移动窗口）
        market_prices = self.prices[max(0, self.current_step-30):self.current_step+1]
        market_log_returns = np.diff(np.log(market_prices))  # 对数收益率序列
        
        # 市场基准收益率（年化处理）
        market_annualized = np.mean(market_log_returns) * 252
        
        # 阿尔法效应计算
        alpha = (strategy_return - market_annualized) * 100  # 缩放系数
        
        # 数值稳定性处理
        if not np.isfinite(alpha):
            alpha = np.clip(alpha, -5, 5)  # 限制异常值
            self.logger.warn(f"非数值阿尔法值，已截断至{alpha:.2f}")
        alpha = np.sign(alpha) * np.log1p(abs(alpha))  # 对数压缩    
        return alpha  
    def calculate_reward0001(self):   
        """计算奖励值"""
        if self.current_step == self.window_size:
            return 0
        else:
            # 单步收益率计算
            current_net = self.net_worth
            previous_net = self.net_worth_history[-2]
            step_return = (current_net - previous_net) / previous_net  # 单步百分比收益

            # 现金惩罚（按交易金额比例）
            if len(self.trade_history.history) > 0:
                last_trade = self.trade_history.history[-1]
                trade_amount = abs(last_trade["shares"] * last_trade["price"])
                cash_penalty = max(0, trade_amount + last_trade["cost"] - self.balance) / self.initial_balance  # 标准化
            else:
                cash_penalty = 0

            # 动态奖励公式
            return_scale = 1       # 单步收益权重
            cash_penalty_scale = 0.01 # 现金惩罚权重
            
            reward = (
                step_return * return_scale 
                # - cash_penalty * cash_penalty_scale
            )

            clip_min, clip_max = -1, 1
                
            reward = np.clip(reward, clip_min, clip_max)
            
            # 记录历史
            self.reward_history.append(reward)
            
            debug_msg = f"Ret:{step_return:.2f} | CashPen:{cash_penalty:.2f}"         
            award_detail = f"{debug_msg} "
            return reward ,award_detail          
    def calculate_reward001(self):   
        """计算奖励值"""
        if self.current_step == self.window_size:
            return 0  
        else:
            current_price = self.prices[self.current_step]
            assets = self.shares_held * current_price #持仓市值
            cash_penalty_proportion = 0.1 #现金不够惩罚比例
            # 现金不足惩罚计算
            cash_penalty = max(0, (assets * cash_penalty_proportion - self.balance))
            adjusted_assets = assets - cash_penalty
            
            # 标准化奖励（考虑回合时长）
            reward =100 * ((adjusted_assets / self.initial_balance) - 1)
            reward /= (self.current_step - self.window_size)
            reward = float(np.clip(reward,-0.1,0.5))
            award_detail = f"cash_penalty:{cash_penalty:.3f} "
            return reward ,award_detail          
    def calculate_reward(self):
       return np.log(self.net_worth_history[-1] / self.net_worth_history[-2]),""
    def calculate_reward111(self):
        """奖励函数，平衡各组件权重"""
        # 获取当前日期的星期信息（周一=0, 周日=6）
        if self.current_date is not None:
            weekday = pd.to_datetime(self.current_date).weekday()
            # 添加周末效应惩罚
            weekend_penalty = 0.5 if weekday in [4,5] else 0  # 周五、周六惩罚
        else:
            weekend_penalty = 0        
        
        # 收益率计算（考虑时间衰减因子）
        steps_decay = 0.95 ** (self.current_step / 100)  # 随时间降低收益权重
        return_pct = (self.net_worth - self.initial_balance) / self.initial_balance
        return_term = return_pct * 50 * steps_decay  # 缩放并加入衰减
        
        # # 动态市场基准（30日移动平均）
        # market_30d = np.mean(self.prices[max(0, self.current_step-30):self.current_step])
        # market_return = (self.prices[self.current_step] - market_30d) / (market_30d + 1e-9)
        # alpha_term = (return_pct - market_return) * 80  # 增强阿尔法效应
        # 对数收益率版阿尔法效应
        alpha_term = self.calculate_alpha()
        
        # 风险控制（平滑处理）
        drawdown = (self.max_net_worth - self.net_worth) / (self.max_net_worth + 1e-9)
        penalty_term = np.tanh(5 * drawdown)  # 降低惩罚系数
        
        # 交易效率奖励（动态调整权重）
        trade_efficiency = self.trade_history.get_recent_efficiency(window=10)
        efficiency_term = np.log1p(trade_efficiency)  # 对数压缩
        efficiency_term=float(np.clip(efficiency_term, 0.0, 2))

        alpha_term *= 0.5
        return_term *= 0.0
        efficiency_term *= 0.0
        penalty_term *= 0.2
        # 组合奖励
        reward = 0.5 * (
            alpha_term +
            return_term +
            efficiency_term -
            penalty_term
        )
        award_detail = f"Alpha_term: {alpha_term:.3f} | Return_term:{return_term:.3f} | Efficiency_term: { efficiency_term:.3f} | penalty_term: {penalty_term:.3f}"
        return float(np.clip(reward, -3.0, 3.0)) /10.0, award_detail  # 放宽负奖励限制      
    def calculate_reward2(self):
        """优化后的奖励函数"""
        # 基础收益率（年化处理）
        annualized_return = (self.net_worth / self.initial_balance - 1) * (252 / (self.current_step + 1))
        
        # 市场适应性奖励（30日移动平均基准）
        market_30d = np.mean(self.prices[self.current_step-30:self.current_step])
        market_return = (self.prices[self.current_step] - market_30d) / market_30d
        alpha_return = annualized_return - market_return * 2  # 增强阿尔法系数
        
        # 风险控制惩罚（非线性调整）
        drawdown = (self.max_net_worth - self.net_worth) / self.max_net_worth
        risk_penalty = np.log1p(10 * drawdown)  # 改用对数函数平滑
        
        # 交易效率奖励（动态权重）
        trade_efficiency = self.trade_history.get_recent_efficiency(window=10)
        efficiency_bonus = 0.5 * np.tanh(trade_efficiency * 5)  # 压缩到[0,0.5]
        
        # 组合奖励
        reward = (
            0.6 * alpha_return * 5 +        # 调低系数
            0.3 * annualized_return * 10 +  # 增强基础收益
            efficiency_bonus -             # 动态效率奖励
            0.4 * risk_penalty             # 降低惩罚权重
        )
        return float(np.clip(reward, -0.5, 3.0))  # 放宽截断范围       
    def calculate_reward1(self):
        """优化版奖励函数"""
        # 基础收益率项（相对变化）
        return_pct = (self.net_worth - self.initial_balance) / self.initial_balance
        scaled_return = np.tanh(return_pct * 10)  # 压缩到[-1,1]
        
        # 动态风险惩罚项
        # recent_returns = np.diff(self.net_worth_history[-50:]) / self.net_worth_history[-51:-1]
        # recent_returns = np.diff(self.net_worth_history[-50:]) / (self.net_worth_history[-51:-1] + 1e-9)
        recent_returns = self.calculate_returns(self.net_worth_history, window=50)
        volatility = np.std(recent_returns) if len(recent_returns) > 5 else 0
        risk_penalty = np.tanh(volatility * 50)  # 波动越大惩罚越强
        
        # 交易频率惩罚项
        trade_freq = self.trade_history.get_trade_frequency(window=50)
        freq_penalty = np.tanh(trade_freq * 5)  # 将频率映射到[0,1]
        
        # 组合奖励
        reward = (
            0.7 * scaled_return -     # 收益主导项
            0.2 * risk_penalty -     # 风险控制项
            0.1 * freq_penalty       # 交易成本项
        )
        return float(np.clip(reward, -1.0, 1.0))       

    def render(self, mode: str = 'human', log_freq: int = 100):
        """增强版可视化方法
        :param mode: 输出模式 
            - 'human': 控制台打印
            - 'file': 写入日志文件
            - 'silent': 静默模式
        :param log_freq: 日志频率（每多少步记录一次）
        """
        if self.current_step % log_freq != 0:
            return
            
        # 关键指标计算
        current_price = self.prices[self.current_step]
        portfolio_return = (self.net_worth / self.initial_balance - 1) * 100
        
        # 构建日志信息
        log_data = {
            "step": self.current_step,
            "date": self.current_date, 
            "price": round(current_price, 2),
            "balance": round(self.balance, 2),
            "shares": round(self.shares_held, 4),
            "net_worth": round(self.net_worth, 2),
            "return(%)": round(portfolio_return, 2),
            "action": self.last_action[0] if hasattr(self, 'last_action') else None,
            "reward": self.last_reward if hasattr(self, 'last_reward') else None,
            "reward_detail": self.last_reward_detail if hasattr(self, 'last_reward_detail') else None,
        }
        
        # 输出控制
        if mode == 'human':
            self._console_render(log_data)
        elif mode == 'file':
            self._file_render(log_data)
            
    def _console_render(self, data: dict):
        """控制台彩色输出"""
        from termcolor import colored
        date_str = colored(str(data['date'])[:10], 'magenta') if data['date'] else ""
        status = colored(f"Step {data['step'] - self.window_size}", 'cyan') 
        price = colored(f"Price: ¥{data['price']}", 'yellow')
        shares = colored(f"Shares: {data['shares']:.0f}", 'yellow')
        action = colored(f"Action: {data['action']:.2f}", 'green' if data['action'] > 0 else 'red')
        reward = colored(f"Reward: {data['reward']:.4f}", 'green' if data['reward'] > 0 else 'red')
        print(f"{date_str} {status} | {price} | {shares} | {action} | Net Worth: ¥{data['net_worth']:.0f} ({data['return(%)']:.1f}%) | {reward} {data['reward_detail']} ")

    def _file_render(self, data: dict):
        """文件日志记录"""
        import csv
        file_path = "./trading_logs.csv"
        write_header = not Path(file_path).exists()
        
        with open(file_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=data.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(data)
    def get_trade_history_df(self):
        """将交易历史转换为DataFrame"""
        if not self.trade_history:  # 处理空交易记录
            return pd.DataFrame()
        df = pd.DataFrame(self.trade_history.history)
        
        # 基础数据类型转换
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        
        return df  
    def get_portfolio_history_df(self):
        """获取投资组合历史"""
        # 计算有效时间范围
        valid_data_length = len(self.net_worth_history) - self.window_size
        date_start = self.window_size - 1  # 初始填充结束位置
        date_end = date_start + valid_data_length
        
        return pd.DataFrame({
            'timestamp': pd.to_datetime(self.dates[date_start:date_end]),
            'portfolio_value': self.net_worth_history[self.window_size:]
        })
    def get_position_history_df(self):
        """获取每日持仓"""
        return pd.DataFrame(self.position_history)
    
    def get_action_history_df(self):
        """获取动作记录"""
        return pd.DataFrame(self.action_history)                           

# 使用示例
if __name__ == "__main__":
    # 参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='configs/train_config.yml',
                        help='训练配置文件路径')
    parser.add_argument('--log_dir', type=str, default='../logs/',
                        help='TensorBoard日志目录')  
    args = parser.parse_args()

    config = load_config(args.config_file)
    # 初始化环境
    dataset = StockDataset(config).build_dataset()
    env = make_vec_env(
        lambda: StockTradingEnv(
          data=dataset,
          feature_names=dataset['feature_names'],
          normalizers=dataset['normalizers'],
          mode='train',
          window_size=config['data']['window_size']
        ),
        n_envs=1,  # 保持单环境
        vec_env_cls=DummyVecEnv
    )    
  
    # 环境检查
    # check_env(env[0])
    
    # 创建LSTM策略模型
    policy_kwargs = dict(
        lstm_hidden_size=256,  
        net_arch=dict(
          pi=[256, 128],  # 显式定义LSTM层
          vf=[128, 64]),
        enable_critic_lstm=False,  # 关闭Critic的LSTM
        optimizer_class = torch.optim.AdamW,
        optimizer_kwargs=dict(weight_decay=1e-4)
    )  
    
    # 创建余弦退火学习率调度（从3e-4到1e-6）
    initial_lr = 3e-4
    final_lr = 1e-6
    lr_schedule = get_schedule_fn(
        lambda progress: final_lr + 0.5*(initial_lr - final_lr)*(1 + np.cos(np.pi*progress))
    )   
    
    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        learning_rate=lr_schedule,  # 直接传入调度器对象  #3e-4, # 调高初始学习率
        n_steps=128,  # 缩短序列长度
        batch_size=512,
        n_epochs=10,
        gamma=0.98, # 延长收益视野
        ent_coef=0.08, # 初始高探索
        use_sde=True,  # 启用状态依赖探索
        policy_kwargs=policy_kwargs,
        verbose=0,
        device='cuda',
        tensorboard_log=f"{args.log_dir}/stock_trading/"
    )
    
    # 创建评估环境并用 Monitor 包裹。
    # Monitor 包装器通常用于强化学习环境中，自动记录每一回合的长度、奖励等指标
    eval_env_raw = make_vec_env(
        lambda: StockTradingEnv(
          data=dataset,
          feature_names=dataset['feature_names'],
          normalizers=dataset['normalizers'],
          mode='val',
          window_size=config['data']['window_size']
        ),
        n_envs=1,  # 保持单环境
        vec_env_cls=DummyVecEnv
    )   
    # eval_env = Monitor(eval_env_raw, './eval_logs')    
    
    # 评估回调
    eval_callback = EvalCallback(
        eval_env=eval_env_raw,
        best_model_save_path="../checkpoints/best_models/",
        eval_freq=2048,
        n_eval_episodes=10,
        verbose=0,
        deterministic=True
    )     
    
    model.learn(
        total_timesteps=3e5,
        progress_bar=True,
        log_interval=10,
        tb_log_name=f"ppo_lstm",
        callback=[eval_callback],
        reset_num_timesteps=True
    )
    print("训练时间步数=",model.num_timesteps)    
    
    # 保存最终模型
    model.save(os.path.join('../checkpoints', 'final_model.zip'))