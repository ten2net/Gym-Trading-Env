import sys

from stable_baselines3 import PPO
sys.path.append("../")
import datetime
import argparse
import numpy as np
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement, ProgressBarCallback

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.utils import get_schedule_fn
import warnings
from envs.env import make_env
from callbacks.profit_curriculum_callback import ProfitCurriculumCallback,EpisodeMetricsCallback
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gymnasium.wrappers import RecordEpisodeStatistics

warnings.filterwarnings("ignore", category=ResourceWarning)
warnings.filterwarnings("ignore", message="sys.meta_path is None, Python is likely shutting down")

class RobustCosineSchedule:
    def __init__(self, initial_lr=1e-6, final_lr=1e-3, warmup_ratio=0.2):
        """
        工业级稳健的学习率调度器
        特点：
        1. 双重数值钳制
        2. 浮点误差补偿
        3. 自动范围校正
        """
        self.initial_lr = float(initial_lr)
        self.final_lr = float(final_lr)
        self.warmup_ratio = float(warmup_ratio)
        
        # 自动校准参数
        self._validate_parameters()
        
    def _validate_parameters(self):
        """参数合法性检查"""
        assert self.initial_lr > 0, "初始学习率必须是正数"
        assert self.final_lr >= self.initial_lr, "峰值学习率不能小于初始值"
        assert 0 < self.warmup_ratio < 1, "预热比例必须在(0,1)区间"
        
        # 防止浮点误差导致的计算问题
        self.epsilon = np.finfo(float).eps * 100
        
    def __call__(self, progress):
        """
        progress: 从1.0（开始）到0.0（结束）的进度
        返回：计算后的学习率
        """
        # 强制进度值合法化
        progress = np.clip(float(progress), 0.0, 1.0)
        
        # 阶段1：线性预热
        if progress > (1 - self.warmup_ratio):
            warmup_progress = (1 - progress) / self.warmup_ratio  # 更直观的计算
            return self._safe_lerp(0.0, self.final_lr, warmup_progress)
        
        # 阶段2：改进型余弦退火
        decay_progress = progress / (1 - self.warmup_ratio)
        cosine_value = np.cos(np.pi * decay_progress)
        
        # 使用线性插值替代原始公式
        return self._safe_lerp(self.final_lr, self.initial_lr, 0.5*(1 + cosine_value))
    
    def _safe_lerp(self, start, end, weight):
        """带保护的线性插值计算"""
        weight = np.clip(weight, 0.0, 1.0)
        # 添加微小偏移防止浮点误差
        return start + (end - start + self.epsilon) * weight  
     
def train(symbol_train: str,
          symbol_eval: str,
          window_size: int | None = None,
          target_return: float = 0.1,  # 策略目标收益率，超过视为成功完成，给予高额奖励
          stop_loss: float = 0.1,  # 最小目标收益率，低于视为失败，给予惩罚
          total_timesteps: int = 1e7
          ):
    # 定义公共环境参数
    common_env_params = {
        'window_size': window_size,
        'eval': False,
        'positions': [0, 0.5, 1],
        'trading_fees': 0.01/100,
        'portfolio_initial_value': 1000000,
        'max_episode_duration': 48 * 10,
        'target_return': target_return,
        'stop_loss': stop_loss,
        'render_mode': "logs",
        'verbose': 0
    }
    # 创建训练环境（可添加训练特有的参数）
    train_env = make_env(
        symbol=symbol_train,
        **common_env_params
    )
    
    # 创建环境并包装为 VecNormalize
    base_env = RecordEpisodeStatistics(train_env)  # 关键步骤！
    train_env = DummyVecEnv([lambda: base_env])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False)
   

    # 创建评估环境（可添加评估特有的参数）
    eval_env = make_env(
        symbol=symbol_eval,
        **{**common_env_params, 'stop_loss': 0.1}   # 评估使用更严格条件,使用字典解包优先级（Python 3.5+）
    )
    
    eval_env = RecordEpisodeStatistics(eval_env)  # 关键步骤！
    eval_env = DummyVecEnv([lambda: eval_env])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)  
    # eval_env = Monitor(eval_env, './eval_logs')  
    # 使用PPO算法训练模型   

    min_lr = 2e-5    # 训练结束时
    max_lr = 2e-4      # 预热结束后的峰值学习率
    warmup_ratio = 0.05    # 前10%训练时间用于预热
    # 在训练神经网络时，前20%训练时间用于预热（Warmup）是一个重要的策略。
    # 其核心好处是稳定初始训练过程，避免模型在初始阶段由于随机权重导致的剧烈波动。
    # ​​问题​​：网络初始权重是随机初始化的，直接使用大学习率会导致：
    #     梯度方向不稳定（不同batch的梯度差异大）
    #     损失函数剧烈震荡
    # ​​解决​​：从小学习率逐步增大，让模型先"摸索"数据分布    
    lr_scheduler = RobustCosineSchedule(
        initial_lr=min_lr,
        final_lr=max_lr,
        warmup_ratio=warmup_ratio
    )   

    device = "cpu" #'cuda' if torch.cuda.is_available() else 'cpu'
    print("torch.cuda.is_available()=", torch.cuda.is_available())
    # net_arch = [ {"pi": [256,256], "vf": [256,256]} ]

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate= lr_scheduler,  
        # policy_kwargs=policy_kwargs,
        n_steps=1024,
        batch_size=512,  # 需满足batch_size <= n_steps
        n_epochs=10,
        clip_range=0.15,
        gamma=0.95,  # 延长收益视野
        ent_coef=0.12,  # 初始高探索
        # gae_lambda=0.98,
        # policy_kwargs={'net_arch': net_arch},
        verbose=0,
        device=device,
        seed=42,
        tensorboard_log=f"../logs/stock_trading2/")
    # 评估回调
    early_stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=100,  # 允许的连续无提升评估次数
        min_evals=50,                  # 最少评估次数后才开始检查
        verbose=1                     # 是否打印日志
    )
    eval_callback = EvalCallback(
        eval_env=eval_env,
        # callback_on_new_best=early_stop_callback,  # 绑定早停回调
        best_model_save_path="../checkpoints/best_models/",
        eval_freq=48 * 22,
        n_eval_episodes=10000,
        verbose=0,
        deterministic=True,
        render=False
    )
    # progressBarCallback = ProgressBarCallback()
    
    # 创建回调实例
    curriculum_callback = ProfitCurriculumCallback()
      
    model.learn(
        total_timesteps=total_timesteps,  #8_000_000,
        progress_bar=True,
        log_interval=10,
        tb_log_name=f"ppo_new_reward",
        callback=[EpisodeMetricsCallback()],
        # callback=[curriculum_callback],
        # callback=[eval_callback,
        #           progressBarCallback],
        reset_num_timesteps=True
    )
    print("训练时间步数=", model.num_timesteps)
    return model


if __name__ == "__main__":
    # 参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol_train', type=str, default='300059',
                        help='训练使用的股票代码')
    parser.add_argument('--symbol_eval', type=str, default='300308',
                        help='评估使用的股票代码')

    parser.add_argument('--window_size', type=int, default=6)
    parser.add_argument('--target_return', type=float, default=0.05)
    parser.add_argument('--stop_loss', type=float, default=0.5)
    parser.add_argument('--total_timesteps', type=int, default=2e6)

    args = parser.parse_args()
    symbol_train = args.symbol_train
    symbol_eval = args.symbol_eval
    window_size = args.window_size
    target_return = args.target_return
    stop_loss = args.stop_loss
    total_timesteps = args.total_timesteps
    
    model = train(
        symbol_train=symbol_train, 
        symbol_eval =symbol_eval, 
        window_size=None,
        target_return=target_return,
        stop_loss=stop_loss,
        total_timesteps=total_timesteps
        )
    # 保存模型
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    model_save_path = f"rppo_trading_model_{timestamp}"
    model.save(model_save_path)
