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
import torch

sys.path.append("../")

warnings.filterwarnings("ignore", category=ResourceWarning)
warnings.filterwarnings("ignore", message="sys.meta_path is None, Python is likely shutting down")


def train(symbol_train: str,
          symbol_eval: str,
          window_size: int | None = None,
          target_return: float = 0.15,  # 策略目标收益率，超过视为成功完成，给予高额奖励
          min_target_return: float = -0.15  # 最小目标收益率，低于视为失败，给予惩罚
          ):
    # 定义公共环境参数
    common_env_params = {
        'window_size': window_size,
        'eval': False,
        'positions': [0, 0.5, 1],
        'trading_fees': 0.01/100,
        'portfolio_initial_value': 1000000.0,
        'max_episode_duration': 1024,
        'target_return': target_return,
        'min_target_return': min_target_return,
        'max_drawdown': -0.8,
        'daily_loss_limit': -0.8,
        'render_mode': "logs",
        'verbose': 0
    }
    # 创建训练环境（可添加训练特有的参数）
    train_env = make_env(
        symbol=symbol_train,
        **common_env_params
    )

    # 创建评估环境（可添加评估特有的参数）
    eval_env = make_env(
        symbol=symbol_eval,
        **{**common_env_params, 'max_drawdown': -0.8}   # 评估使用更严格条件,使用字典解包优先级（Python 3.5+）
    )
    # 使用PPO算法训练模型
    # initial_lr = 1e-6
    # final_lr = 1e-4
    # # 创建余弦退火学习率调度（从3e-4到1e-6）
    # lr_schedule = get_schedule_fn(
    #     lambda progress: final_lr + 0.5 *
    #     (initial_lr - final_lr)*(1 + np.cos(np.pi*progress))
    # )
    # 线性预热+余弦退火
    # 线性预热可以避免训练初期的不稳定
    # 余弦退火提供平滑的衰减
    # 整体训练过程更加稳定可控    

    initial_lr = 1e-6    # 最终学习率（训练结束时）
    final_lr = 1e-4      # 峰值学习率（预热结束后）
    warmup_ratio = 0.4    # 前30%训练时间用于预热

    def lr_lambda(progress):
        if progress > (1 - warmup_ratio):  
            # 线性预热阶段（progress从1.0→0.7）
            # 目标：学习率从0线性增长到final_lr
            warmup_progress = 1 - (progress - (1 - warmup_ratio)) / warmup_ratio  # 映射[1.0, 0.7]→[0,1]
            return final_lr * warmup_progress
        else:
            # 余弦退火阶段（progress从0.7→0）
            # 目标：学习率从final_lr平滑衰减到initial_lr
            decay_progress = (progress / (1 - warmup_ratio))  # 映射[0.7,0]→[1,0]
            return final_lr + 0.5 * (initial_lr - final_lr) * (1 + np.cos(0.5 * np.pi * decay_progress))

    lr_schedule = get_schedule_fn(lr_lambda)

    device = "cpu" #'cuda' if torch.cuda.is_available() else 'cpu'
    print("torch.cuda.is_available()=", torch.cuda.is_available())
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate= lr_schedule,  # 直接传入调度器对象  #3e-4, # 调高初始学习率
        # policy_kwargs=policy_kwargs,
        n_steps=1024,
        batch_size=512,
        n_epochs=10,
        gamma=0.99,  # 延长收益视野
        ent_coef=0.01,  # 初始高探索
        gae_lambda=0.98,
        verbose=0,
        device=device,
        seed=42,
        tensorboard_log=f"../logs/stock_trading2/")
    # # 评估回调
    # early_stop_callback = StopTrainingOnNoModelImprovement(
    #     max_no_improvement_evals=100,  # 允许的连续无提升评估次数
    #     min_evals=5,                  # 最少评估次数后才开始检查
    #     verbose=1                     # 是否打印日志
    # )
    # eval_callback = EvalCallback(
    #     eval_env=eval_env,
    #     callback_on_new_best=early_stop_callback,  # 绑定早停回调
    #     best_model_save_path="../checkpoints/best_models/",
    #     eval_freq=48 * 22,
    #     n_eval_episodes=10,
    #     verbose=0,
    #     deterministic=True,
    #     render=False
    # )
    # progressBarCallback = ProgressBarCallback()

    model.learn(
        total_timesteps=8e6,
        progress_bar=True,
        log_interval=10,
        tb_log_name=f"ppo_new_reward",
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

    parser.add_argument('--window_size', type=int, default=15)
    parser.add_argument('--target_return', type=float, default=0.3)
    parser.add_argument('--min_target_return', type=float, default=0.05)
    parser.add_argument('--total_timesteps', type=int, default=2e6)

    args = parser.parse_args()
    symbol_train = args.symbol_train
    symbol_eval = args.symbol_eval
    model = train(symbol_train, symbol_eval, window_size=3)
    # 保存模型
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    model_save_path = f"rppo_trading_model_{timestamp}"
    model.save(model_save_path)
