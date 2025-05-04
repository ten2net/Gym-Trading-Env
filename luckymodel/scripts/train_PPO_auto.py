import sys
import json

sys.path.append("../")

import optuna
from stable_baselines3 import PPO
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

from stable_baselines3.common.evaluation import evaluate_policy
warnings.filterwarnings("ignore", category=ResourceWarning)
warnings.filterwarnings("ignore", message="sys.meta_path is None, Python is likely shutting down")



def evaluate_model(model, env, n_episodes=10):
    total_rewards = []
    episode_lengths = []
    
    for _ in range(n_episodes):
        obs,_ = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            done = done or truncated  # 处理环境终止条件
            
        total_rewards.append(episode_reward)
        episode_lengths.append(steps)
    
    # 计算统计指标
    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    mean_length = np.mean(episode_lengths)
    
    # 打印诊断信息
    print(f"评估结果 ({n_episodes} episodes):")
    print(f"平均奖励: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"平均回合长度: {mean_length:.1f} steps")
    
    return mean_reward
def train(symbol_train: str,
          symbol_eval: str,
          window_size: int | None = None,
          target_return: float = 3.2,  # 策略目标收益率，超过视为成功完成，给予高额奖励
          min_target_return: float = -3.1  # 最小目标收益率，低于视为失败，给予惩罚
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
    initial_lr = 3e-4
    final_lr = 1e-6
    # 创建余弦退火学习率调度（从3e-4到1e-6）
    lr_schedule = get_schedule_fn(
        lambda progress: final_lr + 0.5 *
        (initial_lr - final_lr)*(1 + np.cos(np.pi*progress))
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("torch.cuda.is_available()=", torch.cuda.is_available())

    def objective(trial):
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
            'n_steps': trial.suggest_categorical('n_steps', [512, 1024, 2048, 4096]),
            'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256]),
            'n_epochs': trial.suggest_int('n_epochs', 5, 20, step=5),
            'gamma': trial.suggest_float('gamma', 0.9, 0.9999),
            'ent_coef': trial.suggest_float('ent_coef', 0.0, 0.1),
            'gae_lambda': trial.suggest_float('gae_lambda', 0.80, 0.99),
            'clip_range': trial.suggest_float('clip_range', 0.1, 0.3),
            'verbose': 0
        }
        
        model = PPO("MlpPolicy", train_env, **params)
        model.learn(total_timesteps=10000,        
                    progress_bar=True,
                    log_interval=10,
                    tb_log_name=f"ppo_new_reward",
                    reset_num_timesteps=True)
        mean_reward = evaluate_model(model, eval_env, n_episodes=10)  # 自定义评估函数
        return mean_reward

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50, show_progress_bar=True)
    
    # 保存最佳参数
    with open("best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=2, ensure_ascii=False)
        
    # 输出最佳结果
    print("\n=== 最佳参数 ===")
    print(study.best_params)
    
    # 用最佳参数训练最终模型
    final_model = PPO(
        "MlpPolicy",
        train_env,
        **study.best_params
    )
    final_model.learn(total_timesteps=500000)
    final_model.save("final_ppo_model")

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
    model = train(symbol_train, symbol_eval, window_size=None)
    # 保存模型
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    model_save_path = f"rppo_trading_model_{timestamp}"
    model.save(model_save_path)
