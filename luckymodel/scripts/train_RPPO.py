import sys
sys.path.append("../")
from envs.env import make_env
import warnings

# from gym_trading_env.environments import TradingEnv
# from env import make_env
from stable_baselines3.common.utils import get_schedule_fn
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import EvalCallback
import numpy as np
import argparse
warnings.filterwarnings("ignore", category=ResourceWarning)


def train(symbol_train: str, symbol_eval: str,window_size:int|None =6):
    train_env = make_env(symbol_train, window_size=window_size, eval=False)
    eval_env = make_env(symbol_eval, window_size=window_size, eval=True)
    # 使用PPO算法训练模型
    initial_lr = 5e-5
    final_lr = 1e-6
    # 创建余弦退火学习率调度（从3e-4到1e-6）
    lr_schedule = get_schedule_fn(
        lambda progress: final_lr + 0.5 *
        (initial_lr - final_lr)*(1 + np.cos(np.pi*progress))
    )

    # 创建LSTM策略模型
    policy_kwargs = dict(
        lstm_hidden_size=256,
        net_arch=dict(
            pi=[128, 128],  # 显式定义LSTM层
            vf=[128, 128]),
        enable_critic_lstm=False,  # 关闭Critic的LSTM
        # optimizer_kwargs=dict(weight_decay=1e-4)
    )
    model = RecurrentPPO(
        "MlpLstmPolicy",
        train_env,
        learning_rate=lr_schedule,  # 直接传入调度器对象  #3e-4, # 调高初始学习率
        policy_kwargs=policy_kwargs,
        n_epochs=10,
        gamma=0.995,  # 延长收益视野
        ent_coef=0.05,  # 初始高探索
        verbose=0,
        device='cpu',
        seed=42,
        tensorboard_log=f"../logs/stock_trading2/")
    # 评估回调
    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path="../checkpoints/best_models/",
        eval_freq=2048,
        n_eval_episodes=10,
        verbose=0,
        deterministic=True
    )

    model.learn(
        total_timesteps=2e6,
        progress_bar=False,
        log_interval=10,
        tb_log_name=f"ppo_example",
        callback=[eval_callback],
        reset_num_timesteps=True
    )
    print("训练时间步数=", model.num_timesteps)
    return model


if __name__ == "__main__":
    # 参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol_train', type=str, default='300502',
                        help='训练使用的股票代码')
    parser.add_argument('--symbol_eval', type=str, default='300308',
                        help='评估使用的股票代码')

    args = parser.parse_args()
    symbol_train = args.symbol_train
    symbol_eval = args.symbol_eval
    model = train(symbol_train, symbol_eval, window_size=6)
    # 保存模型
    model.save("ppo_trading_model")
