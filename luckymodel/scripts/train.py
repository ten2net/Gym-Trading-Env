"""
强化学习模型训练入口
支持功能：
1. 多股票并行训练
2. 自动超参数加载
3. 模型检查点保存
4. TensorBoard监控
"""

import os
import yaml
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from utils.config_parser import load_config
from data.loader.stock_loader import StockDataLoader
from envs.trading_env import StockTradingEnv


def main():
    # 参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train_config.yml',
                        help='训练配置文件路径')
    parser.add_argument('--log_dir', type=str, default='logs/',
                        help='TensorBoard日志目录')
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)
    train_config = config['training']
    algo_config = train_config['algorithm_params']

    # 数据加载
    data_loader = StockDataLoader(train_config['data_settings'])
    train_df = data_loader.load_data()

    # 创建训练环境
    env = DummyVecEnv([lambda: StockTradingEnv(train_df, **config['environment'])])

    # 初始化模型
    model = PPO(
        policy=algo_config['policy'],
        env=env,
        learning_rate=algo_config['learning_rate'],
        n_steps=algo_config['n_steps'],
        batch_size=algo_config['batch_size'],
        n_epochs=algo_config['n_epochs'],
        gamma=algo_config['gamma'],
        tensorboard_log=args.log_dir,
        device=algo_config['device'],
        verbose=1
    )

    # 设置回调函数
    callbacks = []
    # 评估回调
    eval_callback = EvalCallback(
        eval_env=env,
        eval_freq=train_config.get('eval_freq', 10000),
        best_model_save_path='checkpoints/best/',
        log_path='logs/evaluations/',
        deterministic=True
    )
    callbacks.append(eval_callback)

    # 模型保存回调
    checkpoint_callback = CheckpointCallback(
        save_freq=algo_config['n_steps'],
        save_path='checkpoints/',
        name_prefix='model_checkpoint'
    )
    callbacks.append(checkpoint_callback)

    # 开始训练
    model.learn(
        total_timesteps=train_config.get('total_timesteps', 1e6),
        callback=callbacks,
        tb_log_name=algo_config['algorithm']
    )

    # 保存最终模型
    model.save(os.path.join('checkpoints', 'final_model.zip'))


if __name__ == "__main__":
    main()
