"""
模型性能评估脚本
功能包括：
1. 加载验证数据集
2. 计算金融指标（夏普率、最大回撤等）
3. 生成评估报告
4. 可视化结果
"""

import argparse
import numpy as np
from stable_baselines3 import PPO
from utils.config_parser import load_config
from data.loader.stock_loader import StockDataLoader
from envs.trading_env import StockTradingEnv
from evaluators.performance_analyzer import TradingMetricsCalculator
from evaluators.visualization import plot_portfolio_values


def evaluate_model(env, model, num_episodes=10):
    all_returns = []
    metrics = []

    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        episode_returns = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_returns.append(info[0]['portfolio_return'])

        # 计算单次episode指标
        metrics.append({
            'sharpe_ratio': TradingMetricsCalculator.sharpe_ratio(episode_returns),
            'max_drawdown': TradingMetricsCalculator.max_drawdown(episode_returns),
            'total_return': np.prod([1 + r for r in episode_returns]) - 1
        })
        all_returns.extend(episode_returns)

    # 汇总统计
    return {
        'mean_sharpe': np.mean([m['sharpe_ratio'] for m in metrics]),
        'max_drawdown': np.max([m['max_drawdown'] for m in metrics]),
        'avg_return': np.mean([m['total_return'] for m in metrics]),
        'cumulative_returns': all_returns
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True,
                        help='待评估模型路径')
    parser.add_argument('--config', type=str, default='configs/eval_config.yml',
                        help='评估配置文件路径')
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)
    eval_config = config['evaluation']

    # 加载数据
    data_loader = StockDataLoader(eval_config['validation_data'])
    eval_df = data_loader.load_data()

    # 创建评估环境
    env = StockTradingEnv(eval_df, **config['environment'])

    # 加载模型
    model = PPO.load(args.model_path)

    # 运行评估
    results = evaluate_model(
        env, model, num_episodes=eval_config['eval_params']['n_eval_episodes'])

    # 输出结果
    print("\n=== 评估结果 ===")
    print(f"平均夏普比率: {results['mean_sharpe']:.2f}")
    print(f"最大回撤: {results['max_drawdown']:.2%}")
    print(f"平均总收益: {results['avg_return']:.2%}")

    # 可视化
    plot_portfolio_results(results['cumulative_returns'])


if __name__ == "__main__":
    main()
