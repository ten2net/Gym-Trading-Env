"""
模型性能评估脚本
功能包括：
1. 加载验证数据集
2. 计算金融指标（夏普率、最大回撤等）
3. 生成评估报告
4. 可视化结果
"""
import sys
sys.path.append("../")
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sb3_contrib import RecurrentPPO
from envs.env import make_env

class TradingMetricsCalculator:
    """金融指标计算工具类"""
    
    @staticmethod
    def sharpe_ratio(returns):
        """计算年化夏普比率（假设每日收益率）"""
        if len(returns) < 2:
            return 0.0
        
        returns = np.array(returns)
        daily_returns = (returns[1:] - returns[:-1]) / returns[:-1]
        
        if np.std(daily_returns) == 0:
            return 0.0
            
        return np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)

    @staticmethod
    def max_drawdown(returns):
        """计算最大回撤"""
        if len(returns) == 0:
            return 0.0
            
        peak = returns[0]
        max_dd = 0.0
        
        for value in returns:
            if value > peak:
                peak = value
            current_dd = (peak - value) / peak
            if current_dd > max_dd:
                max_dd = current_dd
                
        return max_dd

def evaluate_model(env, model, num_episodes=10):
    """模型评估主函数"""
    
    all_returns = []
    metrics = []

    for _ in range(num_episodes):
        # 获取初始投资组合净值
        obs, info = env.reset()
        episode_returns = [1000000]
        done = False
        truncated = False

        while not done or not truncated:
            action, _ = model.predict(obs, deterministic=False)
            obs, _, done, truncated, info = env.step(action)
            episode_returns.append(info['portfolio_valuation'])
            if truncated or done:
              obs, info = env.reset() 

        # 计算各项指标
        metrics.append({
            'sharpe_ratio': TradingMetricsCalculator.sharpe_ratio(episode_returns),
            'max_drawdown': TradingMetricsCalculator.max_drawdown(episode_returns),
            'total_return': (episode_returns[-1] - episode_returns[0]) / episode_returns[0]
        })
        all_returns.append(episode_returns)

    # 汇总统计结果
    return {
        'mean_sharpe': np.mean([m['sharpe_ratio'] for m in metrics]),
        'max_drawdown': np.max([m['max_drawdown'] for m in metrics]),
        'avg_return': np.mean([m['total_return'] for m in metrics]),
        'cumulative_returns': all_returns
    }

def plot_portfolio_results(returns_list):
    """绘制投资组合净值曲线"""
    
    plt.figure(figsize=(12, 6))
    
    for i, returns in enumerate(returns_list):
        plt.plot(returns, label=f'Episode {i+1}')
        
    plt.title('Portfolio Valuation Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Net Asset Value')
    plt.legend()
    plt.grid(True)
    plt.savefig("1.png")

if __name__ == "__main__":
    # 参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', type=str, default='300301', help='评估使用的股票代码')
    args = parser.parse_args()

    # 创建评估环境
    env = make_env(args.symbol, window_size=3, eval=False)

    # 加载训练好的模型
    model = RecurrentPPO.load("./rppo_trading_model_20250502_0302.zip")

    # 执行评估
    results = evaluate_model(env, model, num_episodes=5)

    # 输出评估报告
    print("\n=== 评估结果 ===")
    print(f"平均夏普比率: {results['mean_sharpe']:.2f}")
    print(f"最大回撤: {results['max_drawdown']:.2%}")
    print(f"平均总收益: {results['avg_return']:.2%}")

    # 可视化净值曲线
    plot_portfolio_results(results['cumulative_returns'])