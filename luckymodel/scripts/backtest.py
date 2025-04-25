"""
完整回测系统实现
功能：
1. 历史数据回测
2. 交易信号生成
3. 绩效分析
4. PDF报告生成
"""

import argparse
import pandas as pd
from backtest.backtest_engine import BacktestEngine
from backtest.exchange import StockExchange
from data.loader.stock_loader import StockDataLoader
from utils.config_parser import load_config
from backtest.report_generator import generate_backtest_report

def run_backtest(config_path, model_path=None):
    # 加载配置
    config = load_config(config_path)
    backtest_config = config['backtesting']
    
    # 初始化交易所
    exchange = StockExchange(
        initial_balance=backtest_config['backtest_params']['initial_balance'],
        commission=backtest_config['backtest_params']['commission'],
        slippage=backtest_config['backtest_params']['slippage']
    )

    # 加载回测数据
    data_loader = StockDataLoader(backtest_config['universe'])
    backtest_data = data_loader.load_data()

    # 初始化回测引擎
    engine = BacktestEngine(
        exchange=exchange,
        data=backtest_data,
        position_limit=backtest_config['backtest_params']['position_limit'],
        risk_free_rate=backtest_config['backtest_params']['risk_free_rate']
    )

    # 运行回测（策略模式选择）
    if model_path:
        # 使用强化学习模型生成信号
        from agents.base_agent import load_trading_agent
        agent = load_trading_agent(model_path)
        results = engine.run_with_agent(agent)
    else:
        # 使用基准策略（如买入持有）
        results = engine.run_benchmark()

    # 生成报告
    report = generate_backtest_report(
        results,
        benchmark=backtest_config['backtest_params']['benchmark'],
        initial_balance=exchange.initial_balance
    )
    
    # 输出结果
    print("\n=== 回测结果摘要 ===")
    print(f"累计收益率: {report['cumulative_return']:.2%}")
    print(f"年化波动率: {report['annual_volatility']:.2%}")
    print(f"夏普比率: {report['sharpe_ratio']:.2f}")
    print(f"最大回撤: {report['max_drawdown']:.2%}")
    
    # 保存详细报告
    report.save('backtest_report.pdf')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/backtest_config.yml',
                      help='回测配置文件路径')
    parser.add_argument('--model', type=str, default=None,
                      help='可选策略模型路径')
    args = parser.parse_args()
    
    run_backtest(args.config, args.model)