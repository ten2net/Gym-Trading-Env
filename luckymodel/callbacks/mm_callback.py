from stable_baselines3.common.callbacks import BaseCallback
import numpy as np


class MultiMetricCallback(BaseCallback):
    """
    自定义回调函数，用于收集多个训练指标
    功能：
    1. 记录自定义指标到Tensorboard
    2. 控制台打印关键指标
    3. 自动保存最佳模型
    """

    def __init__(self, verbose=0, check_freq=1000, log_dir=None, metrics=None):
        super(MultiMetricCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.metrics = metrics or []
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        # 每check_freq步执行一次
        if self.n_calls % self.check_freq != 0:
            return True

        # 收集指标数据
        metric_dict = {}
        for name, func in self.metrics:
            try:
                metric_dict[name] = func(self.training_env.historical_info)
            except Exception as e:
                print(f"计算指标 {name} 失败: {str(e)}")
                metric_dict[name] = None

        # 记录到Tensorboard
        for name, value in metric_dict.items():
            if value is not None:
                self.logger.record(f"metrics/{name}", value)

        # 自动保存最佳模型
        if "portfolio_return" in metric_dict:
            current_reward = metric_dict["portfolio_return"]
            if current_reward > self.best_mean_reward:
                self.best_mean_reward = current_reward
                if self.verbose >= 1:
                    print(f"发现新最佳模型，保存到 {self.log_dir}/best_model.zip")
                self.model.save(f"{self.log_dir}/best_model.zip")

        return True

    def _on_training_end(self) -> None:
        # 训练结束时打印总结报告
        print("\n训练总结:")
        print(f"最佳组合收益率: {self.best_mean_reward:.2f}%")
