from stable_baselines3.common.callbacks import BaseCallback


class EpisodeMetricsCallback(BaseCallback):
    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "episode" in info:
                episode_data = info["episode"]
                # 记录奖励和长度
                self.logger.record("rollout/ep_rew_mean", episode_data["r"])
                self.logger.record("rollout/ep_len_mean", episode_data["l"])
        return True
# 自定义课程学习回调
class ProfitCurriculumCallback(BaseCallback):
    def __init__(self, verbose=0, max_steps=4e6, initial_target=0.02, final_target=0.12,
                 ent_coef_initial=0.08, ent_coef_min=0.001, ent_coef_restore_ratio=0.5):
        """
        :param ent_coef_initial: 熵系数初始值
        :param ent_coef_min: 熵系数最小值
        :param ent_coef_restore_ratio: 难度提升时熵系数的恢复比例（相对于初始值）
        """
        super().__init__(verbose)
        self._last_trigger_step = 0
        self.max_steps = max_steps
        self.initial_target = initial_target
        self.final_target = final_target
        self.ent_coef_initial = ent_coef_initial
        self.ent_coef_min = ent_coef_min
        self.ent_coef_restore_ratio = ent_coef_restore_ratio
        
        # 跟踪上一次的目标值和当前熵系数
        self.last_target = initial_target
        self.current_ent_coef = ent_coef_initial

    def _on_step(self) -> bool:
        warmup_steps = 3e5  # 前30万步高频率更新，这种设计更符合 "先稳后优" 的强化学习调参哲学，能有效平衡探索与利用的矛盾。
        if self.model.num_timesteps < warmup_steps:
            update_interval = 10000
        else:
            update_interval = 2000        
        if (self.model.num_timesteps - self._last_trigger_step) >= update_interval:
            self._update_target()
            self._last_trigger_step = self.model.num_timesteps
        return True

    def _update_target(self):
        total_steps = self.model.num_timesteps
        # 计算新目标值
        # if total_steps <= self.max_steps:
        #     new_target = self.initial_target + (total_steps / self.max_steps) * (self.final_target - self.initial_target)
        # else:
        #     new_target = self.final_target
        # new_target = max(self.initial_target, min(self.final_target, new_target))
        # 每 500k 步提升一次目标值  若目标值（target_return）线性增长过快，策略可能无法适应。
        # 因此，我们将其限制在 0.03 到 0.15 之间，并在每个阶段增加 0.03。
        # 将线性增长改为 ​​分段阶梯式提升​​，每阶段预留足够训练步数：
        stage = min(int(total_steps // 4e5), 10)  # 0~9 共10个阶段
        new_target = self.initial_target + stage * 0.01  # 每阶段增加0.01
        new_target = min(new_target, self.final_target)        
        
        # 检测难度提升（目标值增加超过1%）
        target_abs_increase = new_target - self.last_target
        target_increase = target_abs_increase / self.last_target #相对幅度
        # if target_increase > 0.01:
        if target_abs_increase > 0.005:  # 绝对增幅触发，
            # 恢复熵系数到初始值的指定比例（如50%）
            max_restore_ratio = 1.0  # 允许恢复到初始值，但不超过
            restore_ratio = self.ent_coef_restore_ratio + 0.3 * target_increase  # 目标值增幅越大，恢复比例越高
            restore_ratio = min(restore_ratio, max_restore_ratio)

            self.current_ent_coef = max(
                self.ent_coef_min,
                self.ent_coef_initial * restore_ratio
            )
            self.model.ent_coef = self.current_ent_coef
            self.logger.record("train/ent_coef", self.current_ent_coef)
        
        # 更新环境目标值
        for env_idx in range(self.training_env.num_envs):
            env = self.training_env.envs[env_idx].unwrapped
            try:
                if hasattr(env, 'target_return'):
                    env.target_return = new_target
                    self.logger.record(f"env_{env_idx}/target_return", new_target)
            except Exception as e:
                print(f"Failed to update env {env_idx}: {str(e)}")
        
        self.last_target = new_target
                
class EntropyScheduler(BaseCallback):
    def __init__(self, initial_value, final_value, verbose=0):
        super().__init__(verbose)
        self.initial_value = initial_value
        self.final_value = final_value
        
    def _on_step(self) -> bool:
        # 计算当前进度
        progress = 1.0 - (self.num_timesteps / self.model._total_timesteps)
        
        # 线性衰减
        self.model.ent_coef = self.final_value + (self.initial_value - self.final_value) * progress
        
        # 记录当前ent_coef值
        self.logger.record("train/ent_coef", self.model.ent_coef)
        return True                