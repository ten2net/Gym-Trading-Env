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
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._last_trigger_step = 0  # 记录上次触发步数

    def _on_step(self) -> bool:
        # 动态调整逻辑（每10000步检查一次）
        if (self.model.num_timesteps - self._last_trigger_step) >= 10000:
            self._update_target()
            self._last_trigger_step = self.model.num_timesteps
        return True

    def _update_target(self):
        # 获取当前总步数
        total_steps = self.model.num_timesteps
        
        # 更新所有并行环境的目标
        for env_idx in range(self.training_env.num_envs):
            env = self.training_env.envs[env_idx].unwrapped
            
            if total_steps < 1e6:
                new_target = 0.12
            elif total_steps < 3e6:
                new_target = 0.15
            else:
                new_target = 0.18
                
            # 设置新目标并记录日志
            if env.target_return != new_target:
                env.target_return = new_target
                self.logger.record(f"env_{env_idx}/target_return", new_target)