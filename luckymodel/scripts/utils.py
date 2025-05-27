import numpy as np

class RobustCosineSchedule:
    def __init__(self, initial_lr=1e-6, final_lr=1e-3, warmup_ratio=0.2):
        """
        工业级稳健的学习率调度器
        特点：
        1. 双重数值钳制
        2. 浮点误差补偿
        3. 自动范围校正
        """
        self.initial_lr = float(initial_lr)
        self.final_lr = float(final_lr)
        self.warmup_ratio = float(warmup_ratio)
        
        # 自动校准参数
        self._validate_parameters()
        
    def _validate_parameters(self):
        """参数合法性检查"""
        assert self.initial_lr > 0, "初始学习率必须是正数"
        assert self.final_lr >= self.initial_lr, "峰值学习率不能小于初始值"
        assert 0 < self.warmup_ratio < 1, "预热比例必须在(0,1)区间"
        
        # 防止浮点误差导致的计算问题
        self.epsilon = np.finfo(float).eps * 100
        
    def __call__(self, progress):
        """
        progress: 从1.0（开始）到0.0（结束）的进度
        返回：计算后的学习率
        """
        # 强制进度值合法化
        progress = np.clip(float(progress), 0.0, 1.0)
        
        # 阶段1：线性预热
        if progress > (1 - self.warmup_ratio):
            warmup_progress = (1 - progress) / self.warmup_ratio  # 更直观的计算
            return self._safe_lerp(0.0, self.final_lr, warmup_progress)
        
        # 阶段2：改进型余弦退火
        decay_progress = progress / (1 - self.warmup_ratio)
        cosine_value = np.cos(np.pi * decay_progress)
        
        # 使用线性插值替代原始公式
        return self._safe_lerp(self.final_lr, self.initial_lr, 0.5*(1 + cosine_value))
    
    def _safe_lerp(self, start, end, weight):
        """带保护的线性插值计算"""
        weight = np.clip(weight, 0.0, 1.0)
        # 添加微小偏移防止浮点误差
        return start + (end - start + self.epsilon) * weight
