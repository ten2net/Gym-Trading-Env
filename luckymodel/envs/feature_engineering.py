import numpy as np
import pandas as pd
import talib

def encode_time_features(datetime_index, market_type='A'):
    """
    高级时间特征编码函数，特别优化股票开盘/收盘效应
    :param datetime_index: 日期时间索引 (pandas.DatetimeIndex)
    :param market_type: 市场类型 ('A'表示A股)
    :return: DataFrame包含丰富的时间特征
    """
    if not isinstance(datetime_index, pd.DatetimeIndex):
        raise ValueError("输入必须是DatetimeIndex")
    
    # 创建特征DataFrame
    features = pd.DataFrame(index=datetime_index)
    
    # 基础时间特征
    features['hour'] = datetime_index.hour
    features['minute'] = datetime_index.minute
    features['day_of_week'] = datetime_index.dayofweek
    features['day_of_month'] = datetime_index.day
    features['week_of_year'] = datetime_index.isocalendar().week
    features['month'] = datetime_index.month
    
    # 计算总分钟数（自0:00起）
    total_minutes = features['hour'] * 60 + features['minute']
    
    # ===== 1. 通用时间周期编码 =====
    # 交易日分钟进度（0-1表示交易日进度）
    features['trading_progress'] = 0.0
    
    # 不同市场的交易时间
    if market_type == 'A':  # A股市场
        # 计算自9:30起的分钟数
        morning_minutes = (features['hour'] - 9) * 60 + features['minute'] - 30
        afternoon_minutes = (features['hour'] - 13) * 60 + features['minute']
        
        # 早盘进度 (9:30-11:30)
        morning_mask = (features['hour'] >= 9) & (features['hour'] < 11) | (
            (features['hour'] == 11) & (features['minute'] < 30)
        )
        features.loc[morning_mask, 'trading_progress'] = np.clip(morning_minutes[morning_mask] / 120, 0, 1)
        
        # 午盘进度 (13:00-15:00)
        afternoon_mask = (features['hour'] >= 13) & (features['hour'] < 15)
        features.loc[afternoon_mask, 'trading_progress'] = np.clip(afternoon_minutes[afternoon_mask] / 120, 0, 1)
    else:  # 美股等市场
      pass
      # 实现类似逻辑
    
    # ===== 2. 关键时段标记 =====
    features['is_opening'] = 0  # 开盘前30分钟 (9:30-10:00)
    features['is_early_morning'] = 0  # 早盘时段 (10:00-11:00)
    features['is_midday'] = 0  # 午盘时段 (13:00-14:00)
    features['is_late_afternoon'] = 0  # 尾盘时段 (14:30-15:00)
    features['is_closing'] = 0  # 收盘前15分钟
    
    # A股市场关键时段
    if market_type == 'A':
        # 开盘时段标记
        opening_mask = (features['hour'] == 9) & (features['minute'] >= 30) | (
            (features['hour'] == 10) & (features['minute'] == 0)
        )
        features.loc[opening_mask, 'is_opening'] = 1
        
        # 早盘时段
        early_morning_mask = (features['hour'] == 10) & (features['minute'] > 0) | (
            (features['hour'] == 10) & (features['minute'] <= 59
        ))
        features.loc[early_morning_mask, 'is_early_morning'] = 1
        
        # 午盘时段
        midday_mask = (features['hour'] == 13) | (
            (features['hour'] == 14) & (features['minute'] == 0
        ))
        features.loc[midday_mask, 'is_midday'] = 1
        
        # 尾盘时段
        late_afternoon_mask = (features['hour'] == 14) & (features['minute'] >= 30) & (features['minute'] < 57)
        features.loc[late_afternoon_mask, 'is_late_afternoon'] = 1
        
        # 收盘前
        closing_mask = (features['hour'] == 14) & (features['minute'] >= 57) | (
            (features['hour'] == 14) & (features['minute'] == 58
        )) | (features['hour'] == 14) & (features['minute'] == 59)
        features.loc[closing_mask, 'is_closing'] = 1
    
    # ===== 3. 时间周期性编码 =====
    # 小时级周期特征（正弦/余弦）
    features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
    features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
    
    # 分钟级周期特征
    features['minute_sin'] = np.sin(2 * np.pi * total_minutes / (24 * 60))
    features['minute_cos'] = np.cos(2 * np.pi * total_minutes / (24 * 60))
    
    # ===== 4. 高级复合特征 =====
    # 开盘效应强度（非线性函数）
    features['opening_effect'] = np.exp(-2 * (features['trading_progress'] ** 2))
    
    # 收盘效应强度（线性增长）
    features['closing_effect'] = np.clip(3 * (features['trading_progress'] - 0.7), 0, 1)
    
    # ===== 5. 交易策略相关的特殊特征 =====
    # 开盘波动预期
    features['opening_volatility_expectation'] = 0
    features.loc[features['is_opening'] == 1, 'opening_volatility_expectation'] = 1
    
    # 午盘流动性特征
    features['midday_liquidity'] = 0
    features.loc[features['is_midday'] == 1, 'midday_liquidity'] = 1
    
    # 尾盘资金流特征
    features['late_afternoon_flow'] = np.where(
        features['is_late_afternoon'] == 1,
        features['trading_progress'] * 0.8 + 0.2,
        0
    )
    
    # ===== 6. 季节性/特殊日特征 =====
    # 月度效应（月底资金流动）
    features['month_end_effect'] = np.where(
        features['day_of_month'] >= 25,
        (features['day_of_month'] - 25) / 6,  # 25-31线性增长
        0
    )
    
    # 周五效应
    features['friday_effect'] = (features['day_of_week'] == 4).astype(int)
    
    return features

def prepare_time_features_for_rl(features):
    """准备适用于RL观测空间的时间特征"""
    from sklearn.preprocessing import MinMaxScaler
    import copy
    
    # 创建特征副本防止污染原始数据
    rl_features = copy.deepcopy(features)
    
    # 分层处理不同类型特征
    # 第一层：不需要处理的特征（二值标记、三角函数）
    safe_features = [
        'is_opening', 'is_early_morning', 'is_midday', 
        'is_late_afternoon', 'is_closing', 'friday_effect',
        'hour_sin', 'hour_cos', 'minute_sin', 'minute_cos'
    ]
    
    # 第二层：需要缩放到[0,1]的比例特征
    scaler = MinMaxScaler()
    proportional_features = [
        'trading_progress', 'opening_effect', 'closing_effect',
        'month_end_effect', 'late_afternoon_flow'
    ]
    rl_features[proportional_features] = scaler.fit_transform(
        rl_features[proportional_features]
    )
    
    # 第三层：特殊处理计数特征
    # 将计数特征转换为比例
    rl_features['day_of_week'] /= 6.0      # 周一=0/6≈0.0, 周五=4/6≈0.67, 周六=5/6≈0.83
    rl_features['day_of_month'] /= 31.0     # 1号=0.032, 15号=0.484, 31号=1.0
    rl_features['week_of_year'] /= 52.0
    rl_features['month'] /= 12.0
    
    # 第四层：时间点特征的编码优化
    # 使用复合时间进度替代原始时间计数
    rl_features['composite_time'] = (
        0.7 * rl_features['trading_progress'] + 
        0.3 * ((rl_features['hour'] * 60 + rl_features['minute']) / (16 * 60))
    )
    # 移除原始计数特征
    rl_features = rl_features.drop(columns=['hour', 'minute'])
    
    return rl_features

class FeatureEngineer:
    def __init__(self, window_size: int = 20):
        self.window_size = window_size

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes technical indicators and adds them to the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing price data.

        Returns:
            pd.DataFrame: DataFrame with additional technical indicator columns.
        """
        df = df.copy()
        
        # Simple Moving Average (SMA)
        # df['feature_SMA'] = talib.SMA(df['close'], timeperiod=self.window_size) / df['close']
        
        # Relative Strength Index (RSI)
        df['feature_RSI'] = talib.RSI(df['close'], timeperiod=3) / 100
        
        # ATR
        # df['feature_ATR'] = talib.ATR(df['high'],df['low'],df['close'], timeperiod=self.window_size)
        # AD
        # ad = talib.AD(df['high'],df['low'],df['close'], df['volume'])
        # mean = ad.rolling(5).mean()
        # std = ad.rolling(5).std()
        # df['feature_AD'] =0.1 * (ad - mean) / std        
        # LINEARREG_SLOPE
        # df['feature_LINEARREG_SLOPE'] =10 * talib.LINEARREG_SLOPE(df['high'], timeperiod=self.window_size)
        
        # Moving Average Convergence Divergence (MACD)
        # macd, macd_signal, macd_hist = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        # df['feature_MACD'] = macd
        # df['feature_MACD_Signal'] = macd_signal
        # df['feature_MACD_Hist'] = macd_hist
        
        # Bollinger Bands
        # upper_bb, middle_bb, lower_bb = talib.BBANDS(df['close'], timeperiod=self.window_size, nbdevup=2, nbdevdn=2, matype=0)
        # df['feature_Upper_BB'] = 0.001 * upper_bb / df['close']
        # df['feature_Middle_BB'] = 0.001 * middle_bb / df['close']
        # df['feature_Lower_BB'] = 0.001 * lower_bb / df['close']
        
        # Drop rows with NaN values
        df.dropna(inplace=True)
        
        return df
    def time_features(self, df: pd.DataFrame,key_feature_only=True) -> pd.DataFrame:
        time_features = encode_time_features(df.index, market_type='A')
        prepare_time_features=prepare_time_features_for_rl(time_features)
        if key_feature_only:
           essential_time_features = [
                'hour_sin', 'hour_cos',             # 周期编码
                'minute_sin', 'minute_cos',         # 高频周期
                'day_of_week',                      # 周效应
                'trading_progress',                 # 交易时段进度
                'is_opening', 'is_closing',         # 关键时段
                'opening_effect', 'closing_effect'  # 非线性效应
            ]           
        prepare_time_features = prepare_time_features[essential_time_features].copy()
        new_columns = {col: f"feature_{col}" for col in prepare_time_features.columns}
        prepare_time_features = prepare_time_features.rename(columns=new_columns)
        
        df_with_time_features = pd.concat([df, prepare_time_features], axis=1)        
        return df_with_time_features       