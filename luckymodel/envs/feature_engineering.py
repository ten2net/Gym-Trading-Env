# src/gym_trading_env/utils/feature_engineering.py

import pandas as pd
import talib

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
