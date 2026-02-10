"""データクリーニング・前処理モジュール

欠損値処理、型変換、異常値除去、エンコーディングなどを行います。
"""

import re
from typing import Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


class DataCleaner:
    """データクリーニングクラス"""
    
    def __init__(self):
        self.label_encoders = {}
    
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """データ全体のクリーニングを実行
        
        Args:
            df: 生データのDataFrame
            
        Returns:
            クリーニング済みのDataFrame
        """
        df = df.copy()
        
        # 各カラムの処理
        df = self._clean_finish_position(df)
        df = self._clean_time(df)
        df = self._clean_sex_age(df)
        df = self._clean_numeric_columns(df)
        df = self._encode_categorical(df)
        df = self._handle_missing_values(df)
        df = self._remove_outliers(df)
        
        return df
    
    def _clean_finish_position(self, df: pd.DataFrame) -> pd.DataFrame:
        """着順カラムをクリーニング
        
        Args:
            df: DataFrame
            
        Returns:
            クリーニング済みのDataFrame
        """
        if 'finish_position' in df.columns:
            # 数値以外（取消、除外など）を欠損値に
            df['finish_position'] = pd.to_numeric(df['finish_position'], errors='coerce')
        return df
    
    def _clean_time(self, df: pd.DataFrame) -> pd.DataFrame:
        """タイムを秒数に変換
        
        Args:
            df: DataFrame
            
        Returns:
            クリーニング済みのDataFrame
        """
        if 'time' not in df.columns:
            return df
        
        def time_to_seconds(time_str):
            """タイム文字列を秒数に変換（例: "1:23.4" → 83.4）"""
            if pd.isna(time_str) or time_str == '':
                return np.nan
            
            try:
                # "1:23.4"形式
                match = re.match(r'(\d+):(\d+)\.(\d+)', str(time_str))
                if match:
                    minutes = int(match.group(1))
                    seconds = int(match.group(2))
                    fraction = int(match.group(3))
                    return minutes * 60 + seconds + fraction / 10.0
                
                # "23.4"形式（分なし）
                return float(time_str)
            except:
                return np.nan
        
        df['time_seconds'] = df['time'].apply(time_to_seconds)
        return df
    
    def _clean_sex_age(self, df: pd.DataFrame) -> pd.DataFrame:
        """性齢カラムを分離
        
        Args:
            df: DataFrame
            
        Returns:
            クリーニング済みのDataFrame
        """
        if 'sex_age' not in df.columns:
            return df
        
        def parse_sex_age(sex_age_str):
            """性齢文字列を分離（例: "牡3" → ("牡", 3)）"""
            if pd.isna(sex_age_str) or sex_age_str == '':
                return np.nan, np.nan
            
            try:
                match = re.match(r'([牡牝セ騙])(\d+)', str(sex_age_str))
                if match:
                    return match.group(1), int(match.group(2))
            except:
                pass
            return np.nan, np.nan
        
        sex_age_data = df['sex_age'].apply(parse_sex_age)
        df['sex'] = sex_age_data.apply(lambda x: x[0])
        df['age'] = sex_age_data.apply(lambda x: x[1])
        
        return df
    
    def _clean_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """数値カラムをクリーニング
        
        Args:
            df: DataFrame
            
        Returns:
            クリーニング済みのDataFrame
        """
        numeric_cols = ['frame_number', 'horse_number', 'weight', 
                       'popularity', 'odds', 'distance']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """カテゴリ変数をエンコード
        
        Args:
            df: DataFrame
            
        Returns:
            エンコード済みのDataFrame
        """
        categorical_cols = ['course_type', 'weather', 'track_condition', 'sex']
        
        for col in categorical_cols:
            if col in df.columns:
                # Label Encoding
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    # 欠損値を一時的に文字列に変換
                    valid_mask = df[col].notna()
                    if valid_mask.any():
                        self.label_encoders[col].fit(df.loc[valid_mask, col])
                
                # エンコード実行
                valid_mask = df[col].notna()
                if valid_mask.any():
                    df.loc[valid_mask, f'{col}_encoded'] = self.label_encoders[col].transform(
                        df.loc[valid_mask, col]
                    )
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """欠損値を処理
        
        Args:
            df: DataFrame
            
        Returns:
            欠損値処理済みのDataFrame
        """
        # 数値カラムは中央値で補完
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isna().any():
                df[col].fillna(df[col].median(), inplace=True)
        
        return df
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """異常値を除去
        
        Args:
            df: DataFrame
            
        Returns:
            異常値除去済みのDataFrame
        """
        # タイムの異常値除去（例: 0秒や極端に長いタイム）
        if 'time_seconds' in df.columns:
            df = df[
                (df['time_seconds'] > 50) &  # 50秒未満は異常
                (df['time_seconds'] < 300)   # 5分以上は異常
            ]
        
        # 着順が有効な範囲のみ
        if 'finish_position' in df.columns:
            df = df[
                (df['finish_position'] >= 1) & 
                (df['finish_position'] <= 18)
            ]
        
        return df
    
    def save_encoders(self, filepath: str):
        """エンコーダーを保存
        
        Args:
            filepath: 保存先パス
        """
        import joblib
        joblib.dump(self.label_encoders, filepath)
    
    def load_encoders(self, filepath: str):
        """エンコーダーを読み込み
        
        Args:
            filepath: 読み込み元パス
        """
        import joblib
        self.label_encoders = joblib.load(filepath)
