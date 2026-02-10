"""特徴量エンジニアリングモジュール

過去成績、騎手勝率、距離適性などの特徴量を生成します。
"""

from typing import List
import pandas as pd
import numpy as np


class FeatureEngineer:
    """特徴量生成クラス"""
    
    def __init__(self):
        pass
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """全特徴量を生成
        
        Args:
            df: 前処理済みのDataFrame
            
        Returns:
            特徴量追加済みのDataFrame
        """
        df = df.copy()
        
        # 時系列でソート（重要）
        df = df.sort_values(['race_id', 'horse_id']).reset_index(drop=True)
        
        # 各特徴量を生成
        df = self._create_horse_history_features(df)
        df = self._create_jockey_features(df)
        df = self._create_distance_features(df)
        df = self._create_frame_features(df)
        df = self._create_course_type_features(df)
        
        return df
    
    def _create_horse_history_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """馬の過去成績特徴量
        
        Args:
            df: DataFrame
            
        Returns:
            特徴量追加済みのDataFrame
        """
        if 'horse_id' not in df.columns or 'finish_position' not in df.columns:
            return df
        
        # 過去N走の平均着順
        for n in [3, 5, 10]:
            df[f'horse_avg_finish_{n}'] = df.groupby('horse_id')['finish_position'].transform(
                lambda x: x.shift(1).rolling(window=n, min_periods=1).mean()
            )
        
        # 前走の着順
        df['horse_last_finish'] = df.groupby('horse_id')['finish_position'].shift(1)
        
        # 勝率（過去の1着率）
        df['horse_win_rate'] = df.groupby('horse_id')['finish_position'].transform(
            lambda x: (x.shift(1) == 1).rolling(window=10, min_periods=1).mean()
        )
        
        # 連対率（過去の1-2着率）
        df['horse_place_rate'] = df.groupby('horse_id')['finish_position'].transform(
            lambda x: (x.shift(1) <= 2).rolling(window=10, min_periods=1).mean()
        )
        
        return df
    
    def _create_jockey_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """騎手の特徴量
        
        Args:
            df: DataFrame
            
        Returns:
            特徴量追加済みのDataFrame
        """
        if 'jockey_id' not in df.columns or 'finish_position' not in df.columns:
            return df
        
        # 騎手の勝率
        jockey_stats = df.groupby('jockey_id').agg({
            'finish_position': [
                lambda x: (x == 1).mean(),  # 勝率
                lambda x: (x <= 2).mean(),  # 連対率
                lambda x: (x <= 3).mean(),  # 複勝率
                'mean'  # 平均着順
            ]
        }).reset_index()
        
        jockey_stats.columns = ['jockey_id', 'jockey_win_rate', 'jockey_place_rate', 
                                'jockey_show_rate', 'jockey_avg_finish']
        
        df = df.merge(jockey_stats, on='jockey_id', how='left')
        
        return df
    
    def _create_distance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """距離適性の特徴量
        
        Args:
            df: DataFrame
            
        Returns:
            特徴量追加済みのDataFrame
        """
        if 'distance' not in df.columns or 'horse_id' not in df.columns:
            return df
        
        # 距離カテゴリ（短距離、マイル、中距離、長距離）
        df['distance_category'] = pd.cut(
            df['distance'],
            bins=[0, 1400, 1800, 2200, 4000],
            labels=['短距離', 'マイル', '中距離', '長距離']
        )
        
        # 同距離カテゴリでの過去平均着順
        df['horse_distance_avg_finish'] = df.groupby(['horse_id', 'distance_category'])['finish_position'].transform(
            lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
        )
        
        return df
    
    def _create_frame_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """枠番の特徴量
        
        Args:
            df: DataFrame
            
        Returns:
            特徴量追加済みのDataFrame
        """
        if 'frame_number' not in df.columns:
            return df
        
        # 枠番ごとの平均着順（枠の有利不利）
        frame_stats = df.groupby('frame_number')['finish_position'].agg(['mean', 'std']).reset_index()
        frame_stats.columns = ['frame_number', 'frame_avg_finish', 'frame_std_finish']
        
        df = df.merge(frame_stats, on='frame_number', how='left')
        
        # 内枠/外枠フラグ
        df['is_inner_frame'] = (df['frame_number'] <= 3).astype(int)
        df['is_outer_frame'] = (df['frame_number'] >= 6).astype(int)
        
        return df
    
    def _create_course_type_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """コース適性の特徴量
        
        Args:
            df: DataFrame
            
        Returns:
            特徴量追加済みのDataFrame
        """
        if 'course_type' not in df.columns or 'horse_id' not in df.columns:
            return df
        
        # コースタイプ別の過去平均着順
        df['horse_course_avg_finish'] = df.groupby(['horse_id', 'course_type'])['finish_position'].transform(
            lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
        )
        
        # コースタイプ別の勝率
        df['horse_course_win_rate'] = df.groupby(['horse_id', 'course_type'])['finish_position'].transform(
            lambda x: (x.shift(1) == 1).rolling(window=5, min_periods=1).mean()
        )
        
        return df
    
    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """モデル学習に使用する特徴量のリストを取得
        
        Args:
            df: DataFrame
            
        Returns:
            特徴量名のリスト
        """
        # 基本的な数値特徴量
        base_features = [
            'distance', 'age', 'weight', 'frame_number', 'horse_number'
        ]
        
        # エンコード済みカテゴリ特徴量
        encoded_features = [
            'course_type_encoded', 'weather_encoded', 
            'track_condition_encoded', 'sex_encoded'
        ]
        
        # 生成した特徴量
        generated_features = [
            col for col in df.columns if any([
                col.startswith('horse_avg_finish_'),
                col.startswith('horse_last_'),
                col.startswith('horse_win_'),
                col.startswith('horse_place_'),
                col.startswith('horse_distance_'),
                col.startswith('horse_course_'),
                col.startswith('jockey_win_'),
                col.startswith('jockey_place_'),
                col.startswith('jockey_show_'),
                col.startswith('jockey_avg_'),
                col.startswith('frame_avg_'),
                col.startswith('frame_std_'),
                col.startswith('is_inner_'),
                col.startswith('is_outer_')
            ])
        ]
        
        # 存在する特徴量のみを返す
        all_features = base_features + encoded_features + generated_features
        valid_features = [f for f in all_features if f in df.columns]
        
        # 数値型のみを返す（オブジェクト型を除外）
        numeric_features = [f for f in valid_features if df[f].dtype in ['int64', 'float64', 'int32', 'float32', 'bool']]
        
        return numeric_features
