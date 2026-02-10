"""前処理モジュールのテスト"""

import pytest
import pandas as pd
import numpy as np
from src.preprocessing.cleaner import DataCleaner


class TestDataCleaner:
    """DataCleanerクラスのテスト"""
    
    def setup_method(self):
        """テストのセットアップ"""
        self.cleaner = DataCleaner()
    
    def test_clean_finish_position(self):
        """着順クリーニングのテスト"""
        df = pd.DataFrame({
            'finish_position': ['1', '2', '取消', '4', '除外']
        })
        
        result = self.cleaner._clean_finish_position(df)
        
        assert result['finish_position'].iloc[0] == 1
        assert result['finish_position'].iloc[1] == 2
        assert pd.isna(result['finish_position'].iloc[2])
        assert result['finish_position'].iloc[3] == 4
        assert pd.isna(result['finish_position'].iloc[4])
    
    def test_clean_time_with_minutes(self):
        """タイム変換のテスト（分:秒.小数）"""
        df = pd.DataFrame({
            'time': ['1:23.4', '2:05.6', '']
        })
        
        result = self.cleaner._clean_time(df)
        
        assert result['time_seconds'].iloc[0] == 83.4
        assert result['time_seconds'].iloc[1] == 125.6
        assert pd.isna(result['time_seconds'].iloc[2])
    
    def test_clean_sex_age(self):
        """性齢分離のテスト"""
        df = pd.DataFrame({
            'sex_age': ['牡3', '牝4', 'セ5', '']
        })
        
        result = self.cleaner._clean_sex_age(df)
        
        assert result['sex'].iloc[0] == '牡'
        assert result['age'].iloc[0] == 3
        assert result['sex'].iloc[1] == '牝'
        assert result['age'].iloc[1] == 4
        assert pd.isna(result['sex'].iloc[3])
        assert pd.isna(result['age'].iloc[3])
    
    def test_clean_numeric_columns(self):
        """数値カラムクリーニングのテスト"""
        df = pd.DataFrame({
            'frame_number': ['1', '2', 'invalid'],
            'horse_number': ['5', '6', '7'],
            'distance': ['2000', '1600', '']
        })
        
        result = self.cleaner._clean_numeric_columns(df)
        
        assert result['frame_number'].iloc[0] == 1
        assert result['frame_number'].iloc[1] == 2
        assert pd.isna(result['frame_number'].iloc[2])
        assert result['distance'].iloc[0] == 2000
    
    def test_encode_categorical(self):
        """カテゴリエンコーディングのテスト"""
        df = pd.DataFrame({
            'course_type': ['芝', 'ダート', '芝', 'ダート'],
            'weather': ['晴', '曇', '雨', '晴']
        })
        
        result = self.cleaner._encode_categorical(df)
        
        assert 'course_type_encoded' in result.columns
        assert 'weather_encoded' in result.columns
        assert len(result['course_type_encoded'].unique()) == 2
    
    def test_remove_outliers_time(self):
        """異常値除去のテスト（タイム）"""
        df = pd.DataFrame({
            'time_seconds': [10, 80, 100, 120, 400],
            'finish_position': [1, 2, 3, 4, 5]
        })
        
        result = self.cleaner._remove_outliers(df)
        
        # 50秒未満と300秒以上は除外される
        assert len(result) == 3
        assert 10 not in result['time_seconds'].values
        assert 400 not in result['time_seconds'].values
    
    def test_full_clean_pipeline(self):
        """完全なクリーニングパイプラインのテスト"""
        df = pd.DataFrame({
            'finish_position': ['1', '2', '3'],
            'time': ['1:30.5', '1:31.2', '1:32.0'],
            'sex_age': ['牡3', '牝4', '牡5'],
            'course_type': ['芝', 'ダート', '芝'],
            'distance': [2000, 1600, 2000],
            'frame_number': [1, 2, 3],
            'horse_number': [5, 6, 7],
            'weight': [58, 56, 57],
            'weather': ['晴', '晴', '曇'],
            'track_condition': ['良', '良', '稍重']
        })
        
        result = self.cleaner.clean(df)
        
        # 基本的なチェック
        assert len(result) > 0
        assert 'time_seconds' in result.columns
        assert 'sex' in result.columns
        assert 'age' in result.columns
        assert 'course_type_encoded' in result.columns
