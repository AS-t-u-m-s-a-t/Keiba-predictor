"""特徴量エンジニアリングモジュールのテスト"""

import pytest
import pandas as pd
import numpy as np
from src.features.engineer import FeatureEngineer


class TestFeatureEngineer:
    """FeatureEngineerクラスのテスト"""
    
    def setup_method(self):
        """テストのセットアップ"""
        self.engineer = FeatureEngineer()
    
    def create_sample_data(self):
        """テスト用のサンプルデータを作成"""
        df = pd.DataFrame({
            'race_id': ['R001', 'R001', 'R001', 'R002', 'R002', 'R002'],
            'horse_id': ['H001', 'H002', 'H003', 'H001', 'H002', 'H003'],
            'finish_position': [1, 2, 3, 2, 1, 4],
            'jockey_id': ['J001', 'J002', 'J003', 'J001', 'J002', 'J003'],
            'distance': [2000, 2000, 2000, 1600, 1600, 1600],
            'course_type': ['芝', '芝', '芝', 'ダート', 'ダート', 'ダート'],
            'frame_number': [1, 2, 3, 1, 2, 3],
            'horse_number': [5, 6, 7, 5, 6, 7]
        })
        return df
    
    def test_create_horse_history_features(self):
        """馬の過去成績特徴量のテスト"""
        df = self.create_sample_data()
        
        result = self.engineer._create_horse_history_features(df)
        
        # 特徴量が作成されているか
        assert 'horse_avg_finish_3' in result.columns
        assert 'horse_avg_finish_5' in result.columns
        assert 'horse_last_finish' in result.columns
        assert 'horse_win_rate' in result.columns
        assert 'horse_place_rate' in result.columns
        
        # 最初のレースでは過去データがないのでNaN
        first_race_mask = result['race_id'] == 'R001'
        assert result[first_race_mask]['horse_last_finish'].isna().all()
    
    def test_create_jockey_features(self):
        """騎手特徴量のテスト"""
        df = self.create_sample_data()
        
        result = self.engineer._create_jockey_features(df)
        
        # 騎手統計が追加されているか
        assert 'jockey_win_rate' in result.columns
        assert 'jockey_place_rate' in result.columns
        assert 'jockey_show_rate' in result.columns
        assert 'jockey_avg_finish' in result.columns
        
        # すべての行に値が設定されているか（欠損値はあり得る）
        assert 'jockey_win_rate' in result.columns
    
    def test_create_distance_features(self):
        """距離適性特徴量のテスト"""
        df = self.create_sample_data()
        
        result = self.engineer._create_distance_features(df)
        
        # 距離カテゴリが作成されているか
        assert 'distance_category' in result.columns
        assert 'horse_distance_avg_finish' in result.columns
        
        # 距離カテゴリの値が正しいか
        assert result[result['distance'] == 2000]['distance_category'].iloc[0] == '中距離'
        assert result[result['distance'] == 1600]['distance_category'].iloc[0] == 'マイル'
    
    def test_create_frame_features(self):
        """枠番特徴量のテスト"""
        df = self.create_sample_data()
        
        result = self.engineer._create_frame_features(df)
        
        # 枠番統計が追加されているか
        assert 'frame_avg_finish' in result.columns
        assert 'frame_std_finish' in result.columns
        assert 'is_inner_frame' in result.columns
        assert 'is_outer_frame' in result.columns
        
        # 内枠/外枠フラグが正しく設定されているか
        assert result[result['frame_number'] == 1]['is_inner_frame'].iloc[0] == 1
        assert result[result['frame_number'] == 1]['is_outer_frame'].iloc[0] == 0
    
    def test_create_course_type_features(self):
        """コース適性特徴量のテスト"""
        df = self.create_sample_data()
        
        result = self.engineer._create_course_type_features(df)
        
        # コース適性特徴量が追加されているか
        assert 'horse_course_avg_finish' in result.columns
        assert 'horse_course_win_rate' in result.columns
    
    def test_create_features_full_pipeline(self):
        """完全な特徴量生成パイプラインのテスト"""
        df = self.create_sample_data()
        
        result = self.engineer.create_features(df)
        
        # 元のカラムが保持されているか
        assert 'race_id' in result.columns
        assert 'horse_id' in result.columns
        assert 'finish_position' in result.columns
        
        # 新しい特徴量が追加されているか
        assert 'horse_avg_finish_3' in result.columns
        assert 'jockey_win_rate' in result.columns
        assert 'distance_category' in result.columns
        assert 'frame_avg_finish' in result.columns
        assert 'horse_course_avg_finish' in result.columns
        
        # データ行数が変わっていないか
        assert len(result) == len(df)
    
    def test_get_feature_names(self):
        """特徴量名取得のテスト"""
        df = self.create_sample_data()
        result = self.engineer.create_features(df)
        
        feature_names = self.engineer.get_feature_names(result)
        
        # 特徴量名のリストが返されるか
        assert isinstance(feature_names, list)
        assert len(feature_names) > 0
        
        # 基本特徴量が含まれているか
        assert 'distance' in feature_names
        assert 'frame_number' in feature_names
        assert 'horse_number' in feature_names
    
    def test_feature_consistency(self):
        """特徴量生成の一貫性テスト"""
        df = self.create_sample_data()
        
        # 同じデータで2回実行
        result1 = self.engineer.create_features(df.copy())
        result2 = self.engineer.create_features(df.copy())
        
        # 結果が同じか
        pd.testing.assert_frame_equal(result1, result2)
