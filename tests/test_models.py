"""モデルモジュールのテスト"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from src.models.base import BaseModel
from src.models.lightgbm_model import LightGBMModel
from src.models.neural_net import NeuralNetModel
from src.models.lambdamart import LambdaMARTModel


class TestBaseModel:
    """BaseModelクラスのテスト"""
    
    def test_base_model_is_abstract(self):
        """BaseModelが抽象クラスであることをテスト"""
        with pytest.raises(TypeError):
            BaseModel()
    
    def test_model_has_required_methods(self):
        """BaseModelが必要なメソッドを持つことをテスト"""
        required_methods = ['train', 'predict', 'save', 'load']
        
        for method in required_methods:
            assert hasattr(BaseModel, method)


class TestLightGBMModel:
    """LightGBMModelクラスのテスト"""
    
    def setup_method(self):
        """テストのセットアップ"""
        self.model = LightGBMModel(random_state=42)
    
    def create_sample_data(self, n_samples=100):
        """テスト用のサンプルデータを作成"""
        np.random.seed(42)
        
        X = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples),
            'feature3': np.random.randint(1, 10, n_samples),
            'feature4': np.random.randn(n_samples)
        })
        
        # 目的変数（着順: 1-10）
        y = pd.Series(np.random.randint(1, 11, n_samples))
        
        return X, y
    
    def test_model_initialization(self):
        """モデル初期化のテスト"""
        assert self.model.random_state == 42
        assert self.model.model is None
        assert self.model.is_trained is False
    
    def test_model_training(self):
        """モデル学習のテスト"""
        X, y = self.create_sample_data(n_samples=200)
        
        # 学習実行
        self.model.train(X, y, n_splits=2, num_boost_round=10, early_stopping_rounds=5)
        
        # 学習済みフラグがTrueになっているか
        assert self.model.is_trained is True
        assert self.model.model is not None
        assert self.model.feature_importance_ is not None
        assert self.model.feature_names_ == list(X.columns)
    
    def test_model_prediction(self):
        """モデル予測のテスト"""
        X, y = self.create_sample_data(n_samples=200)
        
        # 学習
        self.model.train(X, y, n_splits=2, num_boost_round=10, early_stopping_rounds=5)
        
        # 予測
        predictions = self.model.predict(X)
        
        # 予測結果の形状が正しいか
        assert len(predictions) == len(X)
        assert isinstance(predictions, np.ndarray)
    
    def test_prediction_without_training(self):
        """未学習での予測がエラーになることをテスト"""
        X, _ = self.create_sample_data()
        
        with pytest.raises(ValueError):
            self.model.predict(X)
    
    def test_predict_proba_win(self):
        """勝率予測のテスト"""
        X, y = self.create_sample_data(n_samples=200)
        
        # 学習
        self.model.train(X, y, n_splits=2, num_boost_round=10, early_stopping_rounds=5)
        
        # 勝率予測
        win_proba = self.model.predict_proba_win(X)
        
        # 勝率が0-1の範囲内か
        assert len(win_proba) == len(X)
        assert (win_proba >= 0).all()
        assert (win_proba <= 1).all()
    
    def test_model_save_and_load(self):
        """モデル保存・読み込みのテスト"""
        X, y = self.create_sample_data(n_samples=200)
        
        # 学習
        self.model.train(X, y, n_splits=2, num_boost_round=10, early_stopping_rounds=5)
        
        # 予測結果を保存
        predictions_before = self.model.predict(X)
        
        # 一時ファイルに保存
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.joblib') as f:
            temp_path = f.name
        
        try:
            # 保存
            self.model.save(temp_path)
            
            # 新しいモデルインスタンスで読み込み
            new_model = LightGBMModel()
            new_model.load(temp_path)
            
            # 読み込み後の予測
            predictions_after = new_model.predict(X)
            
            # 予測結果が同じか
            np.testing.assert_array_almost_equal(predictions_before, predictions_after)
            
            # 学習済みフラグが正しいか
            assert new_model.is_trained is True
            
        finally:
            # 一時ファイルを削除
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_feature_importance(self):
        """特徴量重要度のテスト"""
        X, y = self.create_sample_data(n_samples=200)
        
        # 学習
        self.model.train(X, y, n_splits=2, num_boost_round=10, early_stopping_rounds=5)
        
        # 特徴量重要度が取得できるか
        assert self.model.feature_importance_ is not None
        assert len(self.model.feature_importance_) == len(X.columns)
    
    def test_custom_parameters(self):
        """カスタムパラメータのテスト"""
        custom_params = {
            'learning_rate': 0.1,
            'num_leaves': 20
        }
        
        model = LightGBMModel(params=custom_params)
        
        # カスタムパラメータが反映されているか
        assert model.params['learning_rate'] == 0.1
        assert model.params['num_leaves'] == 20


class TestNeuralNetModel:
    """NeuralNetModelクラスのテスト"""
    
    def test_model_not_implemented(self):
        """ニューラルネットが未実装であることをテスト"""
        model = NeuralNetModel()
        
        X = pd.DataFrame({'feature1': [1, 2, 3]})
        y = pd.Series([1, 2, 3])
        
        # すべてのメソッドがNotImplementedErrorを投げるか
        with pytest.raises(NotImplementedError):
            model.train(X, y)
        
        with pytest.raises(NotImplementedError):
            model.predict(X)
        
        with pytest.raises(NotImplementedError):
            model.save('test.pkl')
        
        with pytest.raises(NotImplementedError):
            model.load('test.pkl')


class TestLambdaMARTModel:
    """LambdaMARTModelクラスのテスト"""
    
    def test_model_not_implemented(self):
        """LambdaMARTが未実装であることをテスト"""
        model = LambdaMARTModel()
        
        X = pd.DataFrame({'feature1': [1, 2, 3]})
        y = pd.Series([1, 2, 3])
        
        # すべてのメソッドがNotImplementedErrorを投げるか
        with pytest.raises(NotImplementedError):
            model.train(X, y)
        
        with pytest.raises(NotImplementedError):
            model.predict(X)
        
        with pytest.raises(NotImplementedError):
            model.save('test.pkl')
        
        with pytest.raises(NotImplementedError):
            model.load('test.pkl')
