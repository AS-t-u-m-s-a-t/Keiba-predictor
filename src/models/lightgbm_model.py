"""LightGBMモデル実装

メインの予測モデルとしてLightGBMを使用します。
"""

from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import joblib

from .base import BaseModel


class LightGBMModel(BaseModel):
    """LightGBM予測モデル
    
    時系列を考慮したクロスバリデーションとハイパーパラメータチューニングに対応。
    """
    
    def __init__(self, random_state: int = 42, params: Optional[Dict[str, Any]] = None):
        """
        Args:
            random_state: 乱数シード
            params: LightGBMのパラメータ
        """
        super().__init__(random_state)
        
        # デフォルトパラメータ
        self.params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': random_state,
        }
        
        # カスタムパラメータで上書き
        if params:
            self.params.update(params)
        
        self.feature_importance_ = None
        self.feature_names_ = None
    
    def train(self, X: pd.DataFrame, y: pd.Series, 
              n_splits: int = 5, num_boost_round: int = 1000,
              early_stopping_rounds: int = 50) -> None:
        """モデルを学習（時系列クロスバリデーション付き）
        
        Args:
            X: 特徴量
            y: 目的変数（着順）
            n_splits: クロスバリデーションの分割数
            num_boost_round: ブースティング回数
            early_stopping_rounds: Early stoppingのラウンド数
        """
        self.feature_names_ = list(X.columns)
        
        # 時系列分割
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        best_score = float('inf')
        best_model = None
        
        print(f"Starting time series cross-validation with {n_splits} splits...")
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            print(f"\nFold {fold}/{n_splits}")
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # LightGBM用データセット作成
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            # 学習
            model = lgb.train(
                self.params,
                train_data,
                num_boost_round=num_boost_round,
                valid_sets=[train_data, val_data],
                valid_names=['train', 'valid'],
                callbacks=[
                    lgb.early_stopping(early_stopping_rounds),
                    lgb.log_evaluation(period=100)
                ]
            )
            
            # バリデーションスコア
            val_pred = model.predict(X_val)
            val_score = np.sqrt(np.mean((val_pred - y_val) ** 2))
            
            print(f"Validation RMSE: {val_score:.4f}")
            
            # ベストモデルを保存
            if val_score < best_score:
                best_score = val_score
                best_model = model
        
        self.model = best_model
        self.is_trained = True
        
        # 特徴量重要度を保存
        self.feature_importance_ = self.model.feature_importance(importance_type='gain')
        
        print(f"\nBest validation RMSE: {best_score:.4f}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """予測を実行
        
        Args:
            X: 特徴量
            
        Returns:
            予測着順
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Call train() first.")
        
        # 特徴量の順序を確認
        if list(X.columns) != self.feature_names_:
            X = X[self.feature_names_]
        
        return self.model.predict(X)
    
    def predict_proba_win(self, X: pd.DataFrame) -> np.ndarray:
        """勝率を予測（着順予測を確率に変換）
        
        Args:
            X: 特徴量
            
        Returns:
            勝率（0-1）
        """
        predictions = self.predict(X)
        
        # 着順予測を勝率に変換（簡易的な方法）
        # 着順が小さいほど勝率が高い
        max_finish = 18  # 最大着順
        win_proba = np.clip(1.0 - (predictions - 1) / max_finish, 0, 1)
        
        return win_proba
    
    def save(self, filepath: str) -> None:
        """モデルを保存
        
        Args:
            filepath: 保存先パス
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Call train() first.")
        
        model_data = {
            'model': self.model,
            'params': self.params,
            'feature_names': self.feature_names_,
            'feature_importance': self.feature_importance_,
            'random_state': self.random_state
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """モデルを読み込み
        
        Args:
            filepath: 読み込み元パス
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.params = model_data['params']
        self.feature_names_ = model_data['feature_names']
        self.feature_importance_ = model_data['feature_importance']
        self.random_state = model_data['random_state']
        self.is_trained = True
        
        print(f"Model loaded from {filepath}")
    
    def plot_feature_importance(self, top_n: int = 20, figsize: tuple = (10, 8)):
        """特徴量重要度をプロット
        
        Args:
            top_n: 表示する上位N個
            figsize: 図のサイズ
        """
        if self.feature_importance_ is None:
            raise ValueError("Model is not trained yet or feature importance is not available.")
        
        # 重要度でソート
        importance_df = pd.DataFrame({
            'feature': self.feature_names_,
            'importance': self.feature_importance_
        }).sort_values('importance', ascending=False).head(top_n)
        
        # プロット
        plt.figure(figsize=figsize)
        plt.barh(range(len(importance_df)), importance_df['importance'])
        plt.yticks(range(len(importance_df)), importance_df['feature'])
        plt.xlabel('Importance (Gain)')
        plt.title('Feature Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        return plt.gcf()
