"""モデル基底クラス（Strategyパターン）

すべてのモデルが継承する抽象基底クラスを定義します。
"""

from abc import ABC, abstractmethod
from typing import Any, Optional
import pandas as pd
import numpy as np


class BaseModel(ABC):
    """予測モデルの抽象基底クラス
    
    Strategyパターンを実装し、異なるモデルを統一インターフェースで扱えるようにします。
    """
    
    def __init__(self, random_state: int = 42):
        """
        Args:
            random_state: 乱数シード
        """
        self.random_state = random_state
        self.model = None
        self.is_trained = False
    
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        """モデルを学習
        
        Args:
            X: 特徴量
            y: 目的変数
            **kwargs: 追加のパラメータ
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """予測を実行
        
        Args:
            X: 特徴量
            
        Returns:
            予測結果
        """
        pass
    
    @abstractmethod
    def save(self, filepath: str) -> None:
        """モデルを保存
        
        Args:
            filepath: 保存先パス
        """
        pass
    
    @abstractmethod
    def load(self, filepath: str) -> None:
        """モデルを読み込み
        
        Args:
            filepath: 読み込み元パス
        """
        pass
    
    def get_model_name(self) -> str:
        """モデル名を取得
        
        Returns:
            モデル名
        """
        return self.__class__.__name__
