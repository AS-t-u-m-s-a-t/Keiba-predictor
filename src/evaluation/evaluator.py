"""評価モジュール

的中率、回収率、Top-N精度などの評価指標を計算します。
"""

from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


class Evaluator:
    """予測モデルの評価クラス"""
    
    def __init__(self):
        pass
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, 
                race_ids: np.ndarray = None, odds: np.ndarray = None) -> Dict[str, float]:
        """総合評価を実行
        
        Args:
            y_true: 実際の着順
            y_pred: 予測着順
            race_ids: レースID（回収率計算用）
            odds: オッズ（回収率計算用）
            
        Returns:
            評価指標の辞書
        """
        results = {}
        
        # 基本的な精度指標
        results['mae'] = self.mean_absolute_error(y_true, y_pred)
        results['rmse'] = self.root_mean_squared_error(y_true, y_pred)
        
        # 的中率
        results['win_accuracy'] = self.win_accuracy(y_true, y_pred)
        results['place_accuracy'] = self.place_accuracy(y_true, y_pred)
        
        # Top-N精度
        results['top3_accuracy'] = self.top_n_accuracy(y_true, y_pred, n=3)
        results['top5_accuracy'] = self.top_n_accuracy(y_true, y_pred, n=5)
        
        # 回収率（オッズ情報がある場合）
        if race_ids is not None and odds is not None:
            results['win_payback_rate'] = self.win_payback_rate(
                y_true, y_pred, race_ids, odds
            )
        
        return results
    
    def mean_absolute_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """平均絶対誤差を計算
        
        Args:
            y_true: 実際の着順
            y_pred: 予測着順
            
        Returns:
            MAE
        """
        return np.mean(np.abs(y_true - y_pred))
    
    def root_mean_squared_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """二乗平均平方根誤差を計算
        
        Args:
            y_true: 実際の着順
            y_pred: 予測着順
            
        Returns:
            RMSE
        """
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    def win_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """1着的中率を計算
        
        Args:
            y_true: 実際の着順
            y_pred: 予測着順
            
        Returns:
            1着的中率（0-1）
        """
        # 予測着順が最も小さい（1着予測）が実際に1着だった割合
        # レース単位でグループ化が必要だが、簡易版として実装
        win_mask = y_true == 1
        pred_win_mask = y_pred == y_pred.min()  # 最小値が予測1着
        
        if win_mask.sum() == 0:
            return 0.0
        
        return np.sum(win_mask & pred_win_mask) / win_mask.sum()
    
    def place_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """複勝的中率を計算（1-3着）
        
        Args:
            y_true: 実際の着順
            y_pred: 予測着順
            
        Returns:
            複勝的中率（0-1）
        """
        place_mask = y_true <= 3
        pred_place_mask = y_pred <= 3
        
        if place_mask.sum() == 0:
            return 0.0
        
        return np.sum(place_mask & pred_place_mask) / place_mask.sum()
    
    def top_n_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray, n: int = 3) -> float:
        """Top-N精度を計算
        
        実際の上位N頭のうち、予測上位N頭に含まれていた割合
        
        Args:
            y_true: 実際の着順
            y_pred: 予測着順
            n: 上位N頭
            
        Returns:
            Top-N精度（0-1）
        """
        top_n_true = y_true <= n
        top_n_pred = y_pred <= n
        
        if top_n_true.sum() == 0:
            return 0.0
        
        return np.sum(top_n_true & top_n_pred) / top_n_true.sum()
    
    def win_payback_rate(self, y_true: np.ndarray, y_pred: np.ndarray,
                        race_ids: np.ndarray, odds: np.ndarray,
                        bet_amount: float = 100.0) -> float:
        """単勝回収率を計算
        
        予測1着の馬に単勝を購入した場合のシミュレーション
        
        Args:
            y_true: 実際の着順
            y_pred: 予測着順
            race_ids: レースID
            odds: オッズ
            bet_amount: 購入金額
            
        Returns:
            回収率（%）
        """
        df = pd.DataFrame({
            'race_id': race_ids,
            'y_true': y_true,
            'y_pred': y_pred,
            'odds': odds
        })
        
        total_bet = 0
        total_return = 0
        
        # レースごとに処理
        for race_id in df['race_id'].unique():
            race_df = df[df['race_id'] == race_id]
            
            # 予測1着の馬を選択
            pred_winner_idx = race_df['y_pred'].idxmin()
            pred_winner = race_df.loc[pred_winner_idx]
            
            # ベット
            total_bet += bet_amount
            
            # 的中したら払い戻し
            if pred_winner['y_true'] == 1:
                total_return += bet_amount * pred_winner['odds']
        
        if total_bet == 0:
            return 0.0
        
        return (total_return / total_bet) * 100
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                            max_position: int = 10, figsize: tuple = (10, 8)):
        """混同行列をプロット
        
        Args:
            y_true: 実際の着順
            y_pred: 予測着順
            max_position: 表示する最大着順
            figsize: 図のサイズ
        """
        # 着順を丸める
        y_true_rounded = np.round(y_true).astype(int)
        y_pred_rounded = np.round(y_pred).astype(int)
        
        # 範囲を制限
        mask = (y_true_rounded <= max_position) & (y_pred_rounded <= max_position)
        y_true_filtered = y_true_rounded[mask]
        y_pred_filtered = y_pred_rounded[mask]
        
        # 混同行列を作成
        cm = confusion_matrix(y_true_filtered, y_pred_filtered,
                            labels=list(range(1, max_position + 1)))
        
        # プロット
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=range(1, max_position + 1),
                   yticklabels=range(1, max_position + 1))
        plt.xlabel('予測着順')
        plt.ylabel('実際の着順')
        plt.title('着順予測の混同行列')
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_accuracy_by_position(self, y_true: np.ndarray, y_pred: np.ndarray,
                                 max_position: int = 10, figsize: tuple = (10, 6)):
        """着順ごとの予測精度をプロット
        
        Args:
            y_true: 実際の着順
            y_pred: 予測着順
            max_position: 表示する最大着順
            figsize: 図のサイズ
        """
        y_true_rounded = np.round(y_true).astype(int)
        y_pred_rounded = np.round(y_pred).astype(int)
        
        accuracies = []
        positions = []
        
        for pos in range(1, max_position + 1):
            mask = y_true_rounded == pos
            if mask.sum() > 0:
                accuracy = np.mean(y_pred_rounded[mask] == pos)
                accuracies.append(accuracy)
                positions.append(pos)
        
        plt.figure(figsize=figsize)
        plt.bar(positions, accuracies)
        plt.xlabel('着順')
        plt.ylabel('的中率')
        plt.title('着順ごとの予測精度')
        plt.xticks(positions)
        plt.ylim(0, 1)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()
