"""
競馬予測AIシステムの使用例

このスクリプトは、システムの主要な機能の使い方を示します。
"""

import pandas as pd
import numpy as np
from pathlib import Path

# モジュールのインポート
from src.scraper.netkeiba import NetkeibaScraper
from src.preprocessing.cleaner import DataCleaner
from src.features.engineer import FeatureEngineer
from src.models.lightgbm_model import LightGBMModel
from src.evaluation.evaluator import Evaluator


def create_sample_data():
    """デモ用のサンプルデータを作成"""
    print("=== サンプルデータ作成 ===")
    
    np.random.seed(42)
    n_samples = 200
    
    # サンプルデータ生成
    data = {
        'race_id': [f'R{i//10:03d}' for i in range(n_samples)],
        'horse_id': [f'H{i%20:03d}' for i in range(n_samples)],
        'horse_name': [f'馬{i%20}号' for i in range(n_samples)],
        'finish_position': np.random.randint(1, 11, n_samples),
        'time': [f"1:{30 + np.random.randint(0, 30)}.{np.random.randint(0, 10)}" for _ in range(n_samples)],
        'sex_age': [f"{np.random.choice(['牡', '牝'])}{np.random.randint(3, 8)}" for _ in range(n_samples)],
        'course_type': np.random.choice(['芝', 'ダート'], n_samples),
        'distance': np.random.choice([1600, 1800, 2000, 2400], n_samples),
        'frame_number': np.random.randint(1, 9, n_samples),
        'horse_number': np.random.randint(1, 19, n_samples),
        'weight': np.random.randint(52, 60, n_samples),
        'weather': np.random.choice(['晴', '曇', '雨'], n_samples),
        'track_condition': np.random.choice(['良', '稍重', '重', '不良'], n_samples),
        'jockey_id': [f'J{i%10:03d}' for i in range(n_samples)],
        'jockey_name': [f'騎手{i%10}' for i in range(n_samples)],
        'odds': np.round(np.random.uniform(1.5, 50, n_samples), 1),
        'popularity': np.random.randint(1, 11, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # データ保存
    output_dir = Path('data/raw')
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / 'sample_races.csv', index=False)
    
    print(f"✓ サンプルデータ作成完了: {len(df)}件")
    print(f"  レース数: {df['race_id'].nunique()}")
    print(f"  馬数: {df['horse_id'].nunique()}")
    
    return df


def preprocess_data(df):
    """データ前処理"""
    print("\n=== データ前処理 ===")
    
    cleaner = DataCleaner()
    df_cleaned = cleaner.clean(df)
    
    print(f"✓ 前処理完了: {len(df_cleaned)}件")
    print(f"  カラム数: {len(df_cleaned.columns)}")
    
    return df_cleaned, cleaner


def create_features(df_cleaned):
    """特徴量生成"""
    print("\n=== 特徴量生成 ===")
    
    engineer = FeatureEngineer()
    df_featured = engineer.create_features(df_cleaned)
    
    feature_names = engineer.get_feature_names(df_featured)
    
    print(f"✓ 特徴量生成完了")
    print(f"  総特徴量数: {len(feature_names)}")
    print(f"  主要な特徴量: {feature_names[:10]}")
    
    # 保存
    output_dir = Path('data/processed')
    output_dir.mkdir(parents=True, exist_ok=True)
    df_featured.to_csv(output_dir / 'featured_data.csv', index=False)
    
    return df_featured, engineer


def train_model(X, y):
    """モデル学習"""
    print("\n=== モデル学習 ===")
    
    model = LightGBMModel(random_state=42)
    print("LightGBMモデルで学習中...")
    
    model.train(
        X, y,
        n_splits=2,  # デモ用に少なめ
        num_boost_round=100,
        early_stopping_rounds=10
    )
    
    print("✓ 学習完了")
    
    # モデル保存
    model_dir = Path('models')
    model_dir.mkdir(exist_ok=True)
    model.save(model_dir / 'demo_model.joblib')
    print(f"  モデル保存: models/demo_model.joblib")
    
    return model


def evaluate_model(model, X, y, df):
    """モデル評価"""
    print("\n=== モデル評価 ===")
    
    # 予測
    predictions = model.predict(X)
    
    # 評価
    evaluator = Evaluator()
    
    race_ids = df['race_id'].values if 'race_id' in df.columns else None
    odds = df['odds'].values if 'odds' in df.columns else None
    
    metrics = evaluator.evaluate(y, predictions, race_ids, odds)
    
    print("✓ 評価完了")
    print(f"  MAE (平均絶対誤差): {metrics['mae']:.3f}")
    print(f"  RMSE (二乗平均平方根誤差): {metrics['rmse']:.3f}")
    print(f"  Top-3精度: {metrics['top3_accuracy']:.2%}")
    print(f"  Top-5精度: {metrics['top5_accuracy']:.2%}")
    
    if 'win_payback_rate' in metrics:
        print(f"  単勝回収率: {metrics['win_payback_rate']:.1f}%")
    
    return metrics


def main():
    """メイン処理"""
    print("=" * 60)
    print("競馬予測AIシステム - デモンストレーション")
    print("=" * 60)
    
    # 1. サンプルデータ作成
    df = create_sample_data()
    
    # 2. データ前処理
    df_cleaned, cleaner = preprocess_data(df)
    
    # 3. 特徴量生成
    df_featured, engineer = create_features(df_cleaned)
    
    # 4. データ準備
    feature_names = engineer.get_feature_names(df_featured)
    X = df_featured[feature_names].fillna(0)
    y = df_featured['finish_position']
    
    print(f"\n=== データ準備完了 ===")
    print(f"  特徴量数: {len(feature_names)}")
    print(f"  サンプル数: {len(X)}")
    
    # 5. モデル学習
    model = train_model(X, y)
    
    # 6. モデル評価
    metrics = evaluate_model(model, X, y, df_featured)
    
    # 7. まとめ
    print("\n" + "=" * 60)
    print("デモンストレーション完了！")
    print("=" * 60)
    print("\n次のステップ:")
    print("1. Streamlit UIを起動: streamlit run src/app/streamlit_app.py")
    print("2. 実データでスクレイピング: src/scraper/netkeiba.py")
    print("3. モデルのチューニング: ハイパーパラメータ調整")
    print("\n詳細はREADME.mdをご覧ください。")


if __name__ == "__main__":
    main()
