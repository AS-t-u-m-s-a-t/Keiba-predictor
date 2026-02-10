"""Streamlitダッシュボード

競馬予測AIのWeb UIを提供します。
"""

import sys
from pathlib import Path

# プロジェクトのルートディレクトリをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from src.preprocessing.cleaner import DataCleaner
from src.features.engineer import FeatureEngineer
from src.models.lightgbm_model import LightGBMModel
from src.models.neural_net import NeuralNetModel
from src.models.lambdamart import LambdaMARTModel
from src.evaluation.evaluator import Evaluator


# ページ設定
st.set_page_config(
    page_title="競馬予測AI",
    page_icon="🏇",
    layout="wide",
    initial_sidebar_state="expanded"
)

# タイトル
st.title("🏇 競馬予測AIシステム")
st.markdown("---")


def load_model(model_type: str, model_path: str = None):
    """モデルを読み込み
    
    Args:
        model_type: モデルタイプ
        model_path: モデルファイルのパス
        
    Returns:
        モデルインスタンス
    """
    if model_type == "LightGBM":
        model = LightGBMModel()
        if model_path:
            try:
                model.load(model_path)
                st.success(f"モデルを読み込みました: {model_path}")
            except Exception as e:
                st.error(f"モデルの読み込みに失敗しました: {e}")
        return model
    elif model_type == "ニューラルネットワーク":
        return NeuralNetModel()
    elif model_type == "LambdaMART":
        return LambdaMARTModel()
    else:
        return None


def preprocess_data(df: pd.DataFrame):
    """データを前処理
    
    Args:
        df: 生データ
        
    Returns:
        前処理済みデータ
    """
    cleaner = DataCleaner()
    engineer = FeatureEngineer()
    
    with st.spinner("データをクリーニング中..."):
        df_cleaned = cleaner.clean(df)
    
    with st.spinner("特徴量を生成中..."):
        df_featured = engineer.create_features(df_cleaned)
    
    return df_featured, engineer


# サイドバー
st.sidebar.header("⚙️ 設定")

# モデル選択
model_type = st.sidebar.selectbox(
    "モデルを選択",
    ["LightGBM", "ニューラルネットワーク", "LambdaMART"],
    help="予測に使用するモデルを選択します"
)

# モデル読み込み
model_file = st.sidebar.file_uploader(
    "学習済みモデルをアップロード（任意）",
    type=['joblib', 'pkl']
)

model_path = None
if model_file:
    # 一時ファイルとして保存
    temp_path = f"/tmp/{model_file.name}"
    with open(temp_path, 'wb') as f:
        f.write(model_file.read())
    model_path = temp_path

model = load_model(model_type, model_path)

# モデルが未実装の場合の警告
if model_type != "LightGBM":
    st.sidebar.warning(f"⚠️ {model_type}は現在未実装です。LightGBMをご使用ください。")

st.sidebar.markdown("---")

# データアップロード
st.header("📊 データアップロード")

uploaded_file = st.file_uploader(
    "レースデータのCSVファイルをアップロード",
    type=['csv']
)

if uploaded_file is not None:
    # データ読み込み
    df = pd.read_csv(uploaded_file)
    
    st.success(f"データを読み込みました: {len(df)}行")
    
    # データプレビュー
    with st.expander("データプレビュー"):
        st.dataframe(df.head(20))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("総レコード数", len(df))
        with col2:
            st.metric("カラム数", len(df.columns))
        with col3:
            st.metric("レースID数", df['race_id'].nunique() if 'race_id' in df.columns else 0)
    
    # 前処理
    st.header("🔧 データ前処理")
    
    if st.button("前処理を実行"):
        try:
            df_processed, engineer = preprocess_data(df)
            st.session_state['df_processed'] = df_processed
            st.session_state['engineer'] = engineer
            st.success("前処理が完了しました！")
            
            # 処理後のデータプレビュー
            with st.expander("前処理後のデータ"):
                st.dataframe(df_processed.head(20))
                st.info(f"特徴量数: {len(df_processed.columns)}")
        
        except Exception as e:
            st.error(f"前処理中にエラーが発生しました: {e}")
    
    # 予測
    if 'df_processed' in st.session_state:
        st.header("🎯 予測")
        
        df_processed = st.session_state['df_processed']
        engineer = st.session_state['engineer']
        
        if model_type == "LightGBM" and model.is_trained:
            if st.button("予測を実行"):
                try:
                    # 特徴量を取得
                    feature_names = engineer.get_feature_names(df_processed)
                    X = df_processed[feature_names]
                    
                    # 予測
                    with st.spinner("予測中..."):
                        predictions = model.predict(X)
                        win_proba = model.predict_proba_win(X)
                    
                    # 結果を追加
                    df_results = df_processed.copy()
                    df_results['予測着順'] = predictions
                    df_results['勝率予測'] = win_proba
                    
                    st.success("予測が完了しました！")
                    
                    # レース単位で結果を表示
                    if 'race_id' in df_results.columns:
                        selected_race = st.selectbox(
                            "表示するレースを選択",
                            df_results['race_id'].unique()
                        )
                        
                        race_results = df_results[df_results['race_id'] == selected_race].copy()
                        race_results = race_results.sort_values('予測着順')
                        
                        # 表示用のカラムを選択
                        display_cols = ['horse_name', '予測着順', '勝率予測']
                        if 'finish_position' in race_results.columns:
                            display_cols.append('finish_position')
                        if 'odds' in race_results.columns:
                            display_cols.append('odds')
                        
                        available_cols = [col for col in display_cols if col in race_results.columns]
                        
                        st.subheader(f"📋 レース {selected_race} の予測結果")
                        st.dataframe(
                            race_results[available_cols].reset_index(drop=True),
                            use_container_width=True
                        )
                        
                        # 勝率予測のグラフ
                        fig = px.bar(
                            race_results,
                            x='horse_name',
                            y='勝率予測',
                            title='各馬の勝率予測',
                            labels={'勝率予測': '勝率', 'horse_name': '馬名'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                except Exception as e:
                    st.error(f"予測中にエラーが発生しました: {e}")
        else:
            st.info("学習済みモデルをアップロードしてください。")
        
        # 評価（実際の着順がある場合）
        if 'finish_position' in df_processed.columns and model_type == "LightGBM" and model.is_trained:
            st.header("📈 モデル評価")
            
            if st.button("評価を実行"):
                try:
                    feature_names = engineer.get_feature_names(df_processed)
                    X = df_processed[feature_names]
                    y_true = df_processed['finish_position']
                    
                    # 予測
                    y_pred = model.predict(X)
                    
                    # 評価
                    evaluator = Evaluator()
                    
                    race_ids = df_processed['race_id'].values if 'race_id' in df_processed.columns else None
                    odds = df_processed['odds'].values if 'odds' in df_processed.columns else None
                    
                    metrics = evaluator.evaluate(y_true, y_pred, race_ids, odds)
                    
                    # メトリクスを表示
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("MAE", f"{metrics['mae']:.3f}")
                    with col2:
                        st.metric("RMSE", f"{metrics['rmse']:.3f}")
                    with col3:
                        st.metric("Top-3精度", f"{metrics['top3_accuracy']:.2%}")
                    with col4:
                        st.metric("Top-5精度", f"{metrics['top5_accuracy']:.2%}")
                    
                    if 'win_payback_rate' in metrics:
                        st.metric("単勝回収率", f"{metrics['win_payback_rate']:.1f}%")
                    
                    # 混同行列
                    st.subheader("混同行列")
                    fig = evaluator.plot_confusion_matrix(y_true, y_pred)
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"評価中にエラーが発生しました: {e}")

# 特徴量重要度
if model_type == "LightGBM" and model.is_trained:
    st.header("📊 特徴量重要度")
    
    if st.button("特徴量重要度を表示"):
        try:
            fig = model.plot_feature_importance()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"特徴量重要度の表示中にエラーが発生しました: {e}")

# フッター
st.markdown("---")
st.markdown("""
### 📝 使い方
1. サイドバーでモデルを選択
2. 学習済みモデルをアップロード（任意）
3. レースデータのCSVをアップロード
4. 前処理を実行
5. 予測を実行
6. 結果を確認

### ℹ️ システム情報
- モデル: Strategyパターンで実装（拡張可能）
- 現在実装済み: LightGBM
- 将来対応予定: ニューラルネットワーク、LambdaMART
""")
